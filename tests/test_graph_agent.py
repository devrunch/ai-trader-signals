"""generate_custom_indicator now writes Pine, not diascript. _write_formula
(the LLM call) is mocked throughout -- consistent with the rest of this
suite's history -- but check_pine_source runs for real against the real
sandbox (app/pine_sandbox/), so these tests prove the retry loop and the
source-text gate actually work end to end, not just that the mocks were
wired right.

FORBIDDEN_CALLS/FORBIDDEN_NAMESPACES/forbidden_call_feedback/synthetic_bars
now live in pine_validation.py, shared with chart_indicators.py's
edit_indicator_source -- both need the identical gate.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.config import get_settings
from app.signals.agent.context import StaticTradingContext
from app.signals.agent.toolbox import AgentToolbox
from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT
from app.signals.agent.tools.pine_validation import (
    FORBIDDEN_CALLS,
    FORBIDDEN_NAMESPACES,
    forbidden_call_feedback,
    synthetic_bars,
)


class _StaticMarket:
    def __init__(self, df):
        self.df = df

    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.df


def _toolbox(trending_frame, llm=None, budget=None):
    return AgentToolbox("RELIANCE", "NSE", trending_frame, context=StaticTradingContext({}),
                        market=_StaticMarket(trending_frame), settings=get_settings(),
                        llm=llm, budget=budget)


GOOD_SOURCE = '//@version=5\nindicator("t")\nplot(ta.ema(close, 20), "EMA20")'
BAD_PLOT_ONLY_BGCOLOR = '//@version=5\nindicator("t")\nbgcolor(close < ta.sma(close, 50) ? color.red : na)'
SYNTAX_ERROR_SOURCE = "this is not pine at all @#$%"


class TestToolFlow:
    @pytest.mark.asyncio
    async def test_valid_on_first_try(self, trending_frame):
        box = _toolbox(trending_frame)
        with patch("app.signals.agent.tools.graph_agent._write_formula", return_value=("main", GOOD_SOURCE)):
            result = await box.execute("generate_custom_indicator", {"description": "20 EMA", "label": "EMA 20"})

        assert result["created"].startswith("pine_")
        assert result["pane"] == "main"
        spec = box.results["custom_indicators"][0]
        assert spec["source"] == GOOD_SOURCE
        assert spec["label"] == "EMA 20"
        assert spec["pane"] == "main"
        assert spec["id"] == result["created"]

    @pytest.mark.asyncio
    async def test_forbidden_call_then_valid_on_retry(self, trending_frame):
        box = _toolbox(trending_frame)
        write_calls = []

        async def fake_write(ctx, description, feedback=None, source=None):
            write_calls.append(feedback)
            return ("main", GOOD_SOURCE) if feedback else ("main", BAD_PLOT_ONLY_BGCOLOR)

        with patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write):
            result = await box.execute("generate_custom_indicator", {"description": "20 EMA"})

        assert len(write_calls) == 2
        assert write_calls[0] is None
        assert "bgcolor(" in write_calls[1]
        assert "created" in result

    @pytest.mark.asyncio
    async def test_syntax_error_then_valid_on_retry(self, trending_frame):
        box = _toolbox(trending_frame)
        write_calls = []

        async def fake_write(ctx, description, feedback=None, source=None):
            write_calls.append(feedback)
            return ("main", GOOD_SOURCE) if feedback else ("main", SYNTAX_ERROR_SOURCE)

        with patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write):
            result = await box.execute("generate_custom_indicator", {"description": "20 EMA"})

        assert len(write_calls) == 2
        assert write_calls[0] is None
        assert write_calls[1]  # the real sandbox's own parse-error message
        assert "created" in result

    @pytest.mark.asyncio
    async def test_invalid_twice_returns_error_without_touching_results(self, trending_frame):
        box = _toolbox(trending_frame)
        with patch("app.signals.agent.tools.graph_agent._write_formula", return_value=("main", SYNTAX_ERROR_SOURCE)):
            result = await box.execute("generate_custom_indicator", {"description": "bad request"})

        assert "error" in result
        assert "custom_indicators" not in box.results

    @pytest.mark.asyncio
    async def test_missing_description_short_circuits(self, trending_frame):
        box = _toolbox(trending_frame)
        result = await box.execute("generate_custom_indicator", {})
        assert "error" in result
        assert "custom_indicators" not in box.results

    @pytest.mark.asyncio
    async def test_no_plot_output_is_rejected(self, trending_frame):
        """A script that runs cleanly but never calls plot() (e.g. only
        computed a value and never displayed it) must not be accepted --
        there is nothing for the chart to show."""
        box = _toolbox(trending_frame)
        no_plot_source = '//@version=5\nindicator("t")\nx = ta.ema(close, 20)'
        with patch("app.signals.agent.tools.graph_agent._write_formula", return_value=("main", no_plot_source)):
            result = await box.execute("generate_custom_indicator", {"description": "nothing"})

        assert "error" in result
        assert "custom_indicators" not in box.results

    @pytest.mark.asyncio
    async def test_content_hashing_makes_identical_sources_idempotent(self, trending_frame):
        box = _toolbox(trending_frame)
        with patch("app.signals.agent.tools.graph_agent._write_formula", return_value=("main", GOOD_SOURCE)):
            first = await box.execute("generate_custom_indicator", {"description": "20 EMA"})
            second = await box.execute("generate_custom_indicator", {"description": "20 EMA"})

        assert first["created"] == second["created"]


class TestForbiddenCallGate:
    """A source-text check, not just a prompt rule -- see FORBIDDEN_CALLS'
    own comment in graph_agent.py for why this needs to be real."""

    def test_clean_source_passes(self):
        assert forbidden_call_feedback(GOOD_SOURCE) is None

    @pytest.mark.parametrize("call", FORBIDDEN_CALLS)
    def test_each_forbidden_call_is_rejected(self, call):
        source = f'//@version=5\nindicator("t")\n{call}close)'
        feedback = forbidden_call_feedback(source)
        assert feedback is not None
        assert call in feedback

    @pytest.mark.parametrize("ns", FORBIDDEN_NAMESPACES)
    def test_each_forbidden_namespace_is_rejected(self, ns):
        source = f'//@version=5\nindicator("t")\nx = {ns}foo'
        feedback = forbidden_call_feedback(source)
        assert feedback is not None
        assert ns in feedback

    @pytest.mark.parametrize("source", [
        'p1 = plot(close, "A")\np2 = plot(open, "B")\nfill(p1, p2, color=color.new(color.blue, 85))',
        'plotshape(close > open, title="Up")',
        'length = input.int(20, title="Length")\nplot(ta.sma(close, length), "SMA")',
    ])
    def test_now_allowed_constructs_pass(self, source):
        """fill()/plotshape()/input.*() all render for real now (fill and
        plotshape were built and verified earlier this session; input.*()
        once the settings gear existed to consume it) -- must not be
        rejected by the same gate that still blocks bgcolor()/plotchar()/
        plotarrow()/strategy.*()/request.security()."""
        assert forbidden_call_feedback(f'//@version=5\nindicator("t")\n{source}') is None


class TestSyntheticBars:
    def test_produces_a_non_degenerate_deterministic_series(self):
        bars = synthetic_bars(80)
        assert len(bars) == 80
        closes = [b["close"] for b in bars]
        assert len(set(closes)) > 1  # not a flat run
        for b in bars:
            assert b["high"] >= max(b["open"], b["close"])
            assert b["low"] <= min(b["open"], b["close"])
            assert b["openTime"] > 0

        # Deterministic -- a fixed seed, not real randomness -- so a retry
        # against the same description reproduces the same synthetic check.
        assert synthetic_bars(80) == bars


class TestSystemPromptContent:
    def test_names_the_never_fake_sophistication_rule_and_its_techniques(self):
        assert "Never fake sophistication" in SYSTEM_PROMPT
        assert "Gaussian" in SYSTEM_PROMPT
        assert "wavelet" in SYSTEM_PROMPT.lower()
        assert "smart money concepts" in SYSTEM_PROMPT.lower()

    def test_states_the_band_naming_convention(self):
        assert '"<Name> Upper"' in SYSTEM_PROMPT or "<Name> Upper" in SYSTEM_PROMPT

    def test_describes_fill_and_plotshape_as_supported(self):
        """Both render for real now (built and verified this session) --
        the prompt must say so, not still ban them."""
        assert "fill(p1, p2, color)" in SYSTEM_PROMPT
        assert "plotshape()" in SYSTEM_PROMPT
        assert "fill(), bgcolor()" not in SYSTEM_PROMPT  # the old combined ban

    def test_describes_input_as_supported(self):
        """The settings gear (Inputs tab) is real now -- the prompt must
        tell the model to use input.*() for tunable values, not still ban
        it as needing a UI that doesn't exist."""
        assert "input.int()" in SYSTEM_PROMPT or "input.*()" in SYSTEM_PROMPT
        assert "settings panel" in SYSTEM_PROMPT or "gear" in SYSTEM_PROMPT

    def test_still_forbids_the_calls_with_no_renderer(self):
        for call in ("bgcolor(", "plotchar(", "plotarrow("):
            assert call in SYSTEM_PROMPT

    def test_forbids_strategy_and_request_security(self):
        assert "strategy.*" in SYSTEM_PROMPT or "strategy." in SYSTEM_PROMPT
        assert "request.security" in SYSTEM_PROMPT


@pytest.mark.asyncio
class TestPromptWorkedExamplesValidateForReal:
    """The worked examples in SYSTEM_PROMPT are what the model pattern-matches
    against -- if one doesn't actually validate against the real sandbox,
    every real request built from it would fail too. Same rigor this
    session already applied to diascript's worked examples."""

    async def _run(self, source: str) -> dict:
        from app.signals.pine.sandbox import run_pine_script
        return await run_pine_script(source, synthetic_bars(60), mode="indicator")

    async def test_ema_diff(self):
        result = await self._run('//@version=5\nindicator("EMA Diff", overlay=false)\nplot(ta.ema(close, 20) - ta.ema(close, 50), "EMA Diff")')
        assert result["ok"] is True
        assert "EMA Diff" in result["plots"]

    async def test_rsi(self):
        result = await self._run('//@version=5\nindicator("RSI 21", overlay=false)\nplot(ta.rsi(close, 21), "RSI")')
        assert result["ok"] is True
        assert "RSI" in result["plots"]

    async def test_gaussian_filter(self):
        source = (
            '//@version=5\nindicator("Gaussian Filter", overlay=true)\n'
            "length = 9\nsigma = 3.0\nsum_w = 0.0\nsum_wv = 0.0\n"
            "for k = 0 to length - 1\n"
            "    w = math.exp(-(k * k) / (2 * sigma * sigma))\n"
            "    sum_w += w\n    sum_wv += w * close[k]\n"
            'plot(sum_wv / sum_w, "Gaussian")'
        )
        result = await self._run(source)
        assert result["ok"] is True
        assert "Gaussian" in result["plots"]

    async def test_stdev_band(self):
        source = (
            '//@version=5\nindicator("StdDev Band", overlay=true)\n'
            "dev = ta.stdev(close, 20) * 2\n"
            'plot(close + dev, "Band Upper")\nplot(close - dev, "Band Lower")'
        )
        result = await self._run(source)
        assert result["ok"] is True
        assert "Band Upper" in result["plots"]
        assert "Band Lower" in result["plots"]

    async def test_wavelet_trend(self):
        source = (
            '//@version=5\nindicator("Wavelet Trend", overlay=true)\n'
            "approx1 = (close + close[1]) / 2\n"
            "approx2 = (approx1 + approx1[2]) / 2\n"
            'plot(approx2, "Trend")'
        )
        result = await self._run(source)
        assert result["ok"] is True
        assert "Trend" in result["plots"]

    async def test_swing_high_line(self):
        source = (
            '//@version=5\nindicator("Swing High", overlay=true)\n'
            "isSwingHigh = high == ta.highest(high, 10)\n"
            "var float lastSwingHigh = na\n"
            "if isSwingHigh\n    lastSwingHigh := high\n"
            'plot(lastSwingHigh, "Last Swing High")'
        )
        result = await self._run(source)
        assert result["ok"] is True
        assert "Last Swing High" in result["plots"]
