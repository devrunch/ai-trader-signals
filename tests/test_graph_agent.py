from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.config import get_settings
from app.signals.agent.context import StaticTradingContext
from app.signals.agent.toolbox import AgentToolbox


class _StaticMarket:
    def __init__(self, df): self.df = df
    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.df


def _toolbox(trending_frame):
    return AgentToolbox("RELIANCE", "NSE", trending_frame, context=StaticTradingContext({}),
                        market=_StaticMarket(trending_frame), settings=get_settings())


@pytest.mark.asyncio
async def test_valid_on_first_try(trending_frame):
    box = _toolbox(trending_frame)
    with (
        patch("app.signals.agent.tools.graph_agent._write_formula",
              return_value="result = line(ema(close, 20))"),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "20 EMA", "label": "EMA 20"})

    assert result["created"].startswith("DIA_CUSTOM_")
    assert box.results["custom_indicators"][0]["source"] == "result = line(ema(close, 20))"
    assert box.results["custom_indicators"][0]["outputName"] == "result"
    assert box.results["custom_indicators"][0]["displayLabel"] == "EMA 20"


@pytest.mark.asyncio
async def test_invalid_then_valid_on_retry(trending_frame):
    box = _toolbox(trending_frame)
    write_calls = []

    async def fake_write(description: str, feedback: str | None = None) -> str:
        write_calls.append(feedback)
        return "result = line(ema(close, 20))" if feedback else "not diascript at all"

    async def fake_validate(source: str, output_name: str) -> dict:
        if source == "not diascript at all":
            return {"valid": False, "error": {"message": "unexpected token"}}
        return {"valid": True, "outputType": "line"}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "20 EMA"})

    assert len(write_calls) == 2
    assert write_calls[0] is None
    assert write_calls[1] == "unexpected token"
    assert "created" in result


@pytest.mark.asyncio
async def test_invalid_twice_returns_error_without_touching_results(trending_frame):
    box = _toolbox(trending_frame)

    async def fake_write(description: str, feedback: str | None = None) -> str:
        return "still not diascript"

    async def fake_validate(source: str, output_name: str) -> dict:
        return {"valid": False, "error": {"message": "unexpected token"}}

    with (
        patch("app.signals.agent.tools.graph_agent._write_formula", new=fake_write),
        patch("app.signals.agent.tools.graph_agent._validate_via_node", new=fake_validate),
    ):
        result = await box.execute("generate_custom_indicator", {"description": "bad request"})

    assert "error" in result
    assert "custom_indicators" not in box.results


@pytest.mark.asyncio
async def test_missing_description_short_circuits(trending_frame):
    box = _toolbox(trending_frame)
    result = await box.execute("generate_custom_indicator", {})
    assert "error" in result
    assert "custom_indicators" not in box.results


@pytest.mark.asyncio
async def test_second_call_in_the_same_turn_gets_a_distinct_name_and_no_seq_leak(trending_frame):
    """Two calls in one turn must not collide on DIA_CUSTOM_1, and the counter
    that tells them apart must never surface as a key in ctx.results — that
    dict is serialised verbatim as the turn's browser-facing payload."""
    box = _toolbox(trending_frame)
    with (
        patch("app.signals.agent.tools.graph_agent._write_formula",
              return_value="result = line(ema(close, 20))"),
        patch("app.signals.agent.tools.graph_agent._validate_via_node",
              return_value={"valid": True, "outputType": "line"}),
    ):
        first = await box.execute("generate_custom_indicator", {"description": "20 EMA"})
        second = await box.execute("generate_custom_indicator", {"description": "50 EMA"})

    assert first["created"] == "DIA_CUSTOM_1"
    assert second["created"] == "DIA_CUSTOM_2"
    assert len(box.results["custom_indicators"]) == 2
    assert "_custom_indicator_seq" not in box.results


def test_system_prompt_never_offers_barcolor_as_a_safe_output_wrapper():
    """barcolor(...) parses fine — diascript-validate only checks that SOME
    output wrapper was used, it doesn't know which ones the real klinecharts
    render adapter supports. The adapter has no case for barcolor (or fill)
    and throws, so the prompt must never suggest it as something to wrap
    `result` in — only ever list it among what NOT to use."""
    from app.signals.agent.tools.graph_agent import SYSTEM_PROMPT

    wrap_rule, _, do_not_use = SYSTEM_PROMPT.partition("Do NOT use")
    assert "barcolor" not in wrap_rule
    assert "barcolor" in do_not_use


@pytest.mark.asyncio
async def test_a_hung_validator_is_killed_and_reaped_not_left_running():
    """asyncio.wait_for cancels the AWAIT on timeout, not the OS process —
    without an explicit kill+wait, a genuinely hung diascript-validate (the
    exact case the timeout exists for) would leak as an orphaned process on
    every retry."""
    from app.signals.agent.tools import graph_agent

    calls = {"kill": 0, "waited": 0}

    class _HangingProc:
        returncode = None

        async def communicate(self, data):
            await asyncio.sleep(10)  # never resolves before the short timeout below

        def kill(self):
            calls["kill"] += 1

        async def wait(self):
            calls["waited"] += 1

    async def fake_create_subprocess_exec(*args, **kwargs):
        return _HangingProc()

    with (
        patch.object(graph_agent.asyncio, "create_subprocess_exec", fake_create_subprocess_exec),
        patch.object(graph_agent, "VALIDATE_TIMEOUT_SECONDS", 0.05),
    ):
        result = await graph_agent._validate_via_node("result = line(close)", "result")

    assert result == {"valid": False, "error": {"message": "validator unavailable"}}
    assert calls["kill"] == 1
    assert calls["waited"] == 1
