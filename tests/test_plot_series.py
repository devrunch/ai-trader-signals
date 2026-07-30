"""
plot_series — the general escape hatch.

"Draw a 5-bar high/low channel" does not fit `draw_on_chart`'s three fixed
shapes (trendline, support/resistance, fibonacci) or a preset indicator
toggle. Rather than refuse, this exposes the SAME validated series allow-list
`build_strategy` already trusts — the model picks a name and bounded params,
never a formula, and nothing here evaluates anything it was not told to.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.config import get_settings
from app.signals.agent.context import StaticTradingContext
from app.signals.agent.toolbox import AgentToolbox
from app.signals.conditions import available_series


class _StaticMarket:
    def __init__(self, df): self.df = df
    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.df


def _toolbox(trending_frame):
    return AgentToolbox("RELIANCE", "NSE", trending_frame, context=StaticTradingContext({}),
                        market=_StaticMarket(trending_frame), settings=get_settings())


@pytest.mark.asyncio
async def test_the_5_bar_high_low_channel_the_demo_needed(trending_frame):
    """The literal request that motivated this: draw the 5-bar highest high
    and lowest low as two lines."""
    box = _toolbox(trending_frame)

    high = await box.execute("plot_series", {"series": "highest", "params": {"length": 5}, "label": "5-bar high"})
    low = await box.execute("plot_series", {"series": "lowest", "params": {"length": 5}, "label": "5-bar low"})

    assert "error" not in high
    assert "error" not in low
    assert high["points_plotted"] > 0
    assert low["points_plotted"] > 0

    lines = [d for d in box.drawings if d["kind"] == "series"]
    assert {d["label"] for d in lines} == {"5-bar high (highest)", "5-bar low (lowest)"}


@pytest.mark.asyncio
async def test_points_are_real_ohlcv_numbers_not_invented_ones(trending_frame):
    """The rolling max of `high` must never exceed the frame's own true max —
    if it did, the "numbers never come from the model" guarantee would be
    broken somewhere in this path."""
    box = _toolbox(trending_frame)
    out = await box.execute("plot_series", {"series": "highest", "params": {"length": 5}})

    points = [d for d in box.drawings if d["kind"] == "series"][0]["points"]
    true_max = float(trending_frame["high"].max())
    assert all(p["value"] <= true_max + 0.01 for p in points)


@pytest.mark.asyncio
async def test_last_value_matches_the_line_it_just_drew(trending_frame):
    """A model narrating "current SMA" from a SEPARATE get_indicators call risks
    a different interval than what was just drawn — chart says one number,
    prose says another (the bug this closes: a live Keltner-channel request
    drew 15m lines but narrated 1h numbers from a second, mismatched call).
    last_value must be exactly the drawn line's own last point, not a fresh
    computation, so quoting it can never disagree with the chart."""
    box = _toolbox(trending_frame)
    out = await box.execute("plot_series", {"series": "sma", "params": {"length": 20}})

    drawn_points = [d for d in box.drawings if d["kind"] == "series"][0]["points"]
    assert out["last_value"] == drawn_points[-1]["value"]


@pytest.mark.asyncio
async def test_a_false_label_cannot_hide_what_was_really_plotted(trending_frame):
    """The bug this closes: live, the model wanted a Keltner band (SMA +/- 2xATR
    — not computable, no arithmetic between series), hit the per-turn call
    limit on a third attempt, then relabeled a plain `close` line "Keltner
    Upper (SMA + 2xATR)". The number was real; the name was a lie. The true
    series name must survive in the label no matter what the model calls it."""
    box = _toolbox(trending_frame)
    out = await box.execute("plot_series", {
        "series": "close", "label": "Keltner Upper (SMA + 2x ATR)",
    })
    assert "error" not in out
    assert out["drawn"] == "close"

    drawing = [d for d in box.drawings if d["kind"] == "series"][0]
    assert drawing["label"] == "Keltner Upper (SMA + 2x ATR) (close)"


@pytest.mark.asyncio
async def test_an_unknown_series_name_lists_what_is_actually_available(trending_frame):
    """The model's feedback loop — same principle as build_strategy's rejected
    specs: say exactly what was wrong so it can retry with a real name."""
    out = await _toolbox(trending_frame).execute("plot_series", {"series": "made_up_thing"})

    assert "error" in out
    assert out["available_series"] == available_series()


@pytest.mark.asyncio
async def test_every_advertised_series_name_actually_plots(trending_frame):
    """The tool schema lists 24 names by hand — this is what stops the list
    drifting from what SERIES actually contains.

    A fresh toolbox per name, not one shared turn: the per-turn call budget
    (`max_calls_per_tool`) caps any single tool at 3 calls, and calling the
    same tool 24 times in one real turn would never happen anyway."""
    for name in available_series():
        out = await _toolbox(trending_frame).execute("plot_series", {"series": name})
        assert "error" not in out, f"{name}: {out.get('error')}"


@pytest.mark.asyncio
async def test_params_are_bounded_the_same_way_a_strategy_spec_is(trending_frame):
    """A caller here gets no more trust than build_strategy's condition tree —
    same _check_params gate, same bounds."""
    out = await _toolbox(trending_frame).execute("plot_series", {
        "series": "highest", "params": {"length": 10_000_000},
    })
    assert "error" in out


@pytest.mark.asyncio
async def test_an_unknown_parameter_name_is_rejected_not_ignored(trending_frame):
    out = await _toolbox(trending_frame).execute("plot_series", {
        "series": "sma", "params": {"lookback": 20},  # not a real param name
    })
    assert "error" in out


@pytest.mark.asyncio
async def test_too_little_history_is_a_clear_error_not_an_empty_line(trending_frame):
    short = trending_frame.tail(2)
    out = await _toolbox(short).execute("plot_series", {"series": "sma", "params": {"length": 50}})
    assert "error" in out


@pytest.mark.asyncio
async def test_the_line_is_capped_so_the_chart_stays_readable(trending_frame):
    """Same principle as MAX_MARKERS for trade markers — every point becomes
    its own chart overlay object on the frontend, so this is a real cost,
    not just a display preference."""
    from app.signals.agent.tools import chart as chart_tools

    long_frame = trending_frame
    while len(long_frame) < chart_tools.MAX_SERIES_POINTS + 100:
        long_frame = pd.concat([long_frame, trending_frame])

    box = _toolbox(long_frame.reset_index(drop=True).set_axis(
        pd.date_range("2024-01-01", periods=len(long_frame), freq="15min"), axis=0
    ))
    out = await box.execute("plot_series", {"series": "close"})
    assert out["points_plotted"] <= chart_tools.MAX_SERIES_POINTS
