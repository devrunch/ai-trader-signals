"""
The condition engine — the DSL the agent emits instead of code.

The security tests come first and matter most. This module exists so an LLM can
describe a strategy WITHOUT the server executing model-written code; if
validation lets something unexpected through, that property is gone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.signals import conditions as C


@pytest.fixture
def frame() -> pd.DataFrame:
    n = 300
    idx = pd.date_range("2026-01-05 09:15", periods=n, freq="15min")
    rng = np.random.default_rng(11)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(close.iloc[0]), "high": close + 0.7,
         "low": close - 0.7, "close": close, "volume": 1000.0},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Safety — the reason this module exists
# ---------------------------------------------------------------------------

def test_unknown_indicator_is_rejected_not_ignored():
    with pytest.raises(C.SpecError, match="Unknown indicator"):
        C.validate_condition({"indicator": "__import__", "op": "<", "value": 1})


def test_indicator_lookup_cannot_reach_arbitrary_attributes():
    """The allow-list is a dict, never getattr on the TA library."""
    for hostile in ("__class__", "__globals__", "os.system", "eval"):
        with pytest.raises(C.SpecError):
            C.validate_condition({"indicator": hostile, "op": ">", "value": 0})


def test_unknown_fields_are_rejected_rather_than_silently_dropped():
    """A typo must fail loudly. Ignoring it would evaluate something other than
    what the user asked for, and report it as if it were the same thing."""
    with pytest.raises(C.SpecError, match="Unexpected fields"):
        C.validate_condition({"indicator": "rsi", "op": "<", "value": 30, "lenght": 14})


def test_out_of_range_parameters_are_rejected():
    with pytest.raises(C.SpecError, match="out of range"):
        C.validate_condition({"indicator": "ema", "params": {"length": 10_000_000},
                              "op": "above", "compare_to": "close"})


def test_unknown_parameter_names_are_rejected():
    with pytest.raises(C.SpecError, match="Unknown parameter"):
        C.validate_condition({"indicator": "rsi", "params": {"window": 14}, "op": "<", "value": 30})


def test_deeply_nested_trees_are_rejected():
    node: dict = {"indicator": "rsi", "op": "<", "value": 30}
    for _ in range(C.MAX_DEPTH + 2):
        node = {"all": [node]}
    with pytest.raises(C.SpecError, match="too deep"):
        C.validate_condition(node)


def test_oversized_trees_are_rejected():
    leaf = {"indicator": "rsi", "op": "<", "value": 30}
    with pytest.raises(C.SpecError, match="too large"):
        C.validate_condition({"all": [dict(leaf) for _ in range(C.MAX_NODES + 5)]})


def test_a_condition_needs_exactly_one_of_value_or_compare_to():
    with pytest.raises(C.SpecError, match="exactly one"):
        C.validate_condition({"indicator": "rsi", "op": "<", "value": 30, "compare_to": "close"})
    with pytest.raises(C.SpecError, match="exactly one"):
        C.validate_condition({"indicator": "rsi", "op": "<"})


def test_unknown_operators_are_rejected():
    with pytest.raises(C.SpecError, match="Unknown operator"):
        C.validate_condition({"indicator": "rsi", "op": "approaches", "value": 30})


def test_short_strategies_are_refused_because_the_account_cannot_short():
    with pytest.raises(C.SpecError, match="long"):
        C.validate_strategy({"entry": {"indicator": "rsi", "op": ">", "value": 70},
                             "exit": {"indicator": "rsi", "op": "<", "value": 30},
                             "side": "short"})


def test_a_strategy_needs_both_entry_and_exit():
    with pytest.raises(C.SpecError, match="both"):
        C.validate_strategy({"entry": {"indicator": "rsi", "op": "<", "value": 30}})


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_a_simple_threshold_evaluates_to_a_boolean_series(frame):
    out = C.evaluate_condition(frame, {"indicator": "close", "op": ">", "value": 0})
    assert out.dtype == bool
    assert len(out) == len(frame)
    assert out.all()


def test_all_is_intersection_and_any_is_union(frame):
    a = {"indicator": "close", "op": ">", "value": 0}       # always true
    b = {"indicator": "close", "op": "<", "value": 0}       # always false
    assert C.evaluate_condition(frame, {"all": [a, b]}).sum() == 0
    assert C.evaluate_condition(frame, {"any": [a, b]}).all()


def test_not_inverts(frame):
    always = {"indicator": "close", "op": ">", "value": 0}
    assert not C.evaluate_condition(frame, {"not": always}).any()


def test_warmup_bars_evaluate_false_rather_than_nan(frame):
    """A rule cannot be said to hold on data that does not exist yet, and False
    is the conservative reading — it declines to trade rather than inventing."""
    out = C.evaluate_condition(frame, {"indicator": "ema", "params": {"length": 200},
                                       "op": "above", "compare_to": "close"})
    assert out.dtype == bool
    assert not out.iloc[:100].any()


def test_crosses_above_fires_only_on_the_crossing_bar():
    idx = pd.date_range("2026-01-05 09:15", periods=6, freq="15min")
    # close crosses the constant 100 exactly once, upward.
    close = pd.Series([98, 99, 101, 102, 103, 104], index=idx, dtype="float64")
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 1000.0}, index=idx)
    out = C.evaluate_condition(df, {"indicator": "close", "op": "crosses_above", "value": 100})
    assert out.sum() == 1
    assert bool(out.iloc[2])


def test_crosses_below_is_the_mirror():
    idx = pd.date_range("2026-01-05 09:15", periods=6, freq="15min")
    close = pd.Series([104, 103, 99, 98, 97, 96], index=idx, dtype="float64")
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 1000.0}, index=idx)
    out = C.evaluate_condition(df, {"indicator": "close", "op": "crosses_below", "value": 100})
    assert out.sum() == 1
    assert bool(out.iloc[2])


def test_a_cross_never_fires_on_the_first_bar():
    """There is no previous bar to have crossed from."""
    idx = pd.date_range("2026-01-05 09:15", periods=3, freq="15min")
    close = pd.Series([200.0, 201.0, 202.0], index=idx)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 1000.0}, index=idx)
    out = C.evaluate_condition(df, {"indicator": "close", "op": "crosses_above", "value": 100})
    assert not bool(out.iloc[0])


def test_two_series_can_be_compared(frame):
    out = C.evaluate_condition(frame, {"indicator": "ema", "params": {"length": 10},
                                       "op": "above", "compare_to": "ema",
                                       "compare_params": {"length": 50}})
    assert out.dtype == bool


def test_vwap_returns_empty_on_a_non_datetime_index(frame):
    """Session-anchored VWAP is undefined without dates; it must not silently
    fall back to a whole-frame cumsum, which is a different number entirely."""
    plain = frame.reset_index(drop=True)
    out = C.evaluate_condition(plain, {"indicator": "vwap", "op": ">", "value": 0})
    assert not out.any()


# ---------------------------------------------------------------------------
# Risk exits and the runner
# ---------------------------------------------------------------------------

def test_stop_and_target_are_extracted_from_the_exit_tree():
    risk = C.extract_risk_exits({"any": [
        {"indicator": "rsi", "op": ">", "value": 65},
        {"type": "stop_loss", "atr_multiple": 1.5},
        {"type": "take_profit", "percent": 4.0},
    ]})
    assert risk["stop_loss"]["atr_multiple"] == 1.5
    assert risk["take_profit"]["percent"] == 4.0


def test_an_exit_type_needs_a_distance():
    with pytest.raises(C.SpecError, match="atr_multiple or percent"):
        C.validate_condition({"type": "stop_loss"})


def test_the_runner_uses_the_stop_declared_in_the_spec(frame):
    res = C.run_strategy(frame, {
        "name": "s", "entry": {"indicator": "rsi", "op": "crosses_above", "value": 30},
        "exit": {"any": [{"indicator": "rsi", "op": ">", "value": 65},
                         {"type": "stop_loss", "percent": 2.0}]},
    })
    assert res["stop_source"] == "spec"
    assert res["stop_pct"] == 2.0


def test_the_runner_reports_signal_counts_so_a_dead_spec_is_obvious(frame):
    """A strategy that never triggers should say so, not report a 0% win rate
    over zero trades as though it had been tested."""
    res = C.run_strategy(frame, {
        "name": "impossible", "entry": {"indicator": "rsi", "op": "<", "value": -5},
        "exit": {"indicator": "rsi", "op": ">", "value": 200},
    })
    assert res["entry_signals"] == 0
    assert res["num_trades"] == 0


def test_the_runner_applies_costs(frame):
    entry = {"indicator": "rsi", "op": "crosses_above", "value": 30}
    exit_ = {"indicator": "rsi", "op": ">", "value": 60}
    spec = {"name": "s", "entry": entry, "exit": exit_}
    gross = C.run_strategy(frame, spec, cost_pct=0.0)
    net = C.run_strategy(frame, spec, cost_pct=0.5)
    if gross["num_trades"]:
        assert net["total_return_pct"] < gross["total_return_pct"]


def test_the_runner_refuses_a_frame_too_short_to_mean_anything(frame):
    assert "error" in C.run_strategy(frame.head(20), {
        "name": "s", "entry": {"indicator": "rsi", "op": "<", "value": 30},
        "exit": {"indicator": "rsi", "op": ">", "value": 70},
    })


def test_the_documented_example_spec_validates_and_runs(frame):
    """The spec printed in docs/agent-roadmap/01-strategy-engine.md. If this
    breaks, the documentation is lying to whoever reads it next."""
    res = C.run_strategy(frame, {
        "name": "RSI dip in uptrend",
        "entry": {"all": [
            {"indicator": "rsi", "params": {"length": 14}, "op": "<", "value": 35},
            {"indicator": "ema", "params": {"length": 50}, "op": "below", "compare_to": "close"},
        ]},
        "exit": {"any": [
            {"indicator": "rsi", "params": {"length": 14}, "op": ">", "value": 65},
            {"type": "stop_loss", "atr_multiple": 1.5},
            {"type": "take_profit", "atr_multiple": 3.0},
        ]},
    })
    assert "error" not in res
    assert res["strategy"] == "RSI dip in uptrend"


# ---------------------------------------------------------------------------
# The agent-facing tool
# ---------------------------------------------------------------------------

class _StaticMarket:
    def __init__(self, df): self.df = df
    async def get_historical_df(self, symbol, exchange="NSE", interval="15m", days=30):
        return self.df


def _toolbox(frame):
    from app.config import get_settings
    from app.signals.agent.context import StaticTradingContext
    from app.signals.agent.toolbox import AgentToolbox
    return AgentToolbox("RELIANCE", "NSE", frame, context=StaticTradingContext({}),
                        market=_StaticMarket(frame), settings=get_settings())


@pytest.mark.asyncio
async def test_the_tool_backtests_a_spec_end_to_end(frame):
    res = await _toolbox(frame).execute("build_strategy", {
        "name": "RSI dip",
        "entry": {"indicator": "rsi", "op": "crosses_above", "value": 30},
        "exit": {"any": [{"indicator": "rsi", "op": ">", "value": 65},
                         {"type": "stop_loss", "atr_multiple": 1.5}]},
    })
    assert "error" not in res, res
    assert res["symbol"] == "RELIANCE"
    assert "num_trades" in res and "win_rate" in res


@pytest.mark.asyncio
async def test_an_invalid_spec_comes_back_with_the_reason_and_the_allow_list(frame):
    """This is the model's feedback loop: a bare failure leaves it guessing,
    so the exact validation error and the available indicators go back to it."""
    res = await _toolbox(frame).execute("build_strategy", {
        "entry": {"indicator": "moon_phase", "op": "<", "value": 1},
        "exit": {"indicator": "rsi", "op": ">", "value": 70},
    })
    assert "Invalid strategy specification" in res["error"]
    assert "moon_phase" in res["error"]
    assert "rsi" in res["available_indicators"]


@pytest.mark.asyncio
async def test_the_result_always_carries_its_caveat(frame):
    """Every performance number this product shows carries its risk context.
    Trade count and in-sample status are the two that matter most here."""
    res = await _toolbox(frame).execute("build_strategy", {
        "entry": {"indicator": "rsi", "op": "crosses_above", "value": 30},
        "exit": {"indicator": "rsi", "op": ">", "value": 65},
    })
    assert "in-sample" in res["caveat"].lower()
    assert "trade count" in res["caveat"].lower()


@pytest.mark.asyncio
async def test_trades_are_plotted_on_the_chart(frame):
    """Every entry and exit the rules produced becomes a chart marker, so the
    user can see WHERE the strategy fired rather than only a summary number."""
    box = _toolbox(frame)
    res = await box.execute("build_strategy", {
        "entry": {"indicator": "rsi", "op": "crosses_above", "value": 30},
        "exit": {"any": [{"indicator": "rsi", "op": ">", "value": 60},
                         {"type": "stop_loss", "atr_multiple": 1.5}]},
    })
    assert res["num_trades"] > 0, "fixture should produce trades"
    markers = [d for d in box.drawings if d["kind"] == "trade_marker"]
    assert markers, "trades were backtested but nothing was drawn"
    assert {"timestamp", "value", "side", "color", "label"} <= set(markers[0])
    # Coordinates come from the backtest, never from the model.
    assert all(isinstance(m["timestamp"], int) for m in markers)


@pytest.mark.asyncio
async def test_exit_markers_are_coloured_by_outcome_not_by_direction(frame):
    """On a long-only strategy every exit is a sell, so colouring by side would
    make every marker identical and convey nothing."""
    box = _toolbox(frame)
    await box.execute("build_strategy", {
        "entry": {"indicator": "rsi", "op": "crosses_above", "value": 30},
        "exit": {"any": [{"indicator": "rsi", "op": ">", "value": 60},
                         {"type": "stop_loss", "atr_multiple": 0.5}]},
    })
    exits = [d for d in box.drawings if d["kind"] == "trade_marker" and d["side"] == "SELL"]
    assert exits
    assert {d["color"] for d in exits} <= {"#16c784", "#f0525d"}
    assert all("%" in d["label"] for d in exits)


@pytest.mark.asyncio
async def test_marker_count_is_capped_so_the_chart_stays_readable(frame):
    from app.signals.agent.tools import strategy as strategy_tools

    box = _toolbox(frame)
    plotted = strategy_tools._plot_trades(
        box.ctx,
        [{"entry_ts": i, "entry_price": 100.0, "exit_ts": i + 1, "exit_price": 101.0,
          "pnl_pct": 1.0, "exit_reason": "signal"} for i in range(200)],
        limit=10,
    )
    assert plotted == 10
    assert len(box.drawings) == 20   # an entry and an exit for each
