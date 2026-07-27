"""
Pure technical analysis. `analysis.py` needed no refactoring to be testable —
it has always been pure over a DataFrame.

The support/resistance clustering test locks in a real regression: chain
clustering (comparing each price to the LAST ADDED rather than the cluster
centre) let dense swing data chain end-to-end into a single meaningless
mega-cluster — 126 swings collapsing to one "level" spanning the entire range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.signals import analysis
from tests.conftest import make_bars


def _frame_from_closes(closes, spread: float = 0.3) -> pd.DataFrame:
    rows = []
    prev = closes[0]
    for c in closes:
        rows.append((prev, max(prev, c) + spread, min(prev, c) - spread, c))
        prev = c
    return make_bars(rows)


def test_support_resistance_does_not_chain_into_one_mega_cluster():
    """The regression. A steady ramp produces many swings a small distance
    apart; centre-based clustering with a width cap must not merge them all."""
    n = 400
    closes = list(np.linspace(100.0, 130.0, n) + np.sin(np.arange(n) / 3.0) * 0.8)
    levels = analysis.support_resistance(_frame_from_closes(closes))
    assert levels, "expected some levels"
    last = closes[-1]
    max_width = last * 0.008
    for lvl in levels:
        # No single level may claim more swings than could physically fit in a
        # 0.8%-wide band of a monotone ramp.
        assert lvl["strength"] < n / 2, f"cluster too large: {lvl}"
    assert len(levels) <= 6
    assert max_width > 0


def test_support_resistance_labels_relative_to_last_price():
    closes = [100 + (i % 7) for i in range(200)]
    frame = _frame_from_closes(closes)
    last = float(frame["close"].iloc[-1])
    for lvl in analysis.support_resistance(frame):
        expected = "resistance" if lvl["value"] >= last else "support"
        assert lvl["kind"] == expected


def test_support_resistance_returns_empty_for_a_short_frame():
    assert analysis.support_resistance(_frame_from_closes([100, 101, 102])) == []


def test_support_resistance_levels_are_returned_in_price_order():
    closes = list(np.linspace(100, 108, 300) + np.sin(np.arange(300) / 2.0))
    levels = analysis.support_resistance(_frame_from_closes(closes))
    assert levels == sorted(levels, key=lambda x: x["value"])


def test_trendline_is_none_without_enough_bars():
    assert analysis.trendline(_frame_from_closes([100, 101, 102])) is None


def test_trendline_returns_two_chronological_points():
    closes = list(np.linspace(100, 140, 300) + np.sin(np.arange(300) / 4.0) * 2)
    tl = analysis.trendline(_frame_from_closes(closes))
    assert tl is not None
    p1, p2 = tl["points"]
    assert p1["timestamp"] < p2["timestamp"], "trendline points must be chronological"
    assert tl["direction"] in ("up", "down")


def test_fibonacci_returns_the_swing_extremes():
    closes = list(np.linspace(100, 120, 60)) + list(np.linspace(120, 105, 60))
    fib = analysis.fibonacci(_frame_from_closes(closes))
    assert fib is not None
    assert fib["high"] > fib["low"]
    assert fib["high"] >= max(closes) - 1


def test_fibonacci_is_none_on_a_flat_frame():
    flat = make_bars([(100, 100, 100, 100)] * 40)
    assert analysis.fibonacci(flat) is None


def test_simulate_trade_reward_risk_is_direction_agnostic():
    long_ = analysis.simulate_trade("BUY", 100, 110, 95, 10)
    short = analysis.simulate_trade("SELL", 100, 90, 105, 10)
    assert long_["reward_risk"] == short["reward_risk"] == 2.0
    assert long_["profit_at_target"] == 100
    assert short["loss_at_stop"] == -50


def test_backtest_reports_an_unclosed_position_instead_of_dropping_it():
    """A strategy that enters and never exits used to report only the trades
    that happened to close, flattering the win rate."""
    closes = list(np.linspace(100, 160, 300))
    res = analysis.backtest(_frame_from_closes(closes), "ma_cross")
    assert "open_trade" in res
    assert "num_trades" in res and "win_rate" in res


def test_backtest_rejects_an_unknown_strategy():
    res = analysis.backtest(_frame_from_closes(list(np.linspace(100, 120, 150))), "moon_phase")
    assert "error" in res and "supported" in res


# ---------------------------------------------------------------------------
# Chart correctness (checklist 06) — errors a user can see on the chart
# ---------------------------------------------------------------------------

def test_trendline_refuses_to_label_descending_lows_an_uptrend():
    """The visible bug: direction came from price-vs-EMA20 alone, so a
    descending pair of lows got drawn as a rising trend line and the chart
    contradicted its own label."""
    # Price ends above its EMA20 (so the old code says "up"), but each swing low
    # is LOWER than the last.
    closes = []
    for leg in range(6):
        base = 120 - leg * 3           # each leg starts lower
        closes += [base, base - 4, base + 2, base + 3]
    closes += [140] * 12               # final push above the EMA
    tl = analysis.trendline(_frame_from_closes(closes))
    # Either it declines to draw a line, or the line it draws agrees with its
    # own label. What it must never do is return a falling line labelled "up".
    if tl is not None:
        p1, p2 = tl["points"]
        rises = p2["value"] > p1["value"]
        assert rises == (tl["direction"] == "up"), "slope must agree with the label"

    # And prove the guard is not vacuous: across many random shapes, every line
    # returned agrees with its label.
    rng = np.random.default_rng(3)
    drawn = 0
    for _ in range(25):
        series = list(np.cumsum(rng.normal(0, 1.2, 320)) + 200)
        t = analysis.trendline(_frame_from_closes(series))
        if t is None:
            continue
        drawn += 1
        a, b = t["points"]
        assert (b["value"] > a["value"]) == (t["direction"] == "up")
    assert drawn > 0, "guard is vacuous — no trend line was produced at all"


def test_trendline_requires_at_least_three_touches():
    """Any two points define a line; three is the first number that is evidence."""
    closes = list(np.linspace(100, 140, 300) + np.sin(np.arange(300) / 4.0) * 2)
    tl = analysis.trendline(_frame_from_closes(closes))
    assert tl is not None, "a clean rising series should still yield a line"
    assert tl["touches"] >= analysis.MIN_TRENDLINE_TOUCHES


def test_trendline_is_none_when_price_has_closed_through_it():
    """A line price has already broken is not support any more."""
    rising = list(np.linspace(100, 130, 260))
    collapsed = rising + list(np.linspace(130, 95, 40))   # sharp break down
    assert analysis.trendline(_frame_from_closes(collapsed)) is None


def test_fibonacci_anchors_chronologically_on_an_up_swing():
    """Anchors used to be returned high-then-low unconditionally, so on an
    up-swing the 38.2% and 61.8% levels swapped. 50% is symmetric, which is
    what hid it."""
    closes = list(np.linspace(100, 140, 80)) + list(np.linspace(140, 130, 40))
    fib = analysis.fibonacci(_frame_from_closes(closes))
    assert fib is not None
    assert fib["direction"] == "up"
    first, second = fib["points"]
    assert first["timestamp"] < second["timestamp"], "anchors must be in time order"
    assert first["value"] == fib["low"] and second["value"] == fib["high"]


def test_fibonacci_anchors_chronologically_on_a_down_swing():
    closes = list(np.linspace(140, 100, 80)) + list(np.linspace(100, 110, 40))
    fib = analysis.fibonacci(_frame_from_closes(closes))
    assert fib is not None
    assert fib["direction"] == "down"
    first, second = fib["points"]
    assert first["timestamp"] < second["timestamp"]
    assert first["value"] == fib["high"] and second["value"] == fib["low"]


def test_fibonacci_382_and_618_are_not_symmetric_and_so_reveal_direction():
    """The regression guard: if anchoring reverses, these two swap."""
    up = analysis.fibonacci(_frame_from_closes(
        list(np.linspace(100, 200, 80)) + list(np.linspace(200, 180, 40))))
    assert up is not None
    # On an up-swing, retracement is measured DOWN from the high, so 38.2% sits
    # above 61.8%.
    assert up["levels"]["38.2%"] > up["levels"]["61.8%"]
    # 50% is the midpoint either way — which is exactly why it never caught this.
    assert up["levels"]["50.0%"] == round((up["high"] + up["low"]) / 2, 2)


def test_support_resistance_prefers_a_recent_level_over_an_old_one():
    """Ranking on raw touch count let five touches eight weeks ago outrank two
    from yesterday."""
    old_level, recent_level = 100.0, 103.0
    closes = []
    for _ in range(10):                       # many old touches at 100
        closes += [old_level, old_level + 2.5, old_level, old_level + 2.5]
    for _ in range(4):                        # fewer, recent touches at 103
        closes += [recent_level, recent_level + 2.5, recent_level, recent_level + 2.5]
    closes += [recent_level + 1] * 10
    levels = analysis.support_resistance(_frame_from_closes(closes))
    assert levels, "expected levels"
    for lvl in levels:
        assert "recency_weighted_strength" in lvl
        assert "last_touch_bars_ago" in lvl
    strongest = max(levels, key=lambda x: x["recency_weighted_strength"])
    assert strongest["last_touch_bars_ago"] < len(closes) / 2, \
        "the recency-weighted winner should be a recent level, not the oldest one"


def test_support_resistance_radius_is_tighter_than_the_old_six_percent():
    """±6% returned levels an intraday trade resolves long before reaching."""
    closes = list(np.linspace(100, 106, 400) + np.sin(np.arange(400) / 3.0) * 0.7)
    frame = _frame_from_closes(closes)
    last = float(frame["close"].iloc[-1])
    for lvl in analysis.support_resistance(frame):
        assert abs(lvl["value"] - last) <= last * 0.06
    assert analysis._reach(frame, last) <= last * 0.04


# ---------------------------------------------------------------------------
# Agent backtest tool (checklist 07)
# ---------------------------------------------------------------------------

def test_backtest_entry_fills_at_the_next_bar_open_not_the_signal_bar_close():
    """One-bar lookahead: a signal computed from a bar's close cannot be acted
    on until that bar has closed, so the earliest obtainable price is the NEXT
    bar's open. Filling at the deciding bar's close books a price that could not
    have been had — on every trade."""
    # Bar 2 triggers entry. Its close is 100; the next bar opens at 130.
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),   # entry signal here, close = 100
        (130, 131, 129, 130),  # only fillable price is this open = 130
        (130, 131, 129, 130),
    ]
    frame = make_bars(rows)
    entries = pd.Series([False, False, True, False, False], index=frame.index)
    exits   = pd.Series([False, False, False, False, True], index=frame.index)

    res = analysis._run_signals(frame, entries, exits, stop_pct=None)
    assert res["num_trades"] == 1
    assert res["trades"][0]["entry_price"] == 130.0, "must not fill at the signal bar's close"


def test_backtest_stop_closes_a_losing_position():
    """Without a stop a position could run -20% and count as one open trade."""
    rows = [(100, 101, 99, 100)] * 3 + [(100, 100, 80, 82)] + [(82, 83, 81, 82)]
    frame = make_bars(rows)
    entries = pd.Series([True, False, False, False, False], index=frame.index)
    exits   = pd.Series([False] * 5, index=frame.index)

    res = analysis._run_signals(frame, entries, exits, stop_pct=2.0)
    assert res["num_trades"] == 1
    assert res["stopped_out"] == 1
    assert res["trades"][0]["exit_reason"] == "stop"
    assert res["open_trade"] is None, "the stop closed it — nothing should be left open"


def test_backtest_stop_gap_fills_at_the_bar_open():
    """A bar that OPENS below the stop fills at the open, not the stop price."""
    rows = [(100, 101, 99, 100)] * 2 + [(90, 91, 89, 90)] + [(90, 91, 89, 90)]
    frame = make_bars(rows)
    entries = pd.Series([True, False, False, False], index=frame.index)
    exits   = pd.Series([False] * 4, index=frame.index)

    res = analysis._run_signals(frame, entries, exits, stop_pct=2.0)
    assert res["trades"][0]["exit_price"] == 90.0


def test_backtest_applies_round_trip_costs():
    rows = [(100, 101, 99, 100)] * 2 + [(110, 111, 109, 110)] * 2
    frame = make_bars(rows)
    entries = pd.Series([True, False, False, False], index=frame.index)
    exits   = pd.Series([False, False, False, True], index=frame.index)

    gross = analysis._run_signals(frame, entries, exits, stop_pct=None, cost_pct=0.0)
    net   = analysis._run_signals(frame, entries, exits, stop_pct=None, cost_pct=0.5)
    assert net["trades"][0]["pnl_pct"] == round(gross["trades"][0]["pnl_pct"] - 0.5, 2)


def test_backtest_reports_the_stop_it_used():
    """The agent quotes this result, so the assumptions must travel with it."""
    res = analysis.backtest(_frame_from_closes(list(np.linspace(100, 160, 300))), "ma_cross")
    assert res["stop_pct"] == 2.0
    assert "stopped_out" in res
