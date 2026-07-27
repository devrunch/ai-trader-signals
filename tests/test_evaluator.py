"""
Outcome evaluation — the two frictions that flatter results if omitted.

This is the number the product uses to tell users whether the system works, and
it previously existed twice in two languages with different rules.
"""
from __future__ import annotations

import pytest

from app.signals.backtest.evaluator import evaluate
from tests.conftest import make_bars

COST = 1.0  # a round 1% so expected P&L is easy to read in the assertions


def test_buy_hits_target():
    forward = make_bars([(100, 101, 99.5, 100.5), (100.5, 106, 100, 105)])
    outcome, exit_price, pnl = evaluate("BUY", 100, 105, 95, forward, cost_pct=COST)
    assert outcome == "TARGET_HIT"
    assert exit_price == 105.0
    assert pnl == pytest.approx(4.0)  # +5% gross - 1% costs


def test_buy_hits_stop_at_stop_price_when_no_gap():
    forward = make_bars([(100, 101, 94, 96)])
    outcome, exit_price, pnl = evaluate("BUY", 100, 105, 95, forward, cost_pct=COST)
    assert outcome == "STOP_HIT"
    assert exit_price == 95.0
    assert pnl == pytest.approx(-6.0)  # -5% gross - 1% costs


def test_buy_gap_through_stop_fills_at_the_open_not_the_stop():
    """The whole point of the gap rule: a bar that OPENS below the stop fills at
    the open. Booking it at the stop price understates the loss."""
    forward = make_bars([(90, 92, 89, 91)])
    outcome, exit_price, pnl = evaluate("BUY", 100, 105, 95, forward, cost_pct=COST)
    assert outcome == "STOP_HIT"
    assert exit_price == 90.0, "gapped stop must fill at the bar open"
    assert pnl == pytest.approx(-11.0)  # -10% gross - 1% costs


def test_sell_hits_target():
    forward = make_bars([(100, 100.5, 94, 95)])
    outcome, exit_price, pnl = evaluate("SELL", 100, 95, 105, forward, cost_pct=COST)
    assert outcome == "TARGET_HIT"
    assert exit_price == 95.0
    assert pnl == pytest.approx(4.0)


def test_sell_gap_through_stop_fills_at_the_open():
    forward = make_bars([(110, 112, 109, 111)])
    outcome, exit_price, pnl = evaluate("SELL", 100, 95, 105, forward, cost_pct=COST)
    assert outcome == "STOP_HIT"
    assert exit_price == 110.0
    assert pnl == pytest.approx(-11.0)


def test_stop_wins_when_both_levels_touch_in_one_bar():
    """A single OHLC bar does not say which level was hit first, so assume the
    adverse one."""
    forward = make_bars([(100, 106, 94, 100)])
    outcome, exit_price, _ = evaluate("BUY", 100, 105, 95, forward, cost_pct=COST)
    assert outcome == "STOP_HIT"
    assert exit_price == 95.0


def test_unresolved_marks_to_last_close_as_open():
    forward = make_bars([(100, 101, 99, 100.5), (100.5, 102, 100, 101)])
    outcome, exit_price, pnl = evaluate("BUY", 100, 105, 95, forward, cost_pct=COST)
    assert outcome == "OPEN"
    assert exit_price == 101.0
    assert pnl == pytest.approx(0.0)  # +1% gross - 1% costs


def test_no_forward_bars_is_open_with_no_exit():
    empty = make_bars([]).iloc[0:0]
    outcome, exit_price, pnl = evaluate("BUY", 100, 105, 95, empty, cost_pct=COST)
    assert (outcome, exit_price, pnl) == ("OPEN", None, 0.0)


def test_costs_change_the_sign_of_a_marginal_trade():
    """The reason costs are not a rounding error at this trade frequency: a
    +0.10% winner is a net loser once 0.12% of round-trip friction is paid."""
    forward = make_bars([(100, 100.2, 99.9, 100.1)])
    _, _, gross = evaluate("BUY", 100, 100.1, 95, forward, cost_pct=0.0)
    _, _, net = evaluate("BUY", 100, 100.1, 95, forward, cost_pct=0.12)
    assert gross > 0
    assert net < 0
