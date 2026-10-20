"""Did the trades it proposed actually work?

Everything this platform produces is a prediction, and until now nothing
recorded whether any of it paid. The settlement rule is pure, so the part that
decides a win from a loss is testable without Redis or a provider — which is
the part worth testing, because the ways a scorecard flatters itself are all
in that rule.
"""
from __future__ import annotations

import pytest

from app.signals import scorecard as S


def proposal(direction="BUY", entry=100.0, stop=95.0, target=110.0,
             order_type="market", verdict="passes_every_gate"):
    return {"trade_id": "T", "symbol": "X", "direction": direction, "entry": entry,
            "stop": stop, "target": target, "order_type": order_type,
            "verdict": verdict, "interval": "1d", "exchange": "NSE"}


def bar(high, low, close=None):
    return {"high": high, "low": low, "close": close if close is not None else low}


class TestSettlement:
    def test_a_long_that_reaches_its_target_is_a_win(self):
        out = S.settle([bar(102, 99), bar(111, 105)], proposal())
        assert out[0] == "target"
        assert out[2] == pytest.approx(10.0)

    def test_a_long_that_hits_its_stop_is_a_loss(self):
        out = S.settle([bar(101, 94)], proposal())
        assert out[0] == "stop"
        assert out[2] == pytest.approx(-5.0)

    def test_a_bar_covering_both_counts_as_the_stop(self):
        """Intrabar order is unknowable from an OHLC bar, and assuming the
        good side first is how a backtest flatters itself."""
        out = S.settle([bar(115, 90)], proposal())
        assert out[0] == "stop"

    def test_a_short_is_scored_the_right_way_round(self):
        short = proposal(direction="SELL", entry=100.0, stop=105.0, target=90.0)
        win = S.settle([bar(101, 89)], short)
        assert win[0] == "target" and win[2] == pytest.approx(10.0)
        loss = S.settle([bar(106, 99)], short)
        assert loss[0] == "stop" and loss[2] == pytest.approx(-5.0)

    def test_neither_side_inside_the_window_expires(self):
        """Not a loss, and not open forever."""
        out = S.settle([bar(101, 99) for _ in range(S.MAX_BARS)], proposal())
        assert out[0] == "expired"

    def test_it_stops_looking_after_the_window(self):
        bars = [bar(101, 99) for _ in range(S.MAX_BARS)] + [bar(999, 998)]
        assert S.settle(bars, proposal())[0] == "expired"


class TestLimitOrders:
    def test_a_limit_that_never_trades_through_is_unfilled(self):
        """Not a loss. Nothing was risked — and counting it as one would make
        the pullback entry modes look worse than market orders for free."""
        p = proposal(order_type="limit", entry=90.0, stop=85.0, target=100.0)
        out = S.settle([bar(101, 95), bar(102, 96)], p)
        assert out[0] == "unfilled"
        assert out[2] is None

    def test_a_limit_fills_when_price_trades_through_it(self):
        p = proposal(order_type="limit", entry=90.0, stop=85.0, target=100.0)
        out = S.settle([bar(95, 89), bar(101, 92)], p)
        assert out[0] == "target"

    def test_bars_held_counts_from_the_fill_not_the_proposal(self):
        p = proposal(order_type="limit", entry=90.0, stop=85.0, target=100.0)
        bars = [bar(95, 94), bar(95, 94), bar(95, 89), bar(101, 92)]
        status, held, _ = S.settle(bars, p)
        assert status == "target" and held == 1

    def test_a_short_limit_fills_on_a_move_up_into_it(self):
        p = proposal(direction="SELL", order_type="limit",
                     entry=110.0, stop=115.0, target=100.0)
        assert S.settle([bar(111, 105), bar(106, 99)], p)[0] == "target"


class TestSummary:
    @pytest.mark.asyncio
    async def test_unfilled_limits_stay_out_of_the_hit_rate(self):
        closed = [
            dict(proposal(), status="target", move_pct=10.0),
            dict(proposal(), status="stop", move_pct=-5.0),
            dict(proposal(), status="unfilled", move_pct=None),
        ]
        out = await S.summary(closed)
        assert out["filled"] == 2
        assert out["unfilled"] == 1
        assert out["hit_rate_pct"] == 50.0

    @pytest.mark.asyncio
    async def test_rejected_proposals_are_recorded_but_not_scored(self):
        """The rejections are the larger claim. A scorecard tracking only the
        trades it liked could never discover that the gates reject winners."""
        closed = [
            dict(proposal(), status="target", move_pct=10.0),
            dict(proposal(verdict="rejected"), status="target", move_pct=20.0),
        ]
        out = await S.summary(closed)
        assert out["recorded"] == 2
        assert out["passed_the_gates"] == 1
        assert out["hit_rate_pct"] == 100.0

    @pytest.mark.asyncio
    async def test_a_thin_sample_says_so_in_the_data(self):
        """A hit rate over a handful of trades reads as a result unless it is
        labelled."""
        out = await S.summary([dict(proposal(), status="target", move_pct=1.0)])
        assert out["trustworthy"] is False
        assert "not a track record" in out["note"]

    @pytest.mark.asyncio
    async def test_an_empty_record_does_not_divide_by_zero(self):
        out = await S.summary([])
        assert out["recorded"] == 0 and "hit_rate_pct" not in out


class TestIdentity:
    def test_the_same_setup_twice_is_one_record(self):
        from datetime import UTC, datetime
        when = datetime(2026, 10, 20, 9, 15, tzinfo=UTC)
        assert S.trade_id("X", "BUY", when, 100.0) == S.trade_id("X", "BUY", when, 100.0)

    def test_the_same_symbol_at_a_different_level_is_a_different_trade(self):
        from datetime import UTC, datetime
        when = datetime(2026, 10, 20, 9, 15, tzinfo=UTC)
        assert S.trade_id("X", "BUY", when, 100.0) != S.trade_id("X", "BUY", when, 105.0)
