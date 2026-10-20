"""The event desk and news feed, as tools.

Asked what could move gold this week, the agent ran two web searches and
answered "sources suggest a relatively light week" — while this process held
the calendar, the forecast, and a measured study of past reactions. These
tests pin the parts that made that failure possible: direction comes from the
published spec and not from a guess, an unavailable calendar is distinguished
from an empty one, and a thin sample is reported as thin.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.events.calendar import CalendarEvent, Impact
from app.signals.agent.tools import desk


def _ctx(symbol="XAUUSD", exchange="FOREX"):
    ctx = MagicMock()
    ctx.symbol, ctx.exchange = symbol, exchange
    return ctx


def event(title="Core CPI m/m", currency="USD", hours=6, impact=Impact.HIGH,
          forecast="0.3%", previous="0.2%"):
    return CalendarEvent(title=title, currency=currency,
                         when=datetime.now(UTC) + timedelta(hours=hours),
                         impact=impact, forecast=forecast, previous=previous)


class TestDrivers:
    def test_gold_is_driven_by_the_dollar(self):
        assert desk.drivers_for("XAUUSD") == ("USD",)

    def test_a_pair_is_driven_by_both_its_legs(self):
        """EURUSD reacts to a German print and a US one. Mapping the symbol to
        one currency would miss half its calendar."""
        assert set(desk.drivers_for("EURUSD")) == {"EUR", "USD"}
        assert set(desk.drivers_for("GBPJPY")) == {"GBP", "JPY"}

    def test_an_unknown_symbol_still_gets_us_data(self):
        assert desk.drivers_for("RELIANCE") == ("USD",)


class TestGetEvents:
    @pytest.mark.asyncio
    async def test_it_carries_the_published_direction_not_a_guess(self):
        """29 of the glossary's events read backwards from the obvious — an
        unemployment beat is bad news. A plausible guess gets those wrong."""
        spec = MagicMock(beat_is_bullish=True, notes="Inflation at the consumer level")
        with patch("app.events.calendar.fetch_week", AsyncMock(return_value=[event()])), \
             patch("app.events.specs.spec_for", return_value=spec):
            out = await desk.get_events(_ctx(), {})
        row = out["events"][0]
        assert row["beat_means"] == "stronger USD"
        assert row["forecast"] == "0.3%"
        assert row["previous"] == "0.2%"
        assert row["in_hours"] > 0

    @pytest.mark.asyncio
    async def test_a_release_with_no_published_direction_says_so(self):
        """Inventing one is how a brief becomes confidently wrong."""
        spec = MagicMock(beat_is_bullish=None, notes=None)
        with patch("app.events.calendar.fetch_week", AsyncMock(return_value=[event()])), \
             patch("app.events.specs.spec_for", return_value=spec):
            out = await desk.get_events(_ctx(), {})
        assert out["events"][0]["beat_means"] is None

    @pytest.mark.asyncio
    async def test_an_unavailable_calendar_is_not_an_empty_calendar(self):
        """Reporting "nothing scheduled" on the morning of a Fed decision is
        the failure this distinction exists to prevent."""
        with patch("app.events.calendar.fetch_week", AsyncMock(return_value=None)):
            out = await desk.get_events(_ctx(), {})
        assert "error" in out and "unavailable" in out["error"]
        assert "events" not in out

    @pytest.mark.asyncio
    async def test_a_genuinely_quiet_week_returns_an_empty_list(self):
        with patch("app.events.calendar.fetch_week", AsyncMock(return_value=[])):
            out = await desk.get_events(_ctx(), {})
        assert out["events"] == [] and "error" not in out

    @pytest.mark.asyncio
    async def test_it_filters_to_the_symbols_own_drivers(self):
        events = [event(currency="USD"), event(currency="JPY", title="BOJ Rate")]
        with patch("app.events.calendar.fetch_week", AsyncMock(return_value=events)), \
             patch("app.events.specs.spec_for", return_value=None):
            out = await desk.get_events(_ctx("XAUUSD"), {})
        assert {r["currency"] for r in out["events"]} == {"USD"}

    @pytest.mark.asyncio
    async def test_the_list_is_capped_so_an_answer_is_not_a_calendar_dump(self):
        many = [event(title=f"Release {i}") for i in range(30)]
        with patch("app.events.calendar.fetch_week", AsyncMock(return_value=many)), \
             patch("app.events.specs.spec_for", return_value=None):
            out = await desk.get_events(_ctx(), {})
        assert len(out["events"]) == desk.MAX_EVENTS
        assert out["total_upcoming"] == 30


class TestGetEventReactions:
    @pytest.mark.asyncio
    async def test_a_thin_sample_is_reported_as_thin(self):
        """A median over three prints spanning two rate regimes looks like
        evidence and is not."""
        stats = MagicMock(sample=2, enough=False, beats=[0.1], misses=[-0.2])
        stats.summary.return_value = None
        with patch("app.events.history.past_prints", return_value=[{}, {}]), \
             patch("app.events.reaction.instances_from_history", return_value=[]), \
             patch("app.events.reaction.measure", AsyncMock(return_value=[])), \
             patch("app.events.reaction.aggregate", return_value=stats):
            out = await desk.get_event_reactions(_ctx(), {"currency": "USD", "title": "Core CPI m/m"})
        assert out["enough"] is False
        assert "too thin" in out["summary"]

    @pytest.mark.asyncio
    async def test_a_real_sample_carries_its_summary(self):
        stats = MagicMock(sample=9, enough=True, beats=[-0.4, -0.6], misses=[0.5])
        stats.summary.return_value = "above forecast: XAUUSD fell 7 of 9, median -0.42% in 60m"
        with patch("app.events.history.past_prints", return_value=[{}] * 9), \
             patch("app.events.reaction.instances_from_history", return_value=[]), \
             patch("app.events.reaction.measure", AsyncMock(return_value=[])), \
             patch("app.events.reaction.aggregate", return_value=stats):
            out = await desk.get_event_reactions(_ctx(), {"currency": "USD", "title": "Core CPI m/m"})
        assert out["sample"] == 9 and "fell 7 of 9" in out["summary"]

    @pytest.mark.asyncio
    async def test_it_refuses_without_a_release_title(self):
        out = await desk.get_event_reactions(_ctx(), {"currency": "USD"})
        assert "error" in out

    @pytest.mark.asyncio
    async def test_a_release_with_no_history_says_so(self):
        with patch("app.events.history.past_prints", return_value=[]):
            out = await desk.get_event_reactions(_ctx(), {"currency": "USD", "title": "New Release"})
        assert "No past prints" in out["error"]


class TestGetNews:
    @staticmethod
    def _result(articles):
        return {"articles": articles, "count": len(articles), "degraded": False}

    @staticmethod
    def _article(headline="Gold rallies", impacts=None, sentiment="POSITIVE"):
        return {"headline": headline, "source": "Reuters", "publishedAt": "2026-10-20T09:00:00Z",
                "sentiment": sentiment, "sentimentAvailable": True,
                "impacts": impacts, "url": "https://example.com/x"}

    @pytest.mark.asyncio
    async def test_it_returns_what_moves_and_why(self):
        impacts = [{"symbol": "XAUUSD", "direction": "UP", "reason": "haven bid"}]
        with patch("app.market.news.get_market_news_result",
                   AsyncMock(return_value=self._result([self._article(impacts=impacts)]))):
            out = await desk.get_news(_ctx(), {"symbol": "XAUUSD"})
        row = out["headlines"][0]
        assert row["moves"][0]["symbol"] == "XAUUSD"
        assert row["moves"][0]["why"] == "haven bid"

    @pytest.mark.asyncio
    async def test_filtering_keeps_stories_the_analysis_could_not_read(self):
        """`impacts: None` means unread, not irrelevant. Dropping those would
        quietly hide every story during an LLM outage."""
        articles = [self._article("unread", impacts=None),
                    self._article("other", impacts=[{"symbol": "INFY"}])]
        with patch("app.market.news.get_market_news_result",
                   AsyncMock(return_value=self._result(articles))):
            out = await desk.get_news(_ctx(), {"symbol": "XAUUSD"})
        assert [r["headline"] for r in out["headlines"]] == ["unread"]

    @pytest.mark.asyncio
    async def test_an_unreadable_sentiment_is_null_not_neutral(self):
        article = self._article()
        article["sentimentAvailable"] = False
        with patch("app.market.news.get_market_news_result",
                   AsyncMock(return_value=self._result([article]))):
            out = await desk.get_news(_ctx(), {})
        assert out["headlines"][0]["sentiment"] is None

    @pytest.mark.asyncio
    async def test_a_dead_feed_is_an_error_not_an_empty_feed(self):
        with patch("app.market.news.get_market_news_result",
                   AsyncMock(side_effect=RuntimeError("down"))):
            out = await desk.get_news(_ctx(), {})
        assert "error" in out and "headlines" not in out

    @pytest.mark.asyncio
    async def test_the_limit_is_capped(self):
        articles = [self._article(f"h{i}") for i in range(40)]
        with patch("app.market.news.get_market_news_result",
                   AsyncMock(return_value=self._result(articles))):
            out = await desk.get_news(_ctx(), {"limit": 99})
        assert len(out["headlines"]) == desk.MAX_HEADLINES


class TestRegistration:
    def test_every_desk_tool_is_advertised_and_handled(self):
        """A schema without a handler is a tool the model will call and get an
        error from; a handler without a schema is dead code."""
        from app.signals.agent.schemas import TOOL_SCHEMAS
        from app.signals.agent.tools import REGISTRY

        advertised = {t["function"]["name"] for t in TOOL_SCHEMAS}
        for name in ("get_events", "get_event_reactions", "get_news", "get_fundamentals"):
            assert name in advertised, f"{name} is not advertised to the model"
            assert name in REGISTRY, f"{name} has no handler"

    def test_each_new_tool_has_a_progress_label(self):
        """An unlabelled tool shows the user a blank step while it runs."""
        from app.signals.agent.events import _TOOL_LABELS
        for name in ("get_events", "get_event_reactions", "get_news", "get_fundamentals"):
            assert _TOOL_LABELS.get(name), f"{name} has no label"


class TestTradeGeometry:
    """The gate that rejected its own output.

    `_geometry` placed the stop at exactly `atr * min_multiple`, then
    `stop_size` recomputed `abs(entry - stop)` and compared it back. At Nifty's
    scale that round trip loses about 1e-10, so the stop the tool had just
    produced was "tighter than 1.0x ATR" than itself:

        stop 274.86 tighter than 1.0x ATR (274.86)
    """

    @staticmethod
    def _frame(rows=120, price=22888.95, swing=300.0):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(3)
        idx = pd.date_range("2026-01-01", periods=rows, freq="1D")
        closes = price + np.cumsum(rng.normal(0, swing / 8, rows))
        return pd.DataFrame({
            "open": closes + rng.normal(0, swing / 20, rows),
            "high": closes + abs(rng.normal(swing / 3, swing / 10, rows)),
            "low": closes - abs(rng.normal(swing / 3, swing / 10, rows)),
            "close": closes,
            "volume": rng.integers(1_000, 9_000, rows).astype(float),
        }, index=idx)

    async def _propose(self, direction, **args):
        from app.signals.agent.toolbox import AgentToolbox
        frame = self._frame()
        box = AgentToolbox("^NSEI", "NSE", frame, None)
        return await box.execute("propose_trade", {"direction": direction,
                                                   "interval": "1d", **args})

    @pytest.mark.asyncio
    async def test_the_stop_it_places_never_fails_its_own_size_gate(self):
        for direction in ("BUY", "SELL"):
            out = await self._propose(direction)
            failed = [g for g in out["gates"] if g["gate"] == "stop_size" and not g["passed"]]
            assert not failed, f"{direction}: {failed}"

    @pytest.mark.asyncio
    async def test_the_stop_sits_inside_the_allowed_atr_band(self):
        for direction in ("BUY", "SELL"):
            out = await self._propose(direction)
            assert 1.0 <= out["stop_in_atr"] <= 3.5, (direction, out["stop_in_atr"])

    @pytest.mark.asyncio
    async def test_the_stop_is_on_the_losing_side_of_the_entry(self):
        buy = await self._propose("BUY")
        assert buy["stop"] < buy["entry"] < buy["target"]
        sell = await self._propose("SELL")
        assert sell["stop"] > sell["entry"] > sell["target"]

    @pytest.mark.asyncio
    async def test_a_rejected_setup_says_which_gate_and_what_would_pass(self):
        """A wall that says "no" teaches the agent nothing. It guessed before,
        and guessing produced the numbers this tool exists to replace."""
        out = await self._propose("BUY", entry=22888.95)
        if out["verdict"] == "rejected":
            assert out["rejected_because"]
            if any(g["gate"] == "reward_risk" and not g["passed"] for g in out["gates"]):
                assert out["to_make_it_work"]["target_for_min_rr"]

    @pytest.mark.asyncio
    async def test_a_short_is_computed_and_flagged_rather_than_deleted(self):
        """long_only exists because the paper account cannot short, not
        because a short is bad analysis."""
        out = await self._propose("SELL")
        assert out["executable_here"] is False
        assert "broker" in out["executable_note"]
        assert out["reward_risk"] is not None

    @pytest.mark.asyncio
    async def test_it_refuses_a_direction_it_was_not_given(self):
        out = await self._propose("MAYBE")
        assert "error" in out and "BUY or SELL" in out["error"]

    @pytest.mark.asyncio
    async def test_every_number_the_model_may_quote_is_computed_here(self):
        out = await self._propose("BUY")
        for key in ("entry", "stop", "target", "risk_per_unit", "reward_per_unit",
                    "reward_risk", "atr", "stop_in_atr", "how_stop_was_placed"):
            assert out.get(key) is not None, key


class TestDerivativeSizing:
    """Nifty cannot be bought. It trades as NFO futures and options, and a
    contract's quantity is lots — one Nifty futures lot is 65 index points of
    exposure. Code that treats that number as shares is wrong by 65x, and
    silently, because the number still looks plausible.
    """

    def test_an_nfo_future_routes_to_kite_as_a_future(self):
        from app.market.contract import AssetClass
        from app.market.symbols import resolve
        for symbol in ("NIFTY1!", "NIFTY26OCTFUT"):
            info = resolve(symbol, "NFO")
            assert info is not None and info.asset_class is AssetClass.FUTURE

    def test_a_strike_routes_as_an_option_not_a_future(self):
        from app.market.contract import AssetClass
        from app.market.symbols import resolve
        for symbol in ("NIFTY26OCT22900CE", "NIFTY26OCT22900PE"):
            assert resolve(symbol, "NFO").asset_class is AssetClass.OPTION

    def test_cash_equity_is_untouched_by_the_nfo_change(self):
        from app.market.contract import AssetClass
        from app.market.symbols import resolve
        assert resolve("RELIANCE", "NSE").asset_class is AssetClass.EQUITY
        assert resolve("^NSEI", "NSE").asset_class is AssetClass.INDEX
        assert resolve("GOLD1!", "MCX").asset_class is AssetClass.COMMODITY

    def test_a_derivative_is_never_sized_against_the_paper_account(self):
        """The paper account is NSE/BSE only and holds simulated money. Sizing
        an NFO contract against it produced "your 1,00,000 account is too small
        for Nifty futures, build capital first" — real advice from a number
        that is not the user's."""
        from app.signals.agent.tools.trade_plan import _lot_sizing
        from app.config import get_settings

        book = {"portfolio": {"totalValue": 100_000, "cashBalance": 100_000}}
        out = _lot_sizing(entry=22_890.0, risk_per_unit=275.0, lot_size=65,
                          book=book, risk_pct=1.0, settings=get_settings(),
                          paper_tradable=False)
        assert out["lots"] is None
        assert out["sized_against"] is None
        assert "account_value" not in out
        assert "paper account" in out["how_to_size"]
        # It still tells them what one lot costs, so they can size themselves.
        assert out["risk_per_lot"] == pytest.approx(275.0 * 65)

    def test_real_capital_the_user_states_is_used(self):
        from app.signals.agent.tools.trade_plan import _lot_sizing
        from app.config import get_settings

        out = _lot_sizing(22_890.0, 275.0, 65, {"portfolio": {}}, 1.0, get_settings(),
                          account_value=10_000_000)
        assert out["lots"] == int(100_000 // (275.0 * 65))
        assert out["sized_against"] == "the capital you gave me"

    def test_an_nse_equity_may_use_the_paper_account_but_it_is_labelled(self):
        """There the paper account IS the account the order would be placed
        in — but it must still be named as simulated."""
        from app.signals.agent.tools.trade_plan import _lot_sizing
        from app.config import get_settings

        book = {"portfolio": {"totalValue": 10_000_000, "cashBalance": 10_000_000}}
        out = _lot_sizing(22_890.0, 275.0, 65, book, 1.0, get_settings(),
                          paper_tradable=True)
        assert out["lots"] == int(100_000 // (275.0 * 65))
        assert "simulated" in out["sized_against"]

    def test_a_lot_count_of_zero_is_not_a_verdict_on_the_trade(self):
        """"Too large for that figure" is arithmetic. "Your account is too
        small, build capital first" is advice, and it was being given off a
        simulator's balance."""
        from app.signals.agent.tools.trade_plan import _lot_sizing
        from app.config import get_settings

        out = _lot_sizing(22_890.0, 275.0, 65, {"portfolio": {}}, 1.0, get_settings(),
                          account_value=100_000)
        assert out["lots"] == 0
        assert "not a judgement about the trade" in out["note"]
        assert "afford" in out["note"]

    def test_margin_is_named_as_the_thing_we_cannot_see(self):
        """Reporting notional as if it were the cash required understates what
        the broker blocks by roughly five times on an index future."""
        from app.signals.agent.tools.trade_plan import _lot_sizing
        from app.config import get_settings

        out = _lot_sizing(22_890.0, 275.0, 65, {"portfolio": {}}, 1.0, get_settings())
        assert "margin" in out["margin_note"].lower()
        assert "not notional" in out["margin_note"]

    def test_no_capital_at_all_still_gives_the_geometry(self):
        from app.signals.agent.tools.trade_plan import _lot_sizing
        from app.config import get_settings

        out = _lot_sizing(22_890.0, 275.0, 65, {"portfolio": {}}, 1.0, get_settings())
        assert out["lots"] is None
        assert out["risk_per_lot"] == pytest.approx(275.0 * 65)
        assert out["notional_per_lot"] == pytest.approx(22_890.0 * 65)

    def test_every_answer_warns_that_the_balance_is_simulated(self):
        """Carried on the result itself, not only when sizing is skipped: the
        agent read the paper balance as real capital and advised from it."""
        import asyncio
        from app.signals.agent.toolbox import AgentToolbox
        frame = TestTradeGeometry._frame()
        box = AgentToolbox("^NSEI", "NSE", frame, None)
        out = asyncio.get_event_loop().run_until_complete(
            box.execute("propose_trade", {"direction": "BUY", "interval": "1d"}))
        assert "PAPER" in out["capital_caveat"]
        assert "build capital" in out["capital_caveat"]
