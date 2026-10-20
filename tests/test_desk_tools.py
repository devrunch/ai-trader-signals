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
        because a short is bad analysis. The flag is a fact about the trade —
        the old wording named the user's paper account and their broker, which
        is the commentary this tool exists to stop making."""
        out = await self._propose("SELL")
        assert out["short"] is True
        assert out["reward_risk"] is not None
        blob = repr(out).lower()
        for word in ("account", "broker", "afford", "capital"):
            assert word not in blob, word

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

    def test_it_reports_what_one_lot_costs_and_nothing_about_money(self):
        """It cannot see what the user trades with. The only balance available
        belongs to a paper simulator, and every attempt to use it became a
        claim about their finances — first "your 1,00,000 account is too small,
        build capital first", then a paragraph explaining why it was not
        advising. Both are noise between candles."""
        from app.signals.agent.tools.trade_plan import _contract_risk

        out = _contract_risk(risk_per_unit=275.0, lot_size=65)
        assert out == {"unit": "lots", "lot_size": 65,
                       "risk_per_lot": pytest.approx(275.0 * 65)}

    def test_the_trade_path_never_reads_an_account(self):
        """A grep, deliberately: the balance kept finding its way back in."""
        from pathlib import Path
        import app.signals.agent.tools.trade_plan as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        for banned in ("ctx.book", "portfolio_tools", "totalValue",
                       "initialCapital", "cashBalance"):
            assert banned not in source, f"{banned} is back in the trade path"

    @pytest.mark.asyncio
    async def test_a_proposed_trade_carries_no_account_language(self):
        from app.signals.agent.toolbox import AgentToolbox
        box = AgentToolbox("^NSEI", "NSE", TestTradeGeometry._frame(), None)
        out = await box.execute("propose_trade", {"direction": "BUY", "interval": "1d"})
        blob = repr(out).lower()
        for word in ("account", "capital", "afford", "simulated", "broker"):
            assert word not in blob, f"{word!r} leaked into the trade"


class TestAnswerShape:
    """A trader reads on a phone, between candles.

    The prompt said "be concise" and gave no shape, so the model produced a
    markdown report: Analysis Summary, Trade Analysis, Critical Constraints,
    Alternatives, Key Levels, Bottom Line. Correct, and nobody reads it.
    """

    @staticmethod
    def _prompt():
        from app.signals.prompts import chat_system_prompt
        return chat_system_prompt("NIFTY1!", "NFO", 22895.2)

    def test_the_first_line_is_the_verdict(self):
        p = self._prompt()
        assert "FIRST LINE is the answer" in p
        assert "no restating the question" in p

    def test_report_furniture_is_named_and_banned(self):
        """Named explicitly, because the model produced every one of these."""
        p = self._prompt()
        for banned in ("Analysis Summary", "Bottom Line", "Key Levels", "'##' headings"):
            assert banned in p, banned

    def test_it_says_when_to_expand(self):
        """A blanket "be short" makes it unhelpful on "why?" — the length has
        to follow the question."""
        p = self._prompt()
        assert "'Why?' earns a paragraph" in p

    def test_the_agent_knows_its_own_confluence_gate(self):
        """Asked "what are the metrics of your confluence?", it answered "I
        don't have a specific confluence or set of metrics that I track" and
        called no tools — while a 17-indicator gate was deciding every trade
        it proposed."""
        p = self._prompt()
        assert "confluence gate" in p
        assert "never say you have no such metric" in p
        for family in ("trend", "momentum", "location"):
            assert family in p

    def test_the_numbers_line_is_shown_not_described(self):
        p = self._prompt()
        assert "Entry 22,894 | Stop 22,806" in p

    def test_a_trade_question_must_cost_a_propose_trade(self):
        """Asked in the terminal, the agent answered from read_chart alone and
        produced no entry, stop or target at all — correct prose, no trade.
        With the card, skipping the tool means the user sees no numbers."""
        p = self._prompt()
        assert "MUST call propose_trade" in p
        assert "is not an answer to that question" in p


class TestTradeCardPlumbing:
    """The card is driven by ctx.results["trade"], not retyped by the model.

    That is the point: a card can only appear when the geometry was actually
    computed, so a turn answering from the chart alone shows no numbers
    instead of plausible ones.
    """

    @pytest.mark.asyncio
    async def test_a_passing_trade_is_published_to_the_ui(self):
        from app.signals.agent.toolbox import AgentToolbox
        box = AgentToolbox("^NSEI", "NSE", TestTradeGeometry._frame(), None)
        await box.execute("propose_trade", {"direction": "BUY", "interval": "1d"})
        card = box.results.get("trade")
        assert card is not None
        for key in ("verdict", "entry", "stop", "target", "reward_risk", "direction"):
            assert key in card, key

    @pytest.mark.asyncio
    async def test_a_rejected_trade_is_published_too(self):
        """A rejection is an answer. Publishing only the passes would leave the
        card blank exactly when the user most needs to see why not."""
        from app.signals.agent.toolbox import AgentToolbox
        box = AgentToolbox("^NSEI", "NSE", TestTradeGeometry._frame(), None)
        out = await box.execute("propose_trade", {"direction": "BUY", "interval": "1d"})
        if out["verdict"] == "rejected":
            assert box.results["trade"]["verdict"] == "rejected"
            assert box.results["trade"]["rejected_because"]

    @pytest.mark.asyncio
    async def test_an_error_publishes_no_card(self):
        """An error is not a trade. Publishing a card for one would put an
        empty grid of levels beside a message saying it could not be done."""
        from app.signals.agent.toolbox import AgentToolbox
        box = AgentToolbox("^NSEI", "NSE", TestTradeGeometry._frame(), None)
        out = await box.execute("propose_trade", {"direction": "MAYBE", "interval": "1d"})
        assert "error" in out
        assert "trade" not in box.results


    @pytest.mark.asyncio
    async def test_the_trade_carries_every_confluence_vote(self):
        """Asked which indicators disagreed, the agent paraphrased the chart,
        because the gate handed it a percentage and nothing else."""
        out = await TestTradeGeometry()._propose("BUY")
        panel = out["confluence"]
        assert panel["votes"] and len(panel["votes"]) >= 14
        assert set(panel["families"]) <= {"trend", "momentum", "location"}
        assert "agreed" in panel and "disagreed" in panel



class TestEntryModes:
    """Entering at the last price is what rejected everything.

    Scanning 29 Nifty names, 24 failed on reward:risk alone — HCLTECH at
    94.4% agreement, POWERGRID at 95.2%. The agreement was fine; a stop one
    ATR from the last traded price leaves no room to the next level, so the
    reward was gone before the setup was judged. Entry at a pullback level
    took the pass rate from 1 of 29 to 8.
    """

    async def _at(self, mode, direction="BUY"):
        from app.signals.agent.toolbox import AgentToolbox
        box = AgentToolbox("^NSEI", "NSE", TestTradeGeometry._frame(), None)
        return await box.execute("propose_trade", {
            "direction": direction, "interval": "1d", "entry_mode": mode})

    @pytest.mark.asyncio
    async def test_market_enters_at_the_last_price(self):
        out = await self._at("market")
        assert out["entry"] == out["last_price"]
        assert out["order_type"] == "market"

    @pytest.mark.asyncio
    async def test_a_level_entry_is_a_limit_order_away_from_the_price(self):
        out = await self._at("level")
        assert out["order_type"] == "limit"
        assert out["distance_from_price_pct"] is not None

    @pytest.mark.asyncio
    async def test_the_drift_gate_only_judges_a_market_order(self):
        """It exists to stop a price being invented far from the market, not
        to forbid a limit order. Applied to one, it rejected every level entry
        on the scan even where reward:risk had improved past the floor."""
        market = await self._at("market")
        level = await self._at("level")
        assert any(g["gate"] == "entry_drift" for g in market["gates"])
        assert not any(g["gate"] == "entry_drift" for g in level["gates"])

    @pytest.mark.asyncio
    async def test_a_retest_target_is_not_the_level_being_retested(self):
        """It was. The target became the very level the entry had just
        crossed, a tenth of an ATR away, so every retest scored R:R 0.1."""
        out = await self._at("retest")
        if out.get("reward_risk") is not None:
            assert out["reward_risk"] > 0.2

    @pytest.mark.asyncio
    async def test_an_explicit_entry_beats_the_mode(self):
        from app.signals.agent.toolbox import AgentToolbox
        box = AgentToolbox("^NSEI", "NSE", TestTradeGeometry._frame(), None)
        out = await box.execute("propose_trade", {
            "direction": "BUY", "interval": "1d", "entry_mode": "level", "entry": 12345.0})
        assert out["entry"] == 12345.0
        assert "gave me" in out["entry_from"]


class TestContextOnTheTrade:
    async def _trade(self, direction="BUY"):
        from app.signals.agent.toolbox import AgentToolbox
        box = AgentToolbox("^NSEI", "NSE", TestTradeGeometry._frame(), None)
        return await box.execute("propose_trade", {"direction": direction, "interval": "1d"})

    @pytest.mark.asyncio
    async def test_it_says_whether_a_trend_reading_means_anything(self):
        """ADX below 20 is a range, and in a range an EMA cross and a
        SuperTrend flip are noise with a direction attached."""
        out = await self._trade()
        assert out["regime"]["kind"] in ("range", "weak", "trending", "unknown")
        assert out["regime"]["note"]

    @pytest.mark.asyncio
    async def test_a_stretched_entry_is_flagged(self):
        """TCS scored 94.4% for a BUY at RSI 73 — high agreement precisely
        because price had already run. rsi_extreme only vetoes past 75."""
        from app.signals.agent.tools.trade_plan import _stretched
        out = _stretched("BUY", entry=110.0, indicators={"ema20": 100.0, "rsi": 73.0}, atr=5.0)
        assert out and out["atr_from_ema20"] == pytest.approx(2.0)
        assert "already happened" in out["note"]

    def test_a_normal_entry_is_not_flagged_as_stretched(self):
        from app.signals.agent.tools.trade_plan import _stretched
        assert _stretched("BUY", 101.0, {"ema20": 100.0, "rsi": 55.0}, 5.0) is None

    def test_selling_above_the_mean_is_not_chasing(self):
        """Stretched only counts in the direction of the trade."""
        from app.signals.agent.tools.trade_plan import _stretched
        assert _stretched("SELL", 110.0, {"ema20": 100.0, "rsi": 73.0}, 5.0) is None


class TestBreakoutProbability:
    """The one measured-probability reading. Browser-only TypeScript until
    now, so every answer about a level was reasoning: the agent could say how
    far a level was and never how often price had actually reached it.

    These mirror the frontend's own tests — both sides must stay in step.
    """

    @staticmethod
    def _bars(rows):
        import pandas as pd
        idx = pd.date_range("2026-01-01", periods=len(rows), freq="1D")
        return pd.DataFrame(
            [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in rows],
            index=idx)

    def test_it_counts_a_level_as_reached_only_when_price_gets_there(self):
        from app.signals import breakout
        green = (100, 110, 100, 108)
        bars = self._bars([green, (108, 115, 105, 100), green, (108, 115, 105, 100),
                           green, (108, 115, 105, 100), green, (108, 109, 105, 100),
                           green])
        out = breakout.compute(bars, 1.0)
        assert out.last_candle_green is True
        assert out.sample == 4
        assert out.upper[0].probability == 75.0

    def test_an_untested_level_is_none_not_zero_percent(self):
        """0% reads as "this cannot happen"."""
        from app.signals import breakout
        doji = (100, 101, 99, 100)
        out = breakout.compute(self._bars([doji, doji, doji]), 1.0)
        assert out.sample == 0
        assert all(lvl.probability is None for lvl in out.upper)

    def test_a_doji_predecessor_is_not_given_a_colour(self):
        from app.signals import breakout
        bars = self._bars([(100, 101, 99, 100), (108, 110, 95, 100),
                           (100, 110, 100, 108), (108, 120, 105, 100),
                           (100, 110, 100, 108)])
        assert breakout.compute(bars, 1.0).sample == 1

    def test_a_further_level_is_never_likelier_than_a_nearer_one(self):
        from app.signals import breakout
        rows = []
        for i in range(20):
            rows.append((100, 110, 100, 108))
            rows.append((108, 110 + (i % 5), 100 - (i % 3), 100))
        rows.append((100, 110, 100, 108))
        out = breakout.compute(self._bars(rows), 0.5)
        probs = [lvl.probability or 0 for lvl in out.upper]
        assert all(probs[i] <= probs[i - 1] for i in range(1, len(probs)))

    def test_bias_declines_to_call_a_coin_flip(self):
        """50.4 against 49.6 is not a read, and presenting it as one is how
        someone trades noise with a number attached."""
        from app.signals.breakout import Breakout, Level, bias
        flip = Breakout(upper=[Level(0, 1, 50.4)], lower=[Level(0, 1, 49.6)],
                        last_candle_green=True, sample=50)
        assert bias(flip) is None
        clear = Breakout(upper=[Level(0, 1, 70.0)], lower=[Level(0, 1, 30.0)],
                         last_candle_green=True, sample=50)
        assert bias(clear) == {"direction": "up", "edge_pct": 40.0}

    def test_reach_probability_answers_about_an_actual_price(self):
        """"How often does it get to 10,982?" is the question an agent gets
        asked, not "what is the ladder of one-percent steps"."""
        from app.signals import breakout
        rows = [(100, 110, 100, 108), (108, 115, 105, 100)] * 15 + [(100, 110, 100, 108)]
        out = breakout.reach_probability(self._bars(rows), target=113.0)
        assert out["direction"] == "above"
        assert out["sample"] > 0
        assert out["probability_pct"] is not None

    def test_a_price_inside_the_last_bar_is_not_a_forecast(self):
        from app.signals import breakout
        rows = [(100, 110, 100, 108), (108, 115, 105, 100), (100, 110, 100, 108)]
        out = breakout.reach_probability(self._bars(rows), target=105.0)
        assert out["probability_pct"] is None
        assert "inside the last bar" in out["note"]

    def test_the_indicator_catalogue_serves_it(self):
        from app.signals import indicators as ind
        assert "breakout" in ind.available_indicators()
        rows = [(100, 110, 100, 108), (108, 115, 105, 100)] * 30 + [(100, 110, 100, 108)]
        out = ind.compute(self._bars(rows), ["breakout"])
        assert out["breakout_sample"] > 0
        assert out["breakout_up_pct"] is not None
