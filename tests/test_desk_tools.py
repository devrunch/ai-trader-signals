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
