"""News the calendar cannot know about, and the paragraph nobody gets unless
they ask.

Two properties carry this file. Selection must be free — it reads the
sentiment and impact analysis the hourly pipeline already paid for, so a test
that lets a model call sneak in here is testing the wrong design. And the
same story must be pushed once, because the pipeline runs hourly and a
headline survives several ticks.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.events import shocks


def article(headline="Fed holds rates steady", description="", impacts=None,
            sentiment="NEUTRAL", article_id="a1"):
    return {"id": article_id, "headline": headline, "description": description,
            "source": "Reuters", "url": f"https://example.com/{article_id}",
            "publishedAt": "2026-09-22T09:00:00+00:00",
            "sentiment": sentiment, "impacts": impacts}


class _FakeRedis:
    def __init__(self):
        self.strings: dict[str, str] = {}

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.strings:
            return None
        self.strings[key] = value
        return True

    async def get(self, key):
        return self.strings.get(key)

    async def aclose(self):
        pass


def _redis(r):
    settings = MagicMock()
    settings.redis_url = "redis://fake"
    return (patch("app.events.shocks._client", return_value=r),
            patch("app.events.shocks.get_settings", return_value=settings))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestSelection:
    def test_an_impact_on_gold_is_selected(self):
        got = shocks.select(article(impacts=[
            {"symbol": "XAUUSD", "direction": "UP", "assetClass": "FOREX",
             "reason": "safe haven bid"}]))
        assert got is not None and "XAUUSD" in got.reason

    def test_geopolitics_the_analyser_scored_as_OTHER_is_still_selected(self):
        """This is most of what the feature exists for. A war headline rarely
        names an instrument, and the analyser is right not to invent one."""
        got = shocks.select(article(
            headline="Israel and Iran trade missile strikes overnight",
            impacts=[{"symbol": "", "assetClass": "OTHER", "reason": ""}]))
        assert got is not None
        assert got.reason in ("missile", "strikes")

    def test_an_ordinary_equity_story_is_not_selected(self):
        """Real analysis, not his problem: he trades gold and the majors."""
        assert shocks.select(article(
            headline="Infosys beats quarterly estimates",
            impacts=[{"symbol": "INFY", "assetClass": "NSE", "reason": "earnings"}])) is None

    def test_a_headline_with_no_impacts_and_no_shock_word_is_not_selected(self):
        assert shocks.select(article(impacts=[])) is None

    def test_an_unanalysable_article_is_not_treated_as_analysed_and_empty(self):
        """`impacts: None` means the analysis could not run; `[]` means it ran
        and honestly found nothing. Neither is a hit, but they are not the
        same fact and the code must not conflate them."""
        assert shocks.select(article(impacts=None)) is None
        assert shocks.select(article(headline="Tariffs raised on steel",
                                     impacts=None)) is not None

    def test_a_word_inside_another_word_is_not_a_shock(self):
        """'warehouse' is not a war. Without word boundaries this fires on
        half the feed and the channel gets muted."""
        assert shocks.select(article(
            headline="Warehouse operator expands in Rotterdam", impacts=[])) is None

    def test_selection_never_calls_a_model(self):
        """Selection is a filter over analysis already paid for. A model call
        here would make the feed cost scale with the feed, which is the thing
        the button exists to avoid."""
        with patch("app.llm.client.get_llm", side_effect=AssertionError("no LLM in selection")):
            assert shocks.select(article(headline="OPEC+ agrees output cut",
                                         impacts=[])) is not None


class TestRanking:
    def test_a_story_naming_an_instrument_outranks_one_that_matched_a_word(self):
        named = shocks.select(article(article_id="a1", impacts=[
            {"symbol": "XAUUSD", "assetClass": "FOREX", "reason": "haven"}]))
        worded = shocks.select(article(article_id="a2", headline="Sanctions widened",
                                       impacts=[]))
        assert shocks.rank([worded, named])[0] is named

    def test_ranking_keeps_the_feeds_own_order_within_a_tier(self):
        first = shocks.select(article(article_id="a1", headline="Tariffs announced",
                                      impacts=[]))
        second = shocks.select(article(article_id="a2", headline="Embargo extended",
                                       impacts=[]))
        assert shocks.rank([first, second]) == [first, second]


# ---------------------------------------------------------------------------
# Not sending the same story all day
# ---------------------------------------------------------------------------

class TestDedupe:
    @pytest.mark.asyncio
    async def test_a_story_is_pushed_once_however_many_ticks_it_survives(self):
        """The pipeline runs hourly and a story stays in the feed for hours."""
        r = _FakeRedis()
        one = shocks.select(article(headline="Tariffs raised on steel", impacts=[]))
        a, b = _redis(r)
        with a, b:
            assert await shocks.unseen([one]) == [one]
            assert await shocks.unseen([one]) == []

    @pytest.mark.asyncio
    async def test_two_ticks_racing_cannot_both_claim_the_same_story(self):
        r = _FakeRedis()
        one = shocks.select(article(headline="Sanctions widened", impacts=[]))
        a, b = _redis(r)
        with a, b:
            first, second = await shocks.unseen([one]), await shocks.unseen([one])
        assert len(first) + len(second) == 1

    @pytest.mark.asyncio
    async def test_nothing_in_means_nothing_out_and_no_redis_call(self):
        with patch("app.events.shocks._client",
                   side_effect=AssertionError("should not touch redis")):
            assert await shocks.unseen([]) == []


# ---------------------------------------------------------------------------
# The message, and the button behind it
# ---------------------------------------------------------------------------

class TestRender:
    def test_the_push_carries_no_interpretation(self):
        """It has to be worth reading without the model call. Interpretation
        is what the button is for."""
        shock = shocks.select(article(
            headline="OPEC+ agrees a surprise output cut", impacts=[]))
        text = shocks.render(shock)
        assert "OPEC+ agrees a surprise output cut" in text
        assert "Reuters" in text
        assert "https://example.com/a1" in text

    @pytest.mark.asyncio
    async def test_the_offer_survives_for_the_button_to_find(self):
        r = _FakeRedis()
        shock = shocks.select(article(headline="Tariffs raised", impacts=[]))
        a, b = _redis(r)
        with a, b:
            token = await shocks.offer(shock)
            assert (await shocks.offered(token))["headline"] == "Tariffs raised"

    @pytest.mark.asyncio
    async def test_a_tap_on_a_story_that_aged_out_returns_nothing(self):
        r = _FakeRedis()
        a, b = _redis(r)
        with a, b:
            assert await shocks.offered("gone") is None

    def test_the_brief_button_fits_telegrams_callback_limit(self):
        from app.events import telegram as tg
        data = tg.brief_button("x" * 24)["inline_keyboard"][0][0]["callback_data"]
        assert len(data.encode()) <= 64


class TestOnDemandOnly:
    @pytest.mark.asyncio
    async def test_pushing_shocks_never_writes_a_summary(self):
        """The whole point: selection is free, prose is not. If this job ever
        calls the model, the feed's cost scales with the feed."""
        from app.worker import news_job

        r = _FakeRedis()
        a, b = _redis(r)
        with a, b, \
             patch("app.events.telegram.send", AsyncMock(return_value=True)), \
             patch("app.events.shock_brief.write",
                   AsyncMock(side_effect=AssertionError("no brief on push"))):
            sent = await news_job.push_shocks([
                article(headline="Tariffs raised on steel", impacts=[]),
                article(article_id="a2", headline="Infosys beats estimates",
                        impacts=[{"symbol": "INFY", "assetClass": "NSE"}]),
            ])
        assert sent == 1, "only the shock should have gone out"

    @pytest.mark.asyncio
    async def test_a_telegram_outage_does_not_fail_the_news_run(self):
        """The terminal reads the published feed whether or not Telegram is
        up; a push problem must not take the run down with it."""
        from app.worker import news_job

        r = _FakeRedis()
        a, b = _redis(r)
        with a, b, patch("app.events.telegram.send",
                         AsyncMock(side_effect=RuntimeError("telegram down"))):
            assert await news_job.push_shocks(
                [article(headline="Tariffs raised", impacts=[])]) == 0

    @pytest.mark.asyncio
    async def test_a_loud_hour_is_capped(self):
        """A phone that buzzes six times an hour gets muted, and a muted
        channel is worth nothing."""
        from app.worker import news_job

        r = _FakeRedis()
        a, b = _redis(r)
        loud = [article(article_id=f"a{i}", headline=f"Sanctions round {i}", impacts=[])
                for i in range(10)]
        with a, b, patch("app.events.telegram.send", AsyncMock(return_value=True)):
            assert await news_job.push_shocks(loud) == shocks.MAX_PER_RUN


class TestFalsePositives:
    """Each of these selected on the live feed and should not have."""

    def test_a_crypto_column_is_not_a_shock(self):
        """Selected live on 'rate hike'. This desk trades gold and the
        majors, and the pipeline's own analyser marks crypto informational
        because no crypto trading exists anywhere in this app."""
        assert shocks.select(article(
            headline="Crypto enjoys bullish bounce post-Fed rate hike: Crypto Week Ahead",
            impacts=[])) is None

    def test_routine_rate_commentary_is_the_calendars_job(self):
        """Scheduled decisions are on the calendar, and the phrase appears in
        every market column written."""
        assert shocks.select(article(
            headline="Analysts split on the pace of the next rate cut",
            impacts=[])) is None

    def test_an_unscheduled_move_is_still_a_shock(self):
        assert shocks.select(article(
            headline="SNB announces an emergency intervention in the franc",
            impacts=[])) is not None

    def test_a_crypto_headline_that_also_moves_gold_is_still_selected(self):
        """The exclusion is for stories that move nothing here, not a veto on
        the word appearing."""
        assert shocks.select(article(
            headline="Bitcoin and gold rally as the dollar slides",
            impacts=[{"symbol": "XAUUSD", "assetClass": "FOREX", "reason": "haven"}])) is not None

    def test_the_war_headline_that_did_select_still_does(self):
        assert shocks.select(article(
            headline="Russia's pro-Putin party set to win election as Ukraine "
                     "exposes war's growing reach into Moscow", impacts=[])) is not None
