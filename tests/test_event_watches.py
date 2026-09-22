"""Watches: the button that closes the loop on a brief.

A brief predicts. A watch reports what actually happened. The tests that
matter here are the ones about *not* reporting: a window whose minute bars have
not arrived yet must be retried, not rendered as a flat market, and it must
eventually stop being retried.

The realised move is measured off the LIVE router, not the historical
Dukascopy feed. Measured on the box, Dukascopy's gold minute bars run six to
twelve hours behind; a watch reporting 35 minutes after a print would have
found nothing, every time.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.events import bot, watches

WHEN = datetime(2026, 9, 22, 12, 30, tzinfo=UTC)


class _FakeRedis:
    def __init__(self):
        self.strings: dict[str, str] = {}
        self.zset: dict[str, float] = {}
        self.members: set[str] = set()
        self.closed = False

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.strings:
            return None
        self.strings[key] = value
        return True

    async def get(self, key):
        return self.strings.get(key)

    async def zadd(self, key, mapping):
        self.zset.update(mapping)

    async def zrangebyscore(self, key, lo, hi):
        hi = float(hi)
        return sorted(m for m, score in self.zset.items() if score <= hi)

    async def zrem(self, key, member):
        self.zset.pop(member, None)

    async def sadd(self, key, *values):
        self.members.update(str(v) for v in values)

    async def srem(self, key, *values):
        self.members.difference_update(str(v) for v in values)

    async def smembers(self, key):
        return set(self.members)

    async def incr(self, key):
        self.strings[key] = str(int(self.strings.get(key, 0)) + 1)
        return int(self.strings[key])

    async def expire(self, key, seconds):
        pass

    async def aclose(self):
        self.closed = True


def _redis(r):
    settings = MagicMock()
    settings.redis_url = "redis://fake"
    settings.telegram_invite_secret = "s3cret"
    settings.telegram_chat_id = ""
    return (
        patch("app.events.watches._client", return_value=r),
        patch("app.events.watches.get_settings", return_value=settings),
        patch("app.events.subscribers._client", return_value=r),
        patch("app.events.subscribers.get_settings", return_value=settings),
        patch("app.events.bot.get_settings", return_value=settings),
    )


class TestOfferAndArm:
    @pytest.mark.asyncio
    async def test_the_same_event_offered_twice_is_one_watch(self):
        """The brief and the nudge both carry the button. Tapping both must
        not queue two reports for the same release."""
        r = _FakeRedis()
        with_ = _redis(r)
        with with_[0], with_[1]:
            t1 = await watches.offer("USD:CPI", "USD", "CPI m/m", WHEN)
            t2 = await watches.offer("USD:CPI", "USD", "CPI m/m", WHEN)
            assert t1 == t2
            await watches.arm(t1, "42")
            await watches.arm(t2, "42")
        assert len(r.zset) == 1

    @pytest.mark.asyncio
    async def test_two_people_tapping_the_same_message_each_get_a_report(self):
        r = _FakeRedis()
        with_ = _redis(r)
        with with_[0], with_[1]:
            token = await watches.offer("USD:CPI", "USD", "CPI m/m", WHEN)
            await watches.arm(token, "1")
            await watches.arm(token, "2")
        assert len(r.zset) == 2

    @pytest.mark.asyncio
    async def test_a_tap_on_an_expired_offer_arms_nothing(self):
        r = _FakeRedis()
        with_ = _redis(r)
        with with_[0], with_[1]:
            assert await watches.arm("deadbeefdead", "42") is None
        assert r.zset == {}

    @pytest.mark.asyncio
    async def test_the_report_is_owed_after_the_release_not_at_it(self):
        """Measuring at T+0 would price a window that does not exist yet."""
        r = _FakeRedis()
        with_ = _redis(r)
        with with_[0], with_[1]:
            token = await watches.offer("USD:CPI", "USD", "CPI m/m", WHEN)
            await watches.arm(token, "42")
            assert await watches.due(now=WHEN) == []
            assert len(await watches.due(now=WHEN + watches.REPORT_DELAY)) == 1


class TestRetry:
    @pytest.mark.asyncio
    async def test_a_window_that_is_not_there_yet_comes_back_later(self):
        r = _FakeRedis()
        with_ = _redis(r)
        with with_[0], with_[1]:
            token = await watches.offer("USD:CPI", "USD", "CPI m/m", WHEN)
            await watches.arm(token, "42")
            member = next(iter(r.zset))
            assert await watches.retry(member, now=WHEN) is True
            assert await watches.due(now=WHEN) == []
            assert len(await watches.due(now=WHEN + timedelta(minutes=11))) == 1

    @pytest.mark.asyncio
    async def test_retrying_forever_is_not_an_option(self):
        """newsd would re-download the same missing window every five minutes
        for as long as Redis remembers it."""
        r = _FakeRedis()
        with_ = _redis(r)
        with with_[0], with_[1]:
            token = await watches.offer("USD:CPI", "USD", "CPI m/m", WHEN)
            await watches.arm(token, "42")
            now = WHEN
            outcomes = []
            for _ in range(watches.MAX_ATTEMPTS + 1):
                if not r.zset:
                    break
                outcomes.append(await watches.retry(next(iter(r.zset)), now=now))
                now += timedelta(minutes=11)
        assert outcomes[-1] is False
        assert r.zset == {}

    @pytest.mark.asyncio
    async def test_a_watch_whose_offer_expired_is_dropped_not_reported(self):
        r = _FakeRedis()
        with_ = _redis(r)
        with with_[0], with_[1]:
            token = await watches.offer("USD:CPI", "USD", "CPI m/m", WHEN)
            await watches.arm(token, "42")
            del r.strings[watches.OFFER_PREFIX + token]
            assert await watches.due(now=WHEN + watches.REPORT_DELAY) == []
        assert r.zset == {}


class TestTheButton:
    @staticmethod
    def _tap(token, chat_id=42):
        return {"update_id": 9, "callback_query": {
            "id": "cb1", "data": f"w:{token}",
            "message": {"chat": {"id": chat_id}}}}

    @pytest.mark.asyncio
    async def test_an_enrolled_chat_tapping_the_button_arms_a_watch(self):
        r = _FakeRedis()
        r.members.add("42")
        r.strings["events.subscribers.seeded"] = "1"
        ack = AsyncMock(return_value=True)
        p = _redis(r)
        with p[0], p[1], p[2], p[3], p[4], \
             patch("app.events.bot.telegram.answer_callback", ack):
            token = await watches.offer("USD:CPI", "USD", "CPI m/m", WHEN)
            reply = await bot.handle(self._tap(token))
        assert len(r.zset) == 1
        assert "CPI m/m" in reply.text
        ack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_tap_from_a_chat_that_never_enrolled_arms_nothing(self):
        """Buttons ride in forwardable messages, so a tap can arrive from
        anywhere the message was pasted."""
        r = _FakeRedis()
        r.strings["events.subscribers.seeded"] = "1"
        ack = AsyncMock(return_value=True)
        p = _redis(r)
        with p[0], p[1], p[2], p[3], p[4], \
             patch("app.events.bot.telegram.answer_callback", ack):
            token = await watches.offer("USD:CPI", "USD", "CPI m/m", WHEN)
            assert await bot.handle(self._tap(token, chat_id=999)) is None
        assert r.zset == {}
        ack.assert_awaited_once_with("cb1", bot.WATCH_DENIED)

    @pytest.mark.asyncio
    async def test_a_tap_is_always_acknowledged_even_when_it_does_nothing(self):
        """Telegram spins the button until answerCallbackQuery, so an
        unacknowledged tap looks broken even when it worked."""
        r = _FakeRedis()
        r.strings["events.subscribers.seeded"] = "1"
        ack = AsyncMock(return_value=True)
        p = _redis(r)
        with p[0], p[1], p[2], p[3], p[4], \
             patch("app.events.bot.telegram.answer_callback", ack):
            await bot.handle({"update_id": 9, "callback_query": {
                "id": "cb2", "data": "garbage", "message": {"chat": {"id": 42}}}})
        ack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_callback_data_fits_telegrams_limit(self):
        """64 bytes, which will not hold an event title — hence the token."""
        from app.events import telegram as tg
        token = watches.token_for("USD:Core CPI m/m (a very long title indeed)", WHEN)
        data = tg.watch_button(token)["inline_keyboard"][0][0]["callback_data"]
        assert len(data.encode()) <= 64


class TestRealisedMove:
    """Measured off the live feed, not the historical one.

    Dukascopy's gold minute bars run six to twelve hours behind on the box
    (T-6h empty, T-12h full). A watch reporting 35 minutes after a print would
    have found nothing, every time. These tests pin the source as much as the
    maths.
    """

    @staticmethod
    def _frame(minutes: int):
        import pandas as pd
        rows, index = [], []
        for minute in range(-1, minutes + 1):
            index.append(WHEN + timedelta(minutes=minute))
            rows.append({"open": 100.0, "high": 102.0, "low": 100.0,
                         "close": 100.0 if minute <= 0 else 100 + minute * 0.1})
        return pd.DataFrame(rows, index=pd.DatetimeIndex(index))

    @pytest.mark.asyncio
    async def test_it_reads_the_live_router_not_the_historical_feed(self):
        from app.events import reaction
        stale = AsyncMock()
        with patch("app.events.reaction.market_data_router.get_historical_df",
                   AsyncMock(return_value=self._frame(30))),              patch("app.events.reaction._window", stale):
            await reaction.realised_move("XAUUSD", WHEN)
        stale.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_empty_feed_reports_nothing(self):
        """Zeros here would turn a late feed into a report of a flat market."""
        from app.events import reaction
        with patch("app.events.reaction.market_data_router.get_historical_df",
                   AsyncMock(return_value=None)):
            assert await reaction.realised_move("XAUUSD", WHEN) is None

    @pytest.mark.asyncio
    async def test_bars_that_stop_short_of_the_marks_report_nothing(self):
        from app.events import reaction
        with patch("app.events.reaction.market_data_router.get_historical_df",
                   AsyncMock(return_value=self._frame(2))):
            assert await reaction.realised_move("XAUUSD", WHEN) is None

    @pytest.mark.asyncio
    async def test_the_fifteen_minute_mark_alone_is_still_worth_reporting(self):
        """Half an answer beats waiting for the other half."""
        from app.events import reaction
        with patch("app.events.reaction.market_data_router.get_historical_df",
                   AsyncMock(return_value=self._frame(16))):
            moved = await reaction.realised_move("XAUUSD", WHEN)
        assert moved["move_15m"] == pytest.approx(1.5, abs=0.01)
        assert moved["move_30m"] is None

    @pytest.mark.asyncio
    async def test_a_complete_window_is_measured_at_both_marks(self):
        from app.events import reaction
        with patch("app.events.reaction.market_data_router.get_historical_df",
                   AsyncMock(return_value=self._frame(30))):
            moved = await reaction.realised_move("XAUUSD", WHEN)
        assert moved["move_15m"] == pytest.approx(1.5, abs=0.01)
        assert moved["move_30m"] == pytest.approx(3.0, abs=0.01)
