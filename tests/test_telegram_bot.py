"""The inbound Telegram path: registry, commands, webhook.

The desk could talk and not listen. These tests cover the listening half —
who is enrolled, what each command answers, and what the webhook does with an
update it cannot handle.

The refusals are tested for being *identical* to each other on purpose. A bot
whose replies differ between "wrong key", "rate limited" and "unknown command"
tells a stranger exactly how far they got.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.events import bot, subscribers


class _FakeRedis:
    """Enough Redis for the registry: a set, some counters, some strings."""

    def __init__(self, initial: dict | None = None):
        self.store: dict[str, object] = dict(initial or {})
        self.expiries: dict[str, int] = {}
        self.closed = False

    async def sadd(self, key, *values):
        s = self.store.setdefault(key, set())
        before = len(s)
        s.update(str(v) for v in values)
        return len(s) - before

    async def srem(self, key, *values):
        s = self.store.get(key)
        if not s:
            return 0
        before = len(s)
        s.difference_update(str(v) for v in values)
        return before - len(s)

    async def smembers(self, key):
        return set(self.store.get(key, set()))

    async def scard(self, key):
        return len(self.store.get(key, set()))

    async def get(self, key):
        v = self.store.get(key)
        return v if v is None else str(v)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def incr(self, key):
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    async def expire(self, key, seconds):
        self.expiries[key] = seconds

    async def aclose(self):
        self.closed = True


def _settings(invite="s3cret", chat_id="", webhook="hook"):
    s = MagicMock()
    s.redis_url = "redis://fake"
    s.telegram_invite_secret = invite
    s.telegram_chat_id = chat_id
    s.telegram_webhook_secret = webhook
    s.telegram_bot_token = "token"
    return s


def _fixture(redis_obj, settings=None):
    """Patch both modules' view of Redis and settings for one test."""
    settings = settings or _settings()
    return (
        patch("app.events.subscribers._client", return_value=redis_obj),
        patch("app.events.subscribers.get_settings", return_value=settings),
        patch("app.events.bot.get_settings", return_value=settings),
    )


def _update(text: str, chat_id: int = 42) -> dict:
    return {"update_id": 1, "message": {"message_id": 7, "chat": {"id": chat_id},
                                        "text": text}}


async def _handle(text, redis_obj, settings=None, chat_id=42):
    a, b, c = _fixture(redis_obj, settings)
    with a, b, c:
        return await bot.handle(_update(text, chat_id))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

class TestSubscribers:
    @pytest.mark.asyncio
    async def test_the_configured_chat_is_seeded_so_deployment_does_not_unsubscribe_it(self):
        """Before this module there was one recipient, in config. An empty
        registry on first boot would silently stop his briefs."""
        r = _FakeRedis()
        a, b, c = _fixture(r, _settings(chat_id="999"))
        with a, b, c:
            assert await subscribers.all() == ["999"]

    @pytest.mark.asyncio
    async def test_a_registry_emptied_by_stop_is_not_re_seeded(self):
        """An empty set means either 'never seeded' or 'they asked to leave'.
        Seeding on emptiness would re-subscribe someone who just left."""
        r = _FakeRedis()
        a, b, c = _fixture(r, _settings(chat_id="999"))
        with a, b, c:
            await subscribers.remove("999")
            assert await subscribers.all() == []

    @pytest.mark.asyncio
    async def test_no_configured_chat_still_marks_the_seed_done(self):
        r = _FakeRedis()
        a, b, c = _fixture(r, _settings(chat_id=""))
        with a, b, c:
            assert await subscribers.all() == []
            await subscribers.add("5")
            assert await subscribers.all() == ["5"]

    @pytest.mark.asyncio
    async def test_adding_is_idempotent(self):
        r = _FakeRedis()
        a, b, c = _fixture(r, _settings(chat_id=""))
        with a, b, c:
            await subscribers.add("5")
            await subscribers.add("5")
            assert await subscribers.all() == ["5"]

    @pytest.mark.asyncio
    async def test_removing_someone_who_was_never_there_is_not_an_error(self):
        r = _FakeRedis()
        a, b, c = _fixture(r, _settings(chat_id=""))
        with a, b, c:
            await subscribers.remove("nobody")
            assert await subscribers.all() == []


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

class TestStart:
    @pytest.mark.asyncio
    async def test_the_right_key_enrolls_the_chat(self):
        r = _FakeRedis()
        reply = await _handle("/start s3cret", r)
        assert reply.chat_id == "42"
        assert "42" in await r.smembers(subscribers.SET_KEY)

    @pytest.mark.asyncio
    async def test_the_wrong_key_does_not_enroll(self):
        r = _FakeRedis()
        reply = await _handle("/start wrong", r)
        assert reply.text == bot.NOT_RECOGNISED
        assert "42" not in await r.smembers(subscribers.SET_KEY)

    @pytest.mark.asyncio
    async def test_a_bare_start_is_refused_exactly_like_a_wrong_key(self):
        """A different reply for 'no key' tells a prober the command takes
        an argument."""
        r = _FakeRedis()
        assert (await _handle("/start", r)).text == bot.NOT_RECOGNISED

    @pytest.mark.asyncio
    async def test_surrounding_whitespace_in_the_key_is_tolerated(self):
        """Phone keyboards add a trailing space; that is not a wrong key."""
        r = _FakeRedis()
        await _handle("/start  s3cret  ", r)
        assert "42" in await r.smembers(subscribers.SET_KEY)

    @pytest.mark.asyncio
    async def test_the_group_chat_command_suffix_is_stripped(self):
        r = _FakeRedis()
        await _handle("/start@adizx_bot s3cret", r)
        assert "42" in await r.smembers(subscribers.SET_KEY)

    @pytest.mark.asyncio
    async def test_an_unset_invite_secret_enrolls_nobody(self):
        """Empty config must not mean empty key accepted — that is a bot open
        to anyone who sends '/start '."""
        r = _FakeRedis()
        reply = await _handle("/start ", r, _settings(invite=""))
        assert reply.text == bot.NOT_RECOGNISED
        assert await r.smembers(subscribers.SET_KEY) == set()


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_the_sixth_attempt_in_an_hour_is_refused(self):
        r = _FakeRedis()
        for _ in range(bot.START_ATTEMPTS_PER_HOUR):
            await _handle("/start wrong", r)
        reply = await _handle("/start s3cret", r)
        assert reply.text == bot.NOT_RECOGNISED
        assert "42" not in await r.smembers(subscribers.SET_KEY), \
            "a correct key must not slip through after the limit trips"

    @pytest.mark.asyncio
    async def test_the_limit_is_per_chat(self):
        r = _FakeRedis()
        for _ in range(bot.START_ATTEMPTS_PER_HOUR + 1):
            await _handle("/start wrong", r, chat_id=1)
        await _handle("/start s3cret", r, chat_id=2)
        assert "2" in await r.smembers(subscribers.SET_KEY)

    @pytest.mark.asyncio
    async def test_the_counter_expires_so_a_lockout_is_not_permanent(self):
        r = _FakeRedis()
        await _handle("/start wrong", r)
        assert r.expiries[bot.ATTEMPTS_PREFIX + "42"] == bot.ATTEMPT_WINDOW_SECONDS


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_removes_an_enrolled_chat(self):
        r = _FakeRedis()
        await _handle("/start s3cret", r)
        await _handle("/stop", r)
        assert await r.smembers(subscribers.SET_KEY) == set()

    @pytest.mark.asyncio
    async def test_stop_is_indistinguishable_from_any_other_word_to_a_stranger(self):
        """A stranger must not learn that /stop exists. Confirming an
        unsubscribe to someone who was never subscribed does exactly that."""
        r = _FakeRedis()
        assert (await _handle("/stop", r)).text == (await _handle("hello", r)).text

    @pytest.mark.asyncio
    async def test_stopping_twice_is_a_no_op_not_an_error(self):
        r = _FakeRedis()
        await _handle("/start s3cret", r)
        await _handle("/stop", r)
        assert (await _handle("/stop", r)).text == bot.ENABLE_HINT
        assert await r.smembers(subscribers.SET_KEY) == set()


class TestStatus:
    @pytest.mark.asyncio
    async def test_status_reports_the_desk_state_to_an_enrolled_chat(self):
        r = _FakeRedis()
        await _handle("/start s3cret", r)
        summary = {"armed": 3, "next_title": "USD Core CPI m/m",
                   "next_due": time.time() + 7200, "last_agenda": time.time() - 3600,
                   "last_watchdog": "clean"}
        with patch("app.events.watchdog.desk_state", AsyncMock(return_value=summary)):
            reply = await _handle("/status", r)
        assert "3" in reply.text and "USD Core CPI m/m" in reply.text

    @pytest.mark.asyncio
    async def test_status_from_a_stranger_gives_the_enable_line_not_a_refusal(self):
        """A refusal that names the command confirms the command exists."""
        r = _FakeRedis()
        assert (await _handle("/status", r)).text == bot.ENABLE_HINT

    @pytest.mark.asyncio
    async def test_status_survives_a_desk_that_cannot_be_read(self):
        r = _FakeRedis()
        await _handle("/start s3cret", r)
        with patch("app.events.watchdog.desk_state",
                   AsyncMock(side_effect=RuntimeError("redis down"))):
            reply = await _handle("/status", r)
        assert reply is not None and reply.text


class TestUnknownInput:
    @pytest.mark.asyncio
    async def test_a_stranger_gets_one_line_and_no_command_list(self):
        r = _FakeRedis()
        reply = await _handle("hello", r)
        assert reply.text == bot.ENABLE_HINT
        assert "/stop" not in reply.text and "/status" not in reply.text

    @pytest.mark.asyncio
    async def test_an_enrolled_chat_sending_nonsense_is_told_what_exists(self):
        r = _FakeRedis()
        await _handle("/start s3cret", r)
        reply = await _handle("what is gold doing", r)
        assert "/status" in reply.text

    @pytest.mark.asyncio
    async def test_a_callback_query_is_ignored_rather_than_half_handled(self):
        """Buttons are the next piece. Answering their taps now would ship a
        half-built protocol."""
        r = _FakeRedis()
        a, b, c = _fixture(r)
        with a, b, c:
            assert await bot.handle({"update_id": 2, "callback_query": {"id": "x"}}) is None

    @pytest.mark.asyncio
    async def test_an_update_with_no_text_is_ignored(self):
        r = _FakeRedis()
        a, b, c = _fixture(r)
        with a, b, c:
            assert await bot.handle({"update_id": 3, "message": {"chat": {"id": 1}}}) is None
