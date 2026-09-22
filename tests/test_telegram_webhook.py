"""The webhook route and the fan-out.

Two rules carry this file. The route never returns a non-2xx once the header
is good — Telegram re-delivers on anything else, so a handler that throws on
one malformed update would otherwise retry it forever. And a fan-out is not a
loop that stops at the first failure: one dead chat must not cost everyone
else their brief.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.events import telegram
from app.events.router import router

HEADER = "X-Telegram-Bot-Api-Secret-Token"


def _client(secret="hook"):
    settings = MagicMock()
    settings.telegram_webhook_secret = secret
    app = FastAPI()
    app.include_router(router, prefix="/telegram")
    return TestClient(app), patch("app.events.router.get_settings", return_value=settings)


def _update(text="/status"):
    return {"update_id": 1, "message": {"chat": {"id": 42}, "text": text}}


class TestTheDoor:
    def test_a_missing_header_is_refused(self):
        client, settings = _client()
        with settings:
            assert client.post("/telegram/webhook", json=_update()).status_code == 401

    def test_a_wrong_header_is_refused(self):
        client, settings = _client()
        with settings:
            resp = client.post("/telegram/webhook", json=_update(), headers={HEADER: "nope"})
        assert resp.status_code == 401

    def test_an_unconfigured_secret_refuses_everything_rather_than_falling_open(self):
        """An empty configured secret matching an empty header would make the
        webhook world-writable."""
        client, settings = _client(secret="")
        with settings:
            assert client.post("/telegram/webhook", json=_update(),
                               headers={HEADER: ""}).status_code == 401

    def test_a_good_header_is_accepted(self):
        client, settings = _client()
        with settings, patch("app.events.router.bot.handle", AsyncMock(return_value=None)):
            assert client.post("/telegram/webhook", json=_update(),
                               headers={HEADER: "hook"}).status_code == 200


class TestRetryLoops:
    def test_a_handler_that_raises_still_returns_200(self):
        """Telegram re-delivers the same update on a non-2xx. A bug that
        throws on one message would become an unbounded retry against a route
        that fails every time."""
        client, settings = _client()
        with settings, patch("app.events.router.bot.handle",
                             AsyncMock(side_effect=RuntimeError("boom"))):
            resp = client.post("/telegram/webhook", json=_update(), headers={HEADER: "hook"})
        assert resp.status_code == 200

    def test_a_body_that_is_not_json_still_returns_200(self):
        client, settings = _client()
        with settings:
            resp = client.post("/telegram/webhook", content=b"not json",
                               headers={HEADER: "hook", "Content-Type": "application/json"})
        assert resp.status_code == 200

    def test_a_reply_that_cannot_be_delivered_does_not_fail_the_request(self):
        client, settings = _client()
        reply = MagicMock(chat_id="42", text="hi")
        with settings, \
             patch("app.events.router.bot.handle", AsyncMock(return_value=reply)), \
             patch("app.events.router.telegram.send_to",
                   AsyncMock(side_effect=httpx.ConnectError("down"))):
            resp = client.post("/telegram/webhook", json=_update(), headers={HEADER: "hook"})
        assert resp.status_code == 200

    def test_a_reply_is_sent_to_the_chat_that_asked(self):
        client, settings = _client()
        reply = MagicMock(chat_id="42", text="hi")
        sender = AsyncMock(return_value=True)
        with settings, \
             patch("app.events.router.bot.handle", AsyncMock(return_value=reply)), \
             patch("app.events.router.telegram.send_to", sender):
            client.post("/telegram/webhook", json=_update(), headers={HEADER: "hook"})
        sender.assert_awaited_once_with("42", "hi")


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.text = text


def _http(responses):
    """An httpx.AsyncClient whose post() returns each response in turn."""
    calls = []

    class _Client:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

        async def post(self, url, json=None):
            calls.append(json)
            result = responses[len(calls) - 1]
            if isinstance(result, Exception):
                raise result
            return result

    return _Client, calls


class TestFanOut:
    @pytest.mark.asyncio
    async def test_one_dead_chat_does_not_cost_the_others_their_brief(self):
        client_cls, calls = _http([_Response(), _Response(403, "blocked"), _Response()])
        with patch("app.events.telegram.subscribers.all",
                   AsyncMock(return_value=["1", "2", "3"])), \
             patch("app.events.telegram.subscribers.remove", AsyncMock()), \
             patch("app.events.telegram.httpx.AsyncClient", client_cls), \
             patch("app.events.telegram.get_settings", return_value=_settings()):
            ok = await telegram.send("hello")
        assert [c["chat_id"] for c in calls] == ["1", "2", "3"]
        assert ok is True, "two of three landed — that is a delivery, not a failure"

    @pytest.mark.asyncio
    async def test_a_chat_that_blocked_the_bot_is_unsubscribed_by_that_act(self):
        client_cls, _ = _http([_Response(403, "Forbidden: bot was blocked by the user")])
        dropped = AsyncMock()
        with patch("app.events.telegram.subscribers.all", AsyncMock(return_value=["9"])), \
             patch("app.events.telegram.subscribers.remove", dropped), \
             patch("app.events.telegram.httpx.AsyncClient", client_cls), \
             patch("app.events.telegram.get_settings", return_value=_settings()):
            await telegram.send("hello")
        dropped.assert_awaited_once_with("9")

    @pytest.mark.asyncio
    async def test_an_unrelated_error_does_not_unsubscribe_anyone(self):
        """A 500 from Telegram is Telegram's problem. Dropping a subscriber
        over it would quietly empty the registry during an outage."""
        client_cls, _ = _http([_Response(500, "internal")])
        dropped = AsyncMock()
        with patch("app.events.telegram.subscribers.all", AsyncMock(return_value=["9"])), \
             patch("app.events.telegram.subscribers.remove", dropped), \
             patch("app.events.telegram.httpx.AsyncClient", client_cls), \
             patch("app.events.telegram.get_settings", return_value=_settings()):
            assert await telegram.send("hello") is False
        dropped.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_empty_registry_is_reported_as_undelivered(self):
        with patch("app.events.telegram.subscribers.all", AsyncMock(return_value=[])), \
             patch("app.events.telegram.get_settings", return_value=_settings()):
            assert await telegram.send("hello") is False

    @pytest.mark.asyncio
    async def test_a_long_message_is_truncated_visibly(self):
        client_cls, calls = _http([_Response()])
        with patch("app.events.telegram.subscribers.all", AsyncMock(return_value=["1"])), \
             patch("app.events.telegram.httpx.AsyncClient", client_cls), \
             patch("app.events.telegram.get_settings", return_value=_settings()):
            await telegram.send("x" * (telegram.MAX_MESSAGE_CHARS + 500))
        assert len(calls[0]["text"]) <= telegram.MAX_MESSAGE_CHARS
        assert "truncated" in calls[0]["text"]


def _settings():
    s = MagicMock()
    s.telegram_bot_token = "token"
    s.telegram_chat_id = ""
    return s
