"""
Contract test for `POST /signals/chat` — the buffered twin of `/chat/stream`.

`chat_stream` (tested in test_agent_events.py) converts a `service.chat()`
exception into a graceful SSE error event; this one must degrade the same
way rather than let the exception turn into a raw 500.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.signals.router import get_service, router


class _BoomingService:
    async def chat(self, *a, **k):
        raise RuntimeError("bedrock is down")


def _client(service):
    app = FastAPI()
    app.include_router(router, prefix="/signals")
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def test_a_failing_chat_call_returns_a_clean_error_not_a_raw_500():
    with _client(_BoomingService()) as client:
        r = client.post("/signals/chat", json={"symbol": "RELIANCE", "message": "hi"})

    assert r.status_code == 502
    # The internal reason must not leak to the client.
    assert "bedrock" not in r.json()["detail"].lower()
