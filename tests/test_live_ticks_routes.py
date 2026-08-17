from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import app.market.router as router_module
from main import app


def test_subscribe_route_calls_live_ticks(monkeypatch):
    live_ticks = AsyncMock()
    monkeypatch.setattr(router_module, "live_ticks", live_ticks)
    client = TestClient(app)

    resp = client.post("/market/internal/live-ticks/subscribe",
                        json={"symbol": "RELIANCE", "exchange": "NSE"})

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    live_ticks.subscribe.assert_awaited_once_with("RELIANCE", "NSE")


def test_unsubscribe_route_calls_live_ticks(monkeypatch):
    live_ticks = AsyncMock()
    monkeypatch.setattr(router_module, "live_ticks", live_ticks)
    client = TestClient(app)

    resp = client.post("/market/internal/live-ticks/unsubscribe",
                        json={"symbol": "RELIANCE", "exchange": "NSE"})

    assert resp.status_code == 200
    live_ticks.unsubscribe.assert_awaited_once_with("RELIANCE", "NSE")
