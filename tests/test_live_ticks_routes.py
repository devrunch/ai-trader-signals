from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import app.market.router as router_module
import main as main_module
from main import app


class _FakeDerivTickerClient:
    """Stands in for the real DerivTickerClient in tests that exercise
    main.lifespan() end to end -- without this, lifespan's own
    `await deriv_ticker.connect()` would spawn a real websocket connection
    attempt to the real Deriv endpoint on every one of these tests."""

    def __init__(self, *args, **kwargs):
        pass

    async def connect(self):
        pass

    async def close(self):
        pass


def test_subscribe_route_calls_live_ticks(monkeypatch):
    live_ticks = AsyncMock()
    live_ticks.subscribe.return_value = True
    monkeypatch.setattr(router_module, "live_ticks", live_ticks)
    client = TestClient(app)

    resp = client.post("/market/internal/live-ticks/subscribe",
                        json={"symbol": "RELIANCE", "exchange": "NSE"})

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    live_ticks.subscribe.assert_awaited_once_with("RELIANCE", "NSE")


def test_subscribe_route_surfaces_a_failed_subscribe(monkeypatch):
    live_ticks = AsyncMock()
    live_ticks.subscribe.return_value = False
    monkeypatch.setattr(router_module, "live_ticks", live_ticks)
    client = TestClient(app)

    resp = client.post("/market/internal/live-ticks/subscribe",
                        json={"symbol": "RELIANCE", "exchange": "NSE"})

    assert resp.status_code == 200
    assert resp.json() == {"ok": False}


def test_unsubscribe_route_calls_live_ticks(monkeypatch):
    live_ticks = AsyncMock()
    monkeypatch.setattr(router_module, "live_ticks", live_ticks)
    client = TestClient(app)

    resp = client.post("/market/internal/live-ticks/unsubscribe",
                        json={"symbol": "RELIANCE", "exchange": "NSE"})

    assert resp.status_code == 200
    live_ticks.unsubscribe.assert_awaited_once_with("RELIANCE", "NSE")


def test_subscribe_route_returns_503_before_startup_completes(monkeypatch):
    monkeypatch.setattr(router_module, "live_ticks", None)
    client = TestClient(app)

    resp = client.post("/market/internal/live-ticks/subscribe",
                        json={"symbol": "RELIANCE", "exchange": "NSE"})

    assert resp.status_code == 503


def test_unsubscribe_route_returns_503_before_startup_completes(monkeypatch):
    monkeypatch.setattr(router_module, "live_ticks", None)
    client = TestClient(app)

    resp = client.post("/market/internal/live-ticks/unsubscribe",
                        json={"symbol": "RELIANCE", "exchange": "NSE"})

    assert resp.status_code == 503


async def test_resubscribe_still_covers_non_kite_symbols_when_kite_attach_fails(monkeypatch):
    """No Zerodha token configured -> _attach_kite_ticker returns False, but a
    non-Kite active symbol (the yfinance poll path) must still be resubscribed;
    only the Kite leg is gated on attach succeeding."""
    monkeypatch.setattr(router_module, "live_ticks", None)

    fake_settings = SimpleNamespace(
        zerodha_api_key="",
        api_service_url="http://api.test",
        internal_api_key="secret",
        redis_url="redis://fake",
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: fake_settings)

    class FakeRedis:
        async def aclose(self):
            pass

    monkeypatch.setattr(main_module.redis, "from_url", lambda url: FakeRedis())

    created_live_ticks = []

    class FakeLiveTicks:
        def __init__(self, *args, **kwargs):
            self.resubscribe_from = AsyncMock()
            self.set_deriv_ticker = AsyncMock()
            created_live_ticks.append(self)

        async def close(self):
            pass

    monkeypatch.setattr(main_module, "LiveTicks", FakeLiveTicks)
    monkeypatch.setattr(main_module, "DerivTickerClient", _FakeDerivTickerClient)

    async def fake_get_with_retry(url, headers, *, what):
        assert "active-symbols" in url
        return SimpleNamespace(json=lambda: [{"symbol": "AAPL", "exchange": "NASDAQ"}])

    monkeypatch.setattr(main_module, "_get_with_retry", fake_get_with_retry)

    async with main_module.lifespan(app):
        # Give the background attach + resubscribe tasks room to run to
        # completion before the context manager's exit cancels them.
        await asyncio.sleep(0.05)

    assert len(created_live_ticks) == 1
    created_live_ticks[0].resubscribe_from.assert_awaited_once_with([("AAPL", "NASDAQ")])


async def test_resubscribe_still_covers_non_kite_symbols_when_kite_attach_raises(monkeypatch):
    """_attach_kite_ticker raising an unexpected exception (not just returning
    False) must not abort _resubscribe_active_symbols before it reaches the
    non-Kite resubscribe."""
    monkeypatch.setattr(router_module, "live_ticks", None)

    fake_settings = SimpleNamespace(
        zerodha_api_key="token",
        api_service_url="http://api.test",
        internal_api_key="secret",
        redis_url="redis://fake",
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: fake_settings)

    class FakeRedis:
        async def aclose(self):
            pass

    monkeypatch.setattr(main_module.redis, "from_url", lambda url: FakeRedis())

    created_live_ticks = []

    class FakeLiveTicks:
        def __init__(self, *args, **kwargs):
            self.resubscribe_from = AsyncMock()
            self.set_deriv_ticker = AsyncMock()
            created_live_ticks.append(self)

        async def close(self):
            pass

    monkeypatch.setattr(main_module, "LiveTicks", FakeLiveTicks)
    monkeypatch.setattr(main_module, "DerivTickerClient", _FakeDerivTickerClient)

    def bad_json():
        raise ValueError("malformed response")

    async def fake_get_with_retry(url, headers, *, what):
        if "active-symbols" in url:
            return SimpleNamespace(json=lambda: [{"symbol": "AAPL", "exchange": "NASDAQ"}])
        return SimpleNamespace(json=bad_json)

    monkeypatch.setattr(main_module, "_get_with_retry", fake_get_with_retry)

    async with main_module.lifespan(app):
        await asyncio.sleep(0.05)

    assert len(created_live_ticks) == 1
    created_live_ticks[0].resubscribe_from.assert_awaited_once_with([("AAPL", "NASDAQ")])


async def test_shutdown_survives_a_failed_background_task(monkeypatch):
    """A background task that already failed with a non-CancelledError
    exception before shutdown begins must not stop the rest of cleanup
    (redis close, executor shutdown) from running."""
    monkeypatch.setattr(router_module, "live_ticks", None)

    fake_settings = SimpleNamespace(
        zerodha_api_key="",
        api_service_url="http://api.test",
        internal_api_key="secret",
        redis_url="redis://fake",
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: fake_settings)

    created_redis = []

    class FakeRedis:
        def __init__(self):
            self.aclose_called = False

        async def aclose(self):
            self.aclose_called = True

    def fake_from_url(url):
        r = FakeRedis()
        created_redis.append(r)
        return r

    monkeypatch.setattr(main_module.redis, "from_url", fake_from_url)

    created_live_ticks = []

    class FakeLiveTicks:
        def __init__(self, *args, **kwargs):
            self.resubscribe_from = AsyncMock(side_effect=RuntimeError("boom"))
            self.set_deriv_ticker = AsyncMock()
            self.close = AsyncMock()
            created_live_ticks.append(self)

    monkeypatch.setattr(main_module, "LiveTicks", FakeLiveTicks)
    monkeypatch.setattr(main_module, "DerivTickerClient", _FakeDerivTickerClient)

    async def fake_get_with_retry(url, headers, *, what):
        assert "active-symbols" in url
        return SimpleNamespace(json=lambda: [{"symbol": "AAPL", "exchange": "NASDAQ"}])

    monkeypatch.setattr(main_module, "_get_with_retry", fake_get_with_retry)

    created_executors = []

    class SpyExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shutdown_calls = []
            created_executors.append(self)

        def shutdown(self, *args, **kwargs):
            self.shutdown_calls.append((args, kwargs))
            super().shutdown(*args, **kwargs)

    monkeypatch.setattr(main_module, "ThreadPoolExecutor", SpyExecutor)

    async with main_module.lifespan(app):
        # Let resubscribe_task fail before the context manager's exit
        # triggers cancel()/await on an already-done, already-failed task.
        await asyncio.sleep(0.05)

    assert created_redis[0].aclose_called
    assert created_executors[0].shutdown_calls
    created_live_ticks[0].close.assert_awaited_once()
