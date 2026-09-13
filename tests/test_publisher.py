from types import SimpleNamespace

import httpx
import pytest

from app.signals import publisher as publisher_mod
from app.signals.publisher import HttpSignalPublisher
from app.signals.types import GeneratedSignal, SignalType

SIGNAL = GeneratedSignal(
    symbol="RELIANCE", exchange="NSE", signal_type=SignalType.BUY, confidence=0.8,
    entry_price=2847.0, target_price=2910.0, stop_loss=2800.0, reasoning="trend", indicators={"rsi": 34.2},
)
SETTINGS = SimpleNamespace(api_service_url="http://api:8000", internal_api_key="k" * 32)


class FakeClient:
    calls: list = []
    status = 201

    def __init__(self, timeout=None):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        FakeClient.calls.append((url, headers, json))
        return httpx.Response(FakeClient.status, request=httpx.Request("POST", url))


@pytest.fixture
def client(monkeypatch):
    FakeClient.calls, FakeClient.status = [], 201
    monkeypatch.setattr(publisher_mod.httpx, "AsyncClient", FakeClient)
    return FakeClient


async def test_a_signal_is_posted_to_the_api_internal_endpoint(client):
    await HttpSignalPublisher(SETTINGS).publish(SIGNAL)

    [(url, headers, body)] = client.calls
    assert url == "http://api:8000/api/internal/signals"
    assert headers == {"x-internal-key": "k" * 32}
    assert body == {
        "symbol": "RELIANCE", "exchange": "NSE", "direction": "BUY", "confidence": 0.8,
        "entry_price": 2847.0, "target_price": 2910.0, "stop_loss": 2800.0,
        "reasoning": "trend", "indicators": {"rsi": 34.2},
    }


async def test_a_rejected_publish_raises_so_the_caller_logs_it(client):
    client.status = 500
    with pytest.raises(httpx.HTTPStatusError):
        await HttpSignalPublisher(SETTINGS).publish(SIGNAL)
