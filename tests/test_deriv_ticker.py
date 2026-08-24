"""
DerivTickerClient — the persistent WS connection for live forex/metals
ticks. A fake DerivProvider stands in for the real network-touching one
(get_quote is mocked directly, no websockets.connect involved here); a fake
websocket object stands in for the persistent connection itself.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.market.deriv_ticker import DerivTickerClient


class FakeWebSocket:
    """Enough of websockets' async connection interface for DerivTickerClient:
    send() records what was sent, and messages queued via push() are what
    `async for raw in ws` yields."""

    def __init__(self):
        self.sent: list[dict] = []
        self._queue: asyncio.Queue = asyncio.Queue()

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def push(self, msg: dict) -> None:
        self._queue.put_nowait(msg)

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self._queue.get()
        return json.dumps(msg)


def _provider(quote: dict | None):
    provider = AsyncMock()
    provider.get_quote = AsyncMock(return_value=quote)
    return provider


BASELINE_QUOTE = {
    "symbol": "XAUUSD", "exchange": "FOREX", "ltp": 2650.0, "change": 45.0,
    "change_percent": 1.73, "open": 2605.0, "high": 2650.5, "low": 2600.0,
    "prev_close": 2605.0, "volume": None, "bid": None, "ask": None,
}


@pytest.mark.asyncio
async def test_subscribe_fails_for_an_unknown_symbol_without_touching_the_socket():
    ws = FakeWebSocket()
    client = DerivTickerClient(on_tick=lambda q: None, provider=_provider(BASELINE_QUOTE))
    client._ws = ws

    ok = await client.subscribe("BTCUSD", "FOREX")

    assert ok is False
    assert ws.sent == []


@pytest.mark.asyncio
async def test_subscribe_fetches_a_baseline_and_sends_the_real_deriv_symbol():
    ws = FakeWebSocket()
    provider = _provider(BASELINE_QUOTE)
    client = DerivTickerClient(on_tick=lambda q: None, provider=provider)
    client._ws = ws

    ok = await client.subscribe("XAUUSD", "FOREX")

    assert ok is True
    provider.get_quote.assert_awaited_once_with("XAUUSD", "FOREX")
    assert ws.sent == [{"ticks": "frxXAUUSD", "subscribe": 1}]


@pytest.mark.asyncio
async def test_subscribe_fails_closed_when_no_baseline_quote_is_available():
    ws = FakeWebSocket()
    client = DerivTickerClient(on_tick=lambda q: None, provider=_provider(None))
    client._ws = ws

    ok = await client.subscribe("XAUUSD", "FOREX")

    assert ok is False
    assert ws.sent == []


@pytest.mark.asyncio
async def test_a_tick_is_shaped_as_a_complete_quote_not_missing_fields():
    """The frontend replaces its whole quote object per tick, never merges
    -- change/open/high/low/prev_close must never go None just because a
    raw Deriv tick itself only carries quote/bid/ask."""
    seen = []
    ws = FakeWebSocket()
    client = DerivTickerClient(on_tick=seen.append, provider=_provider(BASELINE_QUOTE))
    client._ws = ws
    await client.subscribe("XAUUSD", "FOREX")

    client._handle_message({
        "msg_type": "tick",
        "subscription": {"id": "sub-1"},
        "tick": {"symbol": "frxXAUUSD", "quote": 2660.0, "bid": 2659.5, "ask": 2660.5},
    })

    assert len(seen) == 1
    quote = seen[0]
    assert quote["symbol"] == "XAUUSD"
    assert quote["exchange"] == "FOREX"
    assert quote["ltp"] == 2660.0
    assert quote["bid"] == 2659.5
    assert quote["ask"] == 2660.5
    # Derived from the baseline, not left None.
    assert quote["prev_close"] == 2605.0
    assert quote["open"] == 2605.0
    assert quote["change"] == pytest.approx(55.0)
    assert quote["change_percent"] == pytest.approx(round(55.0 / 2605.0 * 100, 4))


@pytest.mark.asyncio
async def test_a_new_high_widens_the_running_high_on_later_ticks():
    seen = []
    ws = FakeWebSocket()
    client = DerivTickerClient(on_tick=seen.append, provider=_provider(BASELINE_QUOTE))
    client._ws = ws
    await client.subscribe("XAUUSD", "FOREX")  # baseline high = 2650.5

    client._handle_message({
        "msg_type": "tick", "subscription": {"id": "sub-1"},
        "tick": {"symbol": "frxXAUUSD", "quote": 2700.0, "bid": 2699.5, "ask": 2700.5},
    })

    assert seen[0]["high"] == 2700.0


@pytest.mark.asyncio
async def test_a_tick_for_an_unknown_deriv_symbol_is_dropped_not_raised():
    seen = []
    client = DerivTickerClient(on_tick=seen.append, provider=_provider(BASELINE_QUOTE))

    client._handle_message({
        "msg_type": "tick", "subscription": {"id": "sub-1"},
        "tick": {"symbol": "frxEURUSD", "quote": 1.1, "bid": 1.09, "ask": 1.11},
    })

    assert seen == []


@pytest.mark.asyncio
async def test_a_non_tick_message_is_ignored():
    seen = []
    client = DerivTickerClient(on_tick=seen.append, provider=_provider(BASELINE_QUOTE))

    client._handle_message({"msg_type": "ping"})

    assert seen == []


@pytest.mark.asyncio
async def test_unsubscribe_sends_forget_with_the_tracked_subscription_id():
    ws = FakeWebSocket()
    client = DerivTickerClient(on_tick=lambda q: None, provider=_provider(BASELINE_QUOTE))
    client._ws = ws
    await client.subscribe("XAUUSD", "FOREX")
    client._handle_message({
        "msg_type": "tick", "subscription": {"id": "sub-42"},
        "tick": {"symbol": "frxXAUUSD", "quote": 2660.0, "bid": 2659.5, "ask": 2660.5},
    })
    ws.sent.clear()

    await client.unsubscribe("XAUUSD", "FOREX")

    assert ws.sent == [{"forget": "sub-42"}]


@pytest.mark.asyncio
async def test_unsubscribe_without_a_prior_subscription_is_a_no_op():
    ws = FakeWebSocket()
    client = DerivTickerClient(on_tick=lambda q: None, provider=_provider(BASELINE_QUOTE))
    client._ws = ws

    await client.unsubscribe("XAUUSD", "FOREX")  # must not raise

    assert ws.sent == []


@pytest.mark.asyncio
async def test_a_tick_after_unsubscribe_is_no_longer_delivered():
    seen = []
    ws = FakeWebSocket()
    client = DerivTickerClient(on_tick=seen.append, provider=_provider(BASELINE_QUOTE))
    client._ws = ws
    await client.subscribe("XAUUSD", "FOREX")
    await client.unsubscribe("XAUUSD", "FOREX")

    client._handle_message({
        "msg_type": "tick", "subscription": {"id": "sub-1"},
        "tick": {"symbol": "frxXAUUSD", "quote": 2660.0, "bid": 2659.5, "ask": 2660.5},
    })

    assert seen == []
