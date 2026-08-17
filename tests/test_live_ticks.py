"""
live_ticks — the exchange router for live price updates. NSE/BSE route to
the Kite ticker; everything else gets a polled loop over the same
market-data path every other quote call already uses. Both publish to the
same Redis channel in the same shape, so nothing downstream needs to know
which one answered.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.market.live_ticks import LiveTicks


def _live_ticks(get_quote=None):
    kite_ticker = MagicMock()
    kite_ticker.subscribe = MagicMock(return_value=True)
    kite_ticker.unsubscribe = MagicMock()
    redis_client = AsyncMock()
    quote_fn = get_quote or AsyncMock(return_value={"symbol": "AAPL", "exchange": "NASDAQ", "ltp": 230.0})
    live = LiveTicks(kite_ticker, redis_client, quote_fn, poll_interval_seconds=0.01)
    return live, kite_ticker, redis_client, quote_fn


@pytest.mark.asyncio
async def test_nse_bse_routes_to_the_kite_ticker():
    live, kite_ticker, redis_client, _ = _live_ticks()

    await live.subscribe("RELIANCE", "NSE")

    kite_ticker.subscribe.assert_called_once_with("RELIANCE", "NSE")


@pytest.mark.asyncio
async def test_other_exchanges_start_a_poll_loop_not_the_kite_ticker():
    live, kite_ticker, redis_client, quote_fn = _live_ticks()

    await live.subscribe("AAPL", "NASDAQ")
    await asyncio.sleep(0.03)
    await live.unsubscribe("AAPL", "NASDAQ")

    kite_ticker.subscribe.assert_not_called()
    assert quote_fn.await_count >= 1
    published = json.loads(redis_client.publish.call_args_list[0].args[1])
    assert published["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_a_second_subscribe_to_an_already_watched_poll_symbol_is_a_noop():
    live, kite_ticker, redis_client, quote_fn = _live_ticks()

    await live.subscribe("AAPL", "NASDAQ")
    await live.subscribe("AAPL", "NASDAQ")
    await asyncio.sleep(0.03)
    tasks_running = len(live._poll_tasks)
    await live.unsubscribe("AAPL", "NASDAQ")

    assert tasks_running == 1


@pytest.mark.asyncio
async def test_unsubscribe_stops_the_poll_loop():
    live, kite_ticker, redis_client, quote_fn = _live_ticks()
    await live.subscribe("AAPL", "NASDAQ")
    await asyncio.sleep(0.03)

    await live.unsubscribe("AAPL", "NASDAQ")
    calls_at_unsubscribe = quote_fn.await_count
    await asyncio.sleep(0.03)

    assert quote_fn.await_count == calls_at_unsubscribe
    assert ("AAPL", "NASDAQ") not in live._poll_tasks


@pytest.mark.asyncio
async def test_resubscribe_from_routes_each_pair_through_the_same_logic():
    live, kite_ticker, redis_client, quote_fn = _live_ticks()

    await live.resubscribe_from([("RELIANCE", "NSE"), ("AAPL", "NASDAQ")])
    await asyncio.sleep(0.03)
    await live.unsubscribe("AAPL", "NASDAQ")

    kite_ticker.subscribe.assert_called_once_with("RELIANCE", "NSE")
    assert quote_fn.await_count >= 1


@pytest.mark.asyncio
async def test_publish_writes_to_the_market_ticks_channel():
    live, _, redis_client, _ = _live_ticks()

    await live.publish({"symbol": "RELIANCE", "exchange": "NSE", "ltp": 1310.0})

    redis_client.publish.assert_called_once()
    channel, message = redis_client.publish.call_args.args
    assert channel == "market:ticks"
    assert json.loads(message)["symbol"] == "RELIANCE"


@pytest.mark.asyncio
async def test_close_cancels_every_running_poll_task():
    live, _, _, _ = _live_ticks()
    await live.subscribe("AAPL", "NASDAQ")
    await live.subscribe("MSFT", "NASDAQ")

    await live.close()

    assert live._poll_tasks == {}
