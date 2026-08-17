"""
Exchange router for live price updates — NSE/BSE go to the real Kite
ticker, everything else gets a poll loop over the same market-data path
every other quote call already uses. Mirrors
MarketDataRouter._provider_for(exchange)'s existing role for
quotes/historical/search; this is the equivalent for live ticks.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

_KITE_EXCHANGES = {"NSE", "BSE"}
_CHANNEL = "market:ticks"


class LiveTicks:
    def __init__(self, kite_ticker, redis_client,
                 get_quote: Callable[[str, str], Awaitable[dict[str, Any] | None]],
                 poll_interval_seconds: float = 5.0):
        self._kite = kite_ticker
        self._redis = redis_client
        self._get_quote = get_quote
        self._poll_interval = poll_interval_seconds
        self._poll_tasks: dict[tuple[str, str], asyncio.Task] = {}

    async def subscribe(self, symbol: str, exchange: str) -> None:
        key = (symbol.upper(), exchange.upper())
        if exchange.upper() in _KITE_EXCHANGES:
            self._kite.subscribe(symbol, exchange)
            return
        if key in self._poll_tasks:
            return
        self._poll_tasks[key] = asyncio.create_task(self._poll_loop(*key))

    async def unsubscribe(self, symbol: str, exchange: str) -> None:
        key = (symbol.upper(), exchange.upper())
        if exchange.upper() in _KITE_EXCHANGES:
            self._kite.unsubscribe(symbol, exchange)
            return
        task = self._poll_tasks.pop(key, None)
        if task:
            task.cancel()

    async def resubscribe_from(self, active: list[tuple[str, str]]) -> None:
        for symbol, exchange in active:
            await self.subscribe(symbol, exchange)

    async def publish(self, payload: dict[str, Any]) -> None:
        await self._redis.publish(_CHANNEL, json.dumps(payload))

    async def close(self) -> None:
        for task in self._poll_tasks.values():
            task.cancel()
        self._poll_tasks.clear()

    async def _poll_loop(self, symbol: str, exchange: str) -> None:
        # bypass_cache: this loop IS the fresh-data source now, shared by
        # every watcher of this symbol — no reason for it to serve its own
        # stale cache entry back to itself every 5s.
        try:
            while True:
                quote = await self._get_quote(symbol, exchange)
                if quote:
                    await self.publish(quote)
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            pass
