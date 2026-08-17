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

from app.market.kite_ticker import KiteTickerClient

logger = logging.getLogger(__name__)

_KITE_EXCHANGES = {"NSE", "BSE"}
_CHANNEL = "market:ticks"


class LiveTicks:
    def __init__(self, kite_ticker: KiteTickerClient | None, redis_client,
                 get_quote: Callable[[str, str], Awaitable[dict[str, Any] | None]],
                 poll_interval_seconds: float = 5.0):
        self._kite = kite_ticker
        self._redis = redis_client
        self._get_quote = get_quote
        self._poll_interval = poll_interval_seconds
        self._poll_tasks: dict[tuple[str, str], asyncio.Task] = {}

    def set_kite_ticker(self, kite_ticker: KiteTickerClient) -> None:
        """Attached once/if a Zerodha token is obtained — the poll path
        above works with no Kite ticker at all, so construction doesn't wait
        on it."""
        self._kite = kite_ticker

    async def subscribe(self, symbol: str, exchange: str) -> bool:
        key = (symbol.upper(), exchange.upper())
        if exchange.upper() in _KITE_EXCHANGES:
            if self._kite is None:
                return False
            # subscribe() can synchronously download the full instrument
            # list on first use — keep that off the event loop.
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._kite.subscribe, symbol, exchange)
        if key in self._poll_tasks:
            return True
        self._poll_tasks[key] = asyncio.create_task(self._poll_loop(*key))
        return True

    async def unsubscribe(self, symbol: str, exchange: str) -> None:
        key = (symbol.upper(), exchange.upper())
        if exchange.upper() in _KITE_EXCHANGES:
            if self._kite is None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._kite.unsubscribe, symbol, exchange)
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
        # get_quote goes through market_data_router, which has its own 45s
        # TTLCache (registry.py QUOTE_TTL_SECONDS) — the real update cadence
        # is bounded by that, not by poll_interval. Most iterations here
        # re-publish an already-cached quote; that's a deliberate, cheap
        # Redis fan-out cost, not a bug.
        try:
            while True:
                try:
                    quote = await self._get_quote(symbol, exchange)
                    if quote:
                        await self.publish(quote)
                except Exception:
                    logger.exception("Live-tick poll iteration failed for %s/%s", symbol, exchange)
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            pass
