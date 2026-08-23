"""
MarketDataRouter — picks which provider answers a request, keyed by exchange.

Today every exchange falls through to YFinanceProvider (free, delayed).
Adding real-time coverage later is just:

    router.providers["NSE"] = DhanProvider(...)
    router.providers["NASDAQ"] = AlpacaProvider(...)

No other file in the app needs to change — market/service.py and
signals/service.py only ever talk to this router, never a vendor directly.

The router also owns the market-data cache. It lives here rather than in
market/service.py because market/service.py is only the REST façade: the chat
agent, the screener and the morning brief all call the router directly, and
they are the paths that re-fetch the same bars four times in a minute. yfinance
is an unofficial scraped API with undocumented rate limiting, so the cache is a
correctness measure as much as a latency one — a throttled request returns None
and surfaces to the user as "not enough data".
"""
from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any

import pandas as pd
from cachetools import TTLCache

from app.config import get_settings
from app.market.providers.base import MarketDataProvider
from app.market.providers.kite_provider import KiteProvider
from app.market.providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)

# A quote is a live price; 45s is the longest a chart or an order-preview can be
# stale before it is misleading.
QUOTE_TTL_SECONDS = 45
# A 15m bar cannot change more often than every 15m, and the in-progress bar is
# the only one that moves. 5 min is a safe compromise for every intraday size.
INTRADAY_TTL_SECONDS = 300
# Daily bars (sensitivity, global cues, 400-day lookbacks) change once a day.
DAILY_TTL_SECONDS = 3600
# A failure is NOT cached for the full TTL — a Yahoo blip would then blank the
# symbol for five minutes. It is remembered just long enough to stop a hot loop
# hammering a vendor that is already refusing us.
NEGATIVE_TTL_SECONDS = 15

_INTRADAY_INTERVALS = frozenset({"1m", "5m", "15m", "1h"})

# Roughly (universe × intervals) with headroom. Bars are the memory cost: ~1500
# rows × 5 float columns ≈ 60 KB each, so 512 entries is ~30 MB worst case
# against the task's 2 GB.
_QUOTE_CACHE_SIZE = 512
_HISTORY_CACHE_SIZE = 512


class MarketDataRouter:
    def __init__(self) -> None:
        self.fallback: MarketDataProvider = YFinanceProvider()
        self.providers: dict[str, MarketDataProvider] = {}

        kite = KiteProvider(get_settings())
        self.providers["NSE"] = kite
        self.providers["BSE"] = kite
        # Commodity futures -- Kite Connect already supports the MCX segment
        # natively, same vendor, same instance, no new provider needed. See
        # KiteProvider's own MCX handling (_ensure_instruments, _resolve_row,
        # the "GOLD1!" continuous-contract convention).
        self.providers["MCX"] = kite

        self._quote_cache: TTLCache = TTLCache(maxsize=_QUOTE_CACHE_SIZE, ttl=QUOTE_TTL_SECONDS)
        self._intraday_cache: TTLCache = TTLCache(maxsize=_HISTORY_CACHE_SIZE, ttl=INTRADAY_TTL_SECONDS)
        self._daily_cache: TTLCache = TTLCache(maxsize=_HISTORY_CACHE_SIZE, ttl=DAILY_TTL_SECONDS)
        self._negative_cache: TTLCache = TTLCache(
            maxsize=_QUOTE_CACHE_SIZE + _HISTORY_CACHE_SIZE, ttl=NEGATIVE_TTL_SECONDS
        )

        # Per-key locks so a cache miss under load does not fan out into N
        # identical yfinance downloads — the first caller fetches, the rest wait
        # and read the cache it fills.
        #
        # Keyed by event loop as well as by cache key: app/worker/tasks.py runs
        # asyncio.run() per symbol, creating and closing a fresh loop each time,
        # and an asyncio.Lock binds to the loop it is first awaited on. A single
        # shared dict of locks would raise "attached to a different loop" the
        # second time round. WeakKeyDictionary lets the per-loop table disappear
        # with the loop.
        self._locks: weakref.WeakKeyDictionary[Any, dict[tuple, asyncio.Lock]] = (
            weakref.WeakKeyDictionary()
        )

    def _provider_for(self, exchange: str) -> MarketDataProvider:
        return self.providers.get(exchange.upper(), self.fallback)

    async def search(self, query: str, limit: int = 8) -> list[dict]:
        """Symbol/company search across every exchange this app covers.

        Kite has no free-text search of its own, so KiteProvider answers from
        its own instrument dump — NSE/BSE only, real listings. The fallback
        vendor covers NASDAQ/NYSE the same way it always has. Both are asked
        and the results concatenated, capped at the combined limit — neither
        vendor knows about the other's half.
        """
        kite = self.providers.get("NSE")
        kite_results = await kite.search(query, limit) if kite is not None else []
        fallback_results = await self.fallback.search(query, limit)
        return (kite_results + fallback_results)[:limit]

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------
    def _locks_for_loop(self) -> dict[tuple, asyncio.Lock]:
        loop = asyncio.get_running_loop()
        table = self._locks.get(loop)
        if table is None:
            table = {}
            self._locks[loop] = table
        return table

    def _history_cache(self, interval: str) -> TTLCache:
        if interval in _INTRADAY_INTERVALS:
            return self._intraday_cache
        if interval == "1d":
            return self._daily_cache
        # Unknown interval — assume the shorter, safer TTL rather than serving
        # an hour-old bar for something that might be a 5m chart.
        return self._intraday_cache

    def _clear_key(self, cache: TTLCache, key: tuple) -> None:
        cache.pop(key, None)
        self._negative_cache.pop(key, None)

    # ------------------------------------------------------------------
    # Public API. Positional signatures are unchanged — app/signals/** calls
    # these — and `bypass_cache` is keyword-only with a safe default.
    # ------------------------------------------------------------------
    async def get_quote(
        self, symbol: str, exchange: str = "NSE", *, bypass_cache: bool = False
    ) -> dict | None:
        key = ("quote", symbol.upper(), exchange.upper())
        cache = self._quote_cache

        if not bypass_cache:
            hit = cache.get(key)
            if hit is not None:
                return hit
            if key in self._negative_cache:
                return None

        locks = self._locks_for_loop()
        lock = locks.get(key)
        if lock is None:
            lock = locks[key] = asyncio.Lock()
        try:
            async with lock:
                # Re-check: while we waited, the caller holding the lock may
                # have filled the cache. This is the whole point of the lock.
                if not bypass_cache:
                    hit = cache.get(key)
                    if hit is not None:
                        return hit
                    if key in self._negative_cache:
                        return None

                provider = self._provider_for(exchange)
                result = await provider.get_quote(symbol, exchange)
                if result is None and provider is not self.fallback:
                    result = await self.fallback.get_quote(symbol, exchange)
                if result is None:
                    self._negative_cache[key] = True
                else:
                    cache[key] = result
                    self._negative_cache.pop(key, None)
                return result
        finally:
            # No await between release and this check, so if nobody re-acquired
            # the lock nobody is waiting on it and the entry can go. Keeps the
            # per-loop table from growing without bound.
            if not lock.locked():
                locks.pop(key, None)

    async def get_historical_df(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "15m",
        days: int = 30,
        *,
        bypass_cache: bool = False,
    ) -> pd.DataFrame | None:
        key = ("hist", symbol.upper(), exchange.upper(), interval, days)
        cache = self._history_cache(interval)

        if not bypass_cache:
            hit = cache.get(key)
            if hit is not None:
                # Hand out a copy: pandas_ta and several call sites add columns
                # to the frame they are given, which would otherwise mutate the
                # cached object under every other caller.
                return hit.copy()
            if key in self._negative_cache:
                return None

        locks = self._locks_for_loop()
        lock = locks.get(key)
        if lock is None:
            lock = locks[key] = asyncio.Lock()
        try:
            async with lock:
                if not bypass_cache:
                    hit = cache.get(key)
                    if hit is not None:
                        return hit.copy()
                    if key in self._negative_cache:
                        return None

                provider = self._provider_for(exchange)
                df = await provider.get_historical_df(symbol, exchange, interval, days)
                if (df is None or df.empty) and provider is not self.fallback:
                    df = await self.fallback.get_historical_df(symbol, exchange, interval, days)
                if df is None or df.empty:
                    self._negative_cache[key] = True
                    return df
                cache[key] = df
                self._negative_cache.pop(key, None)
                return df.copy()
        finally:
            if not lock.locked():
                locks.pop(key, None)

    def invalidate(self, symbol: str, exchange: str = "NSE") -> None:
        """Drop every cached entry for one symbol.

        For the case where something else already knows the cached value is
        wrong (a corporate action, a manual refresh) and `bypass_cache` on a
        single call is not enough.
        """
        sym, exch = symbol.upper(), exchange.upper()
        self._clear_key(self._quote_cache, ("quote", sym, exch))
        for cache in (self._intraday_cache, self._daily_cache):
            for key in [k for k in list(cache.keys()) if k[1] == sym and k[2] == exch]:
                self._clear_key(cache, key)

    def cache_stats(self) -> dict:
        """Sizes only — cheap enough to expose from a readiness/debug endpoint."""
        return {
            "quotes": len(self._quote_cache),
            "intraday": len(self._intraday_cache),
            "daily": len(self._daily_cache),
            "negative": len(self._negative_cache),
        }


# Single shared instance — import this, don't instantiate your own.
market_data_router = MarketDataRouter()
