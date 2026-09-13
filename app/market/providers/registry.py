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
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from cachetools import TTLCache

from app.config import get_settings
from app.market import symbols as symbol_registry
from app.market.bars import to_bars
from app.market.contract import BarsResult, BarsStatus, VolumeSource
from app.market.providers.base import MarketDataProvider
from app.market.providers.deriv_provider import DerivProvider, deriv_symbol_for
from app.market.providers.kite_provider import KiteProvider
from app.market.providers.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)

# A quote is a live price; 45s is the longest a chart or an order-preview can be
# stale before it is misleading. NSE/BSE/MCX/FOREX don't really lean on this
# for "live" feel -- Kite's and Deriv's own tickers push ticks straight
# through, independent of this cache -- so 45s only meaningfully bounds the
# poll-loop exchanges (NASDAQ/NYSE, see live_ticks.py's _poll_loop). FOREX
# used to need its own shorter, dedicated bucket here when it rode that same
# generic poll loop against a credit-metered vendor (Twelve Data); it no
# longer does either, now that live_ticks.py routes FOREX to a real Deriv
# ticker (deriv_ticker.py) with no meaningful rate ceiling.
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

        # Forex majors/minors + precious metals -- neither Kite (India-only)
        # nor yfinance (unreliable in practice for this) cover it. Free,
        # no account, no rate-limit wall -- see deriv_provider.py's own
        # module docstring.
        self.providers["FOREX"] = DerivProvider()

        # The same providers, addressed the way a resolved symbol names them.
        # The exchange-keyed dict above stays for the legacy path; a resolved
        # SymbolInfo says which vendor owns a symbol, which is not always
        # answerable from an exchange string (every forex pair arrives
        # labelled NSE by callers that do not know better).
        self.by_provider: dict[str, MarketDataProvider] = {
            symbol_registry.DERIV: self.providers["FOREX"],
            symbol_registry.KITE: kite,
            symbol_registry.YFINANCE: self.fallback,
        }

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

    async def get_bars(self, symbol: str, interval: str, days: int,
                       exchange: str | None = None) -> BarsResult:
        """Bars, plus why they are what they are.

        The contract path (see docs/superpowers/specs/2026-09-13-market-data-
        contract-design.md). `exchange` is a hint for a genuinely ambiguous
        symbol and nothing more -- the resolver owns the venue, so a caller
        that does not know one can no longer send a request nowhere."""
        info = symbol_registry.resolve(symbol, exchange)
        if info is None:
            return BarsResult([], BarsStatus.UNKNOWN_SYMBOL,
                              reason=f"No provider covers {symbol!r}")

        provider = self.by_provider.get(info.provider, self.fallback)
        caps = provider.capabilities
        if interval not in caps.intervals:
            return BarsResult(
                [], BarsStatus.UNSUPPORTED_INTERVAL, info, caps.volume_source,
                reason=f"{info.exchange} bars are not served at {interval}; try {sorted(caps.intervals)}",
            )

        # Clamped to what THIS vendor serves, not to a shared table of another
        # vendor's limits -- and the caller is told, rather than quietly handed
        # a shorter chart than it asked for.
        allowed = caps.max_days(interval)
        span = max(1, min(days, allowed))
        truncated = span if span < days else None

        df = await self.get_historical_df(symbol, info.exchange, interval, span)
        if df is None or df.empty:
            known = info.authoritative or self._provider_knows(provider, info)
            return self._empty_result(info, caps.volume_source, truncated, known=known)

        return BarsResult(to_bars(df), BarsStatus.OK, info, caps.volume_source,
                          truncated_to_days=truncated)

    @staticmethod
    def _provider_knows(provider, info) -> bool:
        """Whether the vendor can confirm this symbol is real without a call.

        Kite can: it already holds the instrument dump. Nobody else can, and a
        guess in either direction is worse than saying we do not know.
        """
        knows = getattr(provider, "knows_symbol", None)
        return bool(knows and knows(info.symbol, info.exchange))

    def _empty_result(self, info, volume_source: VolumeSource,
                      truncated: int | None, known: bool = True) -> BarsResult:
        """Why nothing came back.

        A closed market is knowable here and is the common case -- forex over a
        weekend, an exchange overnight. With the session open, the providers
        cannot yet tell an empty range from their own failure (both return
        None today); until they are ported to the contract, that case is
        reported as a vendor error, which is the direction that gets looked
        at rather than silently ignored."""
        now = datetime.now(UTC)
        closed = not info.session.is_open(now)
        if closed and known:
            opens = info.session.next_open(now)
            when = f" until {opens:%Y-%m-%d %H:%M} UTC" if opens else ""
            return BarsResult([], BarsStatus.CLOSED_MARKET, info, volume_source,
                              reason=f"{info.exchange} is closed{when}",
                              truncated_to_days=truncated)
        if not known:
            # Both possibilities, neither claimed: an unlisted symbol and a
            # quiet session look identical from here, and picking one is how a
            # typo becomes "the market is closed".
            aside = " (the session is also closed)" if closed else ""
            return BarsResult(
                [], BarsStatus.NO_DATA, info, volume_source,
                reason=f"No bars for {info.symbol} on {info.exchange}; "
                       f"the symbol may not be listed there{aside}",
                truncated_to_days=truncated)
        return BarsResult([], BarsStatus.VENDOR_ERROR, info, volume_source,
                          reason=f"No bars returned for {info.symbol} and the session is open",
                          truncated_to_days=truncated)

    def resolve_exchange(self, symbol: str, exchange: str) -> str:
        """The exchange a request will actually be served from.

        Callers echo this back to the browser rather than what was asked for:
        a chart layout saved as XAUUSD-on-NSE would fail the moment it was
        reopened against a client that trusts the stored exchange."""
        if deriv_symbol_for(symbol) and "FOREX" in self.providers:
            return "FOREX"
        return exchange.upper()

    def _provider_for(self, exchange: str, symbol: str | None = None) -> MarketDataProvider:
        """Exchange picks the provider, except where the symbol overrules it.

        `exchange` defaults to NSE in the frontend client, in the NestJS
        controller and in the FastAPI route, so a caller that does not know the
        exchange -- a saved chart layout, a chat turn, a watchlist row -- is
        indistinguishable from one asking for Indian equity. XAUUSD on NSE is
        not a thing that exists, and used to 404. Deriv's pair table is the
        authority for those 29 symbols, so it wins over the claimed exchange."""
        if symbol and deriv_symbol_for(symbol) and "FOREX" in self.providers:
            return self.providers["FOREX"]
        return self.providers.get(exchange.upper(), self.fallback)

    async def search(self, query: str, limit: int = 8) -> list[dict]:
        """Symbol/company search across every exchange this app covers.

        Kite has no free-text search of its own, so KiteProvider answers from
        its own instrument dump — NSE/BSE/MCX, real listings. The fallback
        vendor covers NASDAQ/NYSE the same way it always has. Deriv (FOREX)
        matches against its own known-pairs table (see deriv_provider.py) --
        29 real instruments, not a live vendor search call. All are asked
        and the results concatenated, capped at the combined limit — none of
        them knows about the others' half.
        """
        kite = self.providers.get("NSE")
        forex = self.providers.get("FOREX")
        kite_results = await kite.search(query, limit) if kite is not None else []
        forex_results = await forex.search(query, limit) if forex is not None else []
        fallback_results = await self.fallback.search(query, limit)
        return (forex_results + kite_results + fallback_results)[:limit]

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

                provider = self._provider_for(exchange, symbol)
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

                provider = self._provider_for(exchange, symbol)
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


# Single shared instance — import this, don't instantiate your own.
market_data_router = MarketDataRouter()
