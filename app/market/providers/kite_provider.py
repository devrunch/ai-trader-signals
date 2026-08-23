"""
KiteProvider — real Zerodha Kite Connect data for NSE/BSE.

Same contract as YFinanceProvider (see providers/base.py): callers never know
which vendor answered. Kite's SDK is synchronous, same as yfinance's, so
every public method wraps its sync half in run_in_executor, identical to
YFinanceProvider's own pattern.

The current access_token lives in NestJS (refreshed daily by
kite_auth.refresh_session, called from a Celery beat task) — this class
fetches and caches it in-process for a few minutes rather than hitting NestJS
on every single Kite call.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import date, datetime, timedelta
from functools import partial
from typing import Any

import httpx
import pandas as pd
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from app.config import Settings
from app.market.intervals import clamp_days

logger = logging.getLogger(__name__)

# Same rationale as yfinance_provider.py's _VENDOR_ERRORS: these degrade to
# None/[], the caller's "no data" path. Anything else is a bug in our own
# code and is re-raised through logger.exception.
_VENDOR_ERRORS = (KiteException, httpx.HTTPError, OSError, KeyError, ValueError, TypeError, IndexError)

_TOKEN_TTL_SECONDS = 300
_INSTRUMENTS_TTL_SECONDS = 86400

_INTERVAL_MAP = {
    "1m": "minute", "5m": "5minute", "15m": "15minute", "30m": "30minute", "1h": "60minute", "1d": "day",
}

# yfinance-style index tickers used elsewhere in this codebase (router.py's
# /market/status, global_cues.py) -- Kite's own instrument dump names the
# same instruments with a human-readable tradingsymbol that has a space in
# it instead. Passing "^NSEI" straight through to Kite's quote()/instruments
# lookup finds nothing and raises KeyError.
_INDEX_ALIASES = {"^NSEI": "NIFTY 50", "^BSESN": "SENSEX"}

# TradingView's own convention for a commodity's continuous, auto-rolling
# contracts (confirmed live: MCX:GOLD1!, MCX:GOLD2!, MCX:GOLDM1!) -- adopted
# here rather than inventing a different one, since that's the shape a user
# coming from TradingView already expects. The digit is the rank by expiry,
# 1 being front month (nearest, not yet expired), 2 the one after that, and
# so on -- never a literal Kite symbol; _resolve_row() below parses it and
# swaps in whichever real dated contract (e.g. "GOLD25DECFUT") is currently
# at that rank, recomputed on every lookup rather than cached as a decision
# -- there is no separate "roll" step to run, because the answer is always
# freshly derived from Kite's own real, vendor-sourced instrument dump
# (refreshed every _INSTRUMENTS_TTL_SECONDS), never a hand-maintained
# mapping that could drift stale.
_CONTINUOUS_SUFFIX = "1!"  # what search() suggests -- always front month
_CONTINUOUS_PATTERN = re.compile(r"^(.+?)(\d+)!$")


def _parse_continuous(symbol: str) -> tuple[str, int] | None:
    """("GOLD", 1) for "GOLD1!", ("GOLD", 2) for "GOLD2!" -- None if the
    symbol isn't shaped like a continuous contract at all."""
    m = _CONTINUOUS_PATTERN.match(symbol)
    return (m.group(1), int(m.group(2))) if m else None


def kite_symbol(symbol: str) -> str:
    """The tradingsymbol Kite's own instrument dump actually uses."""
    return _INDEX_ALIASES.get(symbol.upper(), symbol.upper())


def _as_date(value: Any) -> date:
    """Kite's real SDK returns `expiry` as a `datetime.date` already, but
    the mock fixtures used in tests (and conceivably a future SDK version)
    carry it as a plain "YYYY-MM-DD" string -- coerced here once rather
    than trusting the type at every call site. A missing/unparseable expiry
    sorts last (date.max) instead of raising, since the row it belongs to
    should never have passed `_ensure_instruments()`'s own `r.get("expiry")`
    filter in the first place; this is a second line of defence, not the
    primary guarantee."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            pass
    return date.max


def _matches(haystack: str, needle: str, *, prefix_only: bool = False) -> bool:
    """Prefix or substring match, refusing one that runs into another
    digit right after — "50" matching inside "500" is a different number,
    not a shorter version of the same one."""
    if not needle:
        return False
    positions = [0] if prefix_only else range(len(haystack) - len(needle) + 1)
    ends_in_digit = needle[-1].isdigit()
    for start in positions:
        if haystack[start:start + len(needle)] != needle:
            continue
        end = start + len(needle)
        if not ends_in_digit or end >= len(haystack) or not haystack[end].isdigit():
            return True
    return False


def _score_match(symbol: str, name: str, q: str, q_compact: str, q_spaced: str) -> int | None:
    """Lower is a better match; None means no match at all. Extracted from
    `_search_sync` so MCX's per-commodity-name search (no real per-row
    symbol to match against) can share the exact same ranking rules as
    NSE/BSE's per-instrument search, rather than a second, driftable copy."""
    symbol_l = symbol.lower()
    name_l = name.lower()
    name_compact = name_l.replace(" ", "")

    if symbol_l == q:
        return 0
    if _matches(symbol_l, q, prefix_only=True):
        return 1
    if _matches(name_l, q, prefix_only=True) or _matches(name_l, q_spaced, prefix_only=True):
        return 2
    if _matches(name_compact, q_compact, prefix_only=True):
        return 3
    if _matches(symbol_l, q):
        return 4
    if _matches(name_l, q) or _matches(name_l, q_spaced) or _matches(name_compact, q_compact):
        return 5
    return None


class KiteProvider:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._kite = KiteConnect(api_key=settings.zerodha_api_key)
        self._token_cached_at: float = 0.0
        self._instruments: dict[str, dict[str, dict[str, Any]]] = {}
        self._instruments_loaded_at: float = 0.0

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _ensure_token(self) -> None:
        if self._now() - self._token_cached_at < _TOKEN_TTL_SECONDS and self._kite.access_token:
            return
        resp = httpx.get(
            f"{self._settings.api_service_url}/api/internal/broker/zerodha/session",
            headers={"x-internal-key": self._settings.internal_api_key},
            timeout=10,
        )
        resp.raise_for_status()
        token = resp.json().get("accessToken")
        if not token:
            raise httpx.HTTPError("No Zerodha access token stored yet")
        self._kite.set_access_token(token)
        self._token_cached_at = self._now()

    def _ensure_instruments(self) -> None:
        if self._instruments and self._now() - self._instruments_loaded_at < _INSTRUMENTS_TTL_SECONDS:
            return
        for exch in ("NSE", "BSE"):
            rows = self._kite.instruments(exch)
            # `instrument_type == "EQ"` alone isn't enough: Kite tags some
            # index/benchmark listings (e.g. "NIFTY DIV OPPS 50") as EQ too,
            # and unlike every real tradeable ticker their tradingsymbol is a
            # human-readable name with spaces in it — not chartable, not
            # quotable the normal way, and not something to ever suggest.
            # Real indices (NIFTY 50, SENSEX, ...) also have a space in their
            # tradingsymbol but carry segment "INDICES", which the junk EQ
            # rows don't — so they're let through on that check instead.
            self._instruments[exch] = {
                r["tradingsymbol"]: r for r in rows
                if (r.get("instrument_type") == "EQ" and " " not in r.get("tradingsymbol", ""))
                or r.get("segment") == "INDICES"
            }
        # MCX lists commodity FUTURES, not equities -- there is no "EQ" row
        # to filter for. Options (instrument_type CE/PE) exist on MCX too and
        # are deliberately excluded: futures-only is the scope this was
        # built for, options are a real, separate, more complex concept
        # (strikes, premium, a different P&L model entirely) not attempted
        # here. Every kept row needs a real `expiry` for the continuous-
        # contract resolution in `_resolve_row` to work at all.
        mcx_rows = self._kite.instruments("MCX")
        self._instruments["MCX"] = {
            r["tradingsymbol"]: r for r in mcx_rows
            if r.get("instrument_type") == "FUT" and r.get("expiry")
        }
        self._instruments_loaded_at = self._now()

    def _resolve_row(self, symbol: str, exchange: str) -> dict[str, Any] | None:
        """The instrument row a symbol actually means -- direct lookup for
        everything except an MCX continuous symbol ("GOLD1!", "GOLD2!", ...),
        which resolves to whichever real dated contract currently sits at
        that expiry rank without having expired yet. See
        `_CONTINUOUS_SUFFIX`'s own comment for why this is recomputed on
        every call rather than cached as a decision."""
        exch = exchange.upper()
        sym = kite_symbol(symbol)
        if exch == "MCX":
            parsed = _parse_continuous(sym)
            if parsed:
                name, rank = parsed
                return self._resolve_mcx_continuous(name, rank)
        return self._instruments.get(exch, {}).get(sym)

    def _resolve_mcx_continuous(self, name: str, rank: int = 1) -> dict[str, Any] | None:
        """rank=1 is front month (nearest, not yet expired), rank=2 the one
        after that, and so on -- TradingView's own SYMBOL1!/SYMBOL2!/...
        convention."""
        today = date.today()
        candidates = sorted(
            (row for row in self._instruments.get("MCX", {}).values()
             if str(row.get("name", "")).upper() == name.upper() and _as_date(row.get("expiry")) >= today),
            key=lambda row: _as_date(row["expiry"]),
        )
        if rank < 1 or rank > len(candidates):
            return None
        return candidates[rank - 1]

    # ------------------------------------------------------------------
    # get_quote
    # ------------------------------------------------------------------
    async def get_quote(self, symbol: str, exchange: str) -> dict | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_quote_sync, symbol, exchange)

    def _get_quote_sync(self, symbol: str, exchange: str) -> dict | None:
        try:
            self._ensure_token()
            # Only an MCX continuous symbol ("GOLD1!", "GOLD2!", ...) needs
            # the instrument dump loaded at all -- every other quote (the
            # overwhelming common case) keeps the existing fast, dump-free
            # path exactly as before. Kite's quote() key uses whichever real
            # dated tradingsymbol the continuous one resolves to; the
            # RESPONSE below still echoes back the symbol the caller
            # actually asked for ("GOLD1!"), matching TradingView's own
            # convention of the continuous symbol staying stable while the
            # real contract underneath rolls.
            real_symbol = kite_symbol(symbol)
            if exchange.upper() == "MCX" and _parse_continuous(real_symbol):
                self._ensure_instruments()
                row = self._resolve_row(symbol, exchange)
                if row is None:
                    return None
                real_symbol = row["tradingsymbol"]
            key = f"{exchange.upper()}:{real_symbol}"
            data = self._kite.quote(key)[key]
            ltp = float(data["last_price"])
            ohlc = data["ohlc"]
            prev_close = float(ohlc["close"])
            change = ltp - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0.0
            # Top-of-book bid/ask -- Kite's quote() always includes 5 levels of
            # market depth, we only surface the best price on each side. Absent
            # for the yfinance fallback provider and for pre-market/no-liquidity
            # symbols, so both sides are None rather than a guessed 0.
            depth = data.get("depth") or {}
            buy_levels = depth.get("buy") or []
            sell_levels = depth.get("sell") or []
            bid = float(buy_levels[0]["price"]) if buy_levels else None
            ask = float(sell_levels[0]["price"]) if sell_levels else None
            return {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "ltp": round(ltp, 4),
                "change": round(change, 4),
                "change_percent": round(change_pct, 4),
                "open": float(ohlc["open"]),
                "high": float(ohlc["high"]),
                "low": float(ohlc["low"]),
                "prev_close": round(prev_close, 4),
                "volume": data.get("volume"),
                "bid": bid,
                "ask": ask,
                "spread": round(ask - bid, 4) if bid is not None and ask is not None else None,
            }
        except _VENDOR_ERRORS as e:
            logger.warning("Kite quote fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in Kite quote for %s/%s", symbol, exchange)
            return None

    # ------------------------------------------------------------------
    # get_historical_df
    # ------------------------------------------------------------------
    async def get_historical_df(
        self, symbol: str, exchange: str, interval: str, days: int
    ) -> pd.DataFrame | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._get_historical_df_sync, symbol, exchange, interval, days)
        )

    def _get_historical_df_sync(
        self, symbol: str, exchange: str, interval: str, days: int
    ) -> pd.DataFrame | None:
        try:
            self._ensure_token()
            self._ensure_instruments()
            row = self._resolve_row(symbol, exchange)
            if row is None:
                return None

            to_date = datetime.now()
            from_date = to_date - timedelta(days=clamp_days(interval, days))
            kite_interval = _INTERVAL_MAP.get(interval, "15minute")

            candles = self._kite.historical_data(
                row["instrument_token"], from_date, to_date, kite_interval,
            )
            if not candles:
                return None

            df = pd.DataFrame(candles).set_index("date")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df[["open", "high", "low", "close", "volume"]]
        except _VENDOR_ERRORS as e:
            logger.warning("Kite historical fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in Kite history for %s/%s", symbol, exchange)
            return None

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    async def search(self, query: str, limit: int) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[dict]:
        """Ranked best-first: exact symbol, symbol prefix, name prefix,
        then substring on symbol/name. Every check goes through
        `_matches`, which refuses a match that runs into more digits right
        after it — otherwise "nifty50" prefix-matches "nifty500...", a
        different number wearing the query as its first few characters."""
        try:
            self._ensure_instruments()
            q = query.strip().lower()
            if not q:
                return []
            q_compact = q.replace(" ", "")
            # "nifty50" -> "nifty 50", so a name spelled with the space still matches.
            q_spaced = re.sub(r"([a-z])(\d)", r"\1 \2", q)

            scored: list[tuple[int, dict]] = []
            for exch in ("NSE", "BSE"):
                for symbol, row in self._instruments.get(exch, {}).items():
                    name = row.get("name") or symbol
                    score = _score_match(symbol, name, q, q_compact, q_spaced)
                    if score is not None:
                        scored.append((score, {"symbol": symbol, "name": name, "exchange": exch}))

            # MCX: one result per commodity (its continuous symbol, "GOLD1!"),
            # never one per individual dated contract -- searching "gold"
            # must not flood results with every live expiry month of the
            # same underlying commodity. Matched against the commodity name
            # only; the continuous symbol itself is synthetic (see
            # _CONTINUOUS_SUFFIX) so there's no real "symbol" to match on.
            for name in {row.get("name") for row in self._instruments.get("MCX", {}).values() if row.get("name")}:
                score = _score_match(name, name, q, q_compact, q_spaced)
                if score is not None:
                    scored.append((score, {"symbol": f"{name}{_CONTINUOUS_SUFFIX}", "name": name, "exchange": "MCX"}))

            scored.sort(key=lambda pair: pair[0])
            return [item for _, item in scored[:limit]]
        except _VENDOR_ERRORS as e:
            logger.warning("Kite search failed for %r: %s", query, e)
            return []
        except Exception:
            logger.exception("Unexpected error searching Kite for %r", query)
            return []
