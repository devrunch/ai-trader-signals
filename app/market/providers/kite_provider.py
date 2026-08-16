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
import time
from datetime import datetime, timedelta
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
    "1m": "minute", "5m": "5minute", "15m": "15minute", "1h": "60minute", "1d": "day",
}


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
            self._instruments[exch] = {
                r["tradingsymbol"]: r for r in rows if r.get("instrument_type") == "EQ"
            }
        self._instruments_loaded_at = self._now()

    # ------------------------------------------------------------------
    # get_quote
    # ------------------------------------------------------------------
    async def get_quote(self, symbol: str, exchange: str) -> dict | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_quote_sync, symbol, exchange)

    def _get_quote_sync(self, symbol: str, exchange: str) -> dict | None:
        try:
            self._ensure_token()
            key = f"{exchange.upper()}:{symbol.upper()}"
            data = self._kite.quote(key)[key]
            ltp = float(data["last_price"])
            ohlc = data["ohlc"]
            prev_close = float(ohlc["close"])
            change = ltp - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0.0
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
            }
        except _VENDOR_ERRORS as e:
            logger.warning("Kite quote fetch failed for %s/%s: %s", symbol, exchange, e)
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
            row = self._instruments.get(exchange.upper(), {}).get(symbol.upper())
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

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    async def search(self, query: str, limit: int) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[dict]:
        try:
            self._ensure_instruments()
            q = query.strip().lower()
            if not q:
                return []
            results = []
            for exch in ("NSE", "BSE"):
                for symbol, row in self._instruments.get(exch, {}).items():
                    name = row.get("name") or symbol
                    if q in symbol.lower() or q in name.lower():
                        results.append({"symbol": symbol, "name": name, "exchange": exch})
                        if len(results) >= limit:
                            return results
            return results
        except _VENDOR_ERRORS as e:
            logger.warning("Kite search failed for %r: %s", query, e)
            return []
