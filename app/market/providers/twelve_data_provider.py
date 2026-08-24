"""
TwelveDataProvider — spot gold (XAU/USD) and, if extended later, other forex
pairs. Neither Kite (India-only, no forex/spot-metal coverage) nor yfinance
(XAUUSD=X is flaky/unreliable in practice, confirmed live) actually cover
this, so it's a separate real vendor rather than folded into either.

Deliberately scoped to exactly one symbol for now (XAUUSD) -- search()
returns a small hardcoded entry rather than querying the vendor, since
there is nothing else to search for yet. Extending to more pairs later is
just adding entries to _KNOWN_PAIRS, not a redesign.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from functools import partial

import httpx
import pandas as pd

from app.config import Settings
from app.market.intervals import clamp_days

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.twelvedata.com"

# Same rationale as the other providers' own _VENDOR_ERRORS: these degrade to
# None/[], the caller's "no data" path. Anything else is a bug in our own
# code and is re-raised through logger.exception.
_VENDOR_ERRORS = (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError)

_INTERVAL_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "1d": "1day",
}

# App-facing symbol ("XAUUSD", no slash -- passes the same SYMBOL_RE every
# other symbol in this app already passes) -> Twelve Data's own pair format
# ("XAU/USD"). Only what search() actually offers; extend both together.
_KNOWN_PAIRS = {"XAUUSD": ("XAU/USD", "Gold Spot")}


class TwelveDataProvider:
    def __init__(self, settings: Settings):
        self._api_key = settings.twelve_data_api_key

    def _pair_for(self, symbol: str) -> str | None:
        entry = _KNOWN_PAIRS.get(symbol.upper())
        return entry[0] if entry else None

    # ------------------------------------------------------------------
    # get_quote
    # ------------------------------------------------------------------
    async def get_quote(self, symbol: str, exchange: str) -> dict | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_quote_sync, symbol, exchange)

    def _get_quote_sync(self, symbol: str, exchange: str) -> dict | None:
        pair = self._pair_for(symbol)
        if pair is None or not self._api_key:
            return None
        try:
            resp = httpx.get(f"{_BASE_URL}/quote", params={"symbol": pair, "apikey": self._api_key}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "error" or "close" not in data:
                logger.warning("Twelve Data quote error for %s: %s", pair, data.get("message") or data)
                return None

            ltp = float(data["close"])
            prev_close = float(data.get("previous_close") or ltp)
            change = float(data.get("change") or (ltp - prev_close))
            change_pct = float(data.get("percent_change") or ((change / prev_close * 100) if prev_close else 0.0))

            return {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "ltp": round(ltp, 4),
                "change": round(change, 4),
                "change_percent": round(change_pct, 4),
                "open": float(data["open"]) if data.get("open") else None,
                "high": float(data["high"]) if data.get("high") else None,
                "low": float(data["low"]) if data.get("low") else None,
                "prev_close": round(prev_close, 4),
                # Twelve Data's /quote has no bid/ask/spread for spot metals --
                # never fabricated, same "absent not guessed" rule every other
                # provider here follows for a field the vendor doesn't give.
                "volume": None,
                "bid": None,
                "ask": None,
                "spread": None,
            }
        except _VENDOR_ERRORS as e:
            logger.warning("Twelve Data quote fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in Twelve Data quote for %s/%s", symbol, exchange)
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
        pair = self._pair_for(symbol)
        if pair is None or not self._api_key:
            return None
        try:
            td_interval = _INTERVAL_MAP.get(interval, "1day")
            span = clamp_days(interval, days)
            # Rough bars-per-day so the requested window actually gets covered,
            # capped at Twelve Data's own 5000-point ceiling for one request.
            bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "1d": 1}.get(interval, 1)
            outputsize = min(int(span * bars_per_day) or 1, 5000)

            resp = httpx.get(
                f"{_BASE_URL}/time_series",
                params={"symbol": pair, "interval": td_interval, "outputsize": outputsize, "apikey": self._api_key},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "error" or "values" not in data:
                logger.warning("Twelve Data time_series error for %s: %s", pair, data.get("message") or data)
                return None

            rows = data["values"]
            if not rows:
                return None

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["datetime"])
            df = df.set_index("date").sort_index()
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].astype(float)
            # Spot metals carry no real trade volume from this vendor -- 0,
            # never fabricated, matches how the chart already treats an
            # absent volume series elsewhere.
            df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
            return df[["open", "high", "low", "close", "volume"]]
        except _VENDOR_ERRORS as e:
            logger.warning("Twelve Data historical fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in Twelve Data history for %s/%s", symbol, exchange)
            return None

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    async def search(self, query: str, limit: int) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []
        out = []
        for app_symbol, (pair, name) in _KNOWN_PAIRS.items():
            haystacks = (app_symbol.lower(), name.lower(), pair.lower())
            if any(q in h for h in haystacks):
                out.append({"symbol": app_symbol, "name": name, "exchange": "FOREX"})
        return out[:limit]
