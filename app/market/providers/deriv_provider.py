"""
DerivProvider — forex majors/minors and precious metals (XAU/XAG/XPD/XPT
vs USD) via Deriv's public WebSocket API. Free, no account, no API key, no
rate-limit wall -- confirmed live: all 29 instruments below stream
concurrently on one connection with zero errors, and quote/historical
requests here work the same way against the same public endpoint.

Deriv has no separate plain-REST surface -- every request (even a one-off
quote) goes over WebSocket, so unlike every other provider in this app
(which wrap a synchronous SDK/httpx call in a thread executor), this one is
natively async: each call opens a short-lived connection, sends one
request, reads the matching response, and closes. The always-on connection
for live ticks is a separate, persistent client (deriv_ticker.py) -- same
split this app already has for Kite (KiteProvider's REST calls vs
KiteTickerClient's one persistent socket), not a new pattern.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pandas as pd
import websockets
from websockets.exceptions import WebSocketException

from app.market.intervals import clamp_days

logger = logging.getLogger(__name__)

WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"

# Same rationale as every other provider's own _VENDOR_ERRORS: these degrade
# to None/[], the caller's "no data" path. Anything else is a bug in our own
# code and is re-raised through logger.exception.
_VENDOR_ERRORS = (WebSocketException, OSError, TimeoutError,
                  KeyError, ValueError, TypeError, IndexError)

_GRANULARITY_MAP = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400}
_BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "1d": 1}
# Not vendor-confirmed (Deriv's ticks_history has no documented cap found) --
# matches Twelve Data's own documented 5000-point ceiling as a safe
# assumption; revisit if a real request ever hits it.
_MAX_COUNT = 5000

# App-facing symbol (no slash, matches this app's SYMBOL_RE) -> (Deriv's own
# symbol, display name). All 29 real instruments Deriv's public API lists
# under market "forex"/"commodities" -- confirmed live via active_symbols,
# not guessed.
KNOWN_PAIRS: dict[str, tuple[str, str]] = {
    "AUDCAD": ("frxAUDCAD", "AUD/CAD"), "AUDCHF": ("frxAUDCHF", "AUD/CHF"),
    "AUDJPY": ("frxAUDJPY", "AUD/JPY"), "AUDNZD": ("frxAUDNZD", "AUD/NZD"),
    "AUDUSD": ("frxAUDUSD", "AUD/USD"),
    "EURAUD": ("frxEURAUD", "EUR/AUD"), "EURCAD": ("frxEURCAD", "EUR/CAD"),
    "EURCHF": ("frxEURCHF", "EUR/CHF"), "EURGBP": ("frxEURGBP", "EUR/GBP"),
    "EURJPY": ("frxEURJPY", "EUR/JPY"), "EURNZD": ("frxEURNZD", "EUR/NZD"),
    "EURUSD": ("frxEURUSD", "EUR/USD"),
    "GBPAUD": ("frxGBPAUD", "GBP/AUD"), "GBPCAD": ("frxGBPCAD", "GBP/CAD"),
    "GBPCHF": ("frxGBPCHF", "GBP/CHF"), "GBPJPY": ("frxGBPJPY", "GBP/JPY"),
    "GBPNZD": ("frxGBPNZD", "GBP/NZD"), "GBPUSD": ("frxGBPUSD", "GBP/USD"),
    "NZDJPY": ("frxNZDJPY", "NZD/JPY"), "NZDUSD": ("frxNZDUSD", "NZD/USD"),
    "USDCAD": ("frxUSDCAD", "USD/CAD"), "USDCHF": ("frxUSDCHF", "USD/CHF"),
    "USDJPY": ("frxUSDJPY", "USD/JPY"), "USDMXN": ("frxUSDMXN", "USD/MXN"),
    "USDPLN": ("frxUSDPLN", "USD/PLN"),
    "XAGUSD": ("frxXAGUSD", "Silver/USD"), "XAUUSD": ("frxXAUUSD", "Gold/USD"),
    "XPDUSD": ("frxXPDUSD", "Palladium/USD"), "XPTUSD": ("frxXPTUSD", "Platinum/USD"),
}


def deriv_symbol_for(app_symbol: str) -> str | None:
    entry = KNOWN_PAIRS.get(app_symbol.upper())
    return entry[0] if entry else None


async def _request(payload: dict) -> dict:
    """One request, one response, one short-lived connection -- see this
    module's own docstring for why (no plain REST surface to call
    instead)."""
    async with websockets.connect(WS_URL, open_timeout=10) as ws:
        await ws.send(json.dumps(payload))
        raw = await asyncio.wait_for(ws.recv(), timeout=15)
        return json.loads(raw)


class DerivProvider:
    def _pair_for(self, symbol: str) -> str | None:
        return deriv_symbol_for(symbol)

    # ------------------------------------------------------------------
    # get_quote
    # ------------------------------------------------------------------
    async def get_quote(self, symbol: str, exchange: str) -> dict | None:
        pair = self._pair_for(symbol)
        if pair is None:
            return None
        try:
            # Two daily candles: [yesterday (completed), today (forming)].
            # The forming candle's own close is the latest tick -- Deriv's
            # own semantics, confirmed live -- so this one request covers
            # ltp/open/high/low/prev_close together, no second call needed.
            resp = await _request({
                "ticks_history": pair, "style": "candles",
                "granularity": 86400, "count": 2, "end": "latest",
            })
            if "error" in resp:
                logger.warning("Deriv quote error for %s: %s", pair, resp["error"])
                return None
            candles = resp.get("candles") or []
            if not candles:
                return None

            today = candles[-1]
            prev_close = float(candles[-2]["close"]) if len(candles) > 1 else float(today["open"])
            ltp = float(today["close"])
            change = ltp - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0.0

            return {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "ltp": round(ltp, 5),
                "change": round(change, 5),
                "change_percent": round(change_pct, 4),
                "open": float(today["open"]),
                "high": float(today["high"]),
                "low": float(today["low"]),
                "prev_close": round(prev_close, 5),
                # No real trade volume on a spot/CFD forex feed -- never
                # fabricated, same "absent not guessed" rule every other
                # provider here follows for a field the vendor doesn't give.
                "volume": None,
                "bid": None,
                "ask": None,
                "spread": None,
            }
        except _VENDOR_ERRORS as e:
            logger.warning("Deriv quote fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in Deriv quote for %s/%s", symbol, exchange)
            return None

    # ------------------------------------------------------------------
    # get_historical_df
    # ------------------------------------------------------------------
    async def get_historical_df(
        self, symbol: str, exchange: str, interval: str, days: int
    ) -> pd.DataFrame | None:
        pair = self._pair_for(symbol)
        if pair is None:
            return None
        try:
            granularity = _GRANULARITY_MAP.get(interval, 86400)
            span = clamp_days(interval, days)
            bars_per_day = _BARS_PER_DAY.get(interval, 1)
            count = min(int(span * bars_per_day) or 1, _MAX_COUNT)

            resp = await _request({
                "ticks_history": pair, "style": "candles",
                "granularity": granularity, "count": count, "end": "latest",
            })
            if "error" in resp:
                logger.warning("Deriv time_series error for %s: %s", pair, resp["error"])
                return None
            candles = resp.get("candles") or []
            if not candles:
                return None

            df = pd.DataFrame(candles)
            df["date"] = pd.to_datetime(df["epoch"], unit="s")
            df = df.set_index("date").sort_index()
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].astype(float)
            df["volume"] = 0.0
            return df[["open", "high", "low", "close", "volume"]]
        except _VENDOR_ERRORS as e:
            logger.warning("Deriv historical fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in Deriv history for %s/%s", symbol, exchange)
            return None

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    async def search(self, query: str, limit: int) -> list[dict]:
        q = query.strip().lower()
        if not q:
            return []
        out = []
        for app_symbol, (deriv_sym, name) in KNOWN_PAIRS.items():
            haystacks = (app_symbol.lower(), name.lower(), deriv_sym.lower())
            if any(q in h for h in haystacks):
                out.append({"symbol": app_symbol, "name": name, "exchange": "FOREX"})
        return out[:limit]
