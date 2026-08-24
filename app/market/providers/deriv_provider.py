"""
DerivProvider — forex majors/minors and precious metals (XAU/XAG/XPD/XPT
vs USD) via Deriv's public WebSocket API. Free, no account, no API key.
No rate-limit wall for ordinary use -- confirmed live: all 29 instruments
below stream concurrently on one connection with zero errors, and
quote/historical requests here work the same way against the same public
endpoint. That does NOT extend to bursts of many short-lived connections in
quick succession, though -- see _TICK_BACKFILL_SECONDS's own comment for
where a real, tight rate limit showed up (heavy HTTP 429s from ~70-85
rapid ticks_history requests).

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
import bisect
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
# `count` on a `style: "candles"` request -- not vendor-confirmed (Deriv's
# docs list no documented cap), matches Twelve Data's own documented
# 5000-point ceiling as a safe assumption; revisit if a real request ever
# hits it. NOT the same limit as `style: "ticks"` -- see _TICK_CHUNK_SECONDS,
# confirmed live to silently cap around 1000 regardless of `count`.
_MAX_COUNT = 5000

# ticks_history with style="ticks" silently caps a single response at
# roughly 1000 ticks, anchored to `end`, regardless of the `count` requested
# -- confirmed live: a request for a full hour of XAUUSD ticks came back
# with exactly 1000, covering only the most recent ~17 minutes, no error.
# 900s (15 min) stays comfortably under that for a liquid pair, with margin;
# _fetch_ticks_chunk's own trust check catches it on the rare chunk that
# still exceeds it (a burst of activity), rather than assuming every chunk
# is safe just because it's short.
_TICK_CHUNK_SECONDS = 900
# Deriv's real tick-history rate limit is far tighter than "no rate-limit
# wall" (this module's own top-of-file claim, true for candles/quotes)
# suggests -- confirmed live: backfilling a whole multi-day candle range
# (~70-85 chunks, even bounded to _TICK_BACKFILL_CONCURRENCY in flight at
# once) triggered heavy HTTP 429 rejections, 21-51s chart loads, and STILL
# incomplete coverage anyway -- only whichever chunks happened to dodge the
# limiter came back with real counts. Backfilling the entire fetched range
# was never going to work at any concurrency; instead this only ever
# attempts the most recent _TICK_BACKFILL_SECONDS of the range -- what's
# actually visible on a freshly-loaded chart, not the full history panning
# back could reach. 12 chunks at this width, comfortably under whatever
# threshold triggered the 429s above. Older candles keep the honest 0.0,
# same as a window Deriv's cache never covered at all.
_TICK_BACKFILL_SECONDS = 3 * 3600
# Bounded fan-out for the chunks that ARE requested -- same reasoning as
# ToolContext.gather_bounded in the chat agent, just local here rather than
# shared, since this module has no other concurrency-limiting need.
_TICK_BACKFILL_CONCURRENCY = 8

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


async def _fetch_ticks_chunk(pair: str, chunk_start: int, chunk_end: int) -> list[int] | None:
    """Raw tick epochs in [chunk_start, chunk_end), or None if this one
    chunk's own response can't be trusted.

    Deriv silently narrows a ticks_history response to whatever it actually
    has cached, with no error and without necessarily hitting the `count`
    requested -- confirmed live: a request for a full hour of XAUUSD ticks
    came back with exactly 1000 (well under the 5000 asked for), covering
    only the most recent ~17 minutes. Absence of an error or a full `count`
    is not by itself proof of full coverage, so this is trusted only when
    the earliest tick actually returned reaches back near chunk_start
    (_TICK_CHUNK_SECONDS of slack for a genuine no-tick-right-at-open gap).
    """
    try:
        resp = await _request({
            "ticks_history": pair, "style": "ticks",
            "start": chunk_start, "end": chunk_end, "count": _MAX_COUNT,
        })
    except _VENDOR_ERRORS as e:
        logger.warning("Deriv tick-volume chunk fetch failed for %s: %s", pair, e)
        return None
    if "error" in resp:
        return None
    times = (resp.get("history") or {}).get("times") or []
    if not times:
        return None

    ints = sorted(int(t) for t in times)
    if ints[0] > chunk_start + _TICK_CHUNK_SECONDS:
        return None
    return ints


async def _backfill_tick_volume(
    pair: str, start_epoch: int, end_epoch: int, bucket_starts: list[int]
) -> list[float]:
    """Tick count per candle, as a volume stand-in -- see this module's
    top-of-file rationale for counting ticks at all (no real trade size on
    a spot/CFD forex feed to report, same reason MT4/MT5 use the same
    convention).

    Only ever backfills the most recent _TICK_BACKFILL_SECONDS of
    [start_epoch, end_epoch), not the whole candle range -- see that
    constant's own comment for why (Deriv's real rate limit made
    whole-range backfill both slow and, since the limiter kept knocking out
    chunks anyway, no more complete than this deliberately narrower
    version). One `_fetch_ticks_chunk` request per _TICK_CHUNK_SECONDS-wide
    window within that trailing slice, fanned out with bounded concurrency.
    A chunk whose own response can't be trusted contributes zero counts to
    the candles inside it rather than a guess -- indistinguishable from
    real zero activity in the final numbers, but never a silently wrong
    nonzero one; candles entirely before the trailing slice get the same
    honest 0.0 for the same reason. `bucket_starts` must be sorted
    ascending -- the caller's `candles` list already is.
    """
    backfill_start = max(start_epoch, end_epoch - _TICK_BACKFILL_SECONDS)
    chunk_starts = range(backfill_start, end_epoch, _TICK_CHUNK_SECONDS)
    sem = asyncio.Semaphore(_TICK_BACKFILL_CONCURRENCY)

    async def guarded(chunk_start: int) -> list[int] | None:
        async with sem:
            return await _fetch_ticks_chunk(pair, chunk_start, min(chunk_start + _TICK_CHUNK_SECONDS, end_epoch))

    chunks = await asyncio.gather(*(guarded(cs) for cs in chunk_starts))

    counts = [0] * len(bucket_starts)
    for chunk_times in chunks:
        if not chunk_times:
            continue
        for t in chunk_times:
            idx = bisect.bisect_right(bucket_starts, t) - 1
            if 0 <= idx < len(counts):
                counts[idx] += 1
    return [float(c) for c in counts]


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

            bucket_starts = [int(c["epoch"]) for c in candles]
            tick_volume = await _backfill_tick_volume(
                pair, bucket_starts[0], bucket_starts[-1] + granularity, bucket_starts,
            )

            df = pd.DataFrame(candles)
            df["date"] = pd.to_datetime(df["epoch"], unit="s")
            df = df.set_index("date").sort_index()
            for col in ("open", "high", "low", "close"):
                df[col] = df[col].astype(float)
            df["volume"] = tick_volume
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
