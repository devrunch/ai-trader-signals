"""
MarketDataService — thin façade over MarketDataRouter for the REST API layer.
Converts the router's DataFrame historical shape into the JSON bar shape the
frontend expects; everything else is a pass-through.
"""
from __future__ import annotations

import asyncio
import logging

from app.market.providers.deriv_provider import deriv_symbol_for, tick_volume_since
from app.market.providers.dukascopy_bridge import fetch_ticks
from app.market.providers.registry import market_data_router

logger = logging.getLogger(__name__)

# Volume Footprint/TPO only need whatever's currently visible on the chart,
# not the whole loaded history -- bounding the window keeps a single
# request from turning into a multi-hour Dukascopy backfill. Mirrors the
# same reasoning as MAX_TICK_VOLUME_LOOKBACK_SECONDS on the api-repo side.
MAX_TICKS_WINDOW_SECONDS = 4 * 60 * 60


async def get_quote(symbol: str, exchange: str = "NSE") -> dict | None:
    """Current price data for an equity or forex pair. exchange = NSE | BSE | FOREX | ..."""
    return await market_data_router.get_quote(symbol, exchange)


async def get_tick_volume(symbol: str, since_epoch: int) -> int | None:
    """Real-time ECN tick count since `since_epoch` (Unix seconds) -- the
    still-forming candle's live volume, polled by the chart rather than
    waiting on the next historical fetch. FOREX/metals only (Dukascopy);
    None for anything else, or a real vendor gap -- see tick_volume_since's
    own docstring for why that must not collapse to 0."""
    return await tick_volume_since(symbol, since_epoch)


async def get_ticks(symbol: str, since_epoch: int, until_epoch: int) -> list[dict] | None:
    """Real ECN ticks (mid price) for Volume Footprint/TPO -- FOREX/metals
    only (Dukascopy); None for anything else, a window that's too wide, or
    a real vendor gap. Never a fabricated/interpolated tick list -- a chart
    type built on invented intrabar prices would be worse than no chart
    type at all."""
    if deriv_symbol_for(symbol) is None:
        return None
    if until_epoch - since_epoch > MAX_TICKS_WINDOW_SECONDS or until_epoch <= since_epoch:
        return None
    return await fetch_ticks(symbol.lower(), since_epoch * 1000, until_epoch * 1000)


async def search_symbols(query: str, limit: int = 8) -> list[dict]:
    """Company name / symbol -> matches on exchanges this app can actually chart."""
    return await market_data_router.search(query, limit)


async def get_historical(
    symbol: str,
    exchange: str = "NSE",
    interval: str = "15m",
    days: int = 30,
) -> list[dict]:
    """OHLCV bars as list of dicts with Unix timestamp, for charting."""
    df = await market_data_router.get_historical_df(symbol, exchange, interval, days)
    if df is None or df.empty:
        return []

    bars = []
    for ts, row in df.iterrows():
        bars.append({
            "time": int(ts.timestamp()),
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": int(row.get("volume", 0)),
        })
    return bars


async def get_batch_quotes(symbols: list[str], exchange: str = "NSE") -> list[dict]:
    """Fetch quotes for multiple symbols concurrently.

    Returns only the symbols that answered — the list is shorter than the
    request when the vendor drops some. That shortfall used to be invisible
    (the whole thing was wrapped in `except Exception: return []`, which also
    made a total failure indistinguishable from "no symbols asked for"), so the
    missing symbols are now named in the log and the caller can compare lengths.
    """
    quotes = await asyncio.gather(
        *(get_quote(s, exchange) for s in symbols), return_exceptions=True
    )

    results: list[dict] = []
    missing: list[str] = []
    # strict=True: gather() returns exactly one result per awaitable, so a
    # length mismatch would mean the pairing below is wrong and quotes are
    # being attributed to the wrong symbol — loud is the only safe answer.
    for symbol, quote in zip(symbols, quotes, strict=True):
        if isinstance(quote, dict):
            results.append(quote)
        elif isinstance(quote, BaseException):
            # A provider raising rather than returning None is a bug, not a
            # data outage — gather() only captured it because of
            # return_exceptions, so log it with the traceback.
            logger.error("Quote fetch raised for %s/%s: %r", symbol, exchange, quote)
            missing.append(symbol)
        else:
            missing.append(symbol)

    if missing:
        logger.warning(
            "Batch quotes incomplete for %s — %d of %d missing: %s",
            exchange, len(missing), len(symbols), ", ".join(missing),
        )
    return results
