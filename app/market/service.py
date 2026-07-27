"""
MarketDataService — thin façade over MarketDataRouter for the REST API layer.
Converts the router's DataFrame historical shape into the JSON bar shape the
frontend expects; everything else is a pass-through.
"""
from __future__ import annotations

import asyncio
import logging

from app.market.providers.registry import market_data_router

logger = logging.getLogger(__name__)


async def get_quote(symbol: str, exchange: str = "NSE") -> dict | None:
    """Current price data for an equity or forex pair. exchange = NSE | BSE | FOREX | ..."""
    return await market_data_router.get_quote(symbol, exchange)


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
