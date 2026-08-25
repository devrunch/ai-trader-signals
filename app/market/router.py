import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.market import calendar as market_calendar
from app.market.live_ticks import LiveTicks
from app.market.news import get_market_news_result
from app.market.service import get_batch_quotes, get_historical, get_quote, get_tick_volume, search_symbols

logger = logging.getLogger(__name__)

router = APIRouter()

# Assigned by main.py's lifespan hook at startup.
live_ticks: LiveTicks | None = None


class _SymbolExchange(BaseModel):
    symbol: str
    exchange: str


@router.post("/internal/live-ticks/subscribe")
async def subscribe_live_ticks(body: _SymbolExchange):
    """Called by NestJS on a symbol room's first watcher. No auth guard —
    signals-1 is never internet-facing, same as every route above."""
    if live_ticks is None:
        raise HTTPException(status_code=503, detail="live ticks not ready yet")
    ok = await live_ticks.subscribe(body.symbol, body.exchange)
    return {"ok": ok}


@router.post("/internal/live-ticks/unsubscribe")
async def unsubscribe_live_ticks(body: _SymbolExchange):
    """Called by NestJS on a symbol room's last watcher leaving."""
    if live_ticks is None:
        raise HTTPException(status_code=503, detail="live ticks not ready yet")
    await live_ticks.unsubscribe(body.symbol, body.exchange)
    return {"ok": True}


@router.get("/quote/{symbol}")
async def quote(
    symbol: str,
    exchange: str = Query(default="NSE", description="NSE | BSE | FOREX"),
):
    """Live price for a single symbol. Used by paper trading engine for order execution."""
    data = await get_quote(symbol.upper(), exchange.upper())
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}/{exchange}")
    return data


@router.get("/search")
async def search(q: str = Query(min_length=1, max_length=40)):
    """Company name or symbol -> matches, name attached, on an exchange this
    app can actually chart. Powers the terminal's search box."""
    return {"results": await search_symbols(q)}


@router.get("/historical/{symbol}")
async def historical(
    symbol: str,
    exchange: str = Query(default="NSE"),
    interval: str = Query(default="15m", description="1m | 5m | 15m | 1h | 1d"),
    days: int = Query(default=30),
):
    """OHLCV bars for charting. Works for equities (NSE/BSE) and Forex pairs."""
    bars = await get_historical(symbol.upper(), exchange.upper(), interval, days)
    if not bars:
        raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")
    return {"symbol": symbol.upper(), "exchange": exchange.upper(), "interval": interval, "bars": bars}


@router.get("/tick-volume/{symbol}")
async def tick_volume(
    symbol: str,
    since: int = Query(description="Unix epoch seconds -- real ticks counted from here to now"),
):
    """Live tick-count volume for the chart's still-forming candle, polled
    every few seconds while it's open. FOREX/metals only -- `count: null`
    means this symbol isn't Dukascopy-covered or the vendor call failed,
    never a fabricated 0 (see get_tick_volume's own docstring)."""
    count = await get_tick_volume(symbol.upper(), since)
    return {"symbol": symbol.upper(), "since": since, "count": count}


@router.post("/quotes/batch")
async def batch_quotes(symbols: list[str], exchange: str = Query(default="NSE")):
    """Batch quote fetch for screener — returns available quotes."""
    return await get_batch_quotes(symbols, exchange.upper())


@router.get("/news")
async def news(
    symbols: str | None = Query(default=None, description="Comma-separated NSE tickers"),
    limit: int = Query(default=15, le=30),
):
    """Market news with FinBERT sentiment scoring. Requires NEWS_API_KEY in env."""
    sym_list = [s.strip().upper() for s in symbols.split(",")] if symbols else None
    return await get_market_news_result(sym_list, limit)


@router.get("/status")
async def market_status():
    """NSE market open/closed + Nifty 50 + Sensex snapshot."""
    # Session state comes from app/market/calendar.py — it used to be three
    # inline time() comparisons here, which reported the market open on every
    # trading holiday. The two index quotes are fetched together rather than
    # one after the other; this is a page-load endpoint and each leg is a
    # network round-trip.
    state = market_calendar.session_state()
    # return_exceptions: one index failing must not blank the other, nor the
    # session state, which needs no network at all.
    results: list = await asyncio.gather(
        get_quote("^NSEI", "NSE"),
        get_quote("^BSESN", "BSE"),
        return_exceptions=True,
    )
    for res in results:
        if isinstance(res, BaseException):
            logger.warning("Market status index quote failed: %r", res)
    nifty: dict = results[0] if isinstance(results[0], dict) else {}
    sensex: dict = results[1] if isinstance(results[1], dict) else {}

    return {
        "nse_open": state["is_open"],
        "is_trading_day": state["is_trading_day"],
        "is_holiday": state["is_holiday"],
        "is_square_off": state["is_square_off"],
        "next_open": state["next_open"],
        "holiday_calendar_known": state["holiday_calendar_known"],
        "timestamp": state["timestamp"],
        "nifty50": {
            "ltp": nifty.get("ltp"),
            "change_percent": nifty.get("change_percent"),
        },
        "sensex": {
            "ltp": sensex.get("ltp"),
            "change_percent": sensex.get("change_percent"),
        },
        # A null ltp above means the quote did not arrive, not that the index is
        # at zero. Without this the frontend cannot tell a stale panel from a
        # flat market.
        "degraded": not (nifty and sensex),
    }
