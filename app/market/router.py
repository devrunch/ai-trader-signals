from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.market.live_ticks import LiveTicks
from app.market.service import get_historical, get_quote, get_tick_volume, get_ticks, search_symbols

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


@router.get("/ticks/{symbol}")
async def ticks(
    symbol: str,
    since: int = Query(description="Unix epoch seconds"),
    until: int = Query(description="Unix epoch seconds"),
):
    """Real ECN ticks (mid price) for Volume Footprint/TPO -- FOREX/metals
    only, and only for whatever the chart currently has visible (see
    get_ticks' own MAX_TICKS_WINDOW_SECONDS). `ticks: null` means this
    symbol isn't Dukascopy-covered, the window was rejected, or the vendor
    call failed -- the caller must not render an empty footprint as "no
    trading happened," only as "couldn't check."""
    result = await get_ticks(symbol.upper(), since, until)
    return {"symbol": symbol.upper(), "since": since, "until": until, "ticks": result}
