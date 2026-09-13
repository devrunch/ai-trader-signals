from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from app.market.contract import BarsStatus
from app.market.live_ticks import LiveTicks
from app.market.providers.registry import market_data_router
from app.market.service import get_quote, get_tick_volume, get_ticks, search_symbols

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


# What each outcome means over HTTP. The code is for anything that only
# speaks status codes -- monitoring, a cache, another service -- and the body
# carries the same answer in a form the terminal can render. A closed market
# is deliberately a 200: it is a normal state of the world, and reporting it
# as a 404 is what buried the real failures in noise.
_HTTP_STATUS = {
    BarsStatus.OK: 200,
    BarsStatus.NO_DATA: 200,
    BarsStatus.CLOSED_MARKET: 200,
    BarsStatus.UNSUPPORTED_INTERVAL: 400,
    BarsStatus.UNKNOWN_SYMBOL: 404,
    BarsStatus.VENDOR_ERROR: 503,
    BarsStatus.OUT_OF_RETENTION: 200,
}


@router.get("/historical/{symbol}")
async def historical(
    response: Response,
    symbol: str,
    exchange: str | None = Query(default=None, description="Only disambiguates a dual listing; the symbol decides its venue"),
    interval: str = Query(default="15m", description="1m | 5m | 15m | 1h | 1d"),
    days: int = Query(default=30),
):
    """OHLCV bars for charting, plus why they are what they are.

    Never an unexplained empty chart: the body always carries a `status`,
    and the HTTP code says whether anything is actually wrong. `exchange` is
    optional now -- passing the wrong one (every caller that did not know sent
    NSE) no longer sends the request nowhere."""
    result = await market_data_router.get_bars(symbol.upper(), interval, days, exchange=exchange)
    response.status_code = _HTTP_STATUS.get(result.status, 200)
    if result.status is BarsStatus.VENDOR_ERROR:
        # A vendor outage is worth retrying, unlike a bad request.
        response.headers["Retry-After"] = "30"
    return {
        "symbol": symbol.upper(),
        # The venue it was served from, never the one the caller guessed: a
        # layout saved with the wrong exchange breaks when it is reopened.
        "exchange": result.symbol.exchange if result.symbol else (exchange or "").upper(),
        "interval": interval,
        "bars": result.bars,
        "status": result.status.value,
        "reason": result.reason,
        "volumeSource": result.volume_source.value,
        "truncatedToDays": result.truncated_to_days,
    }


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
