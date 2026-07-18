from fastapi import APIRouter, HTTPException, Query
from app.market.service import get_quote, get_historical, get_batch_quotes

router = APIRouter()


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


@router.post("/quotes/batch")
async def batch_quotes(symbols: list[str], exchange: str = Query(default="NSE")):
    """Batch quote fetch for screener — returns available quotes."""
    return await get_batch_quotes(symbols, exchange.upper())


@router.get("/status")
async def market_status():
    """NSE market open/closed + Nifty 50 + Sensex snapshot."""
    from datetime import datetime, time
    import pytz

    ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(ist)
    market_open = time(9, 15)
    market_close = time(15, 30)
    is_weekday = now_ist.weekday() < 5
    is_market_hours = market_open <= now_ist.time() <= market_close

    nifty = await get_quote("^NSEI", "NSE") or {}
    sensex = await get_quote("^BSESN", "BSE") or {}

    return {
        "nse_open": is_weekday and is_market_hours,
        "timestamp": now_ist.isoformat(),
        "nifty50": {
            "ltp": nifty.get("ltp"),
            "change_percent": nifty.get("change_percent"),
        },
        "sensex": {
            "ltp": sensex.get("ltp"),
            "change_percent": sensex.get("change_percent"),
        },
    }
