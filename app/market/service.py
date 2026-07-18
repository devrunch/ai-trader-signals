"""
MarketDataService — live quotes and historical OHLCV.

Indian equities  : Angel One SmartAPI WebSocket (or yFinance fallback)
Forex            : OANDA v20 streaming (oandapyV20)
Backtesting      : yFinance for both markets
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker helpers
# ---------------------------------------------------------------------------

def _equity_ticker(symbol: str, exchange: str) -> str:
    """RELIANCE + NSE → RELIANCE.NS"""
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    return f"{symbol.upper()}{suffix}"


def _forex_ticker(pair: str) -> str:
    """EURUSD → EURUSD=X  (yFinance format)"""
    pair = pair.upper().replace("/", "").replace("-", "")
    if not pair.endswith("=X"):
        pair = f"{pair}=X"
    return pair


# ---------------------------------------------------------------------------
# Quote (single current price)
# ---------------------------------------------------------------------------

async def get_quote(symbol: str, exchange: str = "NSE") -> Optional[dict]:
    """
    Returns current price data for an equity or forex pair.
    exchange = NSE | BSE | FOREX
    """
    try:
        if exchange.upper() == "FOREX":
            ticker_str = _forex_ticker(symbol)
        else:
            ticker_str = _equity_ticker(symbol, exchange)

        ticker = yf.Ticker(ticker_str)
        info = ticker.fast_info

        def _safe(attr, *fallback_attrs):
            for a in (attr, *fallback_attrs):
                v = getattr(info, a, None)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
            return None

        ltp = _safe("last_price")
        if not ltp:
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                return None
            ltp = float(hist["Close"].iloc[-1])

        prev_close = _safe("previous_close", "regular_market_previous_close") or ltp
        change = ltp - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        return {
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
            "ltp": round(ltp, 4),
            "change": round(change, 4),
            "change_percent": round(change_pct, 4),
            "open": _safe("open"),
            "high": _safe("day_high", "high"),
            "low": _safe("day_low", "low"),
            "prev_close": round(prev_close, 4),
            "volume": _safe("last_volume", "three_month_average_volume"),
        }
    except Exception as e:
        logger.warning("Quote fetch failed for %s/%s: %s", symbol, exchange, e)
        return None


# ---------------------------------------------------------------------------
# Historical OHLCV
# ---------------------------------------------------------------------------

async def get_historical(
    symbol: str,
    exchange: str = "NSE",
    interval: str = "15m",
    days: int = 30,
) -> list[dict]:
    """
    Returns OHLCV bars as list of dicts with Unix timestamp.
    interval: 1m | 5m | 15m | 1h | 1d
    """
    try:
        if exchange.upper() == "FOREX":
            ticker_str = _forex_ticker(symbol)
        else:
            ticker_str = _equity_ticker(symbol, exchange)

        period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "1d": "5y"}
        period = period_map.get(interval, "60d")

        df: pd.DataFrame = yf.download(
            ticker_str,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            return []

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df = df.dropna()

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

    except Exception as e:
        logger.warning("Historical fetch failed for %s/%s: %s", symbol, exchange, e)
        return []


# ---------------------------------------------------------------------------
# Batch quotes (for screener)
# ---------------------------------------------------------------------------

async def get_batch_quotes(symbols: list[str], exchange: str = "NSE") -> list[dict]:
    """Fetch quotes for multiple symbols in one yFinance call."""
    try:
        if exchange.upper() == "FOREX":
            tickers = [_forex_ticker(s) for s in symbols]
        else:
            tickers = [_equity_ticker(s, exchange) for s in symbols]

        data = yf.download(tickers, period="2d", interval="1d", progress=False, auto_adjust=True)
        results = []

        for i, symbol in enumerate(symbols):
            try:
                ticker_str = tickers[i]
                quote = await get_quote(symbol, exchange)
                if quote:
                    results.append(quote)
            except Exception:
                continue

        return results
    except Exception as e:
        logger.warning("Batch quote fetch failed: %s", e)
        return []
