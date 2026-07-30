"""
YFinanceProvider — free, delayed (~15-20 min) data covering equities and forex.
Today's default/fallback provider for every exchange until a real-time vendor
(Dhan for NSE/BSE, Alpaca for NASDAQ/NYSE, ...) is wired in for that exchange.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from functools import partial

import pandas as pd
import yfinance as yf

from app.market.intervals import clamp_days

logger = logging.getLogger(__name__)

# Errors a scraped, unofficial vendor API is *expected* to produce: the network
# failing, Yahoo throttling us into an HTML error page, a symbol that no longer
# exists, a response whose shape changed. These degrade to None (the caller's
# "no data" path). Anything outside this set is a bug in our own code and is
# re-raised through a logger.exception so it is not mistaken for a data outage.
_VENDOR_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError)


def _equity_ticker(symbol: str, exchange: str) -> str:
    """RELIANCE + NSE → RELIANCE.NS"""
    suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
    return f"{symbol.upper()}{suffix}"


# Yahoo's own exchange codes -> the four we actually support. Search returns
# results from every exchange Yahoo lists (Frankfurt, Vienna, Warsaw, ETFs on
# any of them); anything not in this map is dropped rather than shown as a
# suggestion the app cannot correctly chart.
_YAHOO_EXCHANGE = {
    "NMS": "NASDAQ",  # NASDAQ Global Select
    "NGM": "NASDAQ",  # NASDAQ Global Market
    "NCM": "NASDAQ",  # NASDAQ Capital Market
    "NYQ": "NYSE",
    "ASE": "NYSE",    # NYSE American, close enough for viewing
    "NSI": "NSE",
    "BSE": "BSE",
}


def _forex_ticker(pair: str) -> str:
    """EURUSD → EURUSD=X  (yFinance format)"""
    pair = pair.upper().replace("/", "").replace("-", "")
    if not pair.endswith("=X"):
        pair = f"{pair}=X"
    return pair


class YFinanceProvider:
    def _ticker_for(self, symbol: str, exchange: str) -> str:
        if exchange.upper() == "FOREX":
            return _forex_ticker(symbol)
        # Yahoo's index (^NSEI, ^BSESN, ^INDIAVIX), futures (BZ=F) and FX
        # (USDINR=X) symbols are already fully qualified — they carry no
        # exchange suffix and adding one 404s. `/market/status` asked for
        # "^NSEI" on exchange "NSE", got "^NSEI.NS", and had been returning a
        # null Nifty and Sensex ever since.
        upper = symbol.upper()
        if upper.startswith("^") or upper.endswith(("=X", "=F")):
            return upper
        # US and other non-Indian exchanges use the bare ticker, no suffix
        if exchange.upper() not in ("NSE", "BSE"):
            return upper
        return _equity_ticker(symbol, exchange)

    def _get_quote_sync(self, symbol: str, exchange: str = "NSE") -> dict | None:
        try:
            ticker_str = self._ticker_for(symbol, exchange)
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
        except _VENDOR_ERRORS as e:
            logger.warning("YFinance quote fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in YFinance quote for %s/%s", symbol, exchange)
            return None

    def _get_historical_df_sync(
        self, symbol: str, exchange: str = "NSE", interval: str = "15m", days: int = 30
    ) -> pd.DataFrame | None:
        # yfinance caps how far back each intraday interval can go, and rejects a
        # range that touches the exact boundary — so keep a safety margin under the
        # documented max. Respect the caller's window (days) but clamp to this.
        # The table lives in app/market/intervals.py; it used to be inline here
        # and in two other files, with the three copies disagreeing.
        try:
            ticker_str = self._ticker_for(symbol, exchange)
            span = clamp_days(interval, days)
            end = datetime.now()
            start = end - timedelta(days=span)

            df: pd.DataFrame = yf.download(
                ticker_str,
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return None

            # yfinance may return MultiIndex columns like ('Close', 'RELIANCE.NS')
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            return df.dropna()
        except _VENDOR_ERRORS as e:
            logger.warning("YFinance historical fetch failed for %s/%s: %s", symbol, exchange, e)
            return None
        except Exception:
            logger.exception("Unexpected error in YFinance history for %s/%s", symbol, exchange)
            return None

    # ------------------------------------------------------------------
    # Async wrappers.
    #
    # yfinance is fully synchronous. Calling it directly inside `async def`
    # blocks the event loop, so with `--workers 2` the service can serve only
    # ~2 concurrent requests and a slow fetch stalls even /health (which can
    # trip the Docker healthcheck and restart a working container).
    # Offload to the default thread pool instead.
    # ------------------------------------------------------------------
    async def get_quote(self, symbol: str, exchange: str = "NSE") -> dict | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_quote_sync, symbol, exchange)

    async def search(self, query: str, limit: int = 8) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query, limit)

    def _search_sync(self, query: str, limit: int) -> list[dict]:
        """Company name / symbol -> the matches we can actually chart.

        Yahoo's search returns results across every exchange it lists —
        Frankfurt, Vienna, Warsaw, ETFs, depositary receipts. Only NSE, BSE,
        NASDAQ and NYSE map cleanly onto `_ticker_for`; anything else would
        either 404 when selected or (worse) silently chart the wrong
        instrument, so it is filtered out here rather than left for the
        frontend to guess about.
        """
        try:
            results = yf.Search(query, max_results=limit * 2).quotes
        except _VENDOR_ERRORS as e:
            logger.warning("Symbol search failed for %r: %s", query, e)
            return []
        except Exception:
            logger.exception("Unexpected error searching for %r", query)
            return []

        out: list[dict] = []
        seen: set[str] = set()
        for r in results:
            if r.get("quoteType") != "EQUITY":
                continue
            exchange = _YAHOO_EXCHANGE.get(r.get("exchange", ""))
            if not exchange:
                continue
            symbol = str(r.get("symbol", "")).split(".")[0].upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append({
                "symbol": symbol,
                "name": r.get("shortname") or r.get("longname") or symbol,
                "exchange": exchange,
            })
            if len(out) >= limit:
                break
        return out

    async def get_historical_df(
        self, symbol: str, exchange: str = "NSE", interval: str = "15m", days: int = 30
    ) -> pd.DataFrame | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._get_historical_df_sync, symbol, exchange, interval, days)
        )
