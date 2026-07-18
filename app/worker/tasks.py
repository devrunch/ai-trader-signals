from app.worker.celery_app import celery
from app.signals.service import SignalService
import asyncio
import logging

logger = logging.getLogger(__name__)

WATCHLIST = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "BHARTIARTL", "SBIN", "WIPRO", "LT", "AXISBANK",
    "KOTAKBANK", "HINDUNILVR", "ITC", "BAJFINANCE", "MARUTI",
]

_service = SignalService()


@celery.task(name="app.worker.tasks.run_screener")
def run_screener():
    """Screener task — runs every 15 min during NSE market hours via Celery beat."""
    logger.info("Screener started for %d symbols", len(WATCHLIST))
    results = []
    for symbol in WATCHLIST:
        try:
            signal = asyncio.run(
                _service.generate_signal(symbol, "NSE")
            )
            if signal:
                results.append(symbol)
                logger.info("Signal generated: %s", symbol)
        except Exception as e:
            logger.error("Screener failed for %s: %s", symbol, e)
    logger.info("Screener complete — %d signals published", len(results))
    return {"screened": len(WATCHLIST), "signals": len(results), "symbols": results}


@celery.task(name="app.worker.tasks.generate_single")
def generate_single(symbol: str, exchange: str = "NSE"):
    """On-demand signal generation for a single symbol — triggered by NestJS API."""
    try:
        signal = asyncio.run(
            _service.generate_signal(symbol.upper(), exchange.upper())
        )
        if signal is None:
            return {"symbol": symbol, "signal": None}
        return {"symbol": symbol, "signal": signal.__dict__}
    except Exception as e:
        logger.error("generate_single failed for %s: %s", symbol, e)
        return {"symbol": symbol, "error": str(e)}
