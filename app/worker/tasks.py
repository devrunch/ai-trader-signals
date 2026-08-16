from __future__ import annotations

import asyncio
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.config import get_settings
from app.market import calendar as market_calendar
from app.market.providers import kite_auth
from app.signals.service import SignalService
from app.worker.celery_app import celery

logger = logging.getLogger(__name__)


def run_async(coro):
    """`asyncio.run` with a BOUNDED default executor.

    Every blocking call in the data layer (yfinance, boto3) is offloaded with
    `to_thread` / `run_in_executor(None, ...)`, which uses the loop's default
    executor. A fresh `asyncio.run()` loop gets the interpreter default —
    `min(32, cpu_count + 4)` threads, unbounded from the caller's point of
    view. The API process bounds this in its lifespan hook in `main.py`, but a
    Celery worker never runs `main.py`, so it has to be done here too or the
    screener can spawn a thread per symbol against a 0.5 vCPU task.
    """
    async def main():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=16, thread_name_prefix="worker-io") as pool:
            loop.set_default_executor(pool)
            return await coro

    return asyncio.run(main())

# Used only if the watchlist is empty or NestJS is briefly unreachable —
# not the source of truth, that's the dynamic watchlist in MongoDB.
FALLBACK_WATCHLIST = [
    {"symbol": "RELIANCE", "exchange": "NSE"},
    {"symbol": "TCS", "exchange": "NSE"},
    {"symbol": "INFY", "exchange": "NSE"},
]

_service: SignalService | None = None


def service() -> SignalService:
    """Built on first task execution, not at import.

    A module-level `SignalService()` ran at import, which meant `celery -A
    app.worker.celery_app` constructed an LLM client and an SQS client before
    the worker had accepted a single task.
    """
    global _service
    if _service is None:
        _service = SignalService()
    return _service


def _fetch_watchlist() -> list[dict]:
    """Union of every user's watchlist, deduplicated by symbol+exchange."""
    settings = get_settings()
    try:
        resp = httpx.get(
            f"{settings.api_service_url}/api/internal/watchlist",
            headers={"x-internal-key": settings.internal_api_key},
            timeout=10,
        )
        resp.raise_for_status()
        entries = resp.json()
        return entries if entries else FALLBACK_WATCHLIST
    except (httpx.HTTPError, ValueError) as e:
        logger.error("Failed to fetch watchlist from NestJS, using fallback: %s", e)
        return FALLBACK_WATCHLIST


async def _screen(watchlist: list[dict]) -> tuple[list[str], Counter]:
    """Screen the watchlist with bounded concurrency.

    Was a sequential loop with a fresh `asyncio.run()` per symbol — one event
    loop constructed and torn down per symbol, and every network wait serialised.
    Bounded rather than unbounded because yfinance rate-limits aggressively.
    """
    settings = get_settings()
    sem = asyncio.Semaphore(settings.screener_concurrency)
    svc = service()

    async def one(entry: dict):
        symbol, exchange = entry["symbol"], entry.get("exchange", "NSE")
        async with sem:
            try:
                return symbol, await svc.generate(symbol, exchange)
            except Exception:
                logger.exception("Screener failed for %s", symbol)
                return symbol, None

    published: list[str] = []
    # Reasons are counted, not discarded. "0 signals" used to be
    # indistinguishable between "the market was quiet" and "Bedrock is down".
    reasons: Counter = Counter()

    for symbol, result in await asyncio.gather(*(one(e) for e in watchlist)):
        if result is None:
            reasons["task_error"] += 1
        elif result.signal is not None:
            published.append(symbol)
            logger.info("Signal generated: %s", symbol)
        else:
            reasons[result.reason or "unknown"] += 1
    return published, reasons


@celery.task(name="app.worker.tasks.run_screener")
def run_screener():
    """Screener task — runs every 15 min during NSE market hours via Celery beat."""
    # The calendar is the gate, not the beat schedule. Beat narrows the window
    # to 09:20–15:15 IST on weekdays, but it knows nothing about the ~dozen NSE
    # trading holidays a year. A holiday run produces real, stored, scored
    # signals computed on the previous session's prices, which cannot move.
    if not market_calendar.is_market_open():
        state = market_calendar.session_state()
        logger.info("Screener skipped — NSE closed (holiday=%s, trading_day=%s); next open %s",
                    state["is_holiday"], state["is_trading_day"], state["next_open"])
        return {
            "screened": 0,
            "signals": 0,
            "symbols": [],
            "skipped_reasons": {"market_closed": 1},
            "market_closed": True,
        }

    watchlist = _fetch_watchlist()
    logger.info("Screener started for %d symbols", len(watchlist))
    published, reasons = run_async(_screen(watchlist))
    logger.info("Screener complete — %d signals published; skipped: %s",
                len(published), dict(reasons))
    return {
        "screened": len(watchlist),
        "signals": len(published),
        "symbols": published,
        "skipped_reasons": dict(reasons),
    }


@celery.task(name="app.worker.tasks.generate_single")
def generate_single(symbol: str, exchange: str = "NSE"):
    """On-demand signal generation for a single symbol — triggered by NestJS API."""
    try:
        result = run_async(service().generate(symbol.upper(), exchange.upper()))
        if result.signal is None:
            return {"symbol": symbol, "signal": None, "reason": result.reason}
        return {"symbol": symbol, "signal": result.signal.__dict__, "reason": None}
    except Exception as e:
        logger.exception("generate_single failed for %s", symbol)
        return {"symbol": symbol, "error": str(e)}


@celery.task(name="app.worker.tasks.square_off_positions")
def square_off_positions():
    """Close every open paper position at market — 15:20 IST, via Celery beat.

    The product's central claim is that positions are intraday only. Nothing
    enforced it: the paper account held positions indefinitely, so a "no
    overnight risk" strategy was silently carrying every gap.

    The positions live in MongoDB behind NestJS, so this task is a thin trigger
    for the internal square-off endpoint rather than doing the closing itself —
    order execution, cash movement and the WebSocket push all belong to the one
    service that owns them.
    """
    settings = get_settings()
    if not market_calendar.is_trading_day():
        logger.info("Square-off skipped — not an NSE trading day")
        return {"skipped": "not_a_trading_day"}

    try:
        resp = httpx.post(
            f"{settings.api_service_url}/api/internal/paper/square-off",
            headers={"x-internal-key": settings.internal_api_key},
            # Generous: one live price fetch and one order per open position,
            # across every user. Better to wait than to leave positions open
            # overnight because a timeout fired at 15:20:30.
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        # Loud, because the failure mode is silently holding overnight.
        logger.error("Square-off failed — positions may still be open: %s", e)
        return {"error": str(e)}

    logger.info("Square-off complete — %d closed, %d failed: %s",
                data.get("closed", 0), data.get("failed", 0), data.get("details"))
    return data


@celery.task(name="app.worker.tasks.generate_morning_brief")
def generate_morning_brief():
    """Pre-market brief — runs ~06:30 IST, after the US close and before the
    Indian open, so the day's plan is finished before anyone opens the app."""
    from app.signals import brief as brief_mod

    logger.info("Morning brief generation started")

    async def run() -> tuple[dict, bool]:
        # One event loop, not two — `publish` used to run in a second
        # `asyncio.run()`, discarding any connection pooling from the first.
        doc = await brief_mod.generate()
        return doc, await brief_mod.publish(doc)

    try:
        doc, ok = run_async(run())
        logger.info(
            "Morning brief done: bias=%s, %d candidates, published=%s",
            doc["market_read"]["bias"], len(doc["candidates"]), ok,
        )
        return {"date": doc["date"], "candidates": len(doc["candidates"]), "published": ok}
    except Exception:
        logger.exception("Morning brief generation failed")
        return {"error": True}


@celery.task(name="app.worker.tasks.refresh_zerodha_session")
def refresh_zerodha_session():
    """Daily Kite Connect login — 06:00 IST via Celery beat.

    A thin trigger: the actual login lives in kite_auth (proven live before
    this was written), this task just runs it and pushes the result to
    NestJS, the same shape as square_off_positions's call to the internal
    paper-trading endpoint above. On failure, logged loudly and returned —
    not re-raised — because a stale token in NestJS degrades gracefully (the
    router's Kite-then-yfinance fallback picks up every call that fails
    against it) rather than needing this task itself to retry aggressively.
    """
    settings = get_settings()
    try:
        session = kite_auth.refresh_session(settings)
    except kite_auth.KiteAuthError as e:
        logger.error("Zerodha session refresh failed: %s", e)
        return {"ok": False, "error": str(e)}

    try:
        resp = httpx.put(
            f"{settings.api_service_url}/api/internal/broker/zerodha/session",
            headers={"x-internal-key": settings.internal_api_key},
            json={"accessToken": session.access_token},
            timeout=10,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("Zerodha session refreshed but NestJS write failed: %s", e)
        return {"ok": False, "error": str(e)}

    logger.info("Zerodha session refreshed and stored (%d NSE, %d BSE instruments)",
                len(session.nse_instruments), len(session.bse_instruments))
    return {"ok": True}
