"""FastAPI entrypoint for the signals service.

Two health endpoints, deliberately different:

  * ``/health``  — liveness. Static, does no work, cannot fail for a reason
                   outside this process. The container restart policy watches
                   it, so making it touch a dependency means a slow vendor
                   restarts a service that is working fine.
  * ``/ready``   — readiness. Probes the things this service needs to be
                   *useful* (market data, SQS) and is what other services gate
                   their startup on. Cached, so it cannot become a load source.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Response

from app.config import get_settings
from app.market import router as market_router_module
from app.market.kite_ticker import KiteTickerClient
from app.market.live_ticks import LiveTicks
from app.market.providers.registry import market_data_router
from app.market.router import router as market_router
from app.market.service import get_quote
from app.signals.router import router as signals_router

logger = logging.getLogger(__name__)

# Every blocking vendor call (yfinance, boto3) is offloaded with
# `run_in_executor(None, ...)` / `asyncio.to_thread`, both of which use the
# loop's *default* executor. Unbounded, that is min(32, cpu_count + 4) threads
# per worker and grows with load; on the 0.5 vCPU Fargate task that is pure
# context-switching. Bound it explicitly and run a single uvicorn worker
# (docker/signals/Dockerfile) rather than two workers fighting for half a core.
EXECUTOR_MAX_WORKERS = 16

# Readiness is cached so a 15s Docker healthcheck plus an ALB plus a dependent
# service cannot together turn the probe into traffic of its own.
READY_CACHE_SECONDS = 5.0
# The probes themselves must never be the thing that hangs the check.
PROBE_TIMEOUT_SECONDS = 5.0

# docker-compose has `api` wait on `signals` being healthy, not the reverse —
# so when this service's lifespan hook fires, `api` may not be listening yet.
# 6 tries, 5s apart covers a normal `docker compose up` without stalling
# startup indefinitely.
_STARTUP_FETCH_ATTEMPTS = 6
_STARTUP_FETCH_RETRY_SECONDS = 5.0

_ready_cache: dict | None = None
_ready_cached_at: float = 0.0
_ready_lock = asyncio.Lock()


async def _get_with_retry(url: str, headers: dict[str, str], *, what: str) -> httpx.Response | None:
    """One-shot startup fetch, retried up to _STARTUP_FETCH_ATTEMPTS times.
    Returns None (logging an error) once attempts are exhausted, rather than
    raising — a slow-to-start `api` container must not abort this service's
    own startup."""
    for attempt in range(1, _STARTUP_FETCH_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp
        except httpx.HTTPError as e:
            if attempt < _STARTUP_FETCH_ATTEMPTS:
                logger.warning(
                    "%s fetch failed (attempt %d/%d): %s", what, attempt, _STARTUP_FETCH_ATTEMPTS, e
                )
                await asyncio.sleep(_STARTUP_FETCH_RETRY_SECONDS)
            else:
                logger.error("%s fetch failed after %d attempts: %s", what, _STARTUP_FETCH_ATTEMPTS, e)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Captured now, on the main thread, before KiteTickerClient.connect() can
    # start the Kite SDK's own reactor thread — on_tick below runs on that
    # thread and needs this loop, not whatever thread it happens to fire on.
    loop = asyncio.get_running_loop()

    executor = ThreadPoolExecutor(
        max_workers=EXECUTOR_MAX_WORKERS, thread_name_prefix="signals-io"
    )
    loop.set_default_executor(executor)
    logger.info("Default executor bounded to %d threads", EXECUTOR_MAX_WORKERS)

    settings = get_settings()
    redis_client = redis.from_url(settings.redis_url)

    # Constructed unconditionally — the yfinance poll path (NASDAQ/NYSE/...)
    # needs no Kite token at all, so it must not be gated behind one.
    live_ticks = LiveTicks(None, redis_client, get_quote)
    market_router_module.live_ticks = live_ticks

    kite_ticker: KiteTickerClient | None = None

    async def _attach_kite_ticker() -> bool:
        """Best-effort — retried in the background so a slow/unreachable
        NestJS never delays this process binding its port. Returns whether a
        ticker got attached, so resubscribe knows whether there's a Kite leg
        to resubscribe at all."""
        nonlocal kite_ticker
        if not settings.zerodha_api_key:
            logger.warning("No Zerodha access token configured — Kite (NSE/BSE) live ticks disabled, poll path unaffected")
            return False

        # Zerodha's access token is minted daily in NestJS (see KiteProvider's
        # own _ensure_token), not a static setting — fetch the current one the
        # same way.
        resp = await _get_with_retry(
            f"{settings.api_service_url}/api/internal/broker/zerodha/session",
            {"x-internal-key": settings.internal_api_key},
            what="Zerodha session",
        )
        access_token = resp.json().get("accessToken") if resp is not None else None
        if not access_token:
            logger.warning("No Zerodha access token configured — Kite (NSE/BSE) live ticks disabled, poll path unaffected")
            return False

        # Reuse market_data_router's own KiteProvider rather than standing up
        # a second one — a fresh instance would duplicate the instrument-list
        # download and never share its token cache with the router's.
        kite_provider = market_data_router.providers.get("NSE")

        def on_tick(payload: dict) -> None:
            asyncio.run_coroutine_threadsafe(live_ticks.publish(payload), loop)

        kite_ticker = KiteTickerClient(
            api_key=settings.zerodha_api_key,
            access_token=access_token,
            kite_provider=kite_provider,
            on_tick=on_tick,
        )
        kite_ticker.connect()
        live_ticks.set_kite_ticker(kite_ticker)
        return True

    async def _resubscribe_active_symbols(attach_task: asyncio.Task[bool]) -> None:
        # Waits on the attach task rather than running independently — Kite
        # must be attached to live_ticks first, or an NSE/BSE resubscribe
        # hits LiveTicks.subscribe's fail-closed `self._kite is None` path
        # and silently drops that symbol.
        if not await attach_task:
            return
        resp = await _get_with_retry(
            f"{settings.api_service_url}/api/internal/market/active-symbols",
            {"x-internal-key": settings.internal_api_key},
            what="Active symbols",
        )
        if resp is not None:
            active = [(row["symbol"], row["exchange"]) for row in resp.json()]
            await live_ticks.resubscribe_from(active)

    # Both run as background tasks, not awaited here — the NestJS retry loops
    # below can take up to ~85s each and must not delay this process binding
    # its port / answering /health /ready (docker-compose's start_period for
    # this service is far shorter than that worst case).
    kite_attach_task = asyncio.create_task(_attach_kite_ticker())
    resubscribe_task = asyncio.create_task(_resubscribe_active_symbols(kite_attach_task))

    try:
        yield
    finally:
        kite_attach_task.cancel()
        resubscribe_task.cancel()
        for task in (kite_attach_task, resubscribe_task):
            try:
                await task
            except asyncio.CancelledError:
                pass
        await live_ticks.close()
        if kite_ticker:
            kite_ticker.close()
        await redis_client.aclose()
        executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="AI Trader Signals Service", version="0.1.0", lifespan=lifespan)

app.include_router(signals_router, prefix="/signals", tags=["signals"])
app.include_router(market_router, prefix="/market", tags=["market"])


@app.get("/health")
async def health():
    """Liveness. Static by design — see the module docstring."""
    return {"status": "ok", "service": "signals"}


async def _probe_market() -> dict:
    """Can we still get a price?

    Goes through the router, so it reads the router's 45s quote cache and the
    real network cost is paid at most once per cache TTL however often this is
    polled.
    """
    try:
        quote = await asyncio.wait_for(
            market_data_router.get_quote("^NSEI", "NSE"), timeout=PROBE_TIMEOUT_SECONDS
        )
    except TimeoutError:
        return {"ok": False, "detail": "timeout"}
    except Exception as e:  # a provider bug must not 500 the readiness probe
        logger.exception("Readiness market probe raised")
        return {"ok": False, "detail": f"{type(e).__name__}"}
    if quote is None:
        return {"ok": False, "detail": "no quote"}
    return {"ok": True}


def _probe_sqs_sync(queue_url: str, region: str) -> dict:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    client = boto3.client(
        "sqs",
        region_name=region,
        config=Config(
            connect_timeout=2,
            read_timeout=3,
            retries={"max_attempts": 1},
        ),
    )
    try:
        client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["QueueArn"])
        return {"ok": True}
    except (BotoCoreError, ClientError) as e:
        return {"ok": False, "detail": f"{type(e).__name__}"}


async def _probe_sqs() -> dict:
    settings = get_settings()
    if not settings.sqs_signals_queue_url:
        # Not configured is not unready — the compose dev stack runs without a
        # queue. It is reported so it cannot be mistaken for a passing check.
        return {"ok": True, "detail": "not configured"}
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _probe_sqs_sync, settings.sqs_signals_queue_url, settings.aws_region
            ),
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return {"ok": False, "detail": "timeout"}
    except Exception as e:
        logger.exception("Readiness SQS probe raised")
        return {"ok": False, "detail": f"{type(e).__name__}"}


async def _readiness() -> dict:
    global _ready_cache, _ready_cached_at

    now = time.monotonic()
    if _ready_cache is not None and (now - _ready_cached_at) < READY_CACHE_SECONDS:
        return _ready_cache

    async with _ready_lock:
        # Re-check: a concurrent caller may have refreshed while we queued.
        now = time.monotonic()
        if _ready_cache is not None and (now - _ready_cached_at) < READY_CACHE_SECONDS:
            return _ready_cache

        market, sqs = await asyncio.gather(_probe_market(), _probe_sqs())
        checks = {"market_data": market, "sqs": sqs}
        result = {
            "status": "ready" if all(c["ok"] for c in checks.values()) else "not ready",
            "service": "signals",
            "checks": checks,
        }
        _ready_cache = result
        _ready_cached_at = time.monotonic()
        return result


@app.get("/ready")
async def ready(response: Response):
    """Readiness. What dependents should gate startup on, not /health."""
    result = await _readiness()
    if result["status"] != "ready":
        response.status_code = 503
    return result
