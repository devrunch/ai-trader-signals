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

from fastapi import FastAPI, Response

from app.config import get_settings
from app.market.providers.registry import market_data_router
from app.market.router import router as market_router
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

_ready_cache: dict | None = None
_ready_cached_at: float = 0.0
_ready_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    executor = ThreadPoolExecutor(
        max_workers=EXECUTOR_MAX_WORKERS, thread_name_prefix="signals-io"
    )
    asyncio.get_running_loop().set_default_executor(executor)
    logger.info("Default executor bounded to %d threads", EXECUTOR_MAX_WORKERS)
    try:
        yield
    finally:
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
