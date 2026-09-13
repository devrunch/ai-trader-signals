"""Healthchecks.io pings for scheduled jobs.

A missing ping tells us a job stopped running; a `/fail` ping reports a run
that happened but didn't do its job. Each check's ping URL is looked up once
per process with the project API key, keyed by slug. Pinging never raises:
monitoring must not break the job it watches.
"""
from __future__ import annotations

import functools
import json
import logging
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

_CHECKS_API = "https://healthchecks.io/api/v3/checks/"
_TIMEOUT_SECONDS = 10
_IST = ZoneInfo("Asia/Kolkata")

_urls: dict[str, str] | None = None
_lock = threading.Lock()


def _ping_urls() -> dict[str, str]:
    """Slug -> ping URL. A failed lookup isn't cached, so the next run retries."""
    global _urls
    key = get_settings().healthchecks_api_key
    if not key:
        return {}
    with _lock:
        if _urls is None:
            try:
                resp = httpx.get(_CHECKS_API, headers={"X-Api-Key": key}, timeout=_TIMEOUT_SECONDS)
                resp.raise_for_status()
                _urls = {c["slug"]: c["ping_url"] for c in resp.json()["checks"] if c.get("slug")}
            except Exception as e:
                logger.warning("Healthchecks lookup failed: %s", e)
                return {}
        return _urls


def ping(slug: str, ok: bool, body: str = "") -> None:
    url = _ping_urls().get(slug)
    if url is None:
        return
    try:
        httpx.post(url if ok else f"{url}/fail", content=body.encode()[:10_000], timeout=_TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("Healthchecks ping for %s failed: %s", slug, e)


def _did_its_job(result: object) -> bool:
    if not isinstance(result, dict):
        return True
    return not result.get("error") and result.get("ok") is not False and result.get("published") is not False


def monitored(slug):
    """Ping the check named `slug` (or returned by `slug()`) after each run."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = slug() if callable(slug) else slug
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                ping(name, ok=False, body=f"raised {type(e).__name__}: {e}")
                raise
            ping(name, ok=_did_its_job(result), body=json.dumps(result, default=str))
            return result
        return wrapper
    return decorate


def market_overview_slug(now: datetime | None = None) -> str:
    """One task runs both overviews (06:30 and 18:00 IST); each has its own check."""
    now = now or datetime.now(_IST)
    return "market-overview-morning" if now.astimezone(_IST).hour < 12 else "market-overview-evening"
