"""Who watches the desk.

A brief that never fires is invisible: the calendar still lists the event, the
agenda still went out that morning, and nothing anywhere says the Fed brief he
was expecting at 20:00 did not happen. Silent failure on a time-critical path
is worse than no product, because he stops checking for himself.

So arming and firing are both recorded, and a daily sweep compares them. Two
layers of alarm, deliberately:

  * The sweep reports a missed brief -- the desk ran but one job did not.
  * The sweep is itself a Healthchecks job, so if newsd is dead, the missed
    ping raises the alarm nobody inside the box could have raised.

Records live in Redis with a short TTL. This is operational state about the
last few days, not history: losing it costs one sweep, not the ledger.
"""
from __future__ import annotations

import logging
import time

import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)

ARMED_PREFIX = "eventdesk:armed:"
FIRED_PREFIX = "eventdesk:fired:"
# Long enough for a weekend plus the sweep that follows it.
RECORD_TTL_SECONDS = 4 * 24 * 3600
# A brief is due at T-2h; give the job room to run before calling it missed.
GRACE_SECONDS = 15 * 60


def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


async def record_armed(job_id: str, due_epoch: float) -> None:
    """Remember that a brief was promised, so its absence is detectable."""
    try:
        client = _client()
        try:
            await client.set(ARMED_PREFIX + job_id, due_epoch, ex=RECORD_TTL_SECONDS)
        finally:
            await client.aclose()
    except Exception:
        # Bookkeeping must never take down the thing it is bookkeeping for.
        logger.exception("Could not record armed brief %s", job_id)


async def record_fired(job_id: str, outcome: str) -> None:
    try:
        client = _client()
        try:
            await client.set(FIRED_PREFIX + job_id, outcome, ex=RECORD_TTL_SECONDS)
        finally:
            await client.aclose()
    except Exception:
        logger.exception("Could not record fired brief %s", job_id)


async def missed(now: float | None = None) -> list[str]:
    """Briefs that were due and never reported firing."""
    now = now or time.time()
    client = _client()
    try:
        armed = [k async for k in client.scan_iter(match=ARMED_PREFIX + "*", count=200)]
        out = []
        for key in armed:
            due = await client.get(key)
            try:
                due_epoch = float(due)
            except (TypeError, ValueError):
                continue
            if now < due_epoch + GRACE_SECONDS:
                continue                      # not late yet
            job_id = key[len(ARMED_PREFIX):]
            if not await client.get(FIRED_PREFIX + job_id):
                out.append(job_id)
        return out
    finally:
        await client.aclose()


async def sweep() -> dict:
    """The daily check. `ok` is False when a brief was promised and not kept,
    which fails this job's own Healthchecks ping as well as telling him."""
    from app.events import telegram

    try:
        overdue = await missed()
    except Exception as e:
        logger.exception("Watchdog sweep failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if not overdue:
        logger.info("Watchdog: every armed brief fired")
        return {"ok": True, "missed": 0}

    logger.error("Watchdog: %d briefs did not fire: %s", len(overdue), overdue)
    await telegram.send(
        "*Desk warning*\n\n"
        f"{len(overdue)} brief(s) were scheduled and did not send:\n"
        + "\n".join(f"- `{job}`" for job in overdue[:5])
        + "\n\n_you were not told about those events. this is the desk "
          "reporting its own failure._"
    )
    return {"ok": False, "missed": len(overdue), "jobs": overdue[:10]}
