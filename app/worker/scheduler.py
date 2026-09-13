"""In-process scheduler for the recurring jobs.

Replaces Celery beat, which was a whole second process (and a broker round
trip) to fire six cron entries a day. It also schedules to the minute only,
and the event watcher this is groundwork for has to wake seconds before a
release lands.

The jobs themselves live in app/worker/jobs.py as plain functions, because the
Redis job store keeps each job as an import path and resolves it on load.
"""
from __future__ import annotations

import logging

from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings

logger = logging.getLogger(__name__)

IST = "Asia/Kolkata"

# id -> (job import path, cron fields). Same cadences Celery beat ran.
SCHEDULE: dict[str, tuple[str, dict]] = {
    # Hourly, all day: drift_check.check() is cheap on a quiet hour (one real
    # fetch, no LLM call, no alert).
    "drift-check": ("app.worker.jobs:run_drift_check", {"minute": 30}),
    # Odd hours only, to ration Tavily's free credits across callers.
    "reddit-sentiment": ("app.worker.jobs:run_reddit_sentiment",
                         {"minute": 45, "hour": "1,3,5,7,9,11,13,15,17,19,21,23"}),
    # After the US close, before the Indian open.
    "market-overview-morning": ("app.worker.jobs:generate_morning_brief",
                                {"minute": 30, "hour": 6, "day_of_week": "mon-fri"}),
    # And again before the US open, which the 06:30 run is too early for.
    "market-overview-evening": ("app.worker.jobs:generate_morning_brief",
                                {"minute": 0, "hour": 18, "day_of_week": "mon-fri"}),
    # Kite's access token expires around 06:00 IST.
    "kite-session-refresh": ("app.worker.jobs:refresh_zerodha_session", {"minute": 0, "hour": 6}),
    # Intraday means intraday: nothing is held overnight.
    "paper-square-off": ("app.worker.jobs:square_off_positions",
                         {"minute": 20, "hour": 15, "day_of_week": "mon-fri"}),
}

# A deploy restarts this process, and a job due during that restart must still
# run rather than being skipped -- the 15:20 square-off above all. Coalesced,
# so a long outage produces one catch-up run, not a burst of them.
MISFIRE_GRACE_SECONDS = 900

JOB_DEFAULTS = {
    "coalesce": True,
    # The hourly work can outrun its own interval; a second copy would
    # double-spend on the LLM and publish an alert twice.
    "max_instances": 1,
    "misfire_grace_time": MISFIRE_GRACE_SECONDS,
}


def build_scheduler(jobstore: RedisJobStore | None = None) -> AsyncIOScheduler:
    """The scheduler with every job registered. Not started."""
    scheduler = AsyncIOScheduler(timezone=IST, job_defaults=JOB_DEFAULTS)
    if jobstore is not None:
        scheduler.add_jobstore(jobstore)
    for job_id, (func, cron) in SCHEDULE.items():
        scheduler.add_job(
            func, trigger=CronTrigger(timezone=IST, **cron),
            id=job_id, name=job_id, replace_existing=True,
        )
    return scheduler


def redis_jobstore(jobs_key: str, run_times_key: str) -> RedisJobStore:
    """Persisted store, not in-memory: an in-memory store forgets that a job
    was due while the process was down, so misfire_grace_time could never
    apply. Each process passes its own keys -- newsd runs a second scheduler
    against this same Redis, and a shared key would have each of them
    executing the other's jobs."""
    store = RedisJobStore.__new__(RedisJobStore)
    RedisJobStore.__init__(store, jobs_key=jobs_key, run_times_key=run_times_key,
                           **redis_kwargs(get_settings().redis_url))
    return store


def start() -> AsyncIOScheduler:
    """Build, wire to Redis and start. Returns the running scheduler."""
    scheduler = build_scheduler(jobstore=redis_jobstore("apscheduler.jobs", "apscheduler.run_times"))
    scheduler.start()
    logger.info("Scheduler started with %d jobs: %s", len(SCHEDULE), ", ".join(SCHEDULE))
    return scheduler


def redis_kwargs(url: str) -> dict:
    """RedisJobStore takes connection kwargs, not a URL."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    kwargs: dict = {"host": parsed.hostname or "localhost", "port": parsed.port or 6379}
    db = (parsed.path or "").strip("/")
    if db:
        kwargs["db"] = int(db)
    if parsed.password:
        kwargs["password"] = parsed.password
    return kwargs
