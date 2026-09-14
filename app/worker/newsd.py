"""The news engine as its own process.

One job, one schedule, one failure domain. The hourly news pipeline is the
memory-hungry part of the system -- an LLM pass over ~25 articles -- and the
terminal has to keep serving charts when it dies or hits its cap. Replaces the
Celery worker and beat pair, which cost ~283 MB resident between them to fire
one cron entry an hour.

Run it with `python -m app.worker.newsd`.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.events import watcher as event_watcher
from app.worker.scheduler import IST, JOB_DEFAULTS, redis_jobstore

logger = logging.getLogger(__name__)

# Hourly on the hour: NewsAPI's free tier caps at 100 requests a day, which
# 15-minute runs would spend by mid-afternoon.
SCHEDULE: dict[str, tuple[str, dict]] = {
    "news-analysis": ("app.worker.news_job:run_news_analysis", {"minute": 0}),
    # The day's agenda, before the London session opens for him (08:00 Dubai
    # is 04:00 UTC). Free to produce -- calendar plus each release's own
    # published spec, no model in the path.
    "event-agenda": ("app.events.job:run_agenda", {"minute": 0, "hour": 4}),
}

# Its own keys -- see redis_jobstore's docstring.
JOBS_KEY = "newsd.jobs"
RUN_TIMES_KEY = "newsd.run_times"


def build_scheduler(jobstore=None) -> AsyncIOScheduler:
    """The news scheduler with its job registered. Not started."""
    scheduler = AsyncIOScheduler(timezone=IST, job_defaults=JOB_DEFAULTS)
    if jobstore is not None:
        scheduler.add_jobstore(jobstore)
    for job_id, (func, cron) in SCHEDULE.items():
        scheduler.add_job(
            func, trigger=CronTrigger(timezone=IST, **cron),
            id=job_id, name=job_id, replace_existing=True,
        )
    return scheduler


async def serve() -> None:
    scheduler = build_scheduler(redis_jobstore(JOBS_KEY, RUN_TIMES_KEY))
    # The event watcher books its own one-off wake-ups on this scheduler
    # (one per high-impact release, two hours ahead), so it needs the
    # instance rather than a cron entry of its own.
    event_watcher.bind(scheduler)
    scheduler.start()
    logger.info("newsd started with %d job(s): %s", len(SCHEDULE), ", ".join(SCHEDULE))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:      # Windows, i.e. a developer's machine
            pass
    await stop.wait()

    # wait=True: a news run in flight gets to finish and publish within the
    # stop grace period rather than being dropped halfway.
    logger.info("newsd stopping")
    scheduler.shutdown(wait=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(serve())


if __name__ == "__main__":
    main()
