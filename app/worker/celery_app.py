from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery = Celery(
    "signals_worker",
    broker=settings.redis_url,
    include=["app.worker.tasks"],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    result_backend=None,          # no result backend needed
    broker_connection_retry_on_startup=True,
    timezone="Asia/Kolkata",
    enable_utc=True,
    beat_schedule={
        # Everything else moved to the in-process scheduler (see
        # app/worker/scheduler.py). News stays here only until it moves to
        # its own process, where it can be memory-capped away from the
        # terminal; see docs/architecture.
        "news-analysis": {
            "task": "app.worker.tasks.run_news_analysis",
            "schedule": crontab(minute="0"),
        },    },
)
