from celery import Celery
from celery.schedules import crontab
from app.config import settings

celery = Celery("signals_worker", broker=settings.redis_url, backend=settings.redis_url)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    beat_schedule={
        "run-screener": {
            "task": "app.worker.tasks.run_screener",
            "schedule": crontab(minute="*/15", hour="9-15", day_of_week="mon-fri"),
        },
    },
)
