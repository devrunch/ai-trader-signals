from celery import Celery
from celery.schedules import crontab
from app.config import settings

# SQS broker URL — kombu[sqs] handles the transport
# IAM role auth (Fargate): sqs://
# Explicit key auth (local dev): sqs://KEY:SECRET@
if settings.aws_access_key_id:
    _broker = (
        f"sqs://{settings.aws_access_key_id}:{settings.aws_secret_access_key}@"
    )
else:
    _broker = "sqs://"

celery = Celery(
    "signals_worker",
    broker=_broker,
    broker_transport_options={
        "region": settings.aws_region,
        "predefined_queues": {
            "ai-trader-tasks": {"url": settings.sqs_tasks_queue_url},
        },
    },
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    result_backend=None,          # no result backend needed
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_default_queue="ai-trader-tasks",
    beat_schedule={
        "run-screener": {
            "task": "app.worker.tasks.run_screener",
            "schedule": crontab(minute="*/15", hour="9-15", day_of_week="mon-fri"),
        },
    },
)
