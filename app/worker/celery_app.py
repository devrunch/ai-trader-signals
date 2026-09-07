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
    include=["app.worker.tasks"],
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
        # The screener fires only inside the real NSE window, 09:20–15:15 IST.
        # A single `minute="*/15", hour="9-15"` entry also fired at 09:00 and
        # 09:15 (before/at the open, on yesterday's bars) and at 15:45 (after
        # the close), and every one of those runs stored signals that were
        # scored in the published performance statistics. Three entries because
        # crontab cannot express a half-open hour range in one expression.
        # 09:20 rather than 09:15 so the session's first 15m bar has formed;
        # 15:15 rather than 15:30 so there is time to act before square-off.
        # `run_screener` re-checks `is_market_open()` itself — beat only narrows
        # the window, the calendar (holidays included) is the actual gate.
        "run-screener-open": {
            "task": "app.worker.tasks.run_screener",
            "schedule": crontab(minute="20,35,50", hour="9", day_of_week="mon-fri"),
        },
        "run-screener-day": {
            "task": "app.worker.tasks.run_screener",
            "schedule": crontab(minute="*/15", hour="10-14", day_of_week="mon-fri"),
        },
        "run-screener-close": {
            "task": "app.worker.tasks.run_screener",
            "schedule": crontab(minute="0,15", hour="15", day_of_week="mon-fri"),
        },
        # Intraday means intraday: force-close every open paper position at
        # 15:20 IST, the point from which real brokers square off MIS positions.
        # Nothing is held overnight, so nothing carries overnight gap risk.
        "square-off-positions": {
            "task": "app.worker.tasks.square_off_positions",
            "schedule": crontab(minute="20", hour="15", day_of_week="mon-fri"),
        },
        # Pre-market brief: after the US close (~02:00 IST), before the
        # Indian open (09:15). Ready well ahead of the 07:00 notification.
        "morning-brief": {
            "task": "app.worker.tasks.generate_morning_brief",
            "schedule": crontab(minute="30", hour="6", day_of_week="mon-fri"),
        },
        # Second run of the same brief, ahead of the US open (~19:00 IST /
        # 09:30 ET) rather than the Indian one -- picks up the day's own US
        # session macro releases (US data mostly prints 18:00-19:30 IST) that
        # the 06:30 run is too early for. Same task, same "real numbers only"
        # narrative -- just a second real vantage point on the same day, not
        # a different kind of brief.
        "evening-brief": {
            "task": "app.worker.tasks.generate_morning_brief",
            "schedule": crontab(minute="0", hour="18", day_of_week="mon-fri"),
        },
        # Hourly, all day -- moves the real NewsAPI/HF/LLM work off the
        # request path (see run_news_analysis's own docs). NewsAPI's free
        # tier caps at 100 requests/day; a 15-min cadence would burn 96 of
        # those on this alone, leaving almost no room for anything else.
        # Hourly is 24/day -- and NewsAPI's own articles are already ~24h
        # delayed regardless of how often this polls, so there's no real
        # freshness lost. Different minute than drift-check (:30) to spread
        # load across the hour rather than firing both at once.
        "news-analysis": {
            "task": "app.worker.tasks.run_news_analysis",
            "schedule": crontab(minute="0"),
        },
        # Kite Connect's access_token expires ~6am IST. 06:00 is ahead of the
        # 06:30 morning brief and well ahead of the 09:15 open — see the
        # design spec for why a failed refresh here is not an emergency (the
        # router falls back to yfinance for the rest of that day).
        "refresh-zerodha-session": {
            "task": "app.worker.tasks.refresh_zerodha_session",
            "schedule": crontab(minute="0", hour="6"),
        },
        # Every hour, all day -- drift_check.check() is cheap on a quiet hour
        # (one real fetch, no LLM call, no alert), so there's no need to
        # restrict this to market hours the way the screener is.
        "drift-check": {
            "task": "app.worker.tasks.run_drift_check",
            "schedule": crontab(minute="30"),
        },
        # Odd hours only (1,3,5...23 IST) -- keeps Tavily's free-credit usage
        # down, per the explicit call to ration it across callers.
        "reddit-sentiment": {
            "task": "app.worker.tasks.run_reddit_sentiment",
            "schedule": crontab(minute="45", hour="1,3,5,7,9,11,13,15,17,19,21,23"),
        },
    },
)
