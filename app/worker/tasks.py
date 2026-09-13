"""Celery wrappers around the scheduled jobs.

The jobs live in jobs.py; this module only exposes them to the Celery worker,
which still runs the news pipeline until that moves to its own process.
Everything else is scheduled in-process by app/worker/scheduler.py.
"""
from __future__ import annotations

from app.worker import jobs
from app.worker.celery_app import celery

run_screener = celery.task(name="app.worker.tasks.run_screener")(jobs.run_screener)
square_off_positions = celery.task(name="app.worker.tasks.square_off_positions")(jobs.square_off_positions)
run_news_analysis = celery.task(name="app.worker.tasks.run_news_analysis")(jobs.run_news_analysis)
generate_morning_brief = celery.task(name="app.worker.tasks.generate_morning_brief")(jobs.generate_morning_brief)
run_drift_check = celery.task(name="app.worker.tasks.run_drift_check")(jobs.run_drift_check)
run_reddit_sentiment = celery.task(name="app.worker.tasks.run_reddit_sentiment")(jobs.run_reddit_sentiment)
refresh_zerodha_session = celery.task(name="app.worker.tasks.refresh_zerodha_session")(jobs.refresh_zerodha_session)
