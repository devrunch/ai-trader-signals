"""Arming the calendar: one wake-up per event that deserves a brief.

The calendar already says when to look, so this needs no judgement and no
model -- it walks the day's high-impact releases and books a `DateTrigger` two
hours ahead of each. That lead time is the product decision: the brief exists
to be read *before*, since a machine cannot beat the market to the print
itself (see the spec's own section on the physics).

Idempotent by event key, because the sweep runs daily and events live for a
week in the feed: re-arming the same release must replace its wake-up, not add
a second one.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from apscheduler.triggers.date import DateTrigger

from app.events.calendar import CalendarEvent, Impact

logger = logging.getLogger(__name__)

LEAD = timedelta(hours=2)
# An event less than this away when the sweep runs has already lost the point
# of a two-hour warning; it still appears in the daily agenda.
MIN_LEAD = timedelta(minutes=20)

_scheduler = None


def bind(scheduler) -> None:
    """Called once by newsd at startup. Without it, arming is a no-op that
    says so rather than pretending the wake-up exists."""
    global _scheduler
    _scheduler = scheduler


def job_id(event: CalendarEvent) -> str:
    return f"brief:{event.currency}:{event.title}:{event.when:%Y%m%d%H%M}"


def arm(events: list[CalendarEvent], *, now: datetime | None = None) -> list[str]:
    """Book a brief for every high-impact event far enough out. Returns the
    ids armed, so the caller can report a number rather than a shrug."""
    if _scheduler is None:
        logger.warning("No scheduler bound -- %d events not armed", len(events))
        return []

    now = now or datetime.now(UTC)
    armed = []
    for event in events:
        if event.impact is not Impact.HIGH:
            continue
        fire_at = event.when - LEAD
        if fire_at - now < MIN_LEAD - LEAD:      # event itself already passed
            continue
        if fire_at <= now + MIN_LEAD:
            continue
        _scheduler.add_job(
            "app.events.job:send_event_brief",
            trigger=DateTrigger(run_date=fire_at),
            id=job_id(event),
            name=job_id(event),
            replace_existing=True,
            kwargs={"currency": event.currency, "title": event.title,
                    "when_iso": event.when.isoformat()},
        )
        armed.append(job_id(event))
    return armed
