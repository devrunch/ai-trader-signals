"""The daily agenda job.

Runs in newsd, costs nothing, and is the one piece that has to be reliable:
everything else in the event desk is an improvement on top of "he knows what
is coming today".

Reports what happened rather than returning a bare bool -- the heartbeat
decorator reads `ok`, and a run that fetched nothing is a different failure
from one that fetched fine and found a quiet day.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.events import brief as brief_mod
from app.events import history, reaction, telegram, watchdog, watcher
from app.events.agenda import WATCHED_CURRENCIES, build
from app.events.calendar import CalendarEvent, Impact, fetch_week, upcoming
from app.llm.client import get_llm
from app.market.providers.registry import market_data_router
from app.worker import heartbeat

logger = logging.getLogger(__name__)

# Low-impact rows are most of the feed and none of the value: bank holidays,
# second-tier surveys, speeches with no policy content.
AGENDA_IMPACTS = {Impact.HIGH, Impact.MEDIUM}


@heartbeat.monitored("event-agenda")
async def run_agenda() -> dict:
    events = await fetch_week()
    if events is None:
        logger.error("Agenda skipped: calendar feed unavailable")
        return {"ok": False, "error": "calendar_unavailable"}

    relevant = upcoming(events, currencies=WATCHED_CURRENCIES, impacts=AGENDA_IMPACTS)
    message = build(relevant)
    if message is None:
        # A genuinely empty week, said plainly rather than pushed as content.
        logger.info("Agenda: nothing upcoming for %s", sorted(WATCHED_CURRENCIES))
        return {"ok": True, "events": 0, "sent": False}

    sent = await telegram.send(message)
    # Arming happens whether or not the push landed: a delivery failure must
    # not also cost him every brief for the rest of the week.
    armed = await watcher.arm(relevant)
    logger.info("Agenda: %d events (%d high), sent=%s, armed=%d", len(relevant),
                sum(1 for e in relevant if e.impact is Impact.HIGH), sent, len(armed))
    return {"ok": sent, "events": len(relevant), "sent": sent, "armed": len(armed)}


@heartbeat.monitored("event-brief")
async def send_event_brief(currency: str, title: str, when_iso: str) -> dict:
    """One release, about two hours out: spec, measured reaction, situation.

    Scheduled by the watcher off the calendar, so the arguments carry the
    event rather than re-fetching a feed that may have moved on.
    """
    event = CalendarEvent(
        title=title, currency=currency, when=datetime.fromisoformat(when_iso),
        impact=Impact.HIGH, forecast=None, previous=None,
    )
    # The feed carries the forecast, and it is often revised after the sweep
    # that armed this. Worth one fetch; the brief still goes without it.
    fresh = await fetch_week()
    if fresh:
        for candidate in fresh:
            if candidate.key == event.key and candidate.when == event.when:
                event = candidate
                break

    stats = None
    rows = history.past_prints(currency, title)
    if rows:
        instances = reaction.instances_from_history(list(rows), limit=10)
        measured = await reaction.measure(brief_mod.GOLD, instances)
        stats = reaction.aggregate(event.key, brief_mod.GOLD, measured)

    quote = await market_data_router.get_quote(brief_mod.GOLD, "FOREX")
    price = (quote or {}).get("ltp")

    situation = await brief_mod._situation(get_llm(), event, stats, price)
    message = brief_mod.render(event, stats, price, situation, now=datetime.now(UTC))
    sent = await telegram.send(message)
    logger.info("Brief for %s sent=%s (sample=%s, situation=%s)",
                event.key, sent, getattr(stats, "sample", None), bool(situation))
    # Recorded whether or not it sent: the watchdog asks whether the job ran,
    # and a delivery failure is a different fault from a job that never woke.
    await watchdog.record_fired(watcher.job_id(event), "sent" if sent else "send_failed")
    return {"ok": sent, "event": event.key, "sample": getattr(stats, "sample", 0),
            "situation": bool(situation), "sent": sent}


@heartbeat.monitored("event-watchdog")
async def run_watchdog() -> dict:
    """Daily: did every brief we promised actually fire? See watchdog.py."""
    return await watchdog.sweep()
