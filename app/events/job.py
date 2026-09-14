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

from app.events import telegram
from app.events.agenda import WATCHED_CURRENCIES, build
from app.events.calendar import Impact, fetch_week, upcoming

logger = logging.getLogger(__name__)

# Low-impact rows are most of the feed and none of the value: bank holidays,
# second-tier surveys, speeches with no policy content.
AGENDA_IMPACTS = {Impact.HIGH, Impact.MEDIUM}


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
    logger.info("Agenda: %d events (%d high), sent=%s", len(relevant),
                sum(1 for e in relevant if e.impact is Impact.HIGH), sent)
    return {"ok": sent, "events": len(relevant), "sent": sent}
