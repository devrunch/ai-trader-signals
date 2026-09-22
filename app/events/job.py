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
from app.events import history, reaction, telegram, watchdog, watcher, watches
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
        await watchdog.record_run("agenda", "quiet week")
        return {"ok": True, "events": 0, "sent": False}

    sent = await telegram.send(message)
    # Arming happens whether or not the push landed: a delivery failure must
    # not also cost him every brief for the rest of the week.
    armed = await watcher.arm(relevant)
    logger.info("Agenda: %d events (%d high), sent=%s, armed=%d", len(relevant),
                sum(1 for e in relevant if e.impact is Impact.HIGH), sent, len(armed))
    # So /status can answer "when did the desk last look", which is the
    # question someone asks when their phone has been quiet.
    await watchdog.record_run("agenda", f"{len(relevant)} events, {len(armed)} armed")
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
    # The brief and the nudge offer the same watch: the token is derived from
    # the event, so tapping either arms one watch rather than two.
    token = await watches.offer(event.key, currency, title, event.when)
    sent = await telegram.send(message, buttons=telegram.watch_button(token))
    logger.info("Brief for %s sent=%s (sample=%s, situation=%s)",
                event.key, sent, getattr(stats, "sample", None), bool(situation))
    # Recorded whether or not it sent: the watchdog asks whether the job ran,
    # and a delivery failure is a different fault from a job that never woke.
    await watchdog.record_fired(watcher.job_id(event), "sent" if sent else "send_failed")
    return {"ok": sent, "event": event.key, "sample": getattr(stats, "sample", 0),
            "situation": bool(situation), "sent": sent}


@heartbeat.monitored("event-nudge")
async def send_event_nudge(currency: str, title: str, when_iso: str) -> dict:
    """Thirty minutes out: one line and a button.

    Deliberately not a second brief. The brief went two hours ago and holds
    the reasoning; this exists so a release does not arrive while he is
    looking at something else, and so there is somewhere to put the button.
    No model call, so a quiet day costs nothing.
    """
    when = datetime.fromisoformat(when_iso)
    event = CalendarEvent(title=title, currency=currency, when=when,
                          impact=Impact.HIGH, forecast=None, previous=None)
    fresh = await fetch_week()
    for candidate in (fresh or []):
        if candidate.key == event.key and candidate.when == event.when:
            event = candidate
            break

    token = await watches.offer(event.key, currency, title, when)
    line = f"*{currency} {title}* in 30 min"
    if event.forecast:
        line += f"\nForecast {event.forecast}"
        if event.previous:
            line += f" · previous {event.previous}"
    sent = await telegram.send(line, buttons=telegram.watch_button(token))
    await watchdog.record_fired(watcher.nudge_id(event), "sent" if sent else "send_failed")
    logger.info("Nudge for %s sent=%s", event.key, sent)
    return {"ok": sent, "event": event.key, "sent": sent}


@heartbeat.monitored("event-watches")
async def run_watches() -> dict:
    """Report the realised move on every watch whose window has arrived.

    Runs often and does nothing most times. A watch whose minute bars are not
    downloadable yet is retried rather than dropped — the feed runs a few
    minutes behind, which is late, not missing.
    """
    try:
        pending = await watches.due()
    except Exception as e:
        logger.exception("Could not read the watch queue")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    reported = deferred = abandoned = 0
    for watch in pending:
        when = datetime.fromisoformat(watch["when"])
        moved = await reaction.realised_move(brief_mod.GOLD, when)
        if moved is None:
            if await watches.retry(watch["member"]):
                deferred += 1
            else:
                abandoned += 1
                await telegram.send_to(
                    watch["chat_id"],
                    f"*{watch['currency']} {watch['title']}* — no minute data for "
                    "that window, so there is nothing honest to report.")
            continue
        await telegram.send_to(watch["chat_id"], _watch_report(watch, moved))
        await watches.done(watch["member"])
        reported += 1

    if pending:
        logger.info("Watches: %d reported, %d deferred, %d abandoned",
                    reported, deferred, abandoned)
    return {"ok": True, "due": len(pending), "reported": reported,
            "deferred": deferred, "abandoned": abandoned}


def _watch_report(watch: dict, moved: dict) -> str:
    """What actually happened.

    No comparison against the historical distribution: the brief carried that
    two hours ago, and recomputing it here would cost ten minute-bar downloads
    per watch on a job that runs every five minutes. This message answers the
    one question the brief could not — what the move was.
    """
    lines = [f"*{watch['currency']} {watch['title']}* — what happened", ""]
    for label, key in (("15 min", "move_15m"), ("30 min", "move_30m")):
        value = moved.get(key)
        lines.append(f"{label}: {brief_mod.GOLD} {value:+.2f}%" if value is not None
                     else f"{label}: not priced")
    return "\n".join(lines)


@heartbeat.monitored("event-watchdog")
async def run_watchdog() -> dict:
    """Daily: did every brief we promised actually fire? See watchdog.py."""
    return await watchdog.sweep()
