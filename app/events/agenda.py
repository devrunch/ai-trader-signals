"""The daily agenda: what is coming, and which way each outcome cuts.

Deliberately model-free. Everything here is the calendar plus the event's own
published spec, so the agenda costs nothing to produce and cannot hallucinate
a direction. The model's turn comes later, per event, with the situational
layer -- what is different about this one, and what price is already saying.

Gold is only tied to USD events. A GBP inflation surprise moves cable; claiming
it moves gold because "the currency got stronger" is the kind of confident
nonsense that makes a trader stop reading.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.events.calendar import CalendarEvent, Impact
from app.events.specs import spec_for

USER_TZ = ZoneInfo("Asia/Dubai")

# What this user trades: gold and the dollar, with the majors that price it.
WATCHED_CURRENCIES = {"USD", "EUR", "GBP"}
# The reserve currency gold is quoted in -- the only one whose direction can be
# carried through to a gold call without hand-waving.
GOLD_DRIVER = "USD"

_IMPACT_MARK = {Impact.HIGH: "[HIGH]", Impact.MEDIUM: "[med]"}
# A phone message nobody scrolls. Beyond this the agenda stops being read.
MAX_DETAILED = 4
MAX_LISTED = 12


def _branches(event: CalendarEvent) -> list[str]:
    """Both directions, from the event's own spec, or nothing.

    An event whose spec does not state a direction (or which is not in the
    glossary at all) gets no branches rather than a guessed pair.
    """
    spec = spec_for(event.currency, event.title)
    if spec is None or spec.beat_is_bullish is None:
        return []

    stronger, weaker = ("above", "below") if spec.beat_is_bullish else ("below", "above")
    if event.currency == GOLD_DRIVER:
        return [f"{stronger} forecast -> USD stronger -> *gold down*",
                f"{weaker} forecast -> USD weaker -> *gold up*"]
    return [f"{stronger} forecast -> {event.currency} stronger",
            f"{weaker} forecast -> {event.currency} weaker"]


def _detail_block(event: CalendarEvent) -> list[str]:
    local = event.when.astimezone(USER_TZ)
    spec = spec_for(event.currency, event.title)
    lines = [f"*{local:%a %d %b  %H:%M}*  {event.currency} {event.title}"]

    if event.forecast:
        previous = event.previous or "n/a"
        lines.append(f"expected *{event.forecast}*  (previous {previous})")
    if spec and spec.measures:
        lines.append(f"_{spec.measures[:150].rstrip(' ;')}_")
    lines += _branches(event)
    if spec and spec.notes:
        lines.append(f"note: {spec.notes[:200].rstrip(' ;')}")
    lines.append("")
    return lines


def build(events: list[CalendarEvent], *, now: datetime | None = None) -> str | None:
    """The agenda message, or None when there is nothing worth sending.

    None, not an empty message: a quiet week is a real answer and he should not
    get a push that says nothing.
    """
    detailed = [e for e in events if e.impact is Impact.HIGH][:MAX_DETAILED]
    listed = [e for e in events if e not in detailed][:MAX_LISTED]
    if not detailed and not listed:
        return None

    lines = ["*What is coming, and which way it cuts*", ""]
    for event in detailed:
        lines += _detail_block(event)

    if listed:
        lines.append("*also on the calendar*")
        for event in listed:
            local = event.when.astimezone(USER_TZ)
            mark = _IMPACT_MARK.get(event.impact, "")
            expected = f"  (exp {event.forecast})" if event.forecast else ""
            lines.append(f"{mark} `{local:%a %H:%M}` {event.currency} {event.title}{expected}")
        lines.append("")

    lines.append("_times Dubai. direction from each release's own published spec._")
    return "\n".join(lines)
