"""The per-event brief, sent about two hours before a release.

Three layers, in descending order of how much they can be trusted, and the
message says which is which:

  1. **The spec** -- what this release is, which way a beat cuts, and the
     caveat its own publisher attaches. Deterministic.
  2. **The measured reaction** -- what this instrument actually did after past
     instances, split by surprise direction. Arithmetic over our own bars, and
     silent when the sample cannot support a claim.
  3. **The situational read** -- what is different this time. This is the only
     layer a model writes, it is labelled, and the brief is still worth
     reading if it is missing.

Order matters: a model that writes first and cites second will produce
plausible sentences and reach for numbers that support them. Here the numbers
exist before the model is asked anything, and the model is told it may not
contradict them.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from app.events.calendar import CalendarEvent
from app.events.reaction import ReactionStats
from app.events.specs import spec_for
from app.llm.client import LlmClient

logger = logging.getLogger(__name__)

USER_TZ = ZoneInfo("Asia/Dubai")
GOLD = "XAUUSD"
GOLD_DRIVER = "USD"

SITUATION_SYSTEM = (
    "You brief one trader before an economic release. He trades gold and the "
    "dollar. Two or three sentences, no preamble, no pleasantries. Say what is "
    "different about this release and what it hinges on. You may not contradict "
    "the measured statistics you are given, and you may not invent numbers, "
    "levels or positioning data. If you have nothing specific to add beyond "
    "what is already stated, say only what the release hinges on."
)
SITUATION_MAX_TOKENS = 220


def _branches(event: CalendarEvent) -> list[str]:
    spec = spec_for(event.currency, event.title)
    if spec is None or spec.beat_is_bullish is None:
        return []
    stronger, weaker = ("above", "below") if spec.beat_is_bullish else ("below", "above")
    if event.currency == GOLD_DRIVER:
        return [f"{stronger} forecast -> USD stronger -> *gold down*",
                f"{weaker} forecast -> USD weaker -> *gold up*"]
    return [f"{stronger} forecast -> {event.currency} stronger",
            f"{weaker} forecast -> {event.currency} weaker"]


async def _situation(llm: LlmClient, event: CalendarEvent, stats: ReactionStats | None,
                     price: float | None) -> str | None:
    """The model's paragraph, or None. Never a precondition for sending."""
    spec = spec_for(event.currency, event.title)
    facts = [
        f"Release: {event.currency} {event.title}",
        f"Expected: {event.forecast or 'no forecast published'}",
        f"Previous: {event.previous or 'n/a'}",
    ]
    if spec and spec.notes:
        facts.append(f"Publisher's own note: {spec.notes}")
    if stats is not None:
        facts.append(f"Measured reaction: {stats.summary() or 'sample too thin to characterise'}")
    if price is not None:
        facts.append(f"Gold is currently {price:.2f}")

    messages = [{"role": "system", "content": SITUATION_SYSTEM},
                {"role": "user", "content": "\n".join(facts)}]
    try:
        # to_thread: the OpenAI client is synchronous, and this runs on the
        # loop that also serves the rest of newsd.
        resp = await asyncio.to_thread(
            llm.chat, messages=messages, temperature=0.2,
            max_tokens=SITUATION_MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        # Enrichment. A model outage costs the paragraph, never the brief.
        logger.exception("Situational read failed for %s", event.key)
        return None
    return text or None


def render(event: CalendarEvent, stats: ReactionStats | None, price: float | None,
           situation: str | None, *, now: datetime | None = None) -> str:
    local = event.when.astimezone(USER_TZ)
    now = now or datetime.now(UTC)
    minutes = max(0, int((event.when - now).total_seconds() // 60))
    spec = spec_for(event.currency, event.title)

    lines = [f"*{event.currency} {event.title}*",
             f"in {minutes // 60}h {minutes % 60}m  —  {local:%a %d %b %H:%M} your time", ""]

    if event.forecast:
        lines.append(f"expected *{event.forecast}*   (previous {event.previous or 'n/a'})")
    if spec and spec.measures:
        lines.append(f"_{spec.measures[:150].rstrip(' ;')}_")
    lines.append("")

    branches = _branches(event)
    if branches:
        lines += branches
    else:
        # Said out loud: the alternative is a brief that looks complete while
        # quietly having no view.
        lines.append("_no published direction for this release — outcome not mapped_")

    if stats is not None:
        summary = stats.summary()
        lines.append("")
        if summary:
            lines.append(f"*last {stats.sample} prints:* {summary}")
        else:
            lines.append(f"*history:* {stats.sample} past prints priced, "
                         "not enough of a surprise in either direction to characterise")

    if price is not None:
        lines.append(f"gold now *{price:.2f}*")

    if spec and spec.notes:
        lines.append("")
        lines.append(f"note: {spec.notes[:220].rstrip(' ;')}")

    if situation:
        lines.append("")
        lines.append(f"🤖 {situation}")

    return "\n".join(lines)
