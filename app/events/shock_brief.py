"""The paragraph nobody gets unless they ask for it.

Selection is free; explanation is not. Every headline the desk pushes has
already been scored by the hourly pipeline, so deciding it matters costs
nothing extra. Turning one into prose costs a model call, and a feed that did
that for every item would spend the day writing about headlines nobody opened.

So this runs on a button press and nowhere else.

Both sides, never a recommendation. The desk's standing rule is that it feeds
a human: it says what happens if this goes one way and what happens if it goes
the other, and leaves the decision where it belongs.
"""
from __future__ import annotations

import asyncio
import logging

from app.llm.client import LlmClient

logger = logging.getLogger(__name__)

SYSTEM = (
    "You brief an experienced FX and gold trader. He is not a beginner and "
    "does not want definitions. Never tell him what to do, never predict a "
    "direction as fact, and never invent a number, a level or a quote. If a "
    "headline has no clear market consequence, say exactly that in one line "
    "rather than manufacturing one."
)

TEMPLATE = """Headline: {headline}
Source: {source}
Instruments the analysis flagged: {symbols}
{price_line}
Write at most 120 words, in this shape and nothing else:

What it is: one sentence of plain fact.
If it escalates: what typically happens to the dollar and gold, and why.
If it fades: the same, for the other side.
Watch: the one thing that would tell him which way it is going.

Leave out any line you cannot fill honestly."""

MAX_TOKENS = 320
# A tap should feel like a tap. Past this the user has already put the phone
# down, and an answer arriving later is worse than one that never came.
TIMEOUT_SECONDS = 25

UNAVAILABLE = "Could not write that up just now. The headline and source are above."


async def write(llm: LlmClient, offer: dict, price: float | None = None) -> str | None:
    """One brief for one headline. None when the model could not be reached —
    the caller says so plainly rather than pushing an empty message."""
    symbols = ", ".join(offer.get("symbols") or []) or "none named"
    price_line = (f"Gold is trading at {price:.2f}.\n" if price else "")
    prompt = TEMPLATE.format(headline=offer.get("headline", ""),
                             source=offer.get("source", "") or "unknown",
                             symbols=symbols, price_line=price_line)
    try:
        response = await asyncio.wait_for(
            # The client is synchronous; a thread keeps one slow model call
            # from stalling the webhook's event loop and every other tap on it.
            asyncio.to_thread(
                llm.chat,
                temperature=0.2, max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
            ),
            timeout=TIMEOUT_SECONDS,
        )
    except Exception as e:
        # Enrichment, like the event brief's own paragraph: a model outage
        # costs the write-up, never the headline that was already delivered.
        logger.warning("Shock brief failed: %s", e)
        return None

    text = _text_of(response)
    return text.strip() or None


def _text_of(response) -> str:
    """The message content, whatever shape the client handed back."""
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError, TypeError):
        logger.warning("Shock brief response had no readable content")
        return ""
