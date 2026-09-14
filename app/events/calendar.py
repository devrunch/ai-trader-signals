"""The economic calendar: what is scheduled, when, and what is expected.

Source is the weekly JSON feed ForexFactory publishes for tools
(nfs.faireconomy.media), not their calendar pages. The distinction matters:
the feed is the path they publish for this purpose, and their terms explicitly
reserve the calendar product itself.

What the feed gives that nothing else free does is the **forecast** -- and a
print means nothing without it. CPI at 3.1% is bullish or bearish purely
against what was expected, so a calendar without consensus is a clock.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import httpx

logger = logging.getLogger(__name__)

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# The feed is served to tools, not browsers, but a default httpx agent gets
# refused; this is the same identification any calendar widget sends.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-trader event desk)"}
_TIMEOUT = 20.0

# One week of a global calendar is a few hundred rows. Anything far outside
# that means the feed changed shape, not that the world got busy.
_SANE_MIN_EVENTS = 20
_SANE_MAX_EVENTS = 2000


class Impact(StrEnum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    HOLIDAY = "Holiday"


@dataclass(frozen=True)
class CalendarEvent:
    title: str
    currency: str
    when: datetime           # always UTC
    impact: Impact
    forecast: str | None
    previous: str | None

    @property
    def key(self) -> str:
        return f"{self.currency}|{self.title}"


def _clean(value) -> str | None:
    """Empty strings mean 'no forecast published', which is not the same as a
    forecast of zero. Kept as None so nothing downstream can print it as one."""
    text = (value or "").strip()
    return text or None


def parse(rows: list[dict]) -> list[CalendarEvent]:
    """Feed rows -> events, dropping only what cannot be placed in time.

    A row without a parseable date is unusable (we schedule off it); a row
    missing a forecast is perfectly usable and keeps its None.
    """
    events = []
    for row in rows:
        try:
            when = datetime.fromisoformat(row["date"]).astimezone(UTC)
        except (KeyError, ValueError, TypeError):
            continue
        try:
            impact = Impact(row.get("impact"))
        except ValueError:
            impact = Impact.LOW
        events.append(CalendarEvent(
            title=row.get("title", "").strip(),
            currency=(row.get("country") or "").strip().upper(),
            when=when,
            impact=impact,
            forecast=_clean(row.get("forecast")),
            previous=_clean(row.get("previous")),
        ))
    events.sort(key=lambda e: e.when)
    return events


async def fetch_week(client: httpx.AsyncClient | None = None) -> list[CalendarEvent] | None:
    """This week's calendar, or None if the feed could not be trusted.

    None rather than an empty list on failure: an empty calendar is a real
    (if rare) answer, and a caller that cannot tell the two apart will happily
    report "nothing scheduled" on the morning of a Fed decision.
    """
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS)
    try:
        resp = await client.get(FEED_URL, headers=_HEADERS)
        resp.raise_for_status()
        rows = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Calendar feed fetch failed: %s", e)
        return None
    finally:
        if own_client:
            await client.aclose()

    if not isinstance(rows, list) or not _SANE_MIN_EVENTS <= len(rows) <= _SANE_MAX_EVENTS:
        # The characteristic failure of a free feed is not an error page, it is
        # a 200 with the wrong shape. Row count is the cheapest tripwire.
        logger.warning("Calendar feed returned %s rows, outside the sane range",
                       len(rows) if isinstance(rows, list) else type(rows).__name__)
        return None

    events = parse(rows)
    if not events:
        logger.warning("Calendar feed parsed to zero events from %d rows", len(rows))
        return None
    return events


def upcoming(events: list[CalendarEvent], *, now: datetime | None = None,
             currencies: set[str] | None = None,
             impacts: set[Impact] | None = None) -> list[CalendarEvent]:
    """Future events only, filtered to what this user's instruments react to."""
    now = now or datetime.now(UTC)
    out = [e for e in events if e.when > now]
    if currencies:
        out = [e for e in out if e.currency in currencies]
    if impacts:
        out = [e for e in out if e.impact in impacts]
    return out
