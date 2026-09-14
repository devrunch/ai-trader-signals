"""What a calendar event actually is, and which way a beat cuts.

Every recurring economic release carries a published spec: what it measures,
which direction a beat pushes the currency, and the caveat traders care about
("the rate decision is usually priced in, so the Statement overshadows it").
That is the difference between a brief that says *"CPI at 16:30"* and one that
says what to do with it -- and it needs no model to produce.

The glossary is a one-time extraction (event_specs.json, 693 events) rather
than a per-event fetch, because event names recur forever: this month's CPI
carries the same spec as every CPI before it.

The direction is parsed, never assumed. 29 of these events are **inverted** --
Unemployment Rate beating its forecast is bad for the currency, not good -- so
a system that hardcoded "beat = bullish" would state the opposite of the truth
on every one of them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_SPECS_FILE = Path(__file__).with_name("event_specs.json")


@dataclass(frozen=True)
class EventSpec:
    measures: str
    usual_effect: str
    notes: str
    why: str
    source: str

    @property
    def beat_is_bullish(self) -> bool | None:
        """Does an actual above forecast strengthen the currency?

        None when the spec does not say. An unknown direction is reported as
        unknown rather than guessed -- stating the wrong branch is worse than
        stating none, because he acts on it.
        """
        effect = self.usual_effect.lower()
        if "greater than" in effect:
            return True
        if "less than" in effect:
            return False
        return None


@lru_cache(maxsize=1)
def _glossary() -> dict[str, EventSpec]:
    raw = json.loads(_SPECS_FILE.read_text(encoding="utf-8"))
    return {k: EventSpec(**v) for k, v in raw.items()}


def spec_for(currency: str, title: str) -> EventSpec | None:
    """The spec for one event, or None if this release is not in the glossary.

    Misses are expected and fine: one-off events (a summit, an unscheduled
    speech) have no recurring spec, and the brief simply says less about them.
    """
    return _glossary().get(f"{currency.upper()}|{title}")


def known_events() -> int:
    return len(_glossary())
