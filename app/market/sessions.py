"""When each market is open, so that "no bars" can be explained.

An empty range is not self-explanatory. Forex over a weekend, an equity
exchange at 3am, and a vendor that has stopped answering all return nothing;
only the first two are normal, and until now all three surfaced as a 404.

A session is a set of weekly windows in its own timezone. That covers both
shapes this app needs: forex, which opens Sunday evening and runs continuously
to Friday evening, and an exchange that opens and closes every weekday.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

_MINUTES_PER_DAY = 24 * 60
_MINUTES_PER_WEEK = 7 * _MINUTES_PER_DAY
# A holiday can close an exchange for several days running (Christmas, or an
# Indian holiday landing next to a weekend); ten candidates is comfortably past
# the longest such run without being an unbounded search.
_MAX_NEXT_OPEN_CANDIDATES = 10


def _minute_of_week(day: int, at: time) -> int:
    return day * _MINUTES_PER_DAY + at.hour * 60 + at.minute


@dataclass(frozen=True)
class Window:
    """One weekly opening, in the session's own timezone. Monday is 0, matching
    ``datetime.weekday()``.

    ``end_day`` may fall earlier in the week than ``start_day``: forex opens
    Sunday and closes Friday as a single continuous window, not five daily ones.
    """

    start_day: int
    start: time
    end_day: int
    end: time

    def contains(self, moment: datetime) -> bool:
        now = _minute_of_week(moment.weekday(), moment.time())
        opens = _minute_of_week(self.start_day, self.start)
        closes = _minute_of_week(self.end_day, self.end)
        if opens <= closes:
            return opens <= now < closes
        return now >= opens or now < closes      # wraps through Sunday

    def next_start_after(self, moment: datetime) -> datetime:
        """The next time this window opens, strictly after `moment`."""
        days_ahead = (self.start_day - moment.weekday()) % 7
        candidate = datetime.combine(
            (moment + timedelta(days=days_ahead)).date(), self.start, tzinfo=moment.tzinfo,
        )
        if candidate <= moment:
            candidate += timedelta(days=7)
        return candidate


def _never(_: date) -> bool:
    return False


@dataclass(frozen=True)
class SessionSpec:
    timezone: str
    windows: tuple[Window, ...]
    # Takes a date in this session's timezone. Forex has none worth modelling;
    # NSE's list is already maintained in app/market/calendar.py.
    holiday: Callable[[date], bool] = field(default=_never, compare=False)

    def _local(self, moment: datetime) -> datetime:
        return moment.astimezone(ZoneInfo(self.timezone))

    def is_open(self, moment: datetime) -> bool:
        local = self._local(moment)
        if self.holiday(local.date()):
            return False
        return any(window.contains(local) for window in self.windows)

    def next_open(self, moment: datetime) -> datetime | None:
        """When this market opens next, in UTC. None if a holiday run outlasts
        the search -- a caller must be able to say "closed" without claiming a
        reopening time it does not actually know."""
        local = self._local(moment)
        candidates = sorted(w.next_start_after(local) for w in self.windows)
        for _ in range(_MAX_NEXT_OPEN_CANDIDATES):
            if not candidates:
                return None
            soonest = candidates.pop(0)
            if not self.holiday(soonest.date()):
                return soonest.astimezone(ZoneInfo("UTC"))
            candidates.append(soonest + timedelta(days=7))
            candidates.sort()
        return None


def _nse_holiday(day: date) -> bool:
    # Imported lazily: app/market/calendar.py owns the published NSE list, and
    # importing it at module scope would make this module's import order matter.
    from app.market import calendar as nse_calendar

    return nse_calendar.is_holiday(day)


# Forex and metals: one continuous window, Sunday 17:00 New York to Friday
# 17:00 New York -- the convention every FX venue quotes, and the reason a
# Sunday request for 1m gold bars legitimately has nothing to return.
FX = SessionSpec(
    timezone="America/New_York",
    windows=(Window(start_day=6, start=time(17, 0), end_day=4, end=time(17, 0)),),
)

# NSE/BSE equity, straight from the calendar module's own constants.
NSE = SessionSpec(
    timezone="Asia/Kolkata",
    windows=tuple(
        Window(start_day=d, start=time(9, 15), end_day=d, end=time(15, 30)) for d in range(5)
    ),
    holiday=_nse_holiday,
)

# MCX commodity futures run an evening session; the equity holiday list does
# not apply to it, and MCX publishes its own.
MCX = SessionSpec(
    timezone="Asia/Kolkata",
    windows=tuple(
        Window(start_day=d, start=time(9, 0), end_day=d, end=time(23, 30)) for d in range(5)
    ),
)

US_EQUITY = SessionSpec(
    timezone="America/New_York",
    windows=tuple(
        Window(start_day=d, start=time(9, 30), end_day=d, end=time(16, 0)) for d in range(5)
    ),
)

# Anything we have not modelled: always open, so an unknown venue degrades to
# "the vendor had nothing" rather than a confident, wrong "market is closed".
# Each window runs to midnight of the following day, so the week has no seam.
ALWAYS_OPEN = SessionSpec(
    timezone="UTC",
    windows=tuple(
        Window(start_day=d, start=time(0, 0), end_day=(d + 1) % 7, end=time(0, 0))
        for d in range(7)
    ),
)
