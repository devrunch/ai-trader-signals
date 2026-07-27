"""NSE trading calendar — session hours, holidays, and square-off time.

Extracted from `market/router.py`, where "is the market open" was three lines of
inline `time()` comparisons in a route handler with no holiday awareness. That
version reported the market open on Republic Day and on every other trading
holiday, which is the sort of thing that quietly poisons a screener run.

Everything here is IST. The NSE equity cash session is 09:15–15:30 IST, Monday
to Friday, excluding the exchange's published holiday list.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytz

IST = pytz.timezone("Asia/Kolkata")

# NSE equity (capital market) segment regular session.
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Intraday (MIS) positions are force-squared-off by brokers from ~15:20, ten
# minutes before the close. A signal issued after this cannot realistically be
# taken intraday, so callers use this to stop emitting fresh intraday entries.
SQUARE_OFF = time(15, 20)

# ---------------------------------------------------------------------------
# NSE trading holidays 2026 (equity/capital-market segment).
#
# Source: NSE's annual trading-holiday circular for the calendar year, as
# published on nseindia.com (Resources > Exchange Communication > Holidays >
# Trading Holidays). Transcribed by hand.
#
# THIS LIST MUST BE REFRESHED EVERY YEAR. NSE publishes the next year's list in
# December; when it does, add a new frozenset and extend `_HOLIDAYS_BY_YEAR`.
# A missing year is handled conservatively — see `is_trading_day` — but a stale
# list silently reports holidays as trading days, so do not let it lapse.
#
# Dates that fall on a Saturday or Sunday are listed for completeness and are
# already excluded by the weekday check.
#
# Not included, deliberately: Diwali Laxmi Pujan (08-Nov-2026, a Sunday) has a
# special one-hour *Muhurat* trading session whose timing NSE announces
# separately each year. The regular session is closed, so treating the day as a
# holiday is correct for everything in this codebase.
# ---------------------------------------------------------------------------
NSE_HOLIDAYS_2026: frozenset[date] = frozenset({
    date(2026, 1, 26),   # Mon — Republic Day
    date(2026, 2, 15),   # Sun — Mahashivratri
    date(2026, 3, 4),    # Wed — Holi
    date(2026, 3, 21),   # Sat — Id-ul-Fitr (Ramzan Id)
    date(2026, 3, 26),   # Thu — Shri Ram Navami
    date(2026, 3, 31),   # Tue — Mahavir Jayanti
    date(2026, 4, 3),    # Fri — Good Friday
    date(2026, 4, 14),   # Tue — Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Fri — Maharashtra Day
    date(2026, 5, 27),   # Wed — Bakri Id (Id-ul-Adha)
    date(2026, 6, 26),   # Fri — Muharram
    date(2026, 8, 15),   # Sat — Independence Day
    date(2026, 9, 14),   # Mon — Ganesh Chaturthi
    date(2026, 10, 2),   # Fri — Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Tue — Dussehra (Vijaya Dashami)
    date(2026, 11, 10),  # Tue — Diwali Balipratipada
    date(2026, 11, 24),  # Tue — Guru Nanak Jayanti
    date(2026, 12, 25),  # Fri — Christmas
})

_HOLIDAYS_BY_YEAR: dict[int, frozenset[date]] = {
    2026: NSE_HOLIDAYS_2026,
}


def now_ist(now: datetime | None = None) -> datetime:
    """Normalise an optional caller-supplied `now` to a tz-aware IST datetime.

    A naive datetime is *interpreted* as IST rather than as UTC — every caller
    in this codebase that hand-rolls a timestamp means local exchange time.
    """
    if now is None:
        return datetime.now(IST)
    if now.tzinfo is None:
        return IST.localize(now)
    return now.astimezone(IST)


def holidays_known_for(year: int) -> bool:
    """Whether a published holiday list has been transcribed for `year`."""
    return year in _HOLIDAYS_BY_YEAR


def is_holiday(day: date) -> bool:
    """Is `day` an NSE trading holiday?

    Returns False for years we have no list for. That is the honest answer —
    "we don't know" — and `is_trading_day` logs the gap rather than inventing
    an answer. Callers that must not guess should check `holidays_known_for`.
    """
    return day in _HOLIDAYS_BY_YEAR.get(day.year, frozenset())


def is_trading_day(day: date | None = None) -> bool:
    """Weekday and not a published NSE holiday."""
    if day is None:
        day = datetime.now(IST).date()
    if day.weekday() >= 5:
        return False
    return not is_holiday(day)


def is_market_open(now: datetime | None = None) -> bool:
    """Is the NSE equity cash session live right now?"""
    ts = now_ist(now)
    if not is_trading_day(ts.date()):
        return False
    return MARKET_OPEN <= ts.time() <= MARKET_CLOSE


def is_square_off_time(now: datetime | None = None) -> bool:
    """True from 15:20 IST until the close on a trading day.

    Intraday entries taken after this point cannot be managed — the broker will
    square them off within minutes, at whatever price the close gives.
    """
    ts = now_ist(now)
    if not is_trading_day(ts.date()):
        return False
    return SQUARE_OFF <= ts.time() <= MARKET_CLOSE


def next_open(now: datetime | None = None) -> datetime:
    """The next 09:15 IST session start strictly after `now`.

    If the market is currently open, this returns *tomorrow's* open (or the next
    trading day's) — the question "when does the next session start" only has a
    future answer.
    """
    ts = now_ist(now)
    candidate = ts.date()
    # Today only qualifies if the session has not begun yet.
    if not (is_trading_day(candidate) and ts.time() < MARKET_OPEN):
        candidate = candidate + timedelta(days=1)
    # A run of consecutive holidays plus a weekend tops out well under 14 days;
    # the bound stops an unpublished-year gap turning into an infinite loop.
    for _ in range(14):
        if is_trading_day(candidate):
            return IST.localize(datetime.combine(candidate, MARKET_OPEN))
        candidate = candidate + timedelta(days=1)
    raise RuntimeError(f"No trading day found within 14 days of {ts.date()}")


def session_state(now: datetime | None = None) -> dict:
    """Everything a caller usually wants at once, as plain JSON-able values."""
    ts = now_ist(now)
    day = ts.date()
    return {
        "is_open": is_market_open(ts),
        "is_trading_day": is_trading_day(day),
        "is_holiday": is_holiday(day),
        "is_square_off": is_square_off_time(ts),
        "timestamp": ts.isoformat(),
        "next_open": next_open(ts).isoformat(),
        # False means the holiday list for this year has not been transcribed,
        # so `is_open` may report a holiday as a trading day.
        "holiday_calendar_known": holidays_known_for(day.year),
    }
