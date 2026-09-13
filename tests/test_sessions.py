"""Session hours, which are what let an empty chart explain itself.

The case that matters is the one that shipped broken: a Sunday request for 1m
gold bars covers only closed market, and answering "no data" there is a
different statement from "this vendor is down".
"""
from datetime import UTC, date, datetime

from app.market import sessions


def _utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


class TestForex:
    def test_the_weekend_is_closed(self):
        # Saturday, and the Sunday morning that 1m charts kept 404ing on.
        assert not sessions.FX.is_open(_utc(2026, 9, 12, 12))
        assert not sessions.FX.is_open(_utc(2026, 9, 13, 13))

    def test_it_opens_sunday_evening_new_york(self):
        # 17:00 New York on 2026-09-13 is 21:00 UTC (daylight time).
        assert not sessions.FX.is_open(_utc(2026, 9, 13, 20, 59))
        assert sessions.FX.is_open(_utc(2026, 9, 13, 21, 1))

    def test_midweek_runs_through_the_night(self):
        # One continuous window, not five daily ones -- 03:00 UTC Wednesday is
        # the middle of the session, not an overnight gap.
        assert sessions.FX.is_open(_utc(2026, 9, 9, 3))

    def test_it_closes_friday_evening(self):
        assert sessions.FX.is_open(_utc(2026, 9, 11, 20, 59))
        assert not sessions.FX.is_open(_utc(2026, 9, 11, 21, 1))

    def test_next_open_from_the_weekend_is_sunday_evening(self):
        nxt = sessions.FX.next_open(_utc(2026, 9, 12, 12))
        assert nxt == _utc(2026, 9, 13, 21)


class TestNse:
    def test_open_during_the_session_closed_outside_it(self):
        # 2026-09-09 is a Wednesday. 09:15-15:30 IST is 03:45-10:00 UTC.
        assert sessions.NSE.is_open(_utc(2026, 9, 9, 5))
        assert not sessions.NSE.is_open(_utc(2026, 9, 9, 11))

    def test_the_published_holiday_list_is_wired_in(self):
        from app.market import calendar as nse_calendar

        holiday = next(iter(sorted(nse_calendar.NSE_HOLIDAYS_2026)))
        midday = datetime(holiday.year, holiday.month, holiday.day, 6, tzinfo=UTC)
        assert not sessions.NSE.is_open(midday)

    def test_next_open_skips_the_weekend_and_the_holiday_behind_it(self):
        # Monday 2026-09-14 is on NSE's published holiday list, so the next
        # open from Saturday is Tuesday 09:15 IST (03:45 UTC), not Monday.
        nxt = sessions.NSE.next_open(_utc(2026, 9, 12, 12))
        assert nxt.astimezone(UTC) == _utc(2026, 9, 15, 3, 45)


class TestHolidays:
    def test_a_holiday_closes_the_session_and_moves_next_open(self):
        # A session with one known holiday, so this does not depend on which
        # dates a real exchange publishes for a given year.
        spec = sessions.SessionSpec(
            timezone="UTC",
            windows=sessions.NSE.windows,
            holiday=lambda day: day == date(2026, 9, 14),
        )
        monday = _utc(2026, 9, 14, 10)
        assert not spec.is_open(monday)
        assert spec.next_open(_utc(2026, 9, 12, 12)).date() == date(2026, 9, 15)


class TestAlwaysOpen:
    def test_every_hour_of_the_week_is_open(self):
        # The fallback for a venue we have not modelled: it must never claim a
        # market is closed, including across the Sunday-to-Monday seam.
        for day in range(7):
            for hour in (0, 12, 23):
                assert sessions.ALWAYS_OPEN.is_open(_utc(2026, 9, 7 + day, hour))
