"""The calendar feed, and the honesty rules around it.

The failure this guards against is not an outage -- it is a 200 with the wrong
shape, which reads as "nothing scheduled" on the morning of a Fed decision.
"""
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from app.events import calendar as cal

ROW = {
    "title": "Federal Funds Rate", "country": "USD", "impact": "High",
    "date": "2026-09-16T14:00:00-04:00", "forecast": "4.00%", "previous": "3.75%",
}


def _rows(n: int = 30, **overrides) -> list[dict]:
    return [dict(ROW, **overrides) for _ in range(n)]


def _response(payload) -> httpx.Response:
    # raise_for_status needs the request attached, which the bare constructor
    # does not set.
    return httpx.Response(200, json=payload,
                          request=httpx.Request("GET", cal.FEED_URL))


class TestParse:
    def test_times_are_normalised_to_utc(self):
        [event] = cal.parse([ROW])
        assert event.when == datetime(2026, 9, 16, 18, 0, tzinfo=UTC)

    def test_a_missing_forecast_is_none_not_an_empty_string(self):
        # "" would render as a forecast of nothing; None renders as absent.
        [event] = cal.parse([dict(ROW, forecast="", previous="  ")])
        assert event.forecast is None
        assert event.previous is None

    def test_a_row_with_no_usable_date_is_dropped(self):
        # Everything downstream schedules off this timestamp.
        assert cal.parse([dict(ROW, date="not a date")]) == []

    def test_an_unknown_impact_degrades_to_low_rather_than_raising(self):
        [event] = cal.parse([dict(ROW, impact="Catastrophic")])
        assert event.impact is cal.Impact.LOW

    def test_events_come_back_in_time_order(self):
        rows = [dict(ROW, date="2026-09-17T10:00:00-04:00"),
                dict(ROW, date="2026-09-16T10:00:00-04:00")]
        assert [e.when for e in cal.parse(rows)] == sorted(e.when for e in cal.parse(rows))


class TestFetch:
    @pytest.mark.asyncio
    async def test_a_good_feed_returns_events(self):
        client = AsyncMock()
        client.get.return_value = _response(_rows())
        assert len(await cal.fetch_week(client)) == 30

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_none_not_an_empty_calendar(self):
        # None and [] mean different things: "we do not know" versus "nothing
        # is scheduled". Collapsing them is how a Fed day goes unannounced.
        client = AsyncMock()
        client.get.side_effect = httpx.ConnectError("down")
        assert await cal.fetch_week(client) is None

    @pytest.mark.asyncio
    async def test_a_suspiciously_short_feed_is_rejected(self):
        # The characteristic free-feed failure is a 200 with the wrong shape.
        client = AsyncMock()
        client.get.return_value = _response(_rows(3))
        assert await cal.fetch_week(client) is None

    @pytest.mark.asyncio
    async def test_a_feed_that_is_not_a_list_is_rejected(self):
        client = AsyncMock()
        client.get.return_value = _response({"error": "nope"})
        assert await cal.fetch_week(client) is None

    @pytest.mark.asyncio
    async def test_rows_that_all_fail_to_parse_are_rejected(self):
        client = AsyncMock()
        client.get.return_value = _response(_rows(30, date="broken"))
        assert await cal.fetch_week(client) is None


class TestUpcoming:
    def test_past_events_are_excluded(self):
        events = cal.parse(_rows(2))
        after = datetime(2026, 9, 17, tzinfo=UTC)
        assert cal.upcoming(events, now=after) == []

    def test_filters_to_the_currencies_asked_for(self):
        events = cal.parse([ROW, dict(ROW, country="JPY")])
        before = datetime(2026, 9, 1, tzinfo=UTC)
        out = cal.upcoming(events, now=before, currencies={"USD"})
        assert [e.currency for e in out] == ["USD"]
