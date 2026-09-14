"""The agenda, whose whole job is to be right about direction or say nothing.

The 29 inverted events in the glossary are the reason this is tested rather
than eyeballed: "beat = good for the currency" is true for most releases and
exactly backwards for unemployment.
"""
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.events import specs
from app.events.agenda import build
from app.events.calendar import CalendarEvent, Impact


def event(title="Federal Funds Rate", currency="USD", impact=Impact.HIGH,
          forecast="4.00%", previous="3.75%") -> CalendarEvent:
    return CalendarEvent(title=title, currency=currency,
                         when=datetime(2026, 9, 16, 18, 0, tzinfo=UTC),
                         impact=impact, forecast=forecast, previous=previous)


class TestSpecs:
    def test_the_glossary_loaded(self):
        assert specs.known_events() > 500

    def test_a_normal_release_reads_a_beat_as_currency_positive(self):
        assert specs.spec_for("USD", "Core CPI m/m").beat_is_bullish is True

    def test_an_inverted_release_reads_a_beat_as_currency_negative(self):
        # Higher unemployment is bad for the currency. A hardcoded rule would
        # print the opposite branch for all 29 events of this shape.
        assert specs.spec_for("CAD", "Unemployment Rate").beat_is_bullish is False

    def test_an_unknown_release_has_no_opinion(self):
        assert specs.spec_for("USD", "Not A Real Release") is None


class TestAgenda:
    def test_a_dollar_event_carries_the_gold_call(self):
        text = build([event()])
        assert "USD stronger -> *gold down*" in text
        assert "USD weaker -> *gold up*" in text

    def test_a_non_dollar_event_does_not_claim_to_move_gold(self):
        # Sterling inflation moves cable. Routing it to gold through "stronger
        # currency" is the kind of stretch that makes a trader stop reading.
        text = build([event(title="CPI y/y", currency="GBP", forecast="3.1%")])
        assert "GBP stronger" in text
        assert "gold" not in text.lower().split("_times")[0]

    def test_an_event_with_no_known_direction_gets_no_branches(self):
        text = build([event(title="Unscheduled Summit", forecast=None, previous=None)])
        assert "->" not in text

    def test_the_expected_value_is_shown_against_the_previous(self):
        assert "expected *4.00%*  (previous 3.75%)" in build([event()])

    def test_a_missing_forecast_is_not_rendered_as_a_number(self):
        text = build([event(forecast=None)])
        assert "expected" not in text

    def test_a_quiet_week_returns_nothing_rather_than_an_empty_push(self):
        assert build([]) is None

    def test_medium_impact_events_are_listed_not_detailed(self):
        text = build([event(), event(title="Retail Sales m/m", impact=Impact.MEDIUM,
                                     forecast="0.8%")])
        assert "also on the calendar" in text
        assert "[med]" in text

    def test_times_are_shown_in_his_timezone(self):
        # 18:00 UTC is 22:00 in Dubai -- an event he can act on, at an hour
        # that means something to him.
        assert "22:00" in build([event()])


class TestWatchdog:
    """The desk reporting its own failure, which is the only kind of failure
    a single-box time-critical path can afford."""

    @pytest.mark.asyncio
    async def test_a_brief_that_never_fired_is_reported_and_fails_the_check(self):
        from app.events import watchdog

        with patch.object(watchdog, "missed", AsyncMock(return_value=["brief:USD:CPI:202609161800"])), \
             patch("app.events.telegram.send", AsyncMock(return_value=True)) as send:
            result = await watchdog.sweep()

        # ok False is what fails this job's own Healthchecks ping, which is the
        # second alarm: the first is the message he gets.
        assert result["ok"] is False
        assert result["missed"] == 1
        assert "did not send" in send.await_args.args[0]

    @pytest.mark.asyncio
    async def test_a_clean_sweep_says_nothing_to_the_user(self):
        from app.events import watchdog

        with patch.object(watchdog, "missed", AsyncMock(return_value=[])), \
             patch("app.events.telegram.send", AsyncMock()) as send:
            result = await watchdog.sweep()

        assert result == {"ok": True, "missed": 0}
        send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_sweep_that_cannot_run_is_a_failure_not_a_clean_bill(self):
        # Redis down must not read as "every brief fired".
        from app.events import watchdog

        with patch.object(watchdog, "missed", AsyncMock(side_effect=OSError("redis gone"))):
            result = await watchdog.sweep()

        assert result["ok"] is False
        assert "OSError" in result["error"]
