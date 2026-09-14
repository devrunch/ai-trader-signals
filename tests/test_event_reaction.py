"""The reaction study, whose only value is being honest about its sample.

A median over three prints spanning two rate regimes looks exactly like
evidence and is not, so the interesting behaviour here is where it declines to
speak rather than where it speaks.
"""
from datetime import UTC, datetime

from app.events import reaction
from app.events.reaction import Instance, Surprise


def bars(base: float, after: float, *, release_ms: int, count: int = 70) -> list[dict]:
    """One bar a minute: flat at `base` before the release, `after` from then."""
    out = [{"t": release_ms - 60_000 * i, "o": base, "h": base, "l": base, "c": base}
           for i in range(3, 0, -1)]
    out += [{"t": release_ms + 60_000 * i, "o": after, "h": after, "l": after, "c": after}
            for i in range(1, count + 1)]
    return out


class TestParseNumber:
    def test_reads_percentages_and_suffixes(self):
        assert reaction.parse_number("4.00%") == 4.0
        assert reaction.parse_number("8.3K") == 8300.0
        assert reaction.parse_number("-11.0K") == -11000.0

    def test_unreadable_values_are_none_not_zero(self):
        # A print we cannot read must not become a surprise of zero.
        assert reaction.parse_number("n/a") is None
        assert reaction.parse_number("") is None
        assert reaction.parse_number(None) is None


class TestClassify:
    def test_beat_miss_and_inline(self):
        assert reaction.classify(0.4, 0.3) is Surprise.BEAT
        assert reaction.classify(0.2, 0.3) is Surprise.MISS
        assert reaction.classify(0.3, 0.3) is Surprise.INLINE


class TestMoveAfter:
    RELEASE = datetime(2026, 9, 11, 12, 30, tzinfo=UTC)

    def test_measures_from_the_last_bar_before_the_release(self):
        ms = int(self.RELEASE.timestamp() * 1000)
        move = reaction.move_after(bars(100.0, 101.0, release_ms=ms), self.RELEASE, 60)
        assert move == 1.0

    def test_a_window_that_stops_short_reports_nothing(self):
        # Otherwise a truncated download reads as a small reaction.
        ms = int(self.RELEASE.timestamp() * 1000)
        short = bars(100.0, 101.0, release_ms=ms, count=5)
        assert reaction.move_after(short, self.RELEASE, 60) is None
        assert reaction.move_after(short, self.RELEASE, 5) == 1.0

    def test_no_bars_at_all_is_none(self):
        assert reaction.move_after([], self.RELEASE, 60) is None


def instance(surprise: Surprise, move: float | None) -> Instance:
    return Instance(when=datetime(2026, 1, 1, tzinfo=UTC), actual=1.0, forecast=0.5,
                    surprise=surprise, move_15m=move, move_60m=move)


class TestStats:
    def test_a_thin_sample_refuses_to_summarise(self):
        stats = reaction.aggregate("USD|X", "XAUUSD",
                                   [instance(Surprise.BEAT, -0.5) for _ in range(3)])
        assert stats.enough is False
        assert stats.summary() is None

    def test_an_all_inline_history_has_nothing_to_say_about_surprises(self):
        # The Fed funds rate prints exactly as forecast almost every time, so
        # there is no surprise distribution to report -- which is itself the
        # useful finding, and must not be dressed up as one.
        stats = reaction.aggregate("USD|Federal Funds Rate", "XAUUSD",
                                   [instance(Surprise.INLINE, 0.1) for _ in range(10)])
        assert stats.enough is True
        assert stats.summary() is None

    def test_a_real_distribution_is_reported_with_its_count(self):
        moves = [instance(Surprise.BEAT, m) for m in (-0.4, -0.7, -0.2, 0.1)]
        moves += [instance(Surprise.MISS, m) for m in (0.5, 0.3, 0.6)]
        summary = reaction.aggregate("USD|Core CPI m/m", "XAUUSD", moves).summary()
        assert "above forecast" in summary and "3 of 4" in summary
        assert "below forecast" in summary and "3 of 3" in summary

    def test_instances_without_a_priced_window_are_excluded_from_the_sample(self):
        mixed = [instance(Surprise.BEAT, -0.4), instance(Surprise.BEAT, None)]
        assert reaction.aggregate("USD|X", "XAUUSD", mixed).sample == 1


class TestInstancesFromHistory:
    def test_rows_without_both_actual_and_forecast_are_dropped(self):
        rows = [
            {"DateTime": "2026-01-13T13:30:00+00:00", "Actual": "0.3", "Forecast": "0.2"},
            {"DateTime": "2025-12-10T13:30:00+00:00", "Actual": "", "Forecast": "0.2"},
            {"DateTime": "2025-11-13T13:30:00+00:00", "Actual": "0.2", "Forecast": ""},
        ]
        out = reaction.instances_from_history(rows)
        assert len(out) == 1
        assert out[0].surprise is Surprise.BEAT

    def test_newest_first(self):
        rows = [
            {"DateTime": "2025-01-13T13:30:00+00:00", "Actual": "0.3", "Forecast": "0.2"},
            {"DateTime": "2026-01-13T13:30:00+00:00", "Actual": "0.3", "Forecast": "0.2"},
        ]
        out = reaction.instances_from_history(rows)
        assert out[0].when.year == 2026
