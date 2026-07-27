"""
Market-hours gating and the intraday holding cap.

`app/market/calendar.py` was written, tested and then never called, so the
screener ran at 09:00 on yesterday's bars, at 15:45 after the close, and on
every trading holiday — each of those runs producing real signals that were
stored and scored. These tests hold the wiring in place.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from app.signals.backtest.runner import session_forward_window
from app.worker import tasks as tasks_mod


class _FrozenCalendar:
    """Stand-in for `app.market.calendar` with a fixed answer."""

    def __init__(self, *, open_: bool, trading_day: bool = True):
        self._open = open_
        self._trading_day = trading_day

    def is_market_open(self, now=None) -> bool:
        return self._open

    def is_trading_day(self, day=None) -> bool:
        return self._trading_day

    def session_state(self, now=None) -> dict:
        return {
            "is_open": self._open,
            "is_trading_day": self._trading_day,
            "is_holiday": not self._trading_day,
            "next_open": "2026-03-11T09:15:00+05:30",
        }


# ---------------------------------------------------------------------------
# Screener gating
# ---------------------------------------------------------------------------

def test_screener_does_no_work_when_the_market_is_shut(monkeypatch):
    monkeypatch.setattr(tasks_mod, "market_calendar", _FrozenCalendar(open_=False, trading_day=False))

    def _explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("the watchlist must not be fetched while the market is shut")

    monkeypatch.setattr(tasks_mod, "_fetch_watchlist", _explode)
    monkeypatch.setattr(tasks_mod, "service", _explode)

    result = tasks_mod.run_screener()

    assert result["market_closed"] is True
    assert result["signals"] == 0
    # The reason is machine-readable, like every other screener skip reason —
    # "0 signals" must stay distinguishable from "0 signals because we didn't look".
    assert result["skipped_reasons"] == {"market_closed": 1}


def test_screener_runs_when_the_market_is_open(monkeypatch):
    monkeypatch.setattr(tasks_mod, "market_calendar", _FrozenCalendar(open_=True))
    monkeypatch.setattr(tasks_mod, "_fetch_watchlist", lambda: [{"symbol": "TCS", "exchange": "NSE"}])

    def _fake_run_async(coro):
        coro.close()          # never awaited — stops the "coroutine was never awaited" warning
        return ["TCS"], {}

    monkeypatch.setattr(tasks_mod, "run_async", _fake_run_async)

    result = tasks_mod.run_screener()

    assert result["screened"] == 1
    assert result["symbols"] == ["TCS"]
    assert "market_closed" not in result


def test_square_off_is_skipped_on_a_non_trading_day(monkeypatch):
    monkeypatch.setattr(tasks_mod, "market_calendar", _FrozenCalendar(open_=False, trading_day=False))

    def _explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("no square-off request on a holiday")

    monkeypatch.setattr(tasks_mod.httpx, "post", _explode)

    assert tasks_mod.square_off_positions() == {"skipped": "not_a_trading_day"}


def test_square_off_posts_to_the_internal_endpoint(monkeypatch):
    monkeypatch.setattr(tasks_mod, "market_calendar", _FrozenCalendar(open_=False, trading_day=True))
    calls: list[tuple] = []

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"closed": 2, "failed": 0, "details": []}

    def _post(url, **kwargs):
        calls.append((url, kwargs))
        return _Resp()

    monkeypatch.setattr(tasks_mod.httpx, "post", _post)

    result = tasks_mod.square_off_positions()

    assert result["closed"] == 2
    assert calls[0][0].endswith("/api/internal/paper/square-off")
    assert "x-internal-key" in calls[0][1]["headers"]


def test_square_off_failure_is_reported_not_raised(monkeypatch):
    import httpx

    monkeypatch.setattr(tasks_mod, "market_calendar", _FrozenCalendar(open_=False, trading_day=True))

    def _post(url, **kwargs):
        raise httpx.ConnectError("api unreachable")

    monkeypatch.setattr(tasks_mod.httpx, "post", _post)

    # A raising Celery task retries and logs a traceback; what is wanted is a
    # recorded, visible failure, because the consequence is positions held
    # overnight.
    assert "error" in tasks_mod.square_off_positions()


# ---------------------------------------------------------------------------
# Backtest holding period
# ---------------------------------------------------------------------------

def _two_session_frame() -> pd.DataFrame:
    """Eight 15m bars on 10 March, then eight more on the 11th."""
    day1 = pd.date_range("2026-03-10 14:00", periods=8, freq="15min")
    day2 = pd.date_range("2026-03-11 09:15", periods=8, freq="15min")
    idx = day1.append(day2)
    n = len(idx)
    return pd.DataFrame(
        {"open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
         "close": [100.0] * n, "volume": [1000.0] * n},
        index=idx,
    )


def test_forward_window_stops_at_the_end_of_the_signals_own_session():
    df = _two_session_frame()
    # Signal on the sixth bar of day one: two bars of that session remain, and
    # a fixed 40-bar window would have run all the way through the next day.
    forward = session_forward_window(df, idx=5, forward_bars=40)

    assert len(forward) == 2
    assert forward.index.max() < pd.Timestamp("2026-03-11")


def test_forward_window_is_empty_for_a_signal_on_the_last_bar_of_a_session():
    df = _two_session_frame()
    forward = session_forward_window(df, idx=7, forward_bars=40)

    # Empty means the evaluator returns OPEN, which is exactly what a 15:20
    # square-off produces — not a trade resolved by tomorrow's gap.
    assert forward.empty


def test_forward_window_still_honours_the_bar_cap_within_a_session():
    df = _two_session_frame()
    forward = session_forward_window(df, idx=0, forward_bars=3)

    assert len(forward) == 3


def test_forward_window_handles_tz_aware_bars():
    df = _two_session_frame()
    df.index = df.index.tz_localize("Asia/Kolkata")
    forward = session_forward_window(df, idx=5, forward_bars=40)

    assert len(forward) == 2


def test_forward_window_falls_back_to_the_bar_cap_without_a_datetime_index():
    df = _two_session_frame().reset_index(drop=True)
    forward = session_forward_window(df, idx=0, forward_bars=4)

    assert len(forward) == 4


@pytest.mark.parametrize(
    "moment,expected",
    [
        (datetime(2026, 3, 10, 9, 0), False),    # before the open
        (datetime(2026, 3, 10, 9, 20), True),    # first beat slot
        (datetime(2026, 3, 10, 15, 15), True),   # last beat slot
        (datetime(2026, 3, 10, 15, 45), False),  # after the close
        (datetime(2026, 3, 4, 11, 0), False),    # Holi — a published NSE holiday
        (datetime(2026, 3, 8, 11, 0), False),    # Sunday
    ],
)
def test_beat_window_sits_inside_the_real_session(moment, expected):
    """The beat slots the schedule fires on must all be genuinely open."""
    from app.market.calendar import is_market_open

    assert is_market_open(moment) is expected
