"""
Indicator catalogue.

The VWAP tests are the point: this module and `SignalService` used to compute
VWAP two different ways, and the chat agent served the wrong one. There is one
implementation now, and it must be session-anchored.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.signals import indicators as ind
from app.signals.patterns import detect_patterns
from tests.conftest import make_bars


def test_catalogue_is_an_allow_list_not_getattr():
    names = ind.available_indicators()
    assert "rsi" in names and "vwap" in names
    assert ind.compute(make_bars([(1, 1, 1, 1)] * 60), ["__class__"])["_unknown_requested"] == ["__class__"]


def test_unknown_names_are_reported_with_the_available_list():
    out = ind.compute(make_bars([(1, 1, 1, 1)] * 60), ["rsi", "not_a_real_indicator"])
    assert out["_unknown_requested"] == ["not_a_real_indicator"]
    assert "_available" in out


def test_price_returns_last_close(trending_frame):
    out = ind.compute(trending_frame, ["price"])
    assert out["ltp"] == round(float(trending_frame["close"].iloc[-1]), 2)


def test_signal_set_produces_every_key_the_signal_path_reads(trending_frame):
    out = ind.compute(trending_frame, ind.SIGNAL_SET)
    for key in ("ltp", "rsi", "macd", "macd_signal", "ema20", "ema50", "adx", "atr", "supertrend_dir", "vwap"):
        assert key in out, f"{key} missing from SIGNAL_SET output"


def test_vwap_is_session_anchored_not_whole_frame():
    """Two sessions. A whole-frame cumsum would drag the second session's VWAP
    back towards the first session's prices; a session-anchored one must sit
    inside the second session's range."""
    day1 = make_bars([(100, 100, 100, 100)] * 20, start="2026-01-05 09:15")
    day2 = make_bars([(200, 200, 200, 200)] * 20, start="2026-01-06 09:15")
    frame = pd.concat([day1, day2])
    vwap = ind.compute(frame, ["vwap"])["vwap"]
    assert vwap == pytest.approx(200.0), "VWAP must reset at the session boundary"


def test_vwap_returns_none_rather_than_a_wrong_number_on_a_non_datetime_index():
    """The old fallback silently substituted the whole-frame cumsum the code
    itself documents as meaningless, and fed it to the LLM as fact."""
    frame = make_bars([(100, 101, 99, 100)] * 30).reset_index(drop=True)
    assert ind.compute(frame, ["vwap"])["vwap"] is None


def test_insufficient_warmup_yields_none_not_an_exception():
    tiny = make_bars([(100, 101, 99, 100)] * 5)
    out = ind.compute(tiny, ["rsi", "adx"])
    assert out.get("rsi") is None
    assert out.get("adx") is None
    assert "_failed" not in out, "missing warm-up is not a failure"


def test_volume_ratio_flags_a_spike():
    rows = [(100, 101, 99, 100)] * 21
    frame = make_bars(rows)
    frame.iloc[-1, frame.columns.get_loc("volume")] = 10_000.0
    out = ind.compute(frame, ["volume"])
    assert out["volume_ratio"] > 5


# -- patterns ---------------------------------------------------------------

def test_patterns_needs_a_minimum_frame():
    assert "error" in detect_patterns(make_bars([(1, 1, 1, 1)] * 3))


def test_doji_detected():
    rows = [(100, 101, 99, 100)] * 10
    rows.append((100.0, 101.0, 99.0, 100.02))  # body ~0 vs 2.0 range
    found = detect_patterns(make_bars(rows), lookback=3)
    assert any(p["pattern"] == "doji" for p in found["patterns"])


def test_bullish_engulfing_detected():
    rows = [(100, 101, 99, 100)] * 10
    rows.append((100.0, 100.2, 96.0, 96.5))   # down candle
    rows.append((96.0, 101.5, 95.8, 101.0))   # engulfs it
    found = detect_patterns(make_bars(rows), lookback=3)
    assert any(p["pattern"] == "bullish_engulfing" for p in found["patterns"])


def test_pattern_result_always_carries_the_weak_signal_caveat():
    res = detect_patterns(make_bars([(100, 101, 99, 100)] * 30))
    assert "weak signals in isolation" in res["note"]
