"""
The six gates that decide whether a trade is emitted.

Every one FAILS CLOSED — a risk filter that disables itself when its input is
missing is worse than no filter, because it is invisible. Those are the cases
that matter most here.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.signals import validation as v

SETTINGS = SimpleNamespace(
    max_entry_drift_atr=0.25,
    min_quant_confluence=0.75,
    min_reward_risk=1.5,
    min_stop_atr_multiple=1.0,
    max_stop_atr_multiple=3.5,
    max_buy_rsi=75.0,
    min_sell_rsi=25.0,
)

BULLISH = {"ema20": 105.0, "ema50": 100.0, "supertrend_dir": 1, "macd": 1.2, "macd_signal": 0.8, "rsi": 62.0}
BEARISH = {"ema20": 100.0, "ema50": 105.0, "supertrend_dir": -1, "macd": 0.8, "macd_signal": 1.2, "rsi": 38.0}


def buy(entry=100.0, target=110.0, stop=96.0, confidence=0.7):
    return {"signal_type": "BUY", "confidence": confidence, "entry_price": entry,
            "target_price": target, "stop_loss": stop, "reasoning": "x"}


# -- HOLD ------------------------------------------------------------------

def test_hold_is_rejected_before_any_price_validation():
    ok, reason = v.validate({"signal_type": "HOLD"}, BULLISH, atr=4.0, ltp=100.0, settings=SETTINGS)
    assert not ok and "HOLD" in reason


# -- price ordering --------------------------------------------------------

@pytest.mark.parametrize("direction,entry,target,stop", [
    ("BUY", 100, 90, 96),    # target below entry
    ("BUY", 100, 110, 104),  # stop above entry
    ("SELL", 100, 110, 104), # target above entry for a SELL
    ("SELL", 100, 90, 96),   # stop below entry for a SELL
])
def test_price_ordering_rejects_wrong_side(direction, entry, target, stop):
    ok, reason = v.price_ordering(direction, entry, target, stop)
    assert not ok and "wrong side" in reason


def test_price_ordering_accepts_correct_sides():
    assert v.price_ordering("BUY", 100, 110, 96) == (True, None)
    assert v.price_ordering("SELL", 100, 90, 104) == (True, None)


def test_unparseable_prices_are_rejected():
    prices, err = v.parse_prices({"entry_price": "abc", "target_price": 1, "stop_loss": 2})
    assert prices is None and err


# -- entry drift -----------------------------------------------------------

def test_entry_far_from_ltp_is_rejected():
    ok, reason = v.entry_drift(entry=105.0, ltp=100.0, atr=4.0, max_drift_atr=0.25)
    assert not ok and "from LTP" in reason


def test_entry_within_drift_budget_is_accepted():
    assert v.entry_drift(entry=100.5, ltp=100.0, atr=4.0, max_drift_atr=0.25) == (True, None)


# -- quant confluence ------------------------------------------------------

def test_confluence_passes_when_indicators_agree():
    assert v.quant_confluence("BUY", BULLISH, 0.75) == (True, None)


def test_confluence_rejects_when_indicators_disagree():
    ok, reason = v.quant_confluence("BUY", BEARISH, 0.75)
    assert not ok and "confluence" in reason


def test_confluence_fails_closed_when_no_indicator_can_vote():
    """Not 'unanimous' — no confirmation at all."""
    ok, reason = v.quant_confluence("BUY", {}, 0.75)
    assert not ok and "no indicators" in reason


def test_confluence_partial_availability_still_requires_the_ratio():
    """Only RSI available and it disagrees -> 0/1, must reject."""
    ok, _ = v.quant_confluence("BUY", {"rsi": 40.0}, 0.75)
    assert not ok


# -- long only -------------------------------------------------------------

def test_sell_is_rejected_because_phase_1_is_long_only():
    """A SELL is a short sale. The paper account has no short path, so the
    signal is an instruction the user cannot follow — and it used to be counted
    in the published win rate anyway."""
    sell = {"signal_type": "SELL", "confidence": 0.9, "entry_price": 100.0,
            "target_price": 90.0, "stop_loss": 104.0, "reasoning": "x"}
    ok, reason = v.validate(sell, BEARISH, atr=4.0, ltp=100.0, settings=SETTINGS)
    assert not ok and "long-only" in reason


def test_long_only_leaves_buys_alone():
    assert v.long_only({"signal_type": "BUY"}) == (True, None)


# -- RSI extreme veto ------------------------------------------------------

def test_buy_at_an_extreme_rsi_is_vetoed_even_with_full_confluence():
    """RSI 85 is a market that has already run. The confluence gate counts it
    as a vote FOR buying (rsi > 50), which is the bug this veto exists for."""
    stretched = {**BULLISH, "rsi": 85.0}
    assert v.quant_confluence("BUY", stretched, 0.75) == (True, None)

    ok, reason = v.validate(buy(), stretched, atr=4.0, ltp=100.0, settings=SETTINGS)
    assert not ok and "overextended" in reason


def test_sell_at_an_extreme_rsi_is_vetoed():
    ok, reason = v.rsi_extreme("SELL", {"rsi": 18.0}, max_buy_rsi=75.0, min_sell_rsi=25.0)
    assert not ok and "overextended" in reason


@pytest.mark.parametrize("rsi", [50.0, 74.9, 75.0])
def test_buy_below_the_rsi_ceiling_is_allowed(rsi):
    assert v.rsi_extreme("BUY", {"rsi": rsi}, max_buy_rsi=75.0, min_sell_rsi=25.0) == (True, None)


def test_rsi_veto_fails_closed_when_rsi_is_missing():
    ok, reason = v.rsi_extreme("BUY", {}, max_buy_rsi=75.0, min_sell_rsi=25.0)
    assert not ok and "unavailable" in reason


# -- reward:risk -----------------------------------------------------------

def test_reward_risk_below_minimum_is_rejected():
    ok, reason = v.reward_risk(entry=100, target=104, stop=96, minimum=1.5)
    assert not ok and "R:R" in reason


def test_reward_risk_uses_absolute_legs_so_sell_works():
    assert v.reward_risk(entry=100, target=90, stop=104, minimum=1.5) == (True, None)


# -- stop sizing -----------------------------------------------------------

def test_stop_tighter_than_min_atr_multiple_is_rejected():
    ok, reason = v.stop_size(entry=100, stop=99.5, atr=4.0, min_multiple=1.0, max_multiple=3.5)
    assert not ok and "tighter" in reason


def test_stop_wider_than_max_atr_multiple_is_rejected():
    ok, reason = v.stop_size(entry=100, stop=80.0, atr=4.0, min_multiple=1.0, max_multiple=3.5)
    assert not ok and "wider" in reason


def test_stop_within_the_band_is_accepted():
    assert v.stop_size(entry=100, stop=96.0, atr=4.0, min_multiple=1.0, max_multiple=3.5) == (True, None)


# -- end to end ------------------------------------------------------------

def test_a_clean_buy_passes_every_gate():
    assert v.validate(buy(), BULLISH, atr=4.0, ltp=100.0, settings=SETTINGS) == (True, None)


def test_missing_atr_driven_gates_still_reject_a_bad_signal():
    """Indicators absent -> confluence has no voters -> rejected, not passed."""
    ok, _ = v.validate(buy(), {}, atr=4.0, ltp=100.0, settings=SETTINGS)
    assert not ok
