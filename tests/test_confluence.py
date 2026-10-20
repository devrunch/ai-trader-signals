"""The indicator panel behind the direction gate.

Every test here is a defect the four-voter version actually had, measured on
live data across nine instruments:

  * four mirrored voters gave five possible scores, so the gate reduced to
    "3 of 4 agree";
  * a 2-2 split scored 50% for BUY and 50% for SELL, making it a permanent
    no-trade in both directions — Nifty refused long and short with the
    identical number and looked like two assessments when it was one tie;
  * `macd > macd_signal` read UP on 8 of the 9 instruments, so a quarter of
    the signal was an indicator saying the same thing everywhere;
  * three of the four — EMA cross, SuperTrend, RSI-above-50 — were the same
    reading of trend.

The old fixtures passed unchanged because they were uniformly bullish or
uniformly bearish, which is exactly the case that never discriminates.
"""
from __future__ import annotations

import pytest

from app.signals import confluence as C

# A real reading: trend split, momentum leaning up, price mid-range. The shape
# that scored a flat 50/50 before and told the user nothing.
MIXED = {
    "ltp": 22888.95,
    "ema20": 22891.0, "ema50": 23297.0, "ema200": 23500.0,
    "supertrend_dir": 1, "psar": 22700.0,
    "di_plus": 16.7, "di_minus": 22.8, "aroon_osc": -14.3,
    "rsi": 47.7, "macd": -3.9, "macd_signal": -4.2,
    "stoch_k": 42.9, "stoch_d": 33.4, "williams_r": -43.9,
    "cci": -49.7, "roc": -0.03, "tsi": -6.3,
    "bb_mid": 22850.0, "kc_mid": 22860.0, "dc_mid": 22800.0, "vwap": 22870.0,
}


class TestResolution:
    def test_it_asks_far_more_than_four_indicators(self):
        """Four voters gave five possible scores. The threshold is 75, so the
        gate was "3 of 4 agree" and nothing finer."""
        report = C.measure("BUY", MIXED)
        assert len(report.votes) >= 14

    def test_a_mixed_market_is_no_longer_a_flat_fifty(self):
        """The number that made every instrument look identical."""
        report = C.measure("BUY", MIXED)
        assert report.ratio != 50.0

    def test_it_groups_by_what_each_indicator_measures(self):
        report = C.measure("BUY", MIXED)
        assert set(report.families) == {"trend", "momentum", "location"}

    def test_trend_cannot_outvote_the_panel_by_being_measured_six_ways(self):
        """Six trend indicators and seven momentum ones must not make trend
        the louder family. Each contributes one equal share."""
        report = C.measure("BUY", MIXED)
        trend_members = [v for v in report.votes if v.family == "trend"]
        assert len(trend_members) >= 5
        # The score is the mean of family shares, so it sits between them.
        shares = list(report.families.values())
        assert min(shares) <= report.ratio <= max(shares)


class TestWhatItReports:
    def test_it_names_which_indicators_disagreed(self):
        """The old gate returned "quant confluence 50% < 75%" and nothing
        else, so asked which indicators disagreed the agent paraphrased from
        the chart instead — it could not see."""
        report = C.measure("BUY", MIXED)
        assert report.disagreed
        assert all(isinstance(n, str) and n for n in report.disagreed)
        assert set(report.agreed).isdisjoint(report.disagreed)

    def test_every_vote_carries_the_reading_behind_it(self):
        report = C.measure("BUY", MIXED)
        rsi = next(v for v in report.votes if v.name == "rsi")
        assert "47.7" in rsi.detail

    def test_the_summary_says_what_was_against_it(self):
        line = C.measure("BUY", MIXED).summary()
        assert "against:" in line
        assert "%" in line

    def test_the_dict_form_explains_the_weighting(self):
        """A caller handed only a percentage will present it as a verdict."""
        out = C.measure("BUY", MIXED).to_dict()
        assert "trend cannot outvote" in out["how"]
        assert out["families"] and out["votes"]


class TestDirectionSymmetry:
    def test_a_direction_and_its_opposite_still_sum_to_a_hundred(self):
        """Inherent: every voter is a mirror. Kept deliberately — what changed
        is the resolution, so a hard tie is now rare instead of routine."""
        buy = C.measure("BUY", MIXED).ratio
        sell = C.measure("SELL", MIXED).ratio
        assert buy + sell == pytest.approx(100.0)

    def test_the_tie_that_deadlocked_both_directions_is_gone(self):
        """Nifty scored 50% long and 50% short, so neither could ever reach
        75 and the instrument was untradeable in principle, not in fact."""
        buy = C.measure("BUY", MIXED)
        sell = C.measure("SELL", MIXED)
        assert not (buy.ratio == 50.0 and sell.ratio == 50.0)


class TestMissingData:
    def test_an_absent_indicator_does_not_vote_against(self):
        """Absent is not neutral and not bearish. Counting a missing indicator
        as disagreement is how a thin feed quietly becomes a sell signal."""
        full = C.measure("BUY", MIXED)
        thin = C.measure("BUY", {k: v for k, v in MIXED.items()
                                 if k not in ("cci", "tsi", "roc")})
        assert len(thin.votes) == len(full.votes) - 3
        assert "cci" not in [v.name for v in thin.votes]

    def test_nothing_at_all_scores_zero_with_no_voters(self):
        report = C.measure("BUY", {})
        assert report.votes == [] and report.ratio == 0.0

    def test_a_family_with_no_members_is_simply_absent(self):
        """Not counted as 0% — that would drag the mean down for data we never
        had."""
        price_only = {"ltp": 100.0, "bb_mid": 95.0, "kc_mid": 96.0, "dc_mid": 94.0}
        report = C.measure("BUY", price_only)
        assert set(report.families) == {"location"}
        assert report.ratio == 100.0


class TestTheGateItself:
    def test_the_rejection_names_the_dissenters(self):
        from app.signals import validation as v
        ok, reason = v.quant_confluence("BUY", MIXED, 0.75)
        assert not ok
        assert "against:" in reason

    def test_it_still_fails_closed_with_no_indicators(self):
        from app.signals import validation as v
        ok, reason = v.quant_confluence("BUY", {}, 0.75)
        assert not ok and "no indicators" in reason

    def test_a_genuinely_aligned_panel_passes(self):
        from app.signals import validation as v
        aligned = dict(MIXED, ema20=23500.0, ema50=23400.0, ema200=23000.0,
                       supertrend_dir=1, psar=22000.0, di_plus=30.0, di_minus=12.0,
                       aroon_osc=80.0, rsi=62.0, macd=5.0, macd_signal=3.0,
                       stoch_k=70.0, stoch_d=60.0, williams_r=-20.0, cci=120.0,
                       roc=1.5, tsi=8.0, bb_mid=22000.0, kc_mid=22100.0,
                       dc_mid=22050.0, vwap=22080.0)
        assert v.quant_confluence("BUY", aligned, 0.75) == (True, None)
