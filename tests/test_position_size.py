"""
Position sizing — the tool with the worst consequences when it is wrong.

Three independent limits apply and the tightest must win. An earlier version
invented a ₹1,00,000 account value when the portfolio fetch failed and returned
confident, fully-populated sizing; a ₹20,000 account would have been handed 5x
oversize with no indication anything was wrong. That path is tested here first.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.signals.agent import portfolio_tools as pt

SETTINGS = SimpleNamespace(max_position_pct=20.0, min_risk_pct=0.25, max_risk_pct=2.0)


def ctx(total=100_000.0, cash=100_000.0, positions=None):
    return {"portfolio": {"totalValue": total, "cashBalance": cash, "initialCapital": total},
            "positions": positions or []}


# -- the refusal paths -----------------------------------------------------

def test_refuses_to_size_when_the_context_failed():
    out = pt.position_size({"error": "Could not load the user's portfolio right now."}, 100, 95, 1, SETTINGS)
    assert "error" in out
    assert "recommended_shares" not in out


def test_refuses_to_size_when_the_account_value_is_missing():
    out = pt.position_size({"portfolio": {"cashBalance": 5000}}, 100, 95, 1, SETTINGS)
    assert "error" in out and "cannot size a position safely" in out["error"]


def test_refuses_when_entry_equals_stop():
    out = pt.position_size(ctx(), entry=100, stop=100, risk_pct=1, settings=SETTINGS)
    assert "error" in out


# -- the three limits ------------------------------------------------------

def test_risk_is_the_binding_limit():
    # 1% of 100k = 1000 risk budget; 5/share risk -> 200 shares.
    # cash allows 1000, concentration allows 20k/100 = 200 ... make risk tighter.
    out = pt.position_size(ctx(), entry=100, stop=90, risk_pct=1, settings=SETTINGS)
    assert out["shares_allowed_by_risk"] == 100          # 1000 / 10
    assert out["recommended_shares"] == 100
    assert out["limited_by"] == "risk"


def test_cash_is_the_binding_limit():
    out = pt.position_size(ctx(total=100_000, cash=1_000), entry=100, stop=99, risk_pct=2, settings=SETTINGS)
    assert out["shares_allowed_by_cash"] == 10
    assert out["recommended_shares"] == 10
    assert out["limited_by"] == "cash"


def test_concentration_caps_a_tight_stop_on_an_expensive_stock():
    """A ₹3,000 stock with a tight stop would size to ~99% of the account and
    still report '0.5% risk' — true only if the stop holds, which a gap ignores."""
    out = pt.position_size(ctx(total=100_000, cash=100_000), entry=3000, stop=2995,
                           risk_pct=2, settings=SETTINGS)
    assert out["shares_allowed_by_concentration"] == 6   # 20% of 100k / 3000
    assert out["recommended_shares"] == 6
    assert out["limited_by"] == "concentration"
    assert out["position_pct_of_account"] <= SETTINGS.max_position_pct


def test_requested_risk_is_clamped_at_both_ends():
    high = pt.position_size(ctx(), 100, 90, risk_pct=25, settings=SETTINGS)
    assert high["risk_pct_requested"] == SETTINGS.max_risk_pct
    low = pt.position_size(ctx(), 100, 90, risk_pct=0.01, settings=SETTINGS)
    assert low["risk_pct_requested"] == SETTINGS.min_risk_pct


def test_reported_actual_risk_matches_the_shares_actually_recommended():
    out = pt.position_size(ctx(), entry=100, stop=90, risk_pct=1, settings=SETTINGS)
    assert out["actual_risk_amount"] == out["recommended_shares"] * out["per_share_risk"]


def test_the_gap_caveat_is_always_present():
    out = pt.position_size(ctx(), 100, 90, 1, SETTINGS)
    assert "gap" in out["caveat"].lower()


# -- exposure and aggregate risk ------------------------------------------

def test_exposure_reports_no_positions_cleanly():
    assert "note" in pt.analyse_exposure(ctx())


def test_exposure_percentages_use_current_price_when_available():
    c = ctx(total=100_000, cash=50_000, positions=[
        {"symbol": "RELIANCE", "quantity": 10, "averageCost": 2000, "currentPrice": 2500},
    ])
    out = pt.analyse_exposure(c)
    assert out["deployed_value"] == 25_000
    assert out["by_symbol"][0]["pct_of_account"] == 25.0


def test_exposure_falls_back_to_average_cost_without_a_current_price():
    c = ctx(total=100_000, cash=80_000, positions=[
        {"symbol": "TCS", "quantity": 5, "averageCost": 4000},
    ])
    assert pt.analyse_exposure(c)["deployed_value"] == 20_000


def test_portfolio_risk_propagates_a_context_error():
    assert "error" in pt.portfolio_risk({"error": "nope"})


def test_portfolio_risk_leads_with_the_truth_that_no_stops_exist():
    """The tool used to answer "if every stop hits you lose ₹1,000". There is no
    STOP order type and no monitoring loop, so no stops exist — that answer
    converted unbounded risk into a specific, reassuring figure."""
    c = ctx(total=100_000, cash=50_000, positions=[
        {"symbol": "SBIN", "quantity": 100, "averageCost": 500, "currentPrice": 500},
    ])
    out = pt.portfolio_risk(c, stop_pct=2, market_drop_pct=5)

    assert out["stops_exist"] is False
    assert out["open_risk"] == "unbounded"
    assert "SBIN" in out["positions_without_a_declared_stop"]
    # The old key must not come back — a caller reading it would be reading a
    # number that describes protection the user does not have.
    assert "loss_if_every_stop_hits" not in out


def test_portfolio_risk_keeps_the_stop_number_only_as_a_labelled_hypothetical():
    c = ctx(total=100_000, cash=50_000, positions=[
        {"symbol": "SBIN", "quantity": 100, "averageCost": 500, "currentPrice": 500},
    ])
    out = pt.portfolio_risk(c, stop_pct=2, market_drop_pct=5)

    assert out["deployed_value"] == 50_000
    hypo = out["hypothetical_if_you_had_stops"]
    assert hypo["loss_if_every_stop_hit"] == 1_000
    assert "hypothetical" in hypo["note"].lower()
    assert out["what_if_market_drops"]["estimated_loss"] == 2_500


def test_refuses_to_size_against_a_stale_account_mark():
    """`totalValue` is computed from position marks that refresh only on an
    explicit call or the next fill — nothing does it on a schedule. Sizing off
    an hours-old value is confident advice built on a number that is no longer
    true: the same failure as a missing account value, just quieter."""
    c = ctx()
    c["portfolio"]["marksStale"] = True
    c["portfolio"]["marksAsOf"] = "2026-07-26T09:20:00+05:30"

    out = pt.position_size(c, entry=100, stop=90, risk_pct=1, settings=SETTINGS)
    assert "error" in out
    assert "refreshed recently" in out["error"]
    assert "recommended_shares" not in out


def test_a_fresh_mark_sizes_normally():
    c = ctx()
    c["portfolio"]["marksStale"] = False
    out = pt.position_size(c, entry=100, stop=90, risk_pct=1, settings=SETTINGS)
    assert out["recommended_shares"] == 100


def test_an_absent_staleness_flag_does_not_block_sizing():
    """Older API builds do not send the field; missing must not mean stale, or
    every sizing request fails during a rolling deploy."""
    c = ctx()
    c["portfolio"].pop("marksStale", None)
    assert "recommended_shares" in pt.position_size(c, 100, 90, 1, SETTINGS)
