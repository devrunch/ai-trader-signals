"""
Portfolio and risk tools for the chat agent.

Pure functions over a trading-context dict, deliberately separated from the
market-data tools: these are the ones that produce a number a user may act on
with real money, and they need to be testable without a network.
"""
from __future__ import annotations

from typing import Any

from app.config import get_settings


def get_portfolio(ctx: dict) -> Any:
    return ctx.get("portfolio") or ctx


def get_positions(ctx: dict) -> Any:
    if "error" in ctx:
        return ctx
    positions = ctx.get("positions", [])
    if not positions:
        return {"count": 0, "note": "No open positions."}
    return {"count": len(positions), "positions": positions}


def _position_value(p: dict) -> float:
    return p["quantity"] * (p.get("currentPrice") or p["averageCost"])


def analyse_exposure(ctx: dict) -> Any:
    if "error" in ctx:
        return ctx
    positions = ctx.get("positions", [])
    pf = ctx.get("portfolio") or {}
    if not positions:
        return {"note": "No open positions — nothing deployed."}

    total_val = sum(_position_value(p) for p in positions)
    account = pf.get("totalValue") or (total_val + (pf.get("cashBalance") or 0))
    breakdown = sorted(
        ({"symbol": p["symbol"],
          "value": round(_position_value(p), 2),
          "pct_of_account": round(_position_value(p) / account * 100, 1) if account else None}
         for p in positions),
        key=lambda x: -(x["value"] or 0),
    )
    return {
        "deployed_value": round(total_val, 2),
        "cash_free": round(pf.get("cashBalance") or 0, 2),
        "pct_deployed": round(total_val / account * 100, 1) if account else None,
        "by_symbol": breakdown,
        "note": "Sector grouping is not available yet — concentration is per-symbol only.",
    }


def position_size(ctx: dict, entry: float, stop: float, risk_pct: float = 1.0, settings=None) -> Any:
    """How many shares so a stop-out costs no more than `risk_pct` of the account.

    Three independent limits apply and the tightest wins.
    """
    settings = settings or get_settings()
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        return {"error": "Entry and stop must differ."}

    # NEVER invent an account value. An earlier version defaulted to
    # ₹1,00,000 when the portfolio fetch failed and returned confident,
    # fully-populated sizing — a user with a ₹20,000 account would have been
    # handed 5x oversize with no indication anything was wrong.
    if "error" in ctx:
        return ctx
    pf = ctx.get("portfolio") or {}
    account = pf.get("totalValue") or pf.get("initialCapital")
    cash = pf.get("cashBalance")
    if account is None or cash is None:
        return {"error": "Could not read your account balance — cannot size a position safely."}

    # A stale account value is the same failure as a missing one, just quieter.
    # `totalValue` is computed from position marks that update only on an
    # explicit refresh or the next fill — nothing refreshes them on a schedule —
    # so after a morning of movement it can be hours old. Sizing off it would be
    # confident advice built on a number that is no longer true.
    if pf.get("marksStale"):
        return {
            "error": (
                "Your account value is based on position prices that have not been "
                "refreshed recently, so I cannot size a position safely against it. "
                "Refresh your portfolio and ask again."
            ),
            "marks_as_of": pf.get("marksAsOf"),
        }

    # Clamp the requested risk — nothing should let a model or a user pass 25%.
    risk_pct = max(settings.min_risk_pct, min(risk_pct, settings.max_risk_pct))
    risk_amount = account * risk_pct / 100

    shares_by_risk = int(risk_amount // per_share_risk)
    shares_by_cash = int(cash // entry) if entry > 0 else 0
    # Concentration cap: stop-loss risk and position size are different things.
    # Without this, a ₹3,000 stock with a tight stop sizes to ~99% of the
    # account and still reports "0.5% risk" — true only if the stop holds,
    # which a gap ignores.
    max_position_pct = settings.max_position_pct
    shares_by_concentration = int((account * max_position_pct / 100) // entry) if entry > 0 else 0

    shares = max(0, min(shares_by_risk, shares_by_cash, shares_by_concentration))
    limits = {"risk": shares_by_risk, "cash": shares_by_cash, "concentration": shares_by_concentration}
    limited_by = min(limits, key=lambda k: limits[k])

    capital = shares * entry
    return {
        "account_value": round(account, 2),
        "cash_available": round(cash, 2),
        "risk_pct_requested": risk_pct,
        "risk_amount": round(risk_amount, 2),
        "per_share_risk": round(per_share_risk, 2),
        "shares_allowed_by_risk": shares_by_risk,
        "shares_allowed_by_cash": shares_by_cash,
        "shares_allowed_by_concentration": shares_by_concentration,
        "max_position_pct": max_position_pct,
        "recommended_shares": shares,
        "capital_required": round(capital, 2),
        "position_pct_of_account": round(capital / account * 100, 1) if account else None,
        "actual_risk_amount": round(shares * per_share_risk, 2),
        "actual_risk_pct": round(shares * per_share_risk / account * 100, 2) if account else None,
        "limited_by": limited_by,
        "caveat": (
            "Risk assumes the stop fills at your price. A gap through the stop "
            "loses more — position size, not just stop distance, is your real exposure."
        ),
    }


def portfolio_risk(ctx: dict, stop_pct: float = 2.0, market_drop_pct: float = 2.0) -> Any:
    """What the account actually stands to lose — which is not a stop-based number.

    This tool used to answer "if every stop hits, you lose ₹4,200". There is no
    STOP order type in the paper engine and no monitoring loop, so NO STOPS
    EXIST: it was describing protection the user did not have, which is worse
    than saying nothing, because it converts unbounded risk into a specific and
    reassuring figure. It now leads with the truth and reports the stop-based
    number only as the hypothetical it always was.
    """
    if "error" in ctx:
        return ctx
    positions = ctx.get("positions", [])
    pf = ctx.get("portfolio") or {}
    if not positions:
        return {"note": "No open positions — no open risk."}

    account = pf.get("totalValue") or 0
    deployed = sum(_position_value(p) for p in positions)
    # A position only has a bounded loss if someone declared a stop for it —
    # and even then nothing will execute that stop automatically.
    without_stop = [p["symbol"] for p in positions if not p.get("stopLoss")]
    hypothetical_stop_loss = deployed * stop_pct / 100
    loss_on_drop = deployed * market_drop_pct / 100

    result: dict[str, Any] = {
        "stops_exist": False,
        "open_risk": "unbounded",
        "headline": (
            "You have no stop-loss orders — this product has no stop order type and "
            "nothing monitors your positions, so your open risk is unbounded. The "
            f"most you can lose on the ₹{round(deployed, 2)} deployed is all of it."
        ),
        "open_positions": len(positions),
        "deployed_value": round(deployed, 2),
        "deployed_pct_of_account": round(deployed / account * 100, 2) if account else None,
        "positions_without_a_declared_stop": without_stop,
        # Kept, but clearly labelled as the hypothetical it is.
        "hypothetical_if_you_had_stops": {
            "assumed_stop_pct": stop_pct,
            "loss_if_every_stop_hit": round(hypothetical_stop_loss, 2),
            "as_pct_of_account": round(hypothetical_stop_loss / account * 100, 2) if account else None,
            "note": "Hypothetical only. No such stops are placed and none will trigger.",
        },
        "what_if_market_drops": {"drop_pct": market_drop_pct, "estimated_loss": round(loss_on_drop, 2)},
        "caveat": (
            "Positions here move together — the universe is 25 large-cap Nifty names, "
            "5 banks and 4 IT companies — so correlated positions can lose more at once "
            "than a per-position figure suggests. The only real protection today is "
            "position size and the 15:20 square-off."
        ),
    }
    return result


def risk_limits(ctx: dict) -> Any:
    """The account's standing against the hard limits placement enforces.

    Without this the agent keeps proposing trades the API will refuse, which
    reads to the user as the product being broken rather than being careful.
    """
    if "error" in ctx:
        return ctx
    state = ctx.get("riskState")
    if not state:
        return {"error": "Could not read the account's risk limits — do not assume a trade would be allowed."}
    return {
        **state,
        "note": (
            "These are enforced at order placement, not advisory. A blocked order "
            "is refused with this reason."
        ),
    }
