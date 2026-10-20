"""Trade geometry, computed here so the model cannot author it.

The agent used to write its own setups and the arithmetic was wrong in ways
that read as confident. One live answer offered a long AND a short on gold at
the same moment, quoted "35 pip risk vs 90+ pip reward" for a $30 stop on an
instrument with no pips, and placed that stop 0.4x ATR from entry — inside a
single hour's noise, and a setup the signal engine's own `stop_size` gate
rejects outright.

Every one of those checks already existed in `app/signals/validation.py` and
ran on machine-generated signals. Chat could not reach them. So this tool
takes INTENT from the agent — a direction, optionally a level — and computes
the entry, stop, target, reward:risk and position size itself, then runs the
same gates.

It reports rather than refuses. A rejected setup comes back with the gate that
failed, the numbers behind it, and what would have to be true to pass; a wall
that says "no" teaches the agent nothing and gives the user less. The verdict
is unambiguous either way, so a failing trade cannot be mistaken for a
recommendation.

`long_only` is the one gate deliberately not run. It exists because the
built-in paper account cannot short, not because a short is bad analysis — so
a SELL is computed and carries `executable_here: False` instead of vanishing.
"""
from __future__ import annotations

import logging
from typing import Any

from app.signals import analysis, validation
from app.signals import indicators as ind_mod
from app.signals.agent import portfolio_tools
from app.signals.agent.tools.base import BASE_INTERVAL, Handler, ToolContext

logger = logging.getLogger(__name__)

# A target with no level to aim at falls back to a multiple of the risk taken.
_FALLBACK_TARGET_R = 2.0
# Structural stops sit just beyond the level, not on it: price wicks through a
# level and recovers constantly, and a stop exactly on it is a donation.
_STRUCTURE_BUFFER_ATR = 0.15


def _round(symbol: str, value: float) -> float:
    """Forex pairs quote to 4-5 decimals; everything else to 2. Rounding a
    1.1052 entry to 1.11 moves the stop by 40 pips."""
    digits = 4 if len(symbol) == 6 and not symbol.startswith("XA") else 2
    return round(float(value), digits)


def _levels(df, entry: float) -> tuple[list[float], list[float]]:
    """Real touched levels above and below the entry, nearest first."""
    rows = analysis.support_resistance(df) or []
    below = sorted((float(r["value"]) for r in rows if float(r["value"]) < entry), reverse=True)
    above = sorted(float(r["value"]) for r in rows if float(r["value"]) > entry)
    return below, above


def _geometry(direction: str, entry: float, atr: float, below: list[float],
              above: list[float], settings) -> dict:
    """Stop and target from volatility and structure together.

    The stop is the FURTHER of the two candidates — an ATR floor and the level
    price would have to break — because the tighter one is the one that gets
    hit by noise. Then capped, so "safer" cannot become "sloppy".
    """
    min_mult = getattr(settings, "min_stop_atr_multiple", 1.0)
    max_mult = getattr(settings, "max_stop_atr_multiple", 3.5)
    buffer = atr * _STRUCTURE_BUFFER_ATR
    long_side = direction == "BUY"

    atr_stop = entry - atr * min_mult if long_side else entry + atr * min_mult
    structural = (below[0] - buffer) if (long_side and below) else \
                 (above[0] + buffer) if (not long_side and above) else None

    stop = min(atr_stop, structural) if (long_side and structural is not None) else \
           max(atr_stop, structural) if (not long_side and structural is not None) else atr_stop

    # Cap at the outer ATR multiple: a stop beyond it is not caution, it is a
    # position too small to matter or a loss too big to take.
    far = entry - atr * max_mult if long_side else entry + atr * max_mult
    capped = (stop < far) if long_side else (stop > far)
    if capped:
        stop = far

    risk = abs(entry - stop)
    target = (above[0] if (long_side and above) else
              below[0] if (not long_side and below) else None)
    from_level = target is not None
    if target is None:
        target = entry + risk * _FALLBACK_TARGET_R if long_side else entry - risk * _FALLBACK_TARGET_R

    return {"stop": stop, "target": target, "risk_per_unit": risk,
            "stop_capped_at_max_atr": capped, "target_from_level": from_level,
            "stop_from_structure": structural is not None and not capped}


def _gates(direction: str, entry: float, target: float, stop: float,
           indicators: dict, atr: float, ltp: float, settings) -> list[dict]:
    """Every quality gate, each reported whether it passed or failed.

    Reused from the signal engine rather than reimplemented: the thresholds and
    the reasoning behind them live in one place, and a chat answer that
    disagreed with the engine on the same instrument would be the real bug.
    """
    checks = (
        ("price_ordering", validation.price_ordering(direction, entry, target, stop)),
        ("entry_drift", validation.entry_drift(entry, ltp, atr, settings.max_entry_drift_atr)),
        ("indicator_agreement", validation.quant_confluence(
            direction, indicators, settings.min_quant_confluence)),
        ("not_overextended", validation.rsi_extreme(
            direction, indicators, settings.max_buy_rsi, settings.min_sell_rsi)),
        ("reward_risk", validation.reward_risk(entry, target, stop, settings.min_reward_risk)),
        ("stop_size", validation.stop_size(entry, stop, atr,
                                           settings.min_stop_atr_multiple,
                                           settings.max_stop_atr_multiple)),
    )
    return [{"gate": name, "passed": bool(ok), "detail": reason} for name, (ok, reason) in checks]


def _what_would_pass(direction: str, entry: float, stop: float, atr: float,
                     symbol: str, settings) -> dict:
    """The target this setup needs to clear the reward:risk floor.

    A refusal that does not say this leaves the agent guessing, and guessing is
    what produced the numbers this tool exists to replace.
    """
    risk = abs(entry - stop)
    needed = settings.min_reward_risk * risk
    target = entry + needed if direction == "BUY" else entry - needed
    return {"target_for_min_rr": _round(symbol, target),
            "min_reward_risk": settings.min_reward_risk,
            "note": f"At this stop, the target must be at least {_round(symbol, target)} "
                    f"to reach {settings.min_reward_risk}:1."}


async def propose_trade(ctx: ToolContext, args: dict) -> Any:
    """A costed, checked trade for one direction. Intent in, geometry out."""
    direction = str(args.get("direction") or "").strip().upper()
    if direction not in ("BUY", "SELL"):
        return {"error": "direction must be BUY or SELL. Decide which before asking."}

    symbol = str(args.get("symbol") or ctx.symbol).upper()
    interval = str(args.get("interval") or BASE_INTERVAL)
    df = await ctx.frame(args.get("symbol"), interval)
    if df is None or len(df) < 60:
        return {"error": f"Not enough {interval} history for {symbol} to size a trade."}

    indicators = dict(ind_mod.compute(df))
    atr = indicators.get("atr")
    ltp = round(float(df["close"].iloc[-1]), 6)
    if not atr:
        # Without ATR there is no volatility to place a stop against, and a
        # fixed-percentage stop on an unknown instrument is a guess.
        return {"error": "ATR unavailable — cannot place a stop without a volatility measure."}

    entry = float(args.get("entry") or ltp)
    below, above = _levels(df, entry)
    geo = _geometry(direction, entry, atr, below, above, ctx.settings)
    stop, target, risk = geo["stop"], geo["target"], geo["risk_per_unit"]
    reward = abs(target - entry)

    gates = _gates(direction, entry, target, stop, indicators, atr, ltp, ctx.settings)
    failed = [g for g in gates if not g["passed"]]

    sizing = portfolio_tools.position_size(await ctx.book(), entry, stop,
                                           float(args.get("risk_pct") or 1.0),
                                           settings=ctx.settings)

    out: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "direction": direction,
        "verdict": "rejected" if failed else "passes_every_gate",
        "entry": _round(symbol, entry),
        "stop": _round(symbol, stop),
        "target": _round(symbol, target),
        "last_price": _round(symbol, ltp),
        "risk_per_unit": _round(symbol, risk),
        "reward_per_unit": _round(symbol, reward),
        "reward_risk": round(reward / risk, 2) if risk else None,
        "risk_pct_of_entry": round(risk / entry * 100, 2) if entry else None,
        "atr": atr,
        "stop_in_atr": round(risk / atr, 2),
        "how_stop_was_placed": ("structure beyond the nearest level" if geo["stop_from_structure"]
                                else f"{ctx.settings.max_stop_atr_multiple}x ATR cap"
                                if geo["stop_capped_at_max_atr"]
                                else f"{ctx.settings.min_stop_atr_multiple}x ATR floor"),
        "target_source": "next touched level" if geo["target_from_level"]
                         else f"{_FALLBACK_TARGET_R}x risk (no level in the way)",
        "nearest_levels": {"below": [_round(symbol, v) for v in below[:3]],
                           "above": [_round(symbol, v) for v in above[:3]]},
        "gates": gates,
        "position_size": sizing,
        "regime": {"adx": indicators.get("adx"), "rsi": indicators.get("rsi"),
                   "supertrend_dir": indicators.get("supertrend_dir"),
                   "ema20": indicators.get("ema20"), "ema50": indicators.get("ema50")},
    }

    if direction == "SELL":
        # Said plainly rather than letting the gate delete the answer: the
        # paper account cannot short, the user's own broker can.
        out["executable_here"] = False
        out["executable_note"] = ("The built-in paper account is long-only. This short is "
                                  "analysis — place it at your broker if you take it.")

    if failed:
        out["rejected_because"] = [g["detail"] for g in failed]
        if any(g["gate"] == "reward_risk" for g in failed):
            out["to_make_it_work"] = _what_would_pass(direction, entry, stop, atr, symbol, ctx.settings)
        out["note"] = ("This setup fails a gate the signal engine enforces. The numbers are "
                       "real; the trade is not recommended. Say so rather than presenting it.")
    else:
        out["note"] = ("Every gate passed. Numbers are computed here, not estimated — quote "
                       "them as given.")
    return out


TOOLS: dict[str, Handler] = {"propose_trade": propose_trade}
