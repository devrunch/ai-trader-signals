"""
Signal outcome evaluation — THE one implementation.

This logic previously existed twice: here in Python and again in TypeScript at
`ai-trader-api/src/signals/signals.service.ts`. The two answered the same
question — "did this signal hit target or stop first, and what was the P&L" —
with different rules: the TypeScript version filled gapped stops at the exact
stop price and applied no costs. Those are the two numbers the product uses to
tell the user whether the system works, and they systematically disagreed, the
TypeScript one optimistic by at least the cost figure plus every gap.

The rules that matter, and why:

  * Gap-through-stop — if a bar OPENS beyond the stop you fill at the open, not
    at your stop price. Booking every stop at exactly the stop price
    understates losses on gaps, which are common when a trade is held across a
    session boundary.
  * Round-trip costs — brokerage, STT, exchange fees, GST, stamp duty and
    realistic slippage. Zero-cost backtests are not a rounding error at this
    trade frequency; they invert the sign of the expectancy.
  * Stop is checked before target within a bar. A single OHLC bar does not tell
    us which level was touched first, so we assume the adverse one — the
    conservative reading.

Pure: takes a signal-shaped mapping and a DataFrame of the bars that followed,
returns an outcome. No I/O, no config singleton, no class. Directly testable.
"""
from __future__ import annotations

from typing import Literal, NamedTuple

import pandas as pd

from app.config import get_settings
from app.signals.types import GeneratedSignal, SignalType

Outcome = Literal["TARGET_HIT", "STOP_HIT", "OPEN"]


class Evaluation(NamedTuple):
    outcome: Outcome
    exit_price: float | None
    pnl_pct: float


def evaluate(
    direction: str,
    entry: float,
    target: float,
    stop: float,
    forward: pd.DataFrame,
    cost_pct: float | None = None,
) -> Evaluation:
    """Resolve a trade against the bars that actually followed it.

    `forward` must have `open`/`high`/`low`/`close` columns and contain only
    bars strictly AFTER the decision bar — passing the decision bar itself is
    lookahead and will resolve trades that had not started.
    """
    if cost_pct is None:
        cost_pct = get_settings().cost_pct_round_trip

    if forward is None or forward.empty:
        return Evaluation("OPEN", None, 0.0)

    is_buy = str(direction).upper() == "BUY"

    def pnl(exit_price: float) -> float:
        gross = ((exit_price - entry) if is_buy else (entry - exit_price)) / entry * 100
        return round(gross - cost_pct, 2)

    for _, row in forward.iterrows():
        o, hi, lo = float(row["open"]), float(row["high"]), float(row["low"])
        if is_buy:
            if lo <= stop:
                fill = min(o, stop)   # gap-down fills at the open
                return Evaluation("STOP_HIT", round(fill, 2), pnl(fill))
            if hi >= target:
                fill = max(o, target) if o >= target else target
                return Evaluation("TARGET_HIT", round(fill, 2), pnl(fill))
        else:
            if hi >= stop:
                fill = max(o, stop)   # gap-up fills at the open
                return Evaluation("STOP_HIT", round(fill, 2), pnl(fill))
            if lo <= target:
                fill = min(o, target) if o <= target else target
                return Evaluation("TARGET_HIT", round(fill, 2), pnl(fill))

    last_close = float(forward["close"].iloc[-1])
    return Evaluation("OPEN", round(last_close, 2), pnl(last_close))


def walk_forward_eval(signal: GeneratedSignal, forward: pd.DataFrame) -> Evaluation:
    """`evaluate` for a `GeneratedSignal`."""
    return evaluate(
        SignalType(signal.signal_type).value,
        signal.entry_price,
        signal.target_price,
        signal.stop_loss,
        forward,
    )
