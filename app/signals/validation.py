"""
Validation gates applied to the LLM's proposed signal.

Six independent, pure functions. Each takes plain data and returns
`(ok: bool, reason: str | None)` — no logging, no config lookups beyond the
settings object it is handed, no I/O. That makes the rules that actually decide
whether a trade is emitted directly unit-testable with a table of dicts, which
is exactly the code most worth testing and was previously unreachable from a
test because it lived inside a 200-line method on a class whose constructor
built boto3 and OpenAI clients.

Every gate FAILS CLOSED. A risk filter that disables itself when its input is
missing is worse than no filter, because it is invisible.

(Two gates were added after the paragraph above was written — `long_only` and
`rsi_extreme` — so there are eight functions now, not six. Everything else the
paragraph says still holds of all of them.)
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

Result = tuple[bool, str | None]

OK: Result = (True, None)


def hold_is_not_a_trade(data: Mapping[str, Any]) -> Result:
    """HOLD means "no trade" — it must never fall through to R:R and stop
    validation as if it were directional. It was silently mis-scored as a real
    trade in an earlier version, which flattered every performance number."""
    if data.get("signal_type") == "HOLD":
        return False, "HOLD is not a trade"
    return OK


def long_only(data: Mapping[str, Any]) -> Result:
    """Phase 1 is long-only, so a SELL is not a signal — it is an instruction
    the user cannot follow.

    A SELL means sell now and buy back lower: short selling, which needs
    borrowed stock. The paper account rejects any sell without an existing long
    position, so roughly half of everything the engine produced could not be
    traded inside the product — and those untradeable signals were counted in
    the published win rate, which therefore described a system nobody could
    have run. Shorting needs margin rules, borrow availability and a forced
    square-off; that is Phase 2.
    """
    if data.get("signal_type") == "SELL":
        return False, "short selling is not supported (long-only in Phase 1)"
    return OK


def parse_prices(data: Mapping[str, Any]) -> tuple[tuple[float, float, float] | None, str | None]:
    """(entry, target, stop) as floats, or a reason they could not be read."""
    try:
        return (float(data["entry_price"]), float(data["target_price"]), float(data["stop_loss"])), None
    except (KeyError, ValueError, TypeError):
        return None, "entry/target/stop missing or not numeric"


def price_ordering(direction: str, entry: float, target: float, stop: float) -> Result:
    """A BUY's target must sit above entry and its stop below it; the reverse
    for a SELL.

    Caught via backtesting: the LLM occasionally emits a direction with
    target/stop on the wrong side, which silently corrupts evaluation — the eval
    logic branch-mismatches and produces a nonsensical near-instant outcome.
    """
    wants_buy = direction == "BUY"
    ok = (target > entry and stop < entry) if wants_buy else (target < entry and stop > entry)
    if not ok:
        return False, f"target/stop on wrong side of entry for {direction}"
    return OK


def entry_drift(entry: float, ltp: float, atr: float, max_drift_atr: float) -> Result:
    """Entry must be reachable from the last traded price.

    Without this the model can propose an entry the market never touched, and
    the evaluator — which scans forward for stop/target without checking the
    entry ever filled — books P&L on a trade that could not have been taken.
    """
    max_drift = atr * max_drift_atr
    drift = abs(entry - ltp)
    if drift > max_drift:
        return False, f"entry {entry:.2f} is {drift:.2f} from LTP {ltp:.2f} (max {max_drift:.2f})"
    return OK


def quant_confluence(direction: str, indicators: Mapping[str, Any], min_ratio: float) -> Result:
    """The LLM's chosen direction must actually agree with a majority of the
    rule-based indicators it was given, not just narrate around them.

    Backtesting confirmed the LLM's free-form call alone doesn't beat noise;
    require real technical agreement. Fails closed when no indicator can vote —
    that is no confirmation at all, not unanimous confirmation.
    """
    wants_buy = direction == "BUY"
    votes = 0
    total = 0

    if indicators.get("ema20") is not None and indicators.get("ema50") is not None:
        total += 1
        votes += 1 if (indicators["ema20"] > indicators["ema50"]) == wants_buy else 0
    if indicators.get("supertrend_dir") is not None:
        total += 1
        votes += 1 if (indicators["supertrend_dir"] == 1) == wants_buy else 0
    if indicators.get("macd") is not None and indicators.get("macd_signal") is not None:
        total += 1
        votes += 1 if (indicators["macd"] > indicators["macd_signal"]) == wants_buy else 0
    if indicators.get("rsi") is not None:
        total += 1
        votes += 1 if (indicators["rsi"] > 50) == wants_buy else 0

    if total == 0:
        return False, "no indicators available to confirm direction"
    ratio = votes / total
    if ratio < min_ratio:
        return False, f"quant confluence {ratio * 100:.0f}% < {min_ratio * 100:.0f}%"
    return OK


def rsi_extreme(direction: str, indicators: Mapping[str, Any],
                max_buy_rsi: float, min_sell_rsi: float) -> Result:
    """Veto a trade taken into an already-stretched move.

    `quant_confluence` counts RSI as agreeing with BUY whenever `rsi > 50`, so
    RSI at 85 — a market that has already run — votes *for* buying. The gate as
    a whole therefore selects for maximum extension instead of filtering it.
    This is a hard veto that runs regardless of what the rest of the confluence
    says; a proper fix (genuinely orthogonal inputs) is Phase 2.

    Fails closed on a missing RSI: "we could not check" is not "there was no
    extreme". RSI needs only 14 bars, so its absence means the data is wrong.
    """
    rsi = indicators.get("rsi")
    if rsi is None:
        return False, "RSI unavailable — cannot rule out an overextended entry"
    if direction == "BUY" and rsi > max_buy_rsi:
        return False, f"RSI {rsi:.1f} above {max_buy_rsi:.0f} — overextended for a BUY"
    if direction == "SELL" and rsi < min_sell_rsi:
        return False, f"RSI {rsi:.1f} below {min_sell_rsi:.0f} — overextended for a SELL"
    return OK


def reward_risk(entry: float, target: float, stop: float, minimum: float) -> Result:
    """abs() on both legs — target and stop naturally sit on opposite sides of
    entry for BUY vs SELL, so signed deltas flip sign by direction."""
    reward = abs(target - entry)
    risk = abs(entry - stop)
    rr = reward / max(risk, 0.01)
    if rr < minimum:
        return False, f"R:R {rr:.2f} < {minimum:.2f}"
    return OK


def stop_size(entry: float, stop: float, atr: float, min_multiple: float, max_multiple: float) -> Result:
    """Server-side stop-sizing guard — never trust the LLM's stop distance
    blindly. Too tight = stopped out by noise; too wide = sloppy R:R."""
    risk = abs(entry - stop)
    min_stop = atr * min_multiple
    max_stop = atr * max_multiple
    if risk < min_stop:
        return False, f"stop {risk:.2f} tighter than {min_multiple:.1f}x ATR ({min_stop:.2f})"
    if risk > max_stop:
        return False, f"stop {risk:.2f} wider than {max_multiple:.1f}x ATR ({max_stop:.2f})"
    return OK


def validate(data: Mapping[str, Any], indicators: Mapping[str, Any],
             atr: float, ltp: float, settings) -> Result:
    """Run every gate in order, returning the first failure.

    Order matters only for the quality of the rejection reason — the gates are
    independent.
    """
    ok, reason = hold_is_not_a_trade(data)
    if not ok:
        return False, reason

    ok, reason = long_only(data)
    if not ok:
        return False, reason

    prices, err = parse_prices(data)
    if prices is None:
        return False, err
    entry, target, stop = prices
    direction = data.get("signal_type", "")

    for ok, reason in (
        price_ordering(direction, entry, target, stop),
        entry_drift(entry, ltp, atr, settings.max_entry_drift_atr),
        quant_confluence(direction, indicators, settings.min_quant_confluence),
        rsi_extreme(direction, indicators, settings.max_buy_rsi, settings.min_sell_rsi),
        reward_risk(entry, target, stop, settings.min_reward_risk),
        stop_size(entry, stop, atr, settings.min_stop_atr_multiple, settings.max_stop_atr_multiple),
    ):
        if not ok:
            return False, reason
    return OK
