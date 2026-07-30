"""
The condition engine — a validated DSL for expressing rules over price bars.

This is shared machinery. One evaluator drives four features: agent-built
strategies, custom chart drawings, custom alerts, and custom screeners. Build it
once, correctly, and each of those becomes a thin layer on top.

THE SAFETY DECISION, which must not be revisited casually:

    The agent emits declarative SPECIFICATIONS, never executable code.

Having an LLM write Python that the server runs is remote code execution. One
prompt injection — or simply a hallucinated `import os` — compromises the
server, the database and every stored credential. No sandbox configuration makes
that a good trade here. So the model supplies data describing which known
indicators to combine and how; this module interprets that with known-safe
operations, and rejects anything it does not recognise.

Concretely that means:
  * indicator names come from an allow-list, never from `getattr`
  * every numeric parameter is bounded (no `length: 10_000_000`)
  * the condition tree has a depth and node-count cap
  * unknown fields are REJECTED rather than ignored — a typo must fail loudly
    rather than silently evaluating something other than what was asked

The second reason this matters: a rule-based strategy has no LLM in its
evaluation loop, so backtesting it over tens of thousands of bars costs nothing
and is perfectly repeatable. The agent designs; deterministic maths evaluates.
That is the one capability area where quality here is genuinely measurable —
unlike raw LLM signals, where the measurement problem is documented and severe.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Tree limits. Generous enough for any strategy a person would actually describe,
# small enough that a malformed or adversarial spec cannot cost real time.
MAX_DEPTH = 5
MAX_NODES = 40

# Parameter bounds, per parameter name. A period longer than the frame is not a
# strategy, it is a denial of service.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "length": (1, 400),
    "fast": (1, 200),
    "slow": (2, 400),
    "signal": (1, 100),
    "std": (0.1, 5.0),
    "multiplier": (0.1, 10.0),
}

COMPARISON_OPS = {"<", "<=", ">", ">=", "==", "!="}
SERIES_OPS = {"above", "below", "crosses_above", "crosses_below"}
ALL_OPS = COMPARISON_OPS | SERIES_OPS


class SpecError(ValueError):
    """A specification that failed validation. The message is shown to the agent
    so it can correct itself, so it must say precisely what was wrong."""


# ---------------------------------------------------------------------------
# Series builders — the allow-list.
#
# Each returns a full Series aligned to the frame, not a single latest value.
# `indicators.py` deliberately returns only the last value (that is what a
# snapshot needs); a rule has to be evaluated at every bar.
# ---------------------------------------------------------------------------

def _ta():
    # Deferred: pandas_ta is slow to import and most requests never build a spec.
    import pandas_ta as ta
    return ta


def _price(col: str) -> Callable[[pd.DataFrame, dict], pd.Series]:
    return lambda df, p: df[col].astype(float)


def _rsi(df, p):    return _ta().rsi(df["close"], length=int(p.get("length", 14)))
def _sma(df, p):    return _ta().sma(df["close"], length=int(p.get("length", 20)))
def _atr(df, p):    return _ta().atr(df["high"], df["low"], df["close"], length=int(p.get("length", 14)))
def _cci(df, p):    return _ta().cci(df["high"], df["low"], df["close"], length=int(p.get("length", 20)))
def _willr(df, p):  return _ta().willr(df["high"], df["low"], df["close"], length=int(p.get("length", 14)))
def _mfi(df, p):    return _ta().mfi(df["high"], df["low"], df["close"], df["volume"], length=int(p.get("length", 14)))
def _roc(df, p):    return _ta().roc(df["close"], length=int(p.get("length", 10)))


def _ema(df, p):
    return _ta().ema(df["close"], length=int(p.get("length", 20)))


def _macd_part(which: str):
    def build(df, p):
        r = _ta().macd(
            df["close"],
            fast=int(p.get("fast", 12)), slow=int(p.get("slow", 26)), signal=int(p.get("signal", 9)),
        )
        if r is None:
            return pd.Series(index=df.index, dtype="float64")
        col = {"macd": 0, "macd_hist": 1, "macd_signal": 2}[which]
        return r.iloc[:, col]
    return build


def _adx(df, p):
    r = _ta().adx(df["high"], df["low"], df["close"], length=int(p.get("length", 14)))
    if r is None:
        return pd.Series(index=df.index, dtype="float64")
    return r.iloc[:, 0]


def _bband(which: str):
    def build(df, p):
        r = _ta().bbands(df["close"], length=int(p.get("length", 20)), std=float(p.get("std", 2.0)))
        if r is None:
            return pd.Series(index=df.index, dtype="float64")
        like = {"bb_lower": "BBL", "bb_mid": "BBM", "bb_upper": "BBU"}[which]
        sub = r.filter(like=like)
        return sub.iloc[:, 0] if not sub.empty else pd.Series(index=df.index, dtype="float64")
    return build


def _supertrend_dir(df, p):
    r = _ta().supertrend(
        df["high"], df["low"], df["close"],
        length=int(p.get("length", 10)), multiplier=float(p.get("multiplier", 3.0)),
    )
    if r is None:
        return pd.Series(index=df.index, dtype="float64")
    sub = r.filter(like="SUPERTd")
    return sub.iloc[:, 0] if not sub.empty else pd.Series(index=df.index, dtype="float64")


def _vwap(df, p):
    """Session-anchored, matching every other VWAP in this codebase.

    A whole-frame cumsum anchors at the start of the history and is not the VWAP
    any intraday trader means.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(index=df.index, dtype="float64")
    session = pd.Series(df.index, index=df.index).dt.date
    return pv.groupby(session).cumsum() / df["volume"].groupby(session).cumsum()


def _volume_ratio(df, p):
    avg = df["volume"].rolling(int(p.get("length", 20))).mean()
    return df["volume"] / avg


def _highest(df, p):
    """Rolling highest high over `length` bars — the upper half of a Donchian-
    style N-bar range, and the general building block for "N above the high"
    style requests: whatever N and whichever line, this is the one computation
    underneath it."""
    return df["high"].rolling(int(p.get("length", 20))).max()


def _lowest(df, p):
    """Rolling lowest low over `length` bars — the lower half of the same channel."""
    return df["low"].rolling(int(p.get("length", 20))).min()


SERIES: dict[str, Callable[[pd.DataFrame, dict], pd.Series]] = {
    # raw price
    "close": _price("close"), "open": _price("open"),
    "high": _price("high"), "low": _price("low"), "volume": _price("volume"),
    # indicators
    "rsi": _rsi, "ema": _ema, "sma": _sma, "atr": _atr, "adx": _adx,
    "cci": _cci, "williams_r": _willr, "mfi": _mfi, "roc": _roc,
    "macd": _macd_part("macd"), "macd_hist": _macd_part("macd_hist"),
    "macd_signal": _macd_part("macd_signal"),
    "bb_upper": _bband("bb_upper"), "bb_mid": _bband("bb_mid"), "bb_lower": _bband("bb_lower"),
    "supertrend_dir": _supertrend_dir, "vwap": _vwap, "volume_ratio": _volume_ratio,
    "highest": _highest, "lowest": _lowest,
}

EXIT_TYPES = {"stop_loss", "take_profit"}


def available_series() -> list[str]:
    return sorted(SERIES)


def compute_series(df: pd.DataFrame, name: str, params: Any = None) -> pd.Series:
    """A single named, validated series — the entry point for anything that
    wants ONE line, not a boolean condition tree (currently: chart drawing).

    Goes through the same allow-list and the same `_check_params` bounds-check
    as a condition's series does; a caller here gets no more trust than a
    strategy spec does; both name and length are user/model-influenced and
    both go through exactly one gate.
    """
    if name not in SERIES:
        raise SpecError(f"Unknown series '{name}'. Available: {available_series()}")
    checked = _check_params(name, params)
    return _series_for(df, name, checked)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_params(name: str, params: Any) -> dict:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise SpecError(f"'{name}' params must be an object, got {type(params).__name__}")
    out: dict[str, float] = {}
    for key, raw in params.items():
        if key not in PARAM_BOUNDS:
            raise SpecError(
                f"Unknown parameter '{key}' for '{name}'. Allowed: {sorted(PARAM_BOUNDS)}"
            )
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise SpecError(f"Parameter '{key}' must be a number, got {raw!r}") from None
        low, high = PARAM_BOUNDS[key]
        if not (low <= value <= high):
            raise SpecError(f"Parameter '{key}'={value} out of range [{low}, {high}]")
        out[key] = value
    return out


def validate_condition(node: Any, depth: int = 0, counter: list[int] | None = None) -> None:
    """Raise `SpecError` describing precisely what is wrong, or return None."""
    counter = counter if counter is not None else [0]
    counter[0] += 1
    if counter[0] > MAX_NODES:
        raise SpecError(f"Condition tree too large (limit {MAX_NODES} nodes)")
    if depth > MAX_DEPTH:
        raise SpecError(f"Condition tree too deep (limit {MAX_DEPTH})")
    if not isinstance(node, dict):
        raise SpecError(f"Each condition must be an object, got {type(node).__name__}")

    if "all" in node or "any" in node:
        key = "all" if "all" in node else "any"
        extra = set(node) - {key}
        if extra:
            raise SpecError(f"'{key}' node has unexpected fields: {sorted(extra)}")
        children = node[key]
        if not isinstance(children, list) or not children:
            raise SpecError(f"'{key}' must be a non-empty list of conditions")
        for child in children:
            validate_condition(child, depth + 1, counter)
        return

    if "not" in node:
        if set(node) != {"not"}:
            raise SpecError(f"'not' node has unexpected fields: {sorted(set(node) - {'not'})}")
        validate_condition(node["not"], depth + 1, counter)
        return

    if "type" in node:
        kind = node["type"]
        if kind not in EXIT_TYPES:
            raise SpecError(f"Unknown condition type '{kind}'. Allowed: {sorted(EXIT_TYPES)}")
        extra = set(node) - {"type", "atr_multiple", "percent"}
        if extra:
            raise SpecError(f"'{kind}' has unexpected fields: {sorted(extra)}")
        if "atr_multiple" not in node and "percent" not in node:
            raise SpecError(f"'{kind}' needs either atr_multiple or percent")
        for field, (lo, hi) in (("atr_multiple", (0.1, 20.0)), ("percent", (0.05, 90.0))):
            if field in node:
                try:
                    v = float(node[field])
                except (TypeError, ValueError):
                    raise SpecError(f"'{field}' must be a number") from None
                if not (lo <= v <= hi):
                    raise SpecError(f"'{field}'={v} out of range [{lo}, {hi}]")
        return

    if "indicator" not in node:
        raise SpecError(
            "A condition needs 'indicator', or 'all'/'any'/'not', or an exit 'type'. "
            f"Got fields: {sorted(node)}"
        )

    allowed = {"indicator", "params", "op", "value", "compare_to", "compare_params"}
    extra = set(node) - allowed
    if extra:
        # Rejected, not ignored: a typo must fail loudly rather than silently
        # evaluating something other than what was asked for.
        raise SpecError(f"Unexpected fields {sorted(extra)}. Allowed: {sorted(allowed)}")

    name = node["indicator"]
    if name not in SERIES:
        raise SpecError(f"Unknown indicator '{name}'. Available: {available_series()}")
    _check_params(name, node.get("params"))

    op = node.get("op")
    if op not in ALL_OPS:
        raise SpecError(f"Unknown operator {op!r}. Allowed: {sorted(ALL_OPS)}")

    has_value = "value" in node
    has_compare = "compare_to" in node
    if has_value == has_compare:
        raise SpecError("A condition needs exactly one of 'value' or 'compare_to'")
    if has_value:
        try:
            float(node["value"])
        except (TypeError, ValueError):
            raise SpecError(f"'value' must be a number, got {node['value']!r}") from None
        if op in SERIES_OPS and op in {"above", "below"}:
            pass  # comparing a series to a constant level is meaningful
    else:
        target = node["compare_to"]
        if target not in SERIES:
            raise SpecError(f"Unknown compare_to '{target}'. Available: {available_series()}")
        _check_params(target, node.get("compare_params"))


def validate_strategy(spec: Any) -> dict:
    """Validate a full strategy spec and return it normalised."""
    if not isinstance(spec, dict):
        raise SpecError("Strategy must be an object")
    extra = set(spec) - {"name", "entry", "exit", "side"}
    if extra:
        raise SpecError(f"Strategy has unexpected fields: {sorted(extra)}")
    if "entry" not in spec or "exit" not in spec:
        raise SpecError("Strategy needs both 'entry' and 'exit'")

    side = str(spec.get("side", "long")).lower()
    if side != "long":
        # Phase 1 is long-only throughout the product: the paper account has no
        # short path at all, so a short strategy could be backtested but never
        # traded. Allowing it here would recreate the untradeable-output problem
        # that long-only was adopted to remove.
        raise SpecError("Only 'long' strategies are supported (the paper account cannot short)")

    validate_condition(spec["entry"])
    validate_condition(spec["exit"])
    return {
        "name": str(spec.get("name") or "Untitled strategy")[:80],
        "side": side,
        "entry": spec["entry"],
        "exit": spec["exit"],
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _series_for(df: pd.DataFrame, name: str, params: dict) -> pd.Series:
    s = SERIES[name](df, params)
    if s is None:
        return pd.Series(index=df.index, dtype="float64")
    return pd.Series(s, index=df.index).astype("float64")


def evaluate_condition(df: pd.DataFrame, node: dict) -> pd.Series:
    """Evaluate a validated condition into a boolean Series aligned to `df`.

    Bars where an indicator has not warmed up evaluate to False rather than NaN:
    a rule cannot be said to hold on data that does not exist yet, and False is
    the conservative reading — it declines to trade rather than inventing one.
    """
    if "all" in node:
        parts = [evaluate_condition(df, c) for c in node["all"]]
        out = parts[0]
        for p in parts[1:]:
            out = out & p
        return out
    if "any" in node:
        parts = [evaluate_condition(df, c) for c in node["any"]]
        out = parts[0]
        for p in parts[1:]:
            out = out | p
        return out
    if "not" in node:
        return ~evaluate_condition(df, node["not"])
    if "type" in node:
        # Stop/target are handled by the runner, which knows the entry price.
        return pd.Series(False, index=df.index)

    left = _series_for(df, node["indicator"], _check_params(node["indicator"], node.get("params")))
    op = node["op"]

    if "value" in node:
        right_series = pd.Series(float(node["value"]), index=df.index)
    else:
        right_series = _series_for(
            df, node["compare_to"], _check_params(node["compare_to"], node.get("compare_params"))
        )

    if op == "<":   result = left < right_series
    elif op == "<=": result = left <= right_series
    elif op == ">":  result = left > right_series
    elif op == ">=": result = left >= right_series
    elif op == "==": result = left == right_series
    elif op == "!=": result = left != right_series
    elif op == "above": result = left > right_series
    elif op == "below": result = left < right_series
    elif op in ("crosses_above", "crosses_below"):
        prev_l, prev_r = left.shift(1), right_series.shift(1)
        if op == "crosses_above":
            result = (left > right_series) & (prev_l <= prev_r)
        else:
            result = (left < right_series) & (prev_l >= prev_r)
        # A cross needs a previous bar to compare against.
        result = result & prev_l.notna() & prev_r.notna()
    else:  # unreachable after validation
        raise SpecError(f"Unknown operator {op!r}")

    valid = left.notna() & right_series.notna()
    return (result & valid).fillna(False).astype(bool)


def extract_risk_exits(node: dict, out: dict | None = None) -> dict:
    """Collect stop_loss / take_profit settings from an exit tree.

    These cannot be expressed as a boolean series because they depend on the
    entry price, which is not known until a position opens. The runner applies
    them; everything else in the tree evaluates normally.
    """
    out = out if out is not None else {}
    if not isinstance(node, dict):
        return out
    if "type" in node and node["type"] in EXIT_TYPES:
        out[node["type"]] = {
            k: float(node[k]) for k in ("atr_multiple", "percent") if k in node
        }
        return out
    for key in ("all", "any"):
        for child in node.get(key, []) or []:
            extract_risk_exits(child, out)
    if "not" in node:
        extract_risk_exits(node["not"], out)
    return out


def run_strategy(
    df: pd.DataFrame,
    spec: dict,
    stop_pct: float | None = None,
    cost_pct: float = 0.0,
) -> dict:
    """Validate a spec, evaluate it over `df`, and backtest the result.

    The stop is taken from the spec's own `stop_loss` node when it has one —
    expressed either as a percentage or as an ATR multiple, which is converted
    to a percentage using ATR at the time of the run. An ATR-based stop is the
    better choice and the prompt encourages it: a fixed percentage is too tight
    on a volatile name and too loose on a quiet one.

    Entry fills at the NEXT bar's open and round-trip costs are deducted, the
    same rules the rest of the product uses. A spec that looks good here has
    already paid for the two frictions that make naive backtests lie.
    """
    from app.signals import analysis

    validated = validate_strategy(spec)
    if len(df) < 60:
        return {"error": "Not enough history to evaluate a strategy (need 60+ bars)"}

    entries = evaluate_condition(df, validated["entry"])
    exits = evaluate_condition(df, validated["exit"])
    risk = extract_risk_exits(validated["exit"])

    effective_stop = stop_pct
    if "stop_loss" in risk:
        cfg = risk["stop_loss"]
        if "percent" in cfg:
            effective_stop = cfg["percent"]
        elif "atr_multiple" in cfg:
            atr = _series_for(df, "atr", {"length": 14})
            last_atr, last_close = atr.iloc[-1], float(df["close"].iloc[-1])
            if pd.notna(last_atr) and last_close:
                effective_stop = float(last_atr) * cfg["atr_multiple"] / last_close * 100

    result = analysis._run_signals(df, entries, exits, stop_pct=effective_stop, cost_pct=cost_pct)
    result["strategy"] = validated["name"]
    result["spec"] = validated
    result["entry_signals"] = int(entries.sum())
    result["exit_signals"] = int(exits.sum())
    result["bars"] = len(df)
    result["stop_source"] = (
        "spec" if "stop_loss" in risk else "caller" if stop_pct is not None else "none"
    )
    return result
