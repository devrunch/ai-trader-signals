"""
Walk-forward backtest runner.

Runs the SAME production signal logic (regime filter, ATR-grounded prompt,
validation gates) against frozen historical slices. Each slice only sees bars up
to that point (no lookahead); the outcome is resolved against the REAL bars that
actually followed. No FinBERT — point-in-time news isn't available historically
— so this tests the technical logic, which is what's under test.

A standing caveat, recorded here because it keeps being forgotten: at ~100
resolved trades the smallest detectable difference between two runs is ~18
percentage points, while the gap this system needs to close is ~2 points. This
is a CORRECTNESS tool — it has caught three real bugs — not an OPTIMISATION
tool. Do not tune a threshold against it.
"""
from __future__ import annotations

import logging

import pandas as pd

from app.config import get_settings
from app.market.calendar import IST
from app.signals.backtest.evaluator import walk_forward_eval

logger = logging.getLogger(__name__)

NEUTRAL_SENTIMENT = {"label": "neutral", "score": 0.5, "headlines_count": 0}


def _session_dates(index: pd.Index) -> pd.DatetimeIndex | None:
    """One date per bar, as the NSE trading day the bar belongs to.

    Returns None for a non-datetime index — synthetic frames in tests have a
    plain RangeIndex and there is no session to speak of.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return None
    # yfinance returns .NS bars tz-aware in Asia/Kolkata, but a caller may hand
    # over UTC or naive timestamps. Normalise to IST first so "the same session"
    # means the same NSE trading day rather than the same UTC day.
    ist = index.tz_convert(IST) if index.tz is not None else index
    return ist.normalize()


def session_forward_window(df: pd.DataFrame, idx: int, forward_bars: int) -> pd.DataFrame:
    """The bars after `idx` that belong to the SAME session, capped at `forward_bars`.

    Intraday means intraday. A fixed forward window ran up to 10 hours and
    crossed session boundaries, so overnight gap risk was folded into the
    results of a strategy whose whole premise is that it takes none — and since
    the evaluator fills a gapped stop at the bar's open, those gaps were scored
    as losses the trade could never have suffered.

    A trade still unresolved at the last bar of its own session comes back as
    OPEN, which is exactly what the 15:20 square-off produces in live trading.
    """
    forward = df.iloc[idx + 1: idx + 1 + forward_bars]
    if forward.empty:
        return forward
    dates = _session_dates(df.index)
    if dates is None:
        return forward
    return forward[dates[idx + 1: idx + 1 + forward_bars] == dates[idx]]


async def backtest_walkforward(
    service,
    symbols: list[str],
    exchange: str = "NSE",
    points_per_symbol: int = 8,
    warmup_bars: int = 60,
    forward_bars: int = 40,
) -> dict:
    from app.signals import indicators as ind_mod

    settings = get_settings()
    attempts: list[dict] = []

    for symbol in symbols:
        df = await service._fetch_ohlcv(symbol, exchange)
        if df is None or len(df) < warmup_bars + forward_bars + 10:
            attempts.append({"symbol": symbol, "outcome": "INSUFFICIENT_DATA"})
            continue

        usable_start = warmup_bars
        usable_end = len(df) - forward_bars - 1
        if usable_end <= usable_start:
            attempts.append({"symbol": symbol, "outcome": "INSUFFICIENT_DATA"})
            continue
        step = max(1, (usable_end - usable_start) // points_per_symbol)
        slice_points = list(range(usable_start, usable_end, step))[:points_per_symbol]

        for idx in slice_points:
            df_slice = df.iloc[: idx + 1]
            ts = str(df.index[idx])
            try:
                indicators = ind_mod.compute(df_slice, ind_mod.SIGNAL_SET)
            except Exception:
                logger.exception("Indicator computation failed for %s@%s", symbol, ts)
                attempts.append({"symbol": symbol, "timestamp": ts, "outcome": "INDICATOR_ERROR"})
                continue

            adx = indicators.get("adx")
            if adx is not None and adx < settings.min_adx_trend:
                attempts.append({"symbol": symbol, "timestamp": ts, "outcome": "SKIPPED_REGIME", "adx": adx})
                continue

            try:
                signal, _reason = await service._ask_llm(
                    symbol, exchange, df_slice, indicators, NEUTRAL_SENTIMENT, live=False
                )
            except Exception as e:
                logger.warning("Backtest LLM call failed for %s@%s: %s", symbol, ts, e)
                attempts.append({"symbol": symbol, "timestamp": ts, "outcome": "LLM_ERROR"})
                continue

            if signal is None:
                attempts.append({"symbol": symbol, "timestamp": ts, "outcome": "NO_SIGNAL"})
                continue
            if signal.confidence < settings.confidence_threshold:
                attempts.append({"symbol": symbol, "timestamp": ts, "outcome": "LOW_CONFIDENCE",
                                 "confidence": signal.confidence})
                continue

            forward = session_forward_window(df, idx, forward_bars)
            result = walk_forward_eval(signal, forward)
            # snake_case throughout — this endpoint used to return camelCase
            # keys (entryPrice, pnlPct) mixed with snake_case ones (symbol,
            # outcome) in the same object, and the sibling /signals/generate
            # endpoint returned pure snake_case. One convention per language.
            attempts.append({
                "symbol": symbol, "timestamp": ts, "direction": signal.signal_type.value,
                "confidence": signal.confidence, "entry_price": signal.entry_price,
                "target_price": signal.target_price, "stop_loss": signal.stop_loss,
                "outcome": result.outcome, "exit_price": result.exit_price, "pnl_pct": result.pnl_pct,
            })

    closed = [a for a in attempts if a["outcome"] in ("TARGET_HIT", "STOP_HIT")]
    wins = [a for a in closed if a["outcome"] == "TARGET_HIT"]
    avg_pnl = round(sum(a["pnl_pct"] for a in closed) / len(closed), 2) if closed else 0.0
    return {
        "attempts": attempts,
        "summary": {
            "total_slices": len(attempts),
            "signals_generated": len(closed) + len([a for a in attempts if a["outcome"] == "OPEN"]),
            "skipped_regime": len([a for a in attempts if a["outcome"] == "SKIPPED_REGIME"]),
            "no_signal": len([a for a in attempts if a["outcome"] == "NO_SIGNAL"]),
            "resolved": len(closed),
            "wins": len(wins),
            "losses": len(closed) - len(wins),
            "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
            "avg_pnl_pct": avg_pnl,
            "cost_pct_round_trip": settings.cost_pct_round_trip,
        },
    }
