"""
Candlestick pattern detection — native implementations.

pandas_ta only ships three patterns (doji, inside, z) without TA-Lib, and TA-Lib
is a C dependency we would rather not add to the image; the dozen patterns
traders actually use are simple OHLC arithmetic anyway, and implementing them
keeps the definitions explicit.

Pure over a DataFrame — no I/O, no config, no LLM. Directly unit-testable.
"""
from __future__ import annotations

import pandas as pd


def detect_patterns(df: pd.DataFrame, lookback: int = 10) -> dict:
    """Detect classic candlestick patterns over the most recent `lookback` bars.

    Definitions are deliberately conservative (explicit body/shadow ratios) so
    results are reproducible rather than impressionistic.
    """
    if len(df) < 5:
        return {"error": "Not enough bars for pattern detection"}

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, pd.NA)
    upper = h - c.combine(o, max)
    lower = c.combine(o, min) - l
    bull = c > o
    avg_body = body.rolling(14).mean()

    found: list[dict] = []
    start = max(2, len(df) - lookback)

    for i in range(start, len(df)):
        ts = str(df.index[i])
        b, r = body.iloc[i], rng.iloc[i]
        if pd.isna(r) or r == 0:
            continue
        up, lo = upper.iloc[i], lower.iloc[i]
        ab = avg_body.iloc[i] if pd.notna(avg_body.iloc[i]) else body.iloc[:i + 1].mean()

        def add(name, direction, note):
            found.append({"pattern": name, "direction": direction, "at": ts,
                          "close": round(float(c.iloc[i]), 2), "note": note})

        # Single-bar
        if b <= 0.1 * r:
            add("doji", "neutral", "open and close nearly equal — indecision")
        elif lo >= 2 * b and up <= 0.3 * b:
            add("hammer" if c.iloc[i] >= o.iloc[i] else "hanging_man",
                "bullish" if c.iloc[i] >= o.iloc[i] else "bearish",
                "long lower wick — rejection of lower prices")
        elif up >= 2 * b and lo <= 0.3 * b:
            add("shooting_star" if c.iloc[i] < o.iloc[i] else "inverted_hammer",
                "bearish" if c.iloc[i] < o.iloc[i] else "bullish",
                "long upper wick — rejection of higher prices")
        elif b >= 0.9 * r and b > ab:
            add("marubozu_bull" if bull.iloc[i] else "marubozu_bear",
                "bullish" if bull.iloc[i] else "bearish",
                "full-bodied candle, minimal wicks — strong conviction")

        # Two-bar
        po, pc = o.iloc[i - 1], c.iloc[i - 1]
        if bull.iloc[i] and pc < po and c.iloc[i] >= po and o.iloc[i] <= pc:
            add("bullish_engulfing", "bullish", "current body engulfs previous down candle")
        elif (not bull.iloc[i]) and pc > po and c.iloc[i] <= po and o.iloc[i] >= pc:
            add("bearish_engulfing", "bearish", "current body engulfs previous up candle")
        elif h.iloc[i] <= h.iloc[i - 1] and l.iloc[i] >= l.iloc[i - 1]:
            add("inside_bar", "neutral", "range contained by previous bar — compression")

        # Three-bar stars
        if i >= 2:
            o2, c2 = o.iloc[i - 2], c.iloc[i - 2]
            mid2 = (o2 + c2) / 2
            small_mid = body.iloc[i - 1] <= 0.5 * body.iloc[i - 2]
            if c2 < o2 and small_mid and bull.iloc[i] and c.iloc[i] > mid2:
                add("morning_star", "bullish", "three-bar reversal off a down candle")
            elif c2 > o2 and small_mid and (not bull.iloc[i]) and c.iloc[i] < mid2:
                add("evening_star", "bearish", "three-bar reversal off an up candle")

    bulls = sum(1 for f in found if f["direction"] == "bullish")
    bears = sum(1 for f in found if f["direction"] == "bearish")
    return {
        "bars_examined": len(df) - start,
        "patterns_found": len(found),
        "bullish": bulls, "bearish": bears,
        "patterns": found[-12:],
        "note": "Candlestick patterns are weak signals in isolation — confirm with trend, level and volume context.",
    }
