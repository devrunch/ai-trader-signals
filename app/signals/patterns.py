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


# What each pattern means. Keyed by pattern name so a result can carry one
# definition per kind instead of one per occurrence.
NOTES = {
    "doji": "open and close nearly equal — indecision",
    "hammer": "long lower wick — rejection of lower prices",
    "hanging_man": "long lower wick — rejection of lower prices",
    "shooting_star": "long upper wick — rejection of higher prices",
    "inverted_hammer": "long upper wick — rejection of higher prices",
    "marubozu_bull": "full-bodied candle, minimal wicks — strong conviction",
    "marubozu_bear": "full-bodied candle, minimal wicks — strong conviction",
    "bullish_engulfing": "current body engulfs previous down candle",
    "bearish_engulfing": "current body engulfs previous up candle",
    "inside_bar": "range contained by previous bar — compression",
    "morning_star": "three-bar reversal off a down candle",
    "evening_star": "three-bar reversal off an up candle",
}


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

        def add(name, direction):
            found.append({"pattern": name, "direction": direction, "at": ts,
                          "close": round(float(c.iloc[i]), 2)})

        # Single-bar
        if b <= 0.1 * r:
            add("doji", "neutral")
        elif lo >= 2 * b and up <= 0.3 * b:
            bullish_bar = c.iloc[i] >= o.iloc[i]
            add("hammer" if bullish_bar else "hanging_man", "bullish" if bullish_bar else "bearish")
        elif up >= 2 * b and lo <= 0.3 * b:
            bearish_bar = c.iloc[i] < o.iloc[i]
            add("shooting_star" if bearish_bar else "inverted_hammer",
                "bearish" if bearish_bar else "bullish")
        elif b >= 0.9 * r and b > ab:
            add("marubozu_bull" if bull.iloc[i] else "marubozu_bear",
                "bullish" if bull.iloc[i] else "bearish")

        # Two-bar
        po, pc = o.iloc[i - 1], c.iloc[i - 1]
        if bull.iloc[i] and pc < po and c.iloc[i] >= po and o.iloc[i] <= pc:
            add("bullish_engulfing", "bullish")
        elif (not bull.iloc[i]) and pc > po and c.iloc[i] <= po and o.iloc[i] >= pc:
            add("bearish_engulfing", "bearish")
        elif h.iloc[i] <= h.iloc[i - 1] and l.iloc[i] >= l.iloc[i - 1]:
            add("inside_bar", "neutral")

        # Three-bar stars
        if i >= 2:
            o2, c2 = o.iloc[i - 2], c.iloc[i - 2]
            mid2 = (o2 + c2) / 2
            small_mid = body.iloc[i - 1] <= 0.5 * body.iloc[i - 2]
            if c2 < o2 and small_mid and bull.iloc[i] and c.iloc[i] > mid2:
                add("morning_star", "bullish")
            elif c2 > o2 and small_mid and (not bull.iloc[i]) and c.iloc[i] < mid2:
                add("evening_star", "bearish")

    bulls = sum(1 for f in found if f["direction"] == "bullish")
    bears = sum(1 for f in found if f["direction"] == "bearish")
    shown = found[-12:]
    return {
        "bars_examined": len(df) - start,
        "patterns_found": len(found),
        "bullish": bulls, "bearish": bears,
        "patterns": shown,
        # One definition per pattern KIND, not per occurrence. The note is a
        # constant of the pattern, so repeating it on each hit resent the same
        # sentence three and four times inside a single result — and every
        # round after that resent it again.
        "glossary": {f["pattern"]: NOTES[f["pattern"]] for f in shown},
        "note": "Candlestick patterns are weak signals in isolation — confirm with trend, level and volume context.",
    }
