"""Breakout probability: the one measured-probability number the agent lacked.

Given the colour of the previous candle, how often has this market gone on to
exceed that candle's high (or low) by a given percentage?

The question is conditional, which is what makes it more than a volatility
measure: "the last candle was green — historically, how often does the next
one take out its high by 1%?" Counts run over the whole loaded history, so the
numbers are this instrument on this timeframe, not a rule of thumb.

This is a port of the terminal's own implementation
(`lib/indicators/breakout-probability.ts`). It was browser-only TypeScript, so
every answer the agent gave about a level was reasoning — it could compute how
far a level was but never how often price had actually reached it. Both sides
must stay in step; the behaviours the tests pin here are the same ones pinned
there.

Algorithm after Zeiierman's "Breakout Probability (Expo)" (CC BY-NC-SA 4.0),
written from what it computes rather than ported from the Pine source.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Levels either side, including the previous high/low itself at index 0.
MAX_LEVELS = 5
# Below this the counts are noise with a percent sign. The frontend draws an
# untested level unlabelled for the same reason.
MIN_SAMPLE = 10


@dataclass(frozen=True)
class Level:
    index: int
    price: float
    probability: float | None

    def to_dict(self) -> dict:
        return {"step": self.index, "price": round(self.price, 4),
                "probability_pct": self.probability}


@dataclass
class Breakout:
    upper: list[Level]
    lower: list[Level]
    last_candle_green: bool
    sample: int

    def to_dict(self) -> dict:
        return {
            "after": "an up candle" if self.last_candle_green else "a down candle",
            "sample": self.sample,
            "enough": self.sample >= MIN_SAMPLE,
            "upper": [lvl.to_dict() for lvl in self.upper],
            "lower": [lvl.to_dict() for lvl in self.lower],
            "note": ("Share of past bars that followed a candle of this colour and "
                     "then reached each level. Conditional on the previous candle, "
                     "measured on this instrument and timeframe — not a rule of "
                     "thumb."),
        }


def _ratio(hits: int, total: int) -> float | None:
    """No sample is not the same as never happening.

    A 0% on a level that has simply never been tested reads as "this cannot
    happen", which is the one thing the number must not say.
    """
    if total <= 0:
        return None
    return round(hits / total * 100, 2)


def compute(df: pd.DataFrame, step_percent: float = 1.0,
            levels: int = MAX_LEVELS) -> Breakout | None:
    """Walk the history once, counting how often each level was reached.

    Bar i is judged against bar i-1's high/low and attributed to the colour of
    bar i-1, so the counts answer "after a green candle, what happened next".

    `step_percent` is measured against each bar's own close, so a level is the
    same *relative* distance throughout the history rather than a fixed offset
    that would mean something different at 1,800 than at 4,300.
    """
    if df is None or len(df) < 3 or levels < 1:
        return None

    green = red = 0
    green_high = [0] * levels
    green_low = [0] * levels
    red_high = [0] * levels
    red_low = [0] * levels

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    for i in range(1, len(df)):
        prev_green = c[i - 1] > o[i - 1]
        prev_red = c[i - 1] < o[i - 1]
        # A doji is neither, and counting it as one side would quietly bias
        # every number here.
        if not prev_green and not prev_red:
            continue
        if prev_green:
            green += 1
        else:
            red += 1

        step = c[i] * step_percent / 100.0
        for level in range(levels):
            offset = step * level
            reached_high = h[i] >= h[i - 1] + offset
            reached_low = low[i] <= low[i - 1] - offset
            if prev_green:
                green_high[level] += reached_high
                green_low[level] += reached_low
            else:
                red_high[level] += reached_high
                red_low[level] += reached_low

    last_green = bool(c[-1] > o[-1])
    step = c[-1] * step_percent / 100.0
    total = green if last_green else red
    highs = green_high if last_green else red_high
    lows = green_low if last_green else red_low

    return Breakout(
        upper=[Level(i, h[-1] + step * i, _ratio(highs[i], total)) for i in range(levels)],
        lower=[Level(i, low[-1] - step * i, _ratio(lows[i], total)) for i in range(levels)],
        last_candle_green=last_green,
        sample=total,
    )


def bias(result: Breakout) -> dict | None:
    """Which side history favours for the nearest level, and by how much.

    None when the two sides are within a point of each other: calling a
    50.4 / 49.6 split "bullish" is noise dressed as a read.
    """
    up = result.upper[0].probability if result.upper else None
    down = result.lower[0].probability if result.lower else None
    if up is None or down is None:
        return None
    edge = abs(up - down)
    if edge < 1:
        return None
    return {"direction": "up" if up > down else "down", "edge_pct": round(edge, 2)}


def reach_probability(df: pd.DataFrame, target: float,
                      levels: int = MAX_LEVELS) -> dict | None:
    """How often price has reached a specific price from here.

    The question an agent actually gets asked — "how often does it get to
    10,982?" — rather than a ladder of fixed percentage steps. The step is
    derived from the distance to that price, so the nearest level IS the
    target.
    """
    if df is None or len(df) < 3:
        return None
    last_high = float(df["high"].iloc[-1])
    last_low = float(df["low"].iloc[-1])
    close = float(df["close"].iloc[-1])
    if not close:
        return None

    above = target > last_high
    distance = (target - last_high) if above else (last_low - target)
    if distance <= 0:
        # Already inside the last bar's range: no step to measure against, and
        # a probability for a level price has already touched is not a forecast.
        return {"target": target, "probability_pct": None,
                "note": "That price is inside the last bar's range."}

    # One step IS the distance, so level 1 lands exactly on the target.
    step_percent = distance / close * 100.0
    report = compute(df, step_percent=step_percent, levels=2)
    if report is None:
        return None
    side = report.upper if above else report.lower
    return {
        "target": target,
        "direction": "above" if above else "below",
        "probability_pct": side[1].probability if len(side) > 1 else None,
        "sample": report.sample,
        "enough": report.sample >= MIN_SAMPLE,
        "after": "an up candle" if report.last_candle_green else "a down candle",
    }
