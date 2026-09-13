"""Turning a provider's frame into the bars the wire carries.

One conversion, because the rule about volume is easy to get wrong in each
place separately: a bar whose volume was never measured carries null, not
zero. Deriv has no exchange volume at all and its tick-count stand-in is only
fetched for the recent stretch worth paying for, so most of a long forex chart
is legitimately unmeasured -- and a zero there reads as a dead market.
"""
from __future__ import annotations

import pandas as pd


def volume_of(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def to_bars(df: pd.DataFrame) -> list[dict]:
    """OHLCV rows as dicts with Unix timestamps, for charting."""
    bars = []
    for ts, row in df.iterrows():
        bars.append({
            "time": int(ts.timestamp()),
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": volume_of(row.get("volume")),
        })
    return bars
