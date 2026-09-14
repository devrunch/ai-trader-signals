"""What price actually did, last time this number came out.

This is the difference between a brief worth reading and a horoscope. "Gold
may fall on a hot CPI" is something he already knows; "gold fell in 8 of the
last 9 hot prints, median -0.7% within the hour, and the one exception had the
dollar already bid" is a number he can size against.

Two halves, both of which we hold:

  * **the prints** -- every past release with its actual and its forecast, so
    each instance can be labelled a beat, a miss, or in line. A surprise is
    the only thing that moves price; the level itself is old news by the time
    it prints.
  * **the reaction** -- minute bars around each of those timestamps, from
    Dukascopy. Our live vendor keeps 1m bars for seven days, which is why the
    study cannot be built from it.

Every number here is measured, never modelled. Where the sample is too thin to
say anything, it says so instead of averaging three data points into a
forecast.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from statistics import median

from app.market.providers import dukascopy_bridge

logger = logging.getLogger(__name__)

# Below this a distribution is an anecdote. Reported as "too few to say"
# rather than dressed up with a median.
MIN_SAMPLE = 5

# Minute bars either side of the release: enough before it to price the move
# from, enough after for the reaction to finish.
_BEFORE = timedelta(minutes=5)
_AFTER = timedelta(minutes=70)


class Surprise(StrEnum):
    BEAT = "beat"          # actual above forecast
    MISS = "miss"          # actual below forecast
    INLINE = "inline"


@dataclass(frozen=True)
class Instance:
    when: datetime
    actual: float
    forecast: float
    surprise: Surprise
    move_15m: float | None = None      # percent, signed
    move_60m: float | None = None


@dataclass(frozen=True)
class ReactionStats:
    """What the sample says, including when it says nothing."""

    event_key: str
    symbol: str
    sample: int
    beats: list[float]                 # 60m moves after an upside surprise
    misses: list[float]

    @property
    def enough(self) -> bool:
        return self.sample >= MIN_SAMPLE

    def summary(self) -> str | None:
        """One line a human can act on, or None when the sample is too thin.

        None is a real answer here. A median over three prints spanning two
        rate regimes is worse than silence, because it looks like evidence.
        """
        if not self.enough:
            return None
        parts = []
        for label, moves in (("above forecast", self.beats), ("below forecast", self.misses)):
            if len(moves) < 3:
                continue
            down = sum(1 for m in moves if m < 0)
            direction = "fell" if down > len(moves) / 2 else "rose"
            agreed = down if direction == "fell" else len(moves) - down
            parts.append(f"{label}: {self.symbol} {direction} {agreed} of {len(moves)}, "
                         f"median {median(moves):+.2f}% in 60m")
        return "; ".join(parts) or None


def _pct(a: float, b: float) -> float:
    return (b - a) / a * 100.0 if a else 0.0


def classify(actual: float, forecast: float, *, tolerance: float = 1e-9) -> Surprise:
    if actual > forecast + tolerance:
        return Surprise.BEAT
    if actual < forecast - tolerance:
        return Surprise.MISS
    return Surprise.INLINE


def parse_number(raw: str | None) -> float | None:
    """Calendar values are strings with units: '4.00%', '8.3K', '-11.0K'.

    Returns None rather than 0.0 for anything unparseable -- a print we cannot
    read must not become a surprise of zero.
    """
    if raw is None:
        return None
    text = raw.strip().replace("%", "").replace(",", "").replace("$", "")
    if not text:
        return None
    multiplier = 1.0
    if text[-1:].upper() in {"K", "M", "B", "T"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[text[-1].upper()]
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


async def _window(symbol: str, when: datetime) -> list[dict] | None:
    bars = await dukascopy_bridge.fetch_bars(
        symbol.lower(),
        int((when - _BEFORE).timestamp() * 1000),
        int((when + _AFTER).timestamp() * 1000),
    )
    return bars or None


def move_after(bars: list[dict], release: datetime, minutes: int) -> float | None:
    """Percent move from the last bar before the release to `minutes` after it.

    None when the window does not actually cover that far: a study that
    silently reports the move to whatever bar happened to be last would read a
    truncated download as a small reaction.
    """
    if not bars:
        return None
    release_ms = release.timestamp() * 1000
    cutoff = release_ms + minutes * 60_000
    before = [b for b in bars if b["t"] <= release_ms]
    after = [b for b in bars if release_ms < b["t"] <= cutoff]
    if not after:
        return None
    # The window has to actually reach the mark being measured. Counting bars
    # is not enough: a download that stopped five minutes in has bars after the
    # release and would otherwise be reported as the 60-minute move. One bar of
    # slack, since the bar stamped exactly at the cutoff is the one we want.
    if after[-1]["t"] < cutoff - 60_000:
        return None
    base = before[-1]["c"] if before else bars[0]["o"]
    return _pct(base, after[-1]["c"])


async def measure(symbol: str, instances: list[Instance]) -> list[Instance]:
    """Attach the realised move to each instance we can price.

    An instance whose window is missing keeps its None moves and stays in the
    list: how many prints we could not price is itself worth knowing.
    """
    measured = []
    for inst in instances:
        bars = await _window(symbol, inst.when)
        if not bars:
            measured.append(inst)
            continue
        measured.append(Instance(
            when=inst.when, actual=inst.actual, forecast=inst.forecast,
            surprise=inst.surprise,
            move_15m=move_after(bars, inst.when, 15),
            move_60m=move_after(bars, inst.when, 60),
        ))
    return measured


def aggregate(event_key: str, symbol: str, instances: list[Instance]) -> ReactionStats:
    priced = [i for i in instances if i.move_60m is not None]
    return ReactionStats(
        event_key=event_key,
        symbol=symbol,
        sample=len(priced),
        beats=[i.move_60m for i in priced if i.surprise is Surprise.BEAT],
        misses=[i.move_60m for i in priced if i.surprise is Surprise.MISS],
    )


def instances_from_history(rows: list[dict], *, limit: int = 24) -> list[Instance]:
    """Past prints of one event, newest first, keeping only those with both an
    actual and a forecast -- an instance without both cannot be a surprise."""
    out = []
    for row in rows:
        actual = parse_number(row.get("Actual"))
        forecast = parse_number(row.get("Forecast"))
        if actual is None or forecast is None:
            continue
        try:
            when = datetime.fromisoformat(row["DateTime"]).astimezone(UTC)
        except (KeyError, ValueError, TypeError):
            continue
        out.append(Instance(when=when, actual=actual, forecast=forecast,
                            surprise=classify(actual, forecast)))
    out.sort(key=lambda i: i.when, reverse=True)
    return out[:limit]
