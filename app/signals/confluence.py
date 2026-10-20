"""Does the indicator panel actually agree with a chosen direction?

The old gate asked four indicators: `ema20>ema50`, `supertrend`,
`macd>macd_signal` and `rsi>50`. Measured across nine instruments, three
problems fell out of that at once.

Four mirrored voters give five possible scores — 0, 25, 50, 75, 100 — and the
threshold is 75, so the whole gate reduced to "at least 3 of 4 agree". Worse,
every voter flips with direction, so BUY% + SELL% is always exactly 100: a 2-2
split is a *permanent* no-trade in both directions, which is why Nifty refused
long and short with the identical 50% and looked like two assessments when it
was one tie reported twice. And `macd>macd_signal` read UP on 8 of the 9
instruments, so a quarter of the signal was an indicator saying the same thing
everywhere.

Underneath all of it, three of the four are the same reading: an EMA cross,
SuperTrend and RSI-above-50 are all trend direction.

So this asks more indicators, and groups them by what they measure. A family
votes by internal majority and contributes one share; the score is the mean of
the families that could vote. Trend cannot outvote everything by being
measured three ways, one inert indicator costs a fraction of a family rather
than a quarter of the answer, and a tie inside one family no longer deadlocks
the whole panel.

Every vote is returned, not just the ratio. The old gate reported
"quant confluence 50% < 75%" and nothing else, so when asked which indicators
disagreed the agent paraphrased from the chart instead — it could not see.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# What each family is reading. Grouped because the members correlate: an EMA
# cross and SuperTrend disagree only about speed, not about direction.
FAMILIES: dict[str, str] = {
    "trend": "where price is heading",
    "momentum": "how hard it is being pushed",
    "location": "where it sits in its own range",
}


@dataclass
class Vote:
    name: str
    family: str
    agrees: bool
    detail: str

    def to_dict(self) -> dict:
        return {"indicator": self.name, "family": self.family,
                "agrees": self.agrees, "reading": self.detail}


@dataclass
class Confluence:
    direction: str
    ratio: float                       # 0..100, the mean of family shares
    votes: list[Vote] = field(default_factory=list)
    families: dict[str, float] = field(default_factory=dict)

    @property
    def agreed(self) -> list[str]:
        return [v.name for v in self.votes if v.agrees]

    @property
    def disagreed(self) -> list[str]:
        return [v.name for v in self.votes if not v.agrees]

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "agreement_pct": round(self.ratio, 1),
            "families": {k: round(v, 1) for k, v in self.families.items()},
            "agreed": self.agreed,
            "disagreed": self.disagreed,
            "votes": [v.to_dict() for v in self.votes],
            "how": ("Each family votes by internal majority and contributes one "
                    "equal share, so trend cannot outvote the panel by being "
                    "measured three ways."),
        }

    def summary(self) -> str:
        """One line naming what actually disagreed, because "50%" alone is not
        an answer to "which indicators"."""
        if not self.votes:
            return "no indicators could vote"
        parts = [f"{name} {round(share)}%" for name, share in self.families.items()]
        against = ", ".join(self.disagreed[:4]) or "nothing"
        return f"{round(self.ratio)}% ({'; '.join(parts)}) — against: {against}"


def _num(indicators: Mapping[str, Any], key: str) -> float | None:
    value = indicators.get(key)
    return None if value is None else float(value)


def measure(direction: str, indicators: Mapping[str, Any]) -> Confluence:
    """Every indicator that can speak to this direction, grouped by family."""
    wants_buy = direction == "BUY"
    g = lambda k: _num(indicators, k)  # noqa: E731 - local shorthand, read once

    ltp = g("ltp")
    checks: list[tuple[str, str, bool | None, str]] = [
        # -- trend ----------------------------------------------------------
        ("ema20/ema50", "trend",
         None if g("ema20") is None or g("ema50") is None else g("ema20") > g("ema50"),
         f"{g('ema20')} vs {g('ema50')}"),
        ("ema50/ema200", "trend",
         None if g("ema50") is None or g("ema200") is None else g("ema50") > g("ema200"),
         f"{g('ema50')} vs {g('ema200')}"),
        ("supertrend", "trend",
         None if g("supertrend_dir") is None else g("supertrend_dir") == 1,
         f"dir {g('supertrend_dir')}"),
        ("psar", "trend",
         None if g("psar") is None or ltp is None else ltp > g("psar"),
         f"price {ltp} vs psar {g('psar')}"),
        ("+DI/-DI", "trend",
         None if g("di_plus") is None or g("di_minus") is None else g("di_plus") > g("di_minus"),
         f"+DI {g('di_plus')} vs -DI {g('di_minus')}"),
        ("aroon", "trend",
         None if g("aroon_osc") is None else g("aroon_osc") > 0,
         f"osc {g('aroon_osc')}"),

        # -- momentum -------------------------------------------------------
        ("rsi", "momentum", None if g("rsi") is None else g("rsi") > 50, f"{g('rsi')}"),
        ("macd", "momentum",
         None if g("macd") is None or g("macd_signal") is None else g("macd") > g("macd_signal"),
         f"{g('macd')} vs signal {g('macd_signal')}"),
        ("stochastic", "momentum",
         None if g("stoch_k") is None or g("stoch_d") is None else g("stoch_k") > g("stoch_d"),
         f"K {g('stoch_k')} vs D {g('stoch_d')}"),
        ("williams_r", "momentum",
         None if g("williams_r") is None else g("williams_r") > -50, f"{g('williams_r')}"),
        ("cci", "momentum", None if g("cci") is None else g("cci") > 0, f"{g('cci')}"),
        ("roc", "momentum", None if g("roc") is None else g("roc") > 0, f"{g('roc')}"),
        ("tsi", "momentum", None if g("tsi") is None else g("tsi") > 0, f"{g('tsi')}"),

        # -- location -------------------------------------------------------
        ("bollinger mid", "location",
         None if g("bb_mid") is None or ltp is None else ltp > g("bb_mid"),
         f"price {ltp} vs mid {g('bb_mid')}"),
        ("keltner mid", "location",
         None if g("kc_mid") is None or ltp is None else ltp > g("kc_mid"),
         f"price {ltp} vs mid {g('kc_mid')}"),
        ("donchian mid", "location",
         None if g("dc_mid") is None or ltp is None else ltp > g("dc_mid"),
         f"price {ltp} vs mid {g('dc_mid')}"),
        ("vwap", "location",
         None if g("vwap") is None or ltp is None else ltp > g("vwap"),
         f"price {ltp} vs vwap {g('vwap')}"),
    ]

    votes: list[Vote] = []
    for name, family, bullish, detail in checks:
        if bullish is None:
            # Absent, not neutral. An indicator that could not be computed must
            # not be counted as disagreeing — that is how a thin data feed
            # quietly becomes a bearish signal.
            continue
        votes.append(Vote(name=name, family=family, agrees=(bullish == wants_buy),
                          detail=detail))

    families: dict[str, float] = {}
    for family in FAMILIES:
        members = [v for v in votes if v.family == family]
        if members:
            families[family] = 100.0 * sum(v.agrees for v in members) / len(members)

    ratio = sum(families.values()) / len(families) if families else 0.0
    return Confluence(direction=direction, ratio=ratio, votes=votes, families=families)
