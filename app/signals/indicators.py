"""
Expanded indicator catalogue — the single implementation of indicator maths.

Indicators are exposed through an explicit allow-list rather than by passing
user/model input into getattr() on the TA library — that keeps the surface
predictable and prevents arbitrary attribute access.

There used to be a second copy of this maths inside `SignalService`. The two
drifted: the signal path had a session-anchored VWAP while this one still had
the whole-frame cumsum, so the chat agent and the signal it was discussing
reported different VWAPs for the same chart. There is now one implementation
and every caller goes through `compute()`.

Candlestick patterns live in `patterns.py`.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from app.signals.types import Indicators

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicator catalogue — name -> (callable, description)
# Each callable takes the OHLCV frame and returns a dict of latest values.
# ---------------------------------------------------------------------------

def _last(series) -> float | None:
    """Latest finite value of a series, or None.

    None here means "not available" — usually too few warm-up bars for the
    indicator's period, which pandas_ta signals either by returning `None`
    outright or by returning a series whose tail is NaN. Deliberately narrow
    otherwise: a genuine bug in the TA call should surface as an exception in
    `compute()`, which logs it and records the indicator as failed, rather than
    being laundered into a missing value that the confluence gate then silently
    treats as an abstention.
    """
    if series is None:
        return None
    try:
        v = float(series.iloc[-1])
    except (IndexError, KeyError, ValueError, TypeError):
        return None
    if v != v:  # NaN — insufficient warm-up
        return None
    return round(v, 4) if abs(v) < 1000 else round(v, 2)


def _absent(*keys: str) -> dict[str, None]:
    """All-None result for a multi-output indicator pandas_ta declined to
    compute (it returns None rather than an empty frame when the input is
    shorter than the indicator's period)."""
    return {k: None for k in keys}


def _build_catalogue() -> dict[str, tuple[Callable, str]]:
    # Deferred purely because pandas_ta is slow to import — NOT to break a
    # cycle. There is no import cycle anywhere in this package. The catalogue
    # is built once, on first `compute()`.
    import pandas_ta as ta

    def rsi(df):        return {"rsi": _last(ta.rsi(df["close"], length=14))}
    def stoch(df):
        r = ta.stoch(df["high"], df["low"], df["close"])
        if r is None: return _absent("stoch_k", "stoch_d")
        return {"stoch_k": _last(r.iloc[:, 0]), "stoch_d": _last(r.iloc[:, 1])}
    def stochrsi(df):
        r = ta.stochrsi(df["close"])
        if r is None: return _absent("stochrsi_k", "stochrsi_d")
        return {"stochrsi_k": _last(r.iloc[:, 0]), "stochrsi_d": _last(r.iloc[:, 1])}
    def macd(df):
        r = ta.macd(df["close"])
        if r is None: return _absent("macd", "macd_hist", "macd_signal")
        return {"macd": _last(r.iloc[:, 0]), "macd_hist": _last(r.iloc[:, 1]), "macd_signal": _last(r.iloc[:, 2])}
    def willr(df):      return {"williams_r": _last(ta.willr(df["high"], df["low"], df["close"]))}
    def cci(df):        return {"cci": _last(ta.cci(df["high"], df["low"], df["close"]))}
    def mfi(df):        return {"mfi": _last(ta.mfi(df["high"], df["low"], df["close"], df["volume"]))}
    def roc(df):        return {"roc": _last(ta.roc(df["close"]))}
    def tsi(df):
        r = ta.tsi(df["close"])
        if r is None: return _absent("tsi")
        return {"tsi": _last(r.iloc[:, 0] if hasattr(r, "iloc") and getattr(r, "ndim", 1) > 1 else r)}
    def uo(df):         return {"ultimate_osc": _last(ta.uo(df["high"], df["low"], df["close"]))}
    def adx(df):
        r = ta.adx(df["high"], df["low"], df["close"], length=14)
        if r is None: return _absent("adx", "di_plus", "di_minus")
        return {"adx": _last(r["ADX_14"]), "di_plus": _last(r["DMP_14"]), "di_minus": _last(r["DMN_14"])}
    def aroon(df):
        r = ta.aroon(df["high"], df["low"])
        if r is None: return _absent("aroon_up", "aroon_down", "aroon_osc")
        return {"aroon_up": _last(r["AROONU_14"]), "aroon_down": _last(r["AROOND_14"]), "aroon_osc": _last(r["AROONOSC_14"])}
    def ema(df):        return {"ema20": _last(ta.ema(df["close"], length=20)), "ema50": _last(ta.ema(df["close"], length=50)), "ema200": _last(ta.ema(df["close"], length=200))}
    def sma(df):        return {"sma20": _last(ta.sma(df["close"], length=20)), "sma50": _last(ta.sma(df["close"], length=50))}
    def hma(df):        return {"hma": _last(ta.hma(df["close"], length=20))}
    def bbands(df):
        r = ta.bbands(df["close"], length=20)
        if r is None: return _absent("bb_lower", "bb_mid", "bb_upper", "bb_pct")
        return {"bb_lower": _last(r.filter(like="BBL").iloc[:, 0]), "bb_mid": _last(r.filter(like="BBM").iloc[:, 0]),
                "bb_upper": _last(r.filter(like="BBU").iloc[:, 0]), "bb_pct": _last(r.filter(like="BBP").iloc[:, 0])}
    def keltner(df):
        r = ta.kc(df["high"], df["low"], df["close"])
        if r is None: return _absent("kc_lower", "kc_mid", "kc_upper")
        return {"kc_lower": _last(r.iloc[:, 0]), "kc_mid": _last(r.iloc[:, 1]), "kc_upper": _last(r.iloc[:, 2])}
    def donchian(df):
        r = ta.donchian(df["high"], df["low"])
        if r is None: return _absent("dc_lower", "dc_mid", "dc_upper")
        return {"dc_lower": _last(r.iloc[:, 0]), "dc_mid": _last(r.iloc[:, 1]), "dc_upper": _last(r.iloc[:, 2])}
    def supertrend(df):
        r = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3)
        if r is None: return _absent("supertrend_dir")
        d = r.filter(like="SUPERTd").iloc[:, 0]
        return {"supertrend_dir": int(d.iloc[-1]) if pd.notna(d.iloc[-1]) else None}
    def psar(df):
        r = ta.psar(df["high"], df["low"], df["close"])
        if r is None: return _absent("psar")
        long_ = r.filter(like="PSARl")
        short_ = r.filter(like="PSARs")
        val = None
        if not long_.empty and pd.notna(long_.iloc[-1, 0]):
            val = round(float(long_.iloc[-1, 0]), 2)
        elif not short_.empty and pd.notna(short_.iloc[-1, 0]):
            val = round(float(short_.iloc[-1, 0]), 2)
        return {"psar": val}
    def ichimoku(df):
        r = ta.ichimoku(df["high"], df["low"], df["close"])
        if r is None: return _absent("tenkan", "kijun", "senkou_a", "senkou_b")
        vis = r[0] if isinstance(r, tuple) else r
        return {"tenkan": _last(vis["ITS_9"]), "kijun": _last(vis["IKS_26"]),
                "senkou_a": _last(vis["ISA_9"]), "senkou_b": _last(vis["ISB_26"])}
    def atr(df):        return {"atr": _last(ta.atr(df["high"], df["low"], df["close"], length=14))}
    def obv(df):        return {"obv": _last(ta.obv(df["close"], df["volume"]))}
    def cmf(df):        return {"cmf": _last(ta.cmf(df["high"], df["low"], df["close"], df["volume"]))}
    def volume_stats(df):
        v = df["volume"]
        avg = float(v.tail(20).mean())
        cur = float(v.iloc[-1])
        return {"volume": int(cur), "volume_avg20": int(avg),
                "volume_ratio": round(cur / avg, 2) if avg else None}
    def vwap(df):
        # Intraday VWAP MUST reset each session. A plain cumsum over the whole
        # frame anchors at the start of ~58 days of history and produces a slow
        # long-run mean sitting percentage points from price, which is not the
        # VWAP any intraday trader means.
        #
        # If the index is not datetime-like (some providers return an integer
        # index) we return None rather than falling back to that whole-frame
        # cumsum. The old fallback silently substituted the exact number the
        # comment above condemns, and it was then fed to the LLM prompt as fact
        # with no log line. A missing VWAP is handled everywhere downstream; a
        # confidently wrong one is not.
        typical = (df["high"] + df["low"] + df["close"]) / 3
        pv = typical * df["volume"]
        try:
            session = pd.Series(df.index, index=df.index).dt.date
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning("VWAP unavailable — index is not datetime-like (%s)", e)
            return {"vwap": None}
        series = pv.groupby(session).cumsum() / df["volume"].groupby(session).cumsum()
        return {"vwap": _last(series)}
    def price(df):
        # Not an indicator, but every consumer of this dict needs the last
        # traded price alongside it and used to get it from a second code path.
        return {"ltp": round(float(df["close"].iloc[-1]), 2)}

    return {
        "price": (price, "Last traded price"),
        "rsi": (rsi, "Relative Strength Index (14)"),
        "stochastic": (stoch, "Stochastic oscillator %K/%D"),
        "stochrsi": (stochrsi, "Stochastic RSI"),
        "macd": (macd, "MACD line, histogram and signal"),
        "williams_r": (willr, "Williams %R"),
        "cci": (cci, "Commodity Channel Index"),
        "mfi": (mfi, "Money Flow Index (volume-weighted RSI)"),
        "roc": (roc, "Rate of change"),
        "tsi": (tsi, "True Strength Index"),
        "ultimate_oscillator": (uo, "Ultimate Oscillator"),
        "adx": (adx, "ADX trend strength with +DI/-DI"),
        "aroon": (aroon, "Aroon up/down/oscillator"),
        "ema": (ema, "EMA 20/50/200"),
        "sma": (sma, "SMA 20/50"),
        "hma": (hma, "Hull Moving Average"),
        "bollinger": (bbands, "Bollinger Bands"),
        "keltner": (keltner, "Keltner Channels"),
        "donchian": (donchian, "Donchian Channels"),
        "supertrend": (supertrend, "SuperTrend direction"),
        "psar": (psar, "Parabolic SAR"),
        "ichimoku": (ichimoku, "Ichimoku Cloud lines"),
        "atr": (atr, "Average True Range (volatility)"),
        "obv": (obv, "On-Balance Volume"),
        "cmf": (cmf, "Chaikin Money Flow"),
        "volume": (volume_stats, "Current volume vs 20-bar average"),
        "vwap": (vwap, "Volume-weighted average price"),
    }


_CATALOGUE: dict[str, tuple[Callable, str]] | None = None


def catalogue() -> dict[str, tuple[Callable, str]]:
    global _CATALOGUE
    if _CATALOGUE is None:
        _CATALOGUE = _build_catalogue()
    return _CATALOGUE


def available_indicators() -> list[str]:
    return sorted(catalogue().keys())


# The set the signal engine needs. Named so the signal path and any test refer
# to one list rather than repeating it.
SIGNAL_SET = ["price", "rsi", "macd", "ema", "adx", "atr", "supertrend", "vwap"]

DEFAULT_SET = ["rsi", "macd", "ema", "adx", "atr", "supertrend", "vwap", "volume"]


def compute(df: pd.DataFrame, names: list[str] | None = None) -> Indicators:
    """Compute a requested subset of indicators. Unknown names are reported, not
    silently ignored, so the model learns what it may ask for.

    A group that raises is recorded in `_failed` rather than being folded into
    the same "value is None" channel as an indicator that simply lacks warm-up
    bars — the confluence gate treats a None as an abstention, so a genuine
    failure quietly lowers the number of voters and can pass a signal on 2 of 2
    indicators instead of 4 of 4.
    """
    cat = catalogue()
    if not names:
        names = list(DEFAULT_SET)

    out: dict = {}
    unknown: list[str] = []
    failed: list[str] = []
    for raw in names:
        key = str(raw).strip().lower()
        entry = cat.get(key)
        if entry is None:
            unknown.append(raw)
            continue
        try:
            out.update(entry[0](df))
        except Exception as e:
            logger.exception("Indicator %s failed: %s", key, e)
            failed.append(key)
    if unknown:
        out["_unknown_requested"] = unknown
        out["_available"] = available_indicators()
    if failed:
        out["_failed"] = failed
    return out  # type: ignore[return-value]

