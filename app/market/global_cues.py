"""
Global overnight market cues.

The client's core thesis: the US session and overnight global events move the
Indian open. This module collects the inputs that thesis needs — US indices
(which close ~02:00 IST), Asian markets (which open *before* India, making them
the closest leading signal), currency, commodities and volatility gauges.

Every value here is fetched, never estimated. The LLM writes narrative around
these numbers; it does not produce them.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from app.market.providers.registry import market_data_router

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Errors a scraped vendor API is expected to produce — network trouble, a
# throttled response that does not parse, a symbol Yahoo has stopped serving.
# Anything else is our bug and gets a logger.exception, not a shrug.
_VENDOR_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError)

# symbol -> (display name, group, why it matters)
CUES: dict[str, tuple[str, str, str]] = {
    "^IXIC":     ("NASDAQ",      "us",         "US tech close — strongest read-across to Indian IT"),
    "^GSPC":     ("S&P 500",     "us",         "Broad US risk appetite"),
    "^DJI":      ("Dow Jones",   "us",         "US industrials"),
    "^N225":     ("Nikkei 225",  "asia",       "Opens before India — leading regional signal"),
    "^HSI":      ("Hang Seng",   "asia",       "China/HK sentiment"),
    "USDINR=X":  ("USD/INR",     "currency",   "Rupee pressure; proxy for foreign flows"),
    "BZ=F":      ("Brent crude", "commodity",  "Drives OMCs, paints, aviation, tyres"),
    "GC=F":      ("Gold",        "commodity",  "Risk-off gauge"),
    "^VIX":      ("US VIX",      "volatility", "US fear gauge"),
    "^INDIAVIX": ("India VIX",   "volatility", "Indian fear gauge — spikes justify smaller size"),
    "^NSEI":     ("Nifty 50",    "india",      "Prior Indian close, for gap context"),
}


def _fetch_one(symbol: str) -> dict | None:
    try:
        df = yf.download(symbol, period="7d", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 2:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        close = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        if prev == 0:
            return None
        name, group, why = CUES[symbol]
        return {
            "symbol": symbol,
            "name": name,
            "group": group,
            "why": why,
            "value": round(close, 2),
            "change_pct": round((close - prev) / prev * 100, 2),
            "as_of": str(df.index[-1].date()),
        }
    except _VENDOR_ERRORS as e:
        logger.warning("Global cue fetch failed for %s: %s", symbol, e)
        return None
    except Exception:
        logger.exception("Unexpected error fetching global cue %s", symbol)
        return None


async def collect() -> dict:
    """Fetch all global cues concurrently (yfinance is blocking, so via threads)."""
    results = [
        r for r in await asyncio.gather(*(asyncio.to_thread(_fetch_one, sym) for sym in CUES)) if r
    ]

    by_group: dict[str, list[dict]] = {}
    for r in results:
        by_group.setdefault(r["group"], []).append(r)

    # A cue that failed to fetch is silently absent from `results`, which means
    # the US/Asia averages below are computed over whatever happened to arrive —
    # a two-of-three US read looks identical to a three-of-three one. Say so.
    missing = [sym for sym in CUES if sym not in {r["symbol"] for r in results}]
    if missing:
        logger.warning("Global cues incomplete — %d of %d missing: %s",
                       len(missing), len(CUES), ", ".join(missing))

    return {
        "generated_at": datetime.now(IST).isoformat(),
        "cues": results,
        "by_group": by_group,
        "summary": _summarise(results),
        # Visible degradation: consumers can tell a partial overnight picture
        # from a complete one instead of trusting an average of what arrived.
        "degraded": bool(missing),
        "missing_cues": missing,
    }


def _summarise(results: list[dict]) -> dict:
    """Deterministic read of the overnight picture — no model involved."""
    def get(sym: str) -> dict | None:
        return next((r for r in results if r["symbol"] == sym), None)

    us = [r for r in results if r["group"] == "us"]
    asia = [r for r in results if r["group"] == "asia"]
    us_avg = round(sum(r["change_pct"] for r in us) / len(us), 2) if us else None
    asia_avg = round(sum(r["change_pct"] for r in asia) / len(asia), 2) if asia else None

    india_vix = get("^INDIAVIX")
    crude = get("BZ=F")
    usdinr = get("USDINR=X")

    # Simple, explainable bias model — weighted overnight moves.
    score = 0.0
    if us_avg is not None:
        score += us_avg * 1.0
    if asia_avg is not None:
        score += asia_avg * 1.2          # Asia is closer in time to the Indian open
    if usdinr and usdinr["change_pct"] > 0.3:
        score -= 0.3                      # rupee weakness is a headwind
    if india_vix and india_vix["change_pct"] > 5:
        score -= 0.3                      # spiking fear

    if score > 0.5:
        bias, label = "positive", "Constructive open likely"
    elif score < -0.5:
        bias, label = "negative", "Weak open likely"
    else:
        bias, label = "neutral", "Cautious / range-bound"

    # Deliberately NOT called "confidence". In a financial product that word
    # implies calibration — that an 80%-confidence read is right about 80% of
    # the time. This score is a sum of hand-picked weights that has never been
    # measured against next-day Nifty opens, so it supports no such claim. It
    # describes how STRONGLY the overnight cues point one way, which is a
    # statement about the inputs, not about how often they are right.
    signal_strength = "strong" if abs(score) > 1.2 else "moderate" if abs(score) > 0.4 else "weak"

    notes: list[str] = []
    if crude and abs(crude["change_pct"]) >= 2:
        direction = "fell" if crude["change_pct"] < 0 else "rose"
        helped = "OMCs, paints, aviation" if crude["change_pct"] < 0 else "upstream energy"
        notes.append(f"Crude {direction} {abs(crude['change_pct'])}% — favours {helped}.")
    if india_vix and india_vix["change_pct"] > 5:
        notes.append(f"India VIX up {india_vix['change_pct']}% — reduce position size.")
    if us_avg is not None and abs(us_avg) >= 1:
        notes.append(f"US closed {'down' if us_avg < 0 else 'up'} {abs(us_avg)}% on average — watch IT read-across.")

    return {
        "bias": bias,
        "label": label,
        "signal_strength": signal_strength,
        # Kept under the old key so existing consumers do not break, but it now
        # carries the same honest value rather than a calibration claim.
        "confidence": signal_strength,
        "score": round(score, 2),
        "uncalibrated": True,
        "us_avg_pct": us_avg,
        "asia_avg_pct": asia_avg,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Sensitivity — how much does an Indian stock actually react to a global driver?
# ---------------------------------------------------------------------------

# Drivers whose session CLOSES before the Indian market opens, so today's Indian
# move reacts to their PREVIOUS session. Everything else (crude, gold, FX, Asian
# indices) trades during or overlapping Indian hours and is contemporaneous.
#
# Getting this wrong silently destroys the signal: measured on BPCL vs crude,
# lag-1 gives correlation -0.11 (looks like no relationship) while same-day gives
# -0.48 (the real, economically sensible one — oil marketers are squeezed when
# crude rises). Blanket-shifting every driver is wrong.
_LAGGED_DRIVERS = {"^IXIC", "^GSPC", "^DJI", "^VIX"}


async def sensitivity(symbol: str, exchange: str = "NSE", drivers: list[str] | None = None,
                      lookback_days: int = 180) -> dict:
    """Beta of a stock's daily returns to each global driver.

    This is what turns "NASDAQ fell 1%" into "this stock tends to move X% when
    NASDAQ moves 1%". Plain covariance/variance — deterministic and checkable.
    Returns the correlation alongside beta so weak relationships can be ignored
    rather than presented as insight.
    """
    drivers = drivers or ["^IXIC", "BZ=F", "USDINR=X"]
    stock_df = await market_data_router.get_historical_df(symbol, exchange, interval="1d", days=lookback_days)
    if stock_df is None or len(stock_df) < 30:
        return {"symbol": symbol, "error": "Not enough daily history"}

    stock_ret = stock_df["close"].pct_change().dropna()
    stock_ret.index = pd.to_datetime(stock_ret.index).tz_localize(None).normalize()

    out: dict[str, dict] = {}
    for drv in drivers:
        try:
            d = await asyncio.to_thread(
                yf.download, drv, period=f"{lookback_days}d", interval="1d",
                progress=False, auto_adjust=True,
            )
            if d is None or d.empty:
                out[drv] = {"beta": None, "reason": "no driver data"}
                continue
            d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
            drv_ret = d["Close"].pct_change().dropna()
            drv_ret.index = pd.to_datetime(drv_ret.index).tz_localize(None).normalize()

            # Align by session overlap, not by blanket assumption (see _LAGGED_DRIVERS)
            aligned = drv_ret.shift(1) if drv in _LAGGED_DRIVERS else drv_ret
            joined = pd.concat([stock_ret, aligned], axis=1, join="inner").dropna()
            if len(joined) < 25:
                out[drv] = {"beta": None, "reason": f"only {len(joined)} overlapping days"}
                continue

            y, x = joined.iloc[:, 0], joined.iloc[:, 1]
            var = float(x.var())
            if not var:
                out[drv] = {"beta": None, "reason": "driver has no variance"}
                continue
            beta = float(x.cov(y) / var)
            corr = float(x.corr(y))
            out[drv] = {
                "beta": round(beta, 2),
                "correlation": round(corr, 2),
                "n_days": len(joined),
                "lagged": drv in _LAGGED_DRIVERS,
                # Below ~0.25 the relationship is too weak to build a story on.
                "meaningful": abs(corr) >= 0.25,
            }
        except _VENDOR_ERRORS as e:
            logger.warning("Sensitivity %s vs %s failed: %s", symbol, drv, e)
            out[drv] = {"beta": None, "reason": str(e)[:60]}
        except Exception:
            logger.exception("Unexpected error computing sensitivity %s vs %s", symbol, drv)
            out[drv] = {"beta": None, "reason": "internal error"}

    return {"symbol": symbol.upper(), "lookback_days": lookback_days, "drivers": out}
