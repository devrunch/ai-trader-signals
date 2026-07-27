"""
Deterministic technical-analysis engine for the chat agent.

The LLM never invents coordinates — it only chooses WHICH of these computed
artifacts to draw/run and narrates them. Everything here is plain math over the
OHLCV DataFrame (DatetimeIndex; lowercase open/high/low/close/volume columns).

Timestamps are returned as epoch milliseconds (what KLineChart expects).
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _ts_ms(idx) -> int:
    return int(pd.Timestamp(idx).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

def _swing_points(df: pd.DataFrame, window: int = 5):
    """Return (swing_high_positions, swing_low_positions) as integer indices."""
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    n = len(df)
    for i in range(window, n - window):
        seg_h = h[i - window:i + window + 1]
        seg_l = l[i - window:i + window + 1]
        if h[i] == seg_h.max() and h[i] > h[i - 1]:
            highs.append(i)
        if l[i] == seg_l.min() and l[i] < l[i - 1]:
            lows.append(i)
    return highs, lows


# ---------------------------------------------------------------------------
# Support / Resistance — cluster recent swing prices into levels
# ---------------------------------------------------------------------------

def _reach(df: pd.DataFrame, last_price: float) -> float:
    """How far a level can sit from price and still matter within one session.

    Roughly twice the recent typical daily range. A fixed +/-6% (the old value)
    returned levels an intraday trade would resolve long before reaching — a
    stop here sits around 1% away. Scaling off realised range gives a volatile
    stock a wider net than a sleepy one, which a fixed percentage cannot.
    """
    rng = (df["high"] - df["low"]).tail(60)
    typical = float(rng.mean()) if len(rng) else 0.0
    if typical <= 0:
        return last_price * 0.02
    # Clamp so a single wild session cannot open the net back up to the old width.
    return max(last_price * 0.005, min(typical * 2.0, last_price * 0.04))


def support_resistance(df: pd.DataFrame, max_levels: int = 6) -> list[dict]:
    """Cluster recent swing highs/lows into price levels.

    Clustering compares each price to the cluster's running CENTRE and enforces a
    hard width cap. Comparing to the last-added price instead lets dense swing
    data chain end-to-end into a single meaningless mega-cluster (observed: 126
    swings collapsing to one "level" spanning the entire range).

    Levels are ranked by RECENCY-WEIGHTED strength, not raw touch count. Ranking
    on count alone let a level touched five times eight weeks ago outrank one
    touched twice yesterday, and old levels decay in relevance — a fact the
    ranking previously had no way to express.
    """
    if len(df) < 20:
        return []
    highs, lows = _swing_points(df)
    last_price = float(df["close"].iloc[-1])
    n = len(df)

    # Carry the bar position with each price so age can be weighted later.
    points: list[tuple[float, int]] = (
        [(float(df["high"].iloc[i]), i) for i in highs]
        + [(float(df["low"].iloc[i]), i) for i in lows]
    )
    if not points:
        return []

    points.sort(key=lambda t: t[0])
    tol = last_price * 0.004        # a level is +/-0.4% wide
    max_width = last_price * 0.008  # and may never span more than 0.8%

    clusters: list[list[tuple[float, int]]] = []
    for pt in points:
        if clusters:
            c = clusters[-1]
            centre = sum(v for v, _ in c) / len(c)
            if abs(pt[0] - centre) <= tol and (pt[0] - c[0][0]) <= max_width:
                c.append(pt)
                continue
        clusters.append([pt])

    near = _reach(df, last_price)
    # A touch loses half its weight every HALF_LIFE bars. Over ~1,500 bars of
    # 15m history that makes last week decisively more relevant than last month
    # without ever discarding older structure outright.
    half_life = max(20.0, n / 8.0)

    levels: list[dict] = []
    for c in clusters:
        level = round(sum(v for v, _ in c) / len(c), 2)
        if abs(level - last_price) > near:
            continue
        weight = sum(0.5 ** ((n - 1 - pos) / half_life) for _, pos in c)
        levels.append({
            "value": level,
            "kind": "resistance" if level >= last_price else "support",
            "strength": len(c),
            "recency_weighted_strength": round(weight, 2),
            "last_touch_bars_ago": n - 1 - max(pos for _, pos in c),
            "distance_pct": round((level - last_price) / last_price * 100, 2),
        })

    # Strongest by recency-weighted strength, tie-broken by proximity; then
    # presented in price order so the list reads like a ladder.
    levels.sort(key=lambda x: (-float(x["recency_weighted_strength"]),
                               abs(float(x["value"]) - last_price)))
    return sorted(levels[:max_levels], key=lambda x: float(x["value"]))

# ---------------------------------------------------------------------------
# Trend line — fit through recent swing lows (uptrend) or highs (downtrend)
# ---------------------------------------------------------------------------

MIN_TRENDLINE_TOUCHES = 3


def trendline(df: pd.DataFrame) -> dict | None:
    """A trend line, or None when the data does not actually support one.

    Three guards, none of which the previous version had:

      * **The slope must match the label.** Direction came purely from whether
        price sat above EMA20, and the line was then drawn through the last two
        swing lows *without checking they ascend*. A descending pair got
        labelled an uptrend and the chart visibly contradicted itself — an error
        the user can see with their own eyes.
      * **At least three touches.** Any two points define a line; three is the
        first number that is evidence rather than arithmetic.
      * **Not already broken.** A line price has closed through is no longer
        acting as support or resistance, whatever it looked like an hour ago.

    None is now the answer far more often, which is correct — a missing line is
    handled everywhere downstream, a wrong one is not.
    """
    if len(df) < 30:
        return None
    highs, lows = _swing_points(df)
    ema = df["close"].ewm(span=20, adjust=False).mean()
    uptrend = bool(df["close"].iloc[-1] >= ema.iloc[-1])
    pivots = lows if uptrend else highs
    if len(pivots) < MIN_TRENDLINE_TOUCHES:
        return None

    col = "low" if uptrend else "high"
    recent = pivots[-MIN_TRENDLINE_TOUCHES:]
    values = [float(df[col].iloc[i]) for i in recent]

    # Slope must agree with the label: ascending lows for an uptrend, descending
    # highs for a downtrend. Compare the ends rather than demanding strict
    # monotonicity — one shallow pullback should not void a real line.
    if (values[-1] > values[0]) != uptrend:
        return None

    p1, p2 = recent[0], recent[-1]
    v1, v2 = values[0], values[-1]
    span = p2 - p1
    if span <= 0:
        return None

    # Extrapolate to the latest bar; if price has closed through, it is broken.
    slope = (v2 - v1) / span
    projected = v2 + slope * (len(df) - 1 - p2)
    last_close = float(df["close"].iloc[-1])
    if (last_close < projected) if uptrend else (last_close > projected):
        return None

    return {
        "direction": "up" if uptrend else "down",
        "touches": len(recent),
        "points": [
            {"timestamp": _ts_ms(df.index[p1]), "value": round(v1, 2)},
            {"timestamp": _ts_ms(df.index[p2]), "value": round(v2, 2)},
        ],
    }


# ---------------------------------------------------------------------------
# Fibonacci — most recent major swing high → low over the window
# ---------------------------------------------------------------------------

def fibonacci(df: pd.DataFrame) -> dict | None:
    """Fibonacci retracement over the recent swing, anchored CHRONOLOGICALLY.

    A retracement measures how far price has pulled back from a move, so it has
    to know which end came first: a rally from 100 to 120 that pulls back is a
    different picture from a slide from 120 to 100 that bounces.

    This used to return high-then-low unconditionally, so on every up-swing the
    anchors were reversed and the 38.2% and 61.8% levels swapped places. The
    50% level is symmetric and therefore landed correctly either way — which is
    exactly what hid the bug: it looked approximately right and was specifically
    wrong.
    """
    if len(df) < 20:
        return None
    window = df.tail(120)
    hi_idx = window["high"].idxmax()
    lo_idx = window["low"].idxmin()
    hi = round(float(window["high"].max()), 2)
    lo = round(float(window["low"].min()), 2)
    if hi == lo:
        return None

    # Whichever extreme happened first is the anchor; the swing runs to the other.
    low_first = window.index.get_loc(lo_idx) < window.index.get_loc(hi_idx)
    direction = "up" if low_first else "down"
    first = ({"timestamp": _ts_ms(lo_idx), "value": lo} if low_first
             else {"timestamp": _ts_ms(hi_idx), "value": hi})
    second = ({"timestamp": _ts_ms(hi_idx), "value": hi} if low_first
              else {"timestamp": _ts_ms(lo_idx), "value": lo})

    span = hi - lo
    # Retracement levels measured back from the END of the swing.
    levels = {
        f"{pct:.1f}%": round((hi - span * pct / 100) if low_first else (lo + span * pct / 100), 2)
        for pct in (23.6, 38.2, 50.0, 61.8, 78.6)
    }

    return {
        "points": [first, second],
        "direction": direction,
        "high": hi,
        "low": lo,
        "levels": levels,
    }


# ---------------------------------------------------------------------------
# Strategy backtests (long-only, simple, educational)
# ---------------------------------------------------------------------------

def _run_signals(
    df: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    stop_pct: float | None = 2.0,
    cost_pct: float = 0.0,
) -> dict:
    """Simulate a long-only, single-position run over boolean entry/exit series.

    Two corrections over the naive version, both of which flattered results:

      * **Entry fills at the NEXT bar's open, not this bar's close.** A signal
        computed from a bar's close cannot be acted on until that bar has
        closed — by which time the earliest available price is the next bar's
        open. Filling at the close of the deciding bar is one-bar lookahead: it
        books a price that could not have been obtained, on every single trade.
      * **A stop loss exists.** Without one a position could run -20% and still
        count as a single open trade, so the reported returns described a
        strategy nobody would run. The stop is checked against each bar's LOW
        and, if the bar gapped through it, fills at that bar's open.

    `stop_pct=None` disables the stop, which is only useful for comparison.
    """
    op = df["open"].values
    low = df["low"].values
    close = df["close"].values
    ts = df.index
    n = len(df)

    trades: list[dict] = []
    in_pos = False
    entry_price = 0.0
    entry_ts = None
    stop_price = 0.0

    def book(exit_price: float, i: int, reason: str) -> None:
        gross = (exit_price - entry_price) / entry_price * 100
        trades.append({
            "entry_ts": _ts_ms(entry_ts), "entry_price": round(entry_price, 2),
            "exit_ts": _ts_ms(ts[i]), "exit_price": round(exit_price, 2),
            "exit_reason": reason,
            "pnl_pct": round(gross - cost_pct, 2),
        })

    for i in range(n):
        if in_pos:
            # Stop is checked before the exit rule: within one bar we cannot see
            # which came first, so assume the adverse one.
            if stop_pct is not None and low[i] <= stop_price:
                fill = min(float(op[i]), stop_price)   # gap-down fills at the open
                book(fill, i, "stop")
                in_pos = False
                continue
            if bool(exits.iloc[i]):
                book(float(close[i]), i, "signal")
                in_pos = False
                continue

        if not in_pos and bool(entries.iloc[i]) and i + 1 < n:
            in_pos = True
            entry_price = float(op[i + 1])            # next bar's open, not this close
            entry_ts = ts[i + 1]
            stop_price = entry_price * (1 - stop_pct / 100) if stop_pct is not None else 0.0

    wins = [t for t in trades if t["pnl_pct"] > 0]
    total_return = 1.0
    for t in trades:
        total_return *= (1 + t["pnl_pct"] / 100)

    # A position still open at the end of the window is NOT a completed trade
    # and is excluded from the win rate — but it must be reported, not
    # silently dropped. A strategy that enters and never exits would otherwise
    # show a flattering win rate over the handful of trades that did close.
    open_trade = None
    if in_pos:
        mark = float(close[-1])
        open_trade = {
            "entry_ts": _ts_ms(entry_ts), "entry_price": round(entry_price, 2),
            "marked_at": round(mark, 2),
            "unrealised_pct": round((mark - entry_price) / entry_price * 100, 2),
        }

    return {
        "num_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "total_return_pct": round((total_return - 1) * 100, 2),
        "stop_pct": stop_pct,
        "stopped_out": len([t for t in trades if t["exit_reason"] == "stop"]),
        "open_trade": open_trade,
        "trades": trades[-20:],  # cap markers
    }

def backtest(df: pd.DataFrame, strategy: str, stop_pct: float | None = 2.0,
             cost_pct: float = 0.0) -> dict:
    # Deferred purely because pandas_ta is slow to import — NOT to break a
    # cycle. There is no import cycle anywhere in this package.
    import pandas_ta as ta
    close = df["close"]
    strategy = (strategy or "ma_cross").lower()

    if strategy in ("ma_cross", "macross", "ma", "moving_average"):
        fast = ta.ema(close, length=9)
        slow = ta.ema(close, length=21)
        entries = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        exits = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        label = "EMA 9/21 Cross"
    elif strategy in ("rsi",):
        rsi = ta.rsi(close, length=14)
        entries = (rsi > 30) & (rsi.shift(1) <= 30)
        exits = (rsi < 70) & (rsi.shift(1) >= 70)
        label = "RSI(14) 30/70"
    elif strategy in ("macd",):
        macd = ta.macd(close)
        line, sig = macd["MACD_12_26_9"], macd["MACDs_12_26_9"]
        entries = (line > sig) & (line.shift(1) <= sig.shift(1))
        exits = (line < sig) & (line.shift(1) >= sig.shift(1))
        label = "MACD Cross"
    elif strategy in ("bollinger", "boll", "bbands"):
        bb = ta.bbands(close, length=20)
        lower = bb.filter(like="BBL").iloc[:, 0]
        upper = bb.filter(like="BBU").iloc[:, 0]
        entries = close < lower
        exits = close > upper
        label = "Bollinger Band Reversion"
    else:
        return {"error": f"Unknown strategy '{strategy}'", "supported": ["ma_cross", "rsi", "macd", "bollinger"]}

    entries = entries.fillna(False)
    exits = exits.fillna(False)
    result = _run_signals(df, entries, exits, stop_pct=stop_pct, cost_pct=cost_pct)
    result["strategy"] = label
    return result


# ---------------------------------------------------------------------------
# Trade simulation — scenario P&L for a proposed trade
# ---------------------------------------------------------------------------

def simulate_trade(side: str, entry: float, target: float, stop: float, quantity: int = 1) -> dict:
    side = (side or "BUY").upper()
    if side == "BUY":
        reward_ps = target - entry
        risk_ps = entry - stop
    else:
        reward_ps = entry - target
        risk_ps = stop - entry
    rr = round(abs(reward_ps) / max(abs(risk_ps), 0.01), 2)
    return {
        "side": side,
        "entry": round(entry, 2), "target": round(target, 2), "stop": round(stop, 2),
        "quantity": quantity,
        "profit_at_target": round(reward_ps * quantity, 2),
        "loss_at_stop": round(-abs(risk_ps) * quantity, 2),
        "reward_risk": rr,
        "capital_required": round(entry * quantity, 2),
    }
