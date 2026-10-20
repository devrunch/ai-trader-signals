"""Trade geometry, computed here so the model cannot author it.

The agent used to write its own setups and the arithmetic was wrong in ways
that read as confident. One live answer offered a long AND a short on gold at
the same moment, quoted "35 pip risk vs 90+ pip reward" for a $30 stop on an
instrument with no pips, and placed that stop 0.4x ATR from entry — inside a
single hour's noise, and a setup the signal engine's own `stop_size` gate
rejects outright.

Every one of those checks already existed in `app/signals/validation.py` and
ran on machine-generated signals. Chat could not reach them. So this tool
takes INTENT from the agent — a direction, optionally a level — and computes
the entry, stop, target, reward:risk and position size itself, then runs the
same gates.

It reports rather than refuses. A rejected setup comes back with the gate that
failed, the numbers behind it, and what would have to be true to pass; a wall
that says "no" teaches the agent nothing and gives the user less. The verdict
is unambiguous either way, so a failing trade cannot be mistaken for a
recommendation.

`long_only` is the one gate deliberately not run. It exists because the
built-in paper account cannot short, not because a short is bad analysis — so
a SELL is computed and flagged `short: True` instead of vanishing.
"""
from __future__ import annotations

import logging
from typing import Any

from app.signals import analysis, confluence, validation
from app.signals import indicators as ind_mod
from app.market.contract import AssetClass
from app.market import symbols as symbol_registry
from app.signals.agent.tools.base import BASE_INTERVAL, Handler, ToolContext

logger = logging.getLogger(__name__)

# A target with no level to aim at falls back to a multiple of the risk taken.
_FALLBACK_TARGET_R = 2.0
# Structural stops sit just beyond the level, not on it: price wicks through a
# level and recovers constantly, and a stop exactly on it is a donation.
_STRUCTURE_BUFFER_ATR = 0.15
# The ATR floor is placed a hair beyond the minimum, never exactly on it.
# `stop_size` recomputes abs(entry - stop) and compares it to atr * min_mult;
# at Nifty's scale that round trip loses ~1e-10, so a stop sitting precisely on
# the boundary failed the gate that had just produced it ("stop 274.86 tighter
# than 1.0x ATR (274.86)"). Placing a stop at exactly the minimum allowed is
# fragile anyway -- no desk does that either.
_FLOOR_SLACK = 1.002


def _contract(symbol: str, exchange: str) -> tuple[AssetClass | None, dict | None]:
    """What kind of instrument this is, and its lot if it has one.

    A derivative's quantity is lots, not units. One Nifty futures lot is 65
    index points of exposure, so sizing that treats the number as shares is
    wrong by that factor — and silently, because the number still looks
    plausible.
    """
    info = symbol_registry.resolve(symbol, exchange)
    if info is None:
        return None, None
    if info.asset_class not in (AssetClass.FUTURE, AssetClass.OPTION):
        return info.asset_class, None
    try:
        from app.market.providers.registry import market_data_router
        provider = market_data_router.providers.get(exchange.upper())
        spec = provider.contract_spec(symbol, exchange) if provider else None
    except Exception:
        logger.warning("Could not read the contract spec for %s", symbol, exc_info=True)
        spec = None
    return info.asset_class, spec


def _contract_risk(risk_per_unit: float, lot_size: int) -> dict:
    """What one lot costs if the stop is hit. Nothing about anybody's money.

    This used to size against the paper account, which produced "with your
    1,00,000 account, standard Nifty futures are inappropriate - build capital
    first": financial advice derived from a simulator's balance. Replacing it
    with an explanation of why it was not advising was no better - a trader
    reading between candles does not want a paragraph about account
    denomination either.

    So it reports the number and stops. Dividing what you are willing to lose
    by the risk per lot is arithmetic the trader can do, and only they know the
    first figure.
    """
    return {
        "unit": "lots",
        "lot_size": lot_size,
        "risk_per_lot": round(risk_per_unit * lot_size, 2),
    }


# How far past a level a retest entry sits. Price overshoots a level and comes
# back through it constantly; an entry exactly on it fills on noise.
_RETEST_BUFFER_ATR = 0.1
# The timeframe above the one being traded. A 1d long against a weekly
# downtrend is a worse trade than its own score says, and nothing in a
# single-timeframe panel can see that.
_HIGHER = {"1m": "15m", "5m": "1h", "15m": "1h", "30m": "1d", "1h": "1d", "1d": "1d"}


def _entry_for(mode: str, direction: str, ltp: float, atr: float,
               below: list[float], above: list[float]) -> tuple[float, str]:
    """Where the trade is entered, which decides whether it is worth taking.

    Scanning 29 Nifty names, 24 were rejected on reward:risk alone - including
    HCLTECH at 94.4% agreement and POWERGRID at 95.2%. The agreement was fine;
    entering at the last traded price was not. A stop one ATR away leaves no
    room to the next level, so the reward is gone before the setup is judged.

    A trader waits for a pullback. These modes let the geometry say so:

      market  - the last price, what this always did
      level   - the nearest level in the trade's favour, which is where a
                pullback would find support (long) or resistance (short)
      retest  - just through that level, where a break-and-retake fills
    """
    long_side = direction == "BUY"
    candidates = below if long_side else above
    if mode == "market" or not candidates:
        return ltp, ("last price" if mode == "market"
                     else "last price (no level to pull back to)")

    level = candidates[0]
    if mode == "retest":
        # Through the level by a fraction of ATR: a long retest fills slightly
        # below support, a short slightly above resistance.
        offset = atr * _RETEST_BUFFER_ATR
        price = level - offset if long_side else level + offset
        return price, f"retest of {level:g}"
    return level, f"pullback to {level:g}"


async def _higher_timeframe(ctx, symbol: str | None, interval: str,
                            direction: str) -> dict | None:
    """Does the timeframe above agree?

    Not a gate - a caveat. A setup that is clean on its own chart and fighting
    the one above it is still a trade, just a worse one, and the trader should
    be told which.
    """
    higher = _HIGHER.get(interval)
    if not higher or higher == interval:
        return None
    try:
        frame = await ctx.frame(symbol, higher)
        if frame is None or len(frame) < 60:
            return None
        report = confluence.measure(direction, dict(ind_mod.compute(frame)))
    except Exception:
        logger.warning("Higher-timeframe check failed for %s", symbol, exc_info=True)
        return None
    if not report.votes:
        return None
    agrees = report.ratio >= 50
    return {
        "interval": higher,
        "agreement_pct": round(report.ratio, 1),
        "aligned": agrees,
        "note": (f"The {higher} chart agrees ({report.ratio:.0f}%)." if agrees
                 else f"The {higher} chart leans the other way ({report.ratio:.0f}% "
                      f"for this direction). Trading against it is a worse trade "
                      f"than this one's own score suggests."),
    }


def _regime(indicators: dict) -> dict:
    """Is a trend reading worth anything here?

    ADX below 20 is a range, and in a range an EMA cross and a SuperTrend flip
    are noise with a direction attached. Reported so the confluence score can
    be read in context rather than taken at face value.
    """
    adx = indicators.get("adx")
    if adx is None:
        return {"adx": None, "kind": "unknown",
                "note": "ADX unavailable - trend readings cannot be weighted."}
    if adx < 20:
        kind, note = "range", ("ADX {:.0f}: ranging. Trend indicators agreeing here "
                               "means less than the number suggests.").format(adx)
    elif adx < 25:
        kind, note = "weak", "ADX {:.0f}: weak trend.".format(adx)
    else:
        kind, note = "trending", "ADX {:.0f}: trending.".format(adx)
    return {"adx": adx, "kind": kind, "note": note}


def _stretched(direction: str, entry: float, indicators: dict, atr: float) -> dict | None:
    """Has it already made the move the setup is predicting?

    TCS scored 94.4% agreement for a BUY at RSI 73, HCLTECH 94.4% at RSI 69 -
    high agreement precisely BECAUSE price had already run. `rsi_extreme` only
    vetoes past 75, so both sat one point from being blocked for the reason
    that made them look good.

    Distance from the 20-EMA in ATRs catches what an RSI threshold misses: it
    is the same question asked of price rather than of an oscillator.
    """
    ema20 = indicators.get("ema20")
    rsi = indicators.get("rsi")
    if ema20 is None or not atr:
        return None
    extension = (entry - ema20) / atr
    long_side = direction == "BUY"
    # Stretched only in the direction of the trade: buying 2 ATR above the mean
    # is chasing; selling there is not.
    if (long_side and extension < 1.5) or (not long_side and extension > -1.5):
        return None
    return {
        "atr_from_ema20": round(extension, 2),
        "rsi": rsi,
        "note": (f"Price is {abs(extension):.1f} ATR "
                 f"{'above' if long_side else 'below'} the 20-EMA. The agreement is "
                 f"high partly because the move already happened."),
    }


def _round(symbol: str, value: float) -> float:
    """Forex pairs quote to 4-5 decimals; everything else to 2. Rounding a
    1.1052 entry to 1.11 moves the stop by 40 pips."""
    digits = 4 if len(symbol) == 6 and not symbol.startswith("XA") else 2
    return round(float(value), digits)


def _levels(df, entry: float) -> tuple[list[float], list[float]]:
    """Real touched levels above and below the entry, nearest first."""
    rows = analysis.support_resistance(df) or []
    below = sorted((float(r["value"]) for r in rows if float(r["value"]) < entry), reverse=True)
    above = sorted(float(r["value"]) for r in rows if float(r["value"]) > entry)
    return below, above


def _geometry(direction: str, entry: float, atr: float, below: list[float],
              above: list[float], settings) -> dict:
    """Stop and target from volatility and structure together.

    The stop is the FURTHER of the two candidates — an ATR floor and the level
    price would have to break — because the tighter one is the one that gets
    hit by noise. Then capped, so "safer" cannot become "sloppy".
    """
    min_mult = getattr(settings, "min_stop_atr_multiple", 1.0)
    max_mult = getattr(settings, "max_stop_atr_multiple", 3.5)
    buffer = atr * _STRUCTURE_BUFFER_ATR
    long_side = direction == "BUY"

    floor = atr * min_mult * _FLOOR_SLACK
    atr_stop = entry - floor if long_side else entry + floor
    structural = (below[0] - buffer) if (long_side and below) else \
                 (above[0] + buffer) if (not long_side and above) else None

    stop = min(atr_stop, structural) if (long_side and structural is not None) else \
           max(atr_stop, structural) if (not long_side and structural is not None) else atr_stop

    # Cap at the outer ATR multiple: a stop beyond it is not caution, it is a
    # position too small to matter or a loss too big to take.
    ceiling = atr * max_mult / _FLOOR_SLACK
    far = entry - ceiling if long_side else entry + ceiling
    capped = (stop < far) if long_side else (stop > far)
    if capped:
        stop = far

    risk = abs(entry - stop)
    min_rr = getattr(settings, "min_reward_risk", 1.5)
    ahead = above if long_side else below

    # The nearest level is usually a minor one a few ticks away, and aiming at
    # it threw away the trade: measured across 29 Nifty names, entering at a
    # level fixed the entry and the reward:risk still came back 0.2-1.4 because
    # the target was whatever happened to be closest. A retest entry was worse
    # still - its target became the very level it had just crossed, 0.1 ATR
    # off, so every retest scored R:R 0.1.
    #
    # So levels too close to the entry are not targets at all, and among the
    # rest the nearest one that actually clears the reward:risk floor is
    # preferred. Failing that, the furthest known level - which is an honest
    # "this is all the structure there is", not a number invented to pass.
    too_close = atr * 0.3
    usable = [lvl for lvl in ahead if abs(lvl - entry) > too_close]
    worth_it = [lvl for lvl in usable if abs(lvl - entry) >= risk * min_rr]

    if worth_it:
        target, from_level = worth_it[0], True
    elif usable:
        target, from_level = usable[-1], True
    else:
        target, from_level = (entry + risk * _FALLBACK_TARGET_R if long_side
                              else entry - risk * _FALLBACK_TARGET_R), False

    return {"stop": stop, "target": target, "risk_per_unit": risk,
            "stop_capped_at_max_atr": capped, "target_from_level": from_level,
            "levels_ahead": len(usable),
            "stop_from_structure": structural is not None and not capped}


def _gates(direction: str, entry: float, target: float, stop: float,
           indicators: dict, atr: float, ltp: float, settings,
           *, at_market: bool = True) -> list[dict]:
    """Every quality gate, each reported whether it passed or failed.

    Reused from the signal engine rather than reimplemented: the thresholds and
    the reasoning behind them live in one place, and a chat answer that
    disagreed with the engine on the same instrument would be the real bug.
    """
    checks = [
        ("price_ordering", validation.price_ordering(direction, entry, target, stop)),
        ("indicator_agreement", validation.quant_confluence(
            direction, indicators, settings.min_quant_confluence)),
        ("not_overextended", validation.rsi_extreme(
            direction, indicators, settings.max_buy_rsi, settings.min_sell_rsi)),
        ("reward_risk", validation.reward_risk(entry, target, stop, settings.min_reward_risk)),
        ("stop_size", validation.stop_size(entry, stop, atr,
                                           settings.min_stop_atr_multiple,
                                           settings.max_stop_atr_multiple)),
    ]
    if at_market:
        # Only meaningful for an order going in now. The gate exists to stop a
        # price being invented far from the market, not to forbid a limit
        # order - and a pullback entry is away from the last price by design.
        # Applied to one, it rejected every level entry on the 29-name scan
        # even where the reward:risk had improved past the floor.
        checks.insert(1, ("entry_drift", validation.entry_drift(
            entry, ltp, atr, settings.max_entry_drift_atr)))
    return [{"gate": name, "passed": bool(ok), "detail": reason} for name, (ok, reason) in checks]


def _what_would_pass(direction: str, entry: float, stop: float, atr: float,
                     symbol: str, settings) -> dict:
    """The target this setup needs to clear the reward:risk floor.

    A refusal that does not say this leaves the agent guessing, and guessing is
    what produced the numbers this tool exists to replace.
    """
    risk = abs(entry - stop)
    needed = settings.min_reward_risk * risk
    target = entry + needed if direction == "BUY" else entry - needed
    return {"target_for_min_rr": _round(symbol, target),
            "min_reward_risk": settings.min_reward_risk,
            "note": f"At this stop, the target must be at least {_round(symbol, target)} "
                    f"to reach {settings.min_reward_risk}:1."}


async def propose_trade(ctx: ToolContext, args: dict) -> Any:
    """A costed, checked trade for one direction. Intent in, geometry out."""
    direction = str(args.get("direction") or "").strip().upper()
    if direction not in ("BUY", "SELL"):
        return {"error": "direction must be BUY or SELL. Decide which before asking."}

    symbol = str(args.get("symbol") or ctx.symbol).upper()
    interval = str(args.get("interval") or BASE_INTERVAL)
    df = await ctx.frame(args.get("symbol"), interval)
    if df is None or len(df) < 60:
        return {"error": f"Not enough {interval} history for {symbol} to size a trade."}

    indicators = dict(ind_mod.compute(df))
    atr = indicators.get("atr")
    ltp = round(float(df["close"].iloc[-1]), 6)
    if not atr:
        # Without ATR there is no volatility to place a stop against, and a
        # fixed-percentage stop on an unknown instrument is a guess.
        return {"error": "ATR unavailable — cannot place a stop without a volatility measure."}

    mode = str(args.get("entry_mode") or "market").lower()
    below_now, above_now = _levels(df, ltp)
    if args.get("entry"):
        entry, entry_from = float(args["entry"]), "the price you gave me"
    else:
        entry, entry_from = _entry_for(mode, direction, ltp, atr, below_now, above_now)
    below, above = _levels(df, entry)
    geo = _geometry(direction, entry, atr, below, above, ctx.settings)
    stop, target, risk = geo["stop"], geo["target"], geo["risk_per_unit"]
    reward = abs(target - entry)

    at_market = mode == "market" and not args.get("entry")
    gates = _gates(direction, entry, target, stop, indicators, atr, ltp, ctx.settings,
                   at_market=at_market)
    failed = [g for g in gates if not g["passed"]]
    # Every vote, not just the percentage. Asked which indicators disagreed,
    # the agent used to paraphrase the chart because the gate told it only a
    # ratio.
    panel = confluence.measure(direction, indicators).to_dict()

    exchange = str(args.get("exchange") or ctx.exchange).upper()
    asset_class, spec = _contract(symbol, exchange)
    # The account is never consulted. It cannot see what the user actually
    # trades with - the only balance here belongs to a paper simulator - and
    # every attempt to use it turned into a claim about their finances.
    sizing = (_contract_risk(risk, spec["lot_size"])
              if spec and spec.get("lot_size")
              else {"unit": "units"})

    out: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "direction": direction,
        "verdict": "rejected" if failed else "passes_every_gate",
        "entry": _round(symbol, entry),
        "stop": _round(symbol, stop),
        "target": _round(symbol, target),
        "last_price": _round(symbol, ltp),
        "risk_per_unit": _round(symbol, risk),
        "reward_per_unit": _round(symbol, reward),
        "reward_risk": round(reward / risk, 2) if risk else None,
        "risk_pct_of_entry": round(risk / entry * 100, 2) if entry else None,
        "atr": atr,
        "stop_in_atr": round(risk / atr, 2),
        "how_stop_was_placed": ("structure beyond the nearest level" if geo["stop_from_structure"]
                                else f"{ctx.settings.max_stop_atr_multiple}x ATR cap"
                                if geo["stop_capped_at_max_atr"]
                                else f"{ctx.settings.min_stop_atr_multiple}x ATR floor"),
        "target_source": "next touched level" if geo["target_from_level"]
                         else f"{_FALLBACK_TARGET_R}x risk (no level in the way)",
        "nearest_levels": {"below": [_round(symbol, v) for v in below[:3]],
                           "above": [_round(symbol, v) for v in above[:3]]},
        "gates": gates,
        "confluence": panel,
        "entry_from": entry_from,
        "entry_mode": mode,
        # Named, because a limit order that never fills is not a loss but it is
        # not a trade either, and the two must not read the same.
        "order_type": "market" if at_market else "limit",
        "distance_from_price_pct": round(abs(entry - ltp) / ltp * 100, 2) if ltp else None,
        "regime": _regime(indicators),
        "position_size": sizing,
        # The handful of readings worth quoting beside a verdict. `regime`
        # above says whether a trend reading means anything at all.
        "key_indicators": {"adx": indicators.get("adx"), "rsi": indicators.get("rsi"),
                           "supertrend_dir": indicators.get("supertrend_dir"),
                           "ema20": indicators.get("ema20"), "ema50": indicators.get("ema50")},
    }

    if spec:
        out["contract"] = spec
    stretched = _stretched(direction, entry, indicators, atr)
    if stretched:
        out["stretched"] = stretched
    higher = await _higher_timeframe(ctx, args.get("symbol"), interval, direction)
    if higher:
        out["higher_timeframe"] = higher

    ctx.results["trade"] = out
    if asset_class is AssetClass.OPTION:
        # The gates measure price. An option's premium is direction, implied
        # volatility and time decay fused together: an ATR stop on it is not a
        # stop on the underlying, reward:risk is not linear in the index move,
        # and indicator agreement is computed on a series that decays to zero
        # by expiry. Reported, not silently applied to the wrong thing.
        out["verdict"] = "not_applicable"
        out["gates"] = []
        out["note"] = (
            "These gates measure price, and an option's premium is not price — it "
            "carries implied volatility and time decay too. An ATR stop here is not "
            "a stop on Nifty, and reward:risk does not hold as the index moves. "
            "Analyse the underlying, then choose a strike. Do not present this "
            "geometry as an option trade."
        )
        out.pop("rejected_because", None)
        out.pop("to_make_it_work", None)
        ctx.results["trade"] = out
        return out

    if direction == "SELL":
        # Said as a fact about the instrument, not about the user. The earlier
        # wording named the paper account and their broker, which is the
        # commentary this tool exists to stop making.
        out["short"] = True

    if failed:
        out["rejected_because"] = [g["detail"] for g in failed]
        if any(g["gate"] == "reward_risk" for g in failed):
            out["to_make_it_work"] = _what_would_pass(direction, entry, stop, atr, symbol, ctx.settings)
        out["note"] = ("This setup fails a gate the signal engine enforces. The numbers are "
                       "real; the trade is not recommended. Say so rather than presenting it.")
    else:
        out["note"] = ("Every gate passed. Numbers are computed here, not estimated — quote "
                       "them as given.")

    # The terminal renders this as a card beside the answer. Published here
    # rather than left for the model to retype: a card can only appear when the
    # geometry was actually computed, so a turn that answered from read_chart
    # alone shows no numbers instead of plausible prose.
    ctx.results["trade"] = out

    return out


TOOLS: dict[str, Handler] = {"propose_trade": propose_trade}
