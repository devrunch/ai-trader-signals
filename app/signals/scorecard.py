"""Did the trades it proposed actually work?

Every number this platform produces is a prediction with no scorecard. The
confluence panel, the entry modes, the reward:risk floor — all of it was tuned
by reasoning about what ought to work, and nothing anywhere records whether
MARUTI SELL at 11,475 made money. Without this, every later improvement is
made blind, and "the agent got better" is an opinion.

So each trade that passes every gate is recorded, and a job walks the open
ones forward against real bars: did price reach the target before the stop, or
the other way, or neither inside the window?

Three decisions worth stating.

A rejected trade is recorded too. The rejections are the larger claim — most
days most names are refused — and a scorecard that only tracked the trades it
liked could never discover that the gates reject winners.

The window closes. A trade neither stopped nor targeted after `MAX_BARS` is
`expired`, not open forever and not counted as a loss. An entry that never
filled is `unfilled`, which is also not a loss: a limit order that did not
trigger risked nothing.

Nothing here feeds back into a decision automatically. It is a record, read by
a human and by `summary()`. A system that retunes its own thresholds against
its own recent results fits the last month and calls it learning.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)

OPEN_KEY = "scorecard:open"          # sorted set, scored by proposal time
RECORD_PREFIX = "scorecard:trade:"
CLOSED_KEY = "scorecard:closed"      # list, newest first
# Long enough to judge a swing trade, short enough that a verdict arrives while
# the setup is still recognisable.
MAX_BARS = 20
# A record outlives its own window by a wide margin so a sweep that misses a
# day still finds it.
RECORD_TTL_SECONDS = 90 * 24 * 3600
MAX_CLOSED = 500


@dataclass
class Proposal:
    trade_id: str
    symbol: str
    exchange: str
    interval: str
    direction: str
    verdict: str
    entry: float
    stop: float
    target: float
    reward_risk: float | None
    confluence_pct: float | None
    entry_mode: str
    order_type: str
    proposed_at: str
    # Filled in by the sweep.
    status: str = "open"             # open | target | stop | expired | unfilled
    outcome_at: str | None = None
    bars_held: int | None = None
    move_pct: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


def trade_id(symbol: str, direction: str, when: datetime, entry: float) -> str:
    """Stable per symbol, side, bar and price.

    Stable so the same setup proposed twice in one session is one record. The
    entry is in the key because the same symbol and side at a different level
    is a different trade, not a repeat.
    """
    return f"{symbol}:{direction}:{when:%Y%m%d%H%M}:{entry:.4f}"


async def record(trade: dict) -> str | None:
    """Remember a costed trade. Returns its id, or None if it was unusable."""
    needed = ("symbol", "direction", "entry", "stop", "target", "verdict")
    if any(trade.get(k) is None for k in needed):
        return None

    now = datetime.now(UTC)
    tid = trade_id(trade["symbol"], trade["direction"], now, float(trade["entry"]))
    proposal = Proposal(
        trade_id=tid,
        symbol=trade["symbol"],
        exchange=str(trade.get("exchange") or ""),
        interval=str(trade.get("interval") or ""),
        direction=trade["direction"],
        verdict=trade["verdict"],
        entry=float(trade["entry"]),
        stop=float(trade["stop"]),
        target=float(trade["target"]),
        reward_risk=trade.get("reward_risk"),
        confluence_pct=(trade.get("confluence") or {}).get("agreement_pct"),
        entry_mode=str(trade.get("entry_mode") or "market"),
        order_type=str(trade.get("order_type") or "market"),
        proposed_at=now.isoformat(),
    )
    try:
        client = _client()
        try:
            # NX so a re-proposal of the same setup does not reset its clock
            # and quietly extend the window it is being judged over.
            written = await client.set(RECORD_PREFIX + tid, json.dumps(proposal.to_dict()),
                                       ex=RECORD_TTL_SECONDS, nx=True)
            if written:
                await client.zadd(OPEN_KEY, {tid: now.timestamp()})
        finally:
            await client.aclose()
    except Exception:
        # Bookkeeping must never take down the thing it is bookkeeping for.
        logger.exception("Could not record the proposed trade %s", tid)
        return None
    return tid


async def open_trades(limit: int = 200) -> list[dict]:
    client = _client()
    try:
        ids = await client.zrange(OPEN_KEY, 0, limit - 1)
        out = []
        for tid in ids:
            raw = await client.get(RECORD_PREFIX + tid)
            if raw:
                out.append(json.loads(raw))
            else:
                # The record expired before its verdict. Dropping the pointer
                # keeps the queue from growing a tail of ghosts.
                await client.zrem(OPEN_KEY, tid)
        return out
    finally:
        await client.aclose()


async def close(record_dict: dict, status: str, bars_held: int,
                move_pct: float | None) -> None:
    record_dict = dict(record_dict)
    record_dict.update(status=status, bars_held=bars_held, move_pct=move_pct,
                       outcome_at=datetime.now(UTC).isoformat())
    tid = record_dict["trade_id"]
    client = _client()
    try:
        await client.set(RECORD_PREFIX + tid, json.dumps(record_dict),
                         ex=RECORD_TTL_SECONDS)
        await client.zrem(OPEN_KEY, tid)
        await client.lpush(CLOSED_KEY, json.dumps(record_dict))
        await client.ltrim(CLOSED_KEY, 0, MAX_CLOSED - 1)
    finally:
        await client.aclose()


def settle(bars: list[dict], proposal: dict) -> tuple[str, int, float | None] | None:
    """Walk bars forward from the proposal and decide what happened.

    Pure, so the rule is testable without Redis or a provider.

    A bar whose range covers both the stop and the target is counted as a STOP.
    Intrabar order is unknowable from an OHLC bar, and assuming the good side
    first is how a backtest flatters itself.
    """
    if not bars:
        return None
    long_side = proposal["direction"] == "BUY"
    entry = float(proposal["entry"])
    stop = float(proposal["stop"])
    target = float(proposal["target"])
    limit_order = proposal.get("order_type") == "limit"

    filled = not limit_order
    fill_index = 0
    for i, bar in enumerate(bars[:MAX_BARS]):
        high, low = float(bar["high"]), float(bar["low"])

        if not filled:
            # A limit fills when price trades through it.
            if (long_side and low <= entry) or (not long_side and high >= entry):
                filled, fill_index = True, i
            continue

        hit_stop = low <= stop if long_side else high >= stop
        hit_target = high >= target if long_side else low <= target
        if hit_stop:
            return "stop", i - fill_index, _move(entry, stop, long_side)
        if hit_target:
            return "target", i - fill_index, _move(entry, target, long_side)

    if not filled:
        # Never triggered. Not a loss — nothing was risked.
        return ("unfilled", len(bars[:MAX_BARS]), None)
    last = float(bars[min(len(bars), MAX_BARS) - 1]["close"])
    return "expired", len(bars[:MAX_BARS]) - fill_index, _move(entry, last, long_side)


def _move(entry: float, exit_price: float, long_side: bool) -> float | None:
    if not entry:
        return None
    raw = (exit_price - entry) / entry * 100.0
    return round(raw if long_side else -raw, 2)


async def summary(closed: list[dict] | None = None) -> dict:
    """What the record actually says. Honest about a thin sample."""
    if closed is None:
        client = _client()
        try:
            rows = await client.lrange(CLOSED_KEY, 0, MAX_CLOSED - 1)
        finally:
            await client.aclose()
        closed = [json.loads(r) for r in rows]

    taken = [t for t in closed if t.get("verdict") == "passes_every_gate"]
    filled = [t for t in taken if t.get("status") in ("target", "stop", "expired")]
    wins = [t for t in filled if t.get("status") == "target"]
    losses = [t for t in filled if t.get("status") == "stop"]

    out = {
        "recorded": len(closed),
        "passed_the_gates": len(taken),
        "filled": len(filled),
        "unfilled": sum(1 for t in taken if t.get("status") == "unfilled"),
        "target_hit": len(wins),
        "stopped": len(losses),
        "expired": sum(1 for t in filled if t.get("status") == "expired"),
    }
    if filled:
        out["hit_rate_pct"] = round(100 * len(wins) / len(filled), 1)
        moves = [t["move_pct"] for t in filled if t.get("move_pct") is not None]
        out["avg_move_pct"] = round(sum(moves) / len(moves), 2) if moves else None
    # Said in the data, not left to the reader. A hit rate over a handful of
    # trades is noise, and it reads as a result unless it is labelled.
    out["trustworthy"] = len(filled) >= 30
    out["note"] = ("Fewer than 30 settled trades — this is a tally, not a track "
                   "record." if len(filled) < 30 else
                   "Settled trades only; unfilled limits are excluded from the hit rate.")
    return out
