"""
Hourly market-drift check.

Compares the current global-cue snapshot (global_cues.collect(), the same
real fetch the twice-daily brief uses) against the snapshot this function
itself stored an hour ago, and flags anything that moved past a material
threshold since then. No LLM call: each cue's own curated "why" (see
global_cues.CUES) is already the grounded explanation for what it means --
writing a prose narrative on top would only add fabrication risk without
adding real information. Cheap by design: most hours nothing crosses the
threshold and this returns None without ever building an alert.
"""
from __future__ import annotations

import json
import logging

import redis.asyncio as redis

from app.config import get_settings
from app.market import global_cues

logger = logging.getLogger(__name__)

_SNAPSHOT_KEY = "drift:cues:snapshot"
# A few hours, not a day -- generous over the 1-hour cadence so one missed
# run self-heals, but short enough that a long gap (a redeploy, an outage)
# correctly falls back to "no prior snapshot" instead of diffing against a
# stale multi-day-old picture.
_SNAPSHOT_TTL_SECONDS = 4 * 60 * 60


async def check() -> dict | None:
    """Returns an alert dict (`type: "drift"`) when something moved past
    threshold since the last check. Returns None for "nothing material
    moved" (the common case), "couldn't fetch cues", and "no prior
    snapshot yet to diff against" alike -- the caller (the Celery task)
    skips on None either way, so collapsing those here is fine, unlike the
    fetch-layer functions this calls, which keep them apart for their own
    callers."""
    settings = get_settings()
    cues = await global_cues.collect()
    current = {c["symbol"]: c["value"] for c in cues["cues"]}
    if not current:
        return None

    r = redis.from_url(settings.redis_url)
    try:
        raw = await r.get(_SNAPSHOT_KEY)
        previous = json.loads(raw) if raw else None
        await r.set(_SNAPSHOT_KEY, json.dumps(current), ex=_SNAPSHOT_TTL_SECONDS)
    finally:
        await r.aclose()

    if not previous:
        return None

    moved = []
    for symbol, value in current.items():
        prev_value = previous.get(symbol)
        if not prev_value:
            continue
        pct = (value - prev_value) / prev_value * 100
        if abs(pct) >= settings.drift_check_move_threshold:
            name, _group, why = global_cues.CUES.get(symbol, (symbol, "", ""))
            moved.append({"symbol": symbol, "name": name, "why": why, "pct": round(pct, 2), "value": value})

    if not moved:
        return None

    moved.sort(key=lambda m: abs(m["pct"]), reverse=True)
    biggest = moved[0]
    return {
        "type": "drift",
        "title": f"{biggest['name']} {biggest['pct']:+.2f}% in the last hour",
        "body": "; ".join(f"{m['name']} {m['pct']:+.2f}% ({m['why']})" for m in moved[:5]),
        "symbols": [m["symbol"] for m in moved],
        "data": {"moved": moved, "cues_generated_at": cues["generated_at"]},
    }
