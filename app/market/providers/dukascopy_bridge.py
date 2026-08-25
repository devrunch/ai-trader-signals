"""
Bridge to app/dukascopy_bridge/get_ticks.mjs -- a one-shot Node subprocess,
the same shape app/signals/pine/sandbox.py's older one-shot predecessor
used (run_pine.mjs) before that moved to a persistent server for its own,
much higher call volume (every indicator attach). Dukascopy calls happen
far less often -- once per historical fetch, bounded by the same intraday
cache TTL (registry.py's INTRADAY_TTL_SECONDS) that already limits how
often get_historical_df's own vendor call runs -- so a fresh process per
call is the simpler, correct choice here, not a premature optimization
skipped.

dukascopy-node has no plain-REST surface of its own either (same reason
this app already has a Node bridge for Pine) -- it's a real npm package
with no Python equivalent, and reimplementing Dukascopy's raw .bi5 tick-
file format in Python would be real, error-prone reverse-engineering
against an undocumented binary format the maintained JS library already
gets right.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BRIDGE_DIR = Path(__file__).resolve().parents[2] / "dukascopy_bridge"
_SCRIPT = _BRIDGE_DIR / "get_ticks.mjs"

# Same rationale as every other provider's own _VENDOR_ERRORS: these degrade
# to None, the caller's "no data" path. Anything else is a bug in our own
# code and is re-raised through logger.exception.
_VENDOR_ERRORS = (asyncio.TimeoutError, OSError, ValueError, TypeError, KeyError)


async def fetch_tick_timestamps(instrument: str, from_ms: int, to_ms: int, timeout_s: float = 15.0) -> list[int] | None:
    """Raw tick epoch-milliseconds for `instrument` in [from_ms, to_ms), or
    None on any failure -- a real vendor gap and a subprocess/parse error
    are both "we don't have this," not different cases the caller needs to
    tell apart.

    Dukascopy's own publish lag (confirmed live: ~15-20 minutes behind
    real time) means a range reaching up to "now" will come back missing
    its most recent stretch -- not a bug here, the caller's bucketing
    already treats an uncovered candle as 0.0, the same honest fallback
    used everywhere else in this app for a window a vendor hasn't
    published yet.
    """
    payload = json.dumps({"instrument": instrument, "fromMs": from_ms, "toMs": to_ms})
    try:
        proc = await asyncio.create_subprocess_exec(
            "node", str(_SCRIPT),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(_BRIDGE_DIR),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(payload.encode()), timeout=timeout_s)
        if proc.returncode != 0:
            logger.warning(
                "dukascopy bridge exited %s for %s: %s",
                proc.returncode, instrument, stderr.decode(errors="replace")[:300],
            )
            return None
        return json.loads(stdout.decode())
    except _VENDOR_ERRORS as e:
        logger.warning("dukascopy bridge failed for %s: %s", instrument, e)
        return None
    except Exception:
        logger.exception("Unexpected error calling dukascopy bridge for %s", instrument)
        return None
