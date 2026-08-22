"""Runs Pine source in the isolated Node sandbox (app/pine_sandbox/run_pine.mjs).

Mirrors graph_agent.py's own diascript-validate subprocess pattern: a JSON
message over stdin, a JSON result over stdout, never trusted in-process.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_SANDBOX_DIR = Path(__file__).resolve().parents[2] / "pine_sandbox"
_RUN_SCRIPT = _SANDBOX_DIR / "run_pine.mjs"

# The production box runs at its memory ceiling (t4g.small, 2GB -- confirmed
# live at 67MB free with 500MB already in swap under normal load). Spawning
# a Node subprocess means the OS faults in its binary/library pages, and
# under swap pressure that can take long enough that asyncio's subprocess
# transport closes before the payload write finishes ("did not complete:
# ...the handler is closed") -- observed clustering exactly when several
# indicators attach at once and spawn several Node processes simultaneously.
# Serializing spawns doesn't fix the box being undersized for the stack it
# runs, but it stops our own request pattern from being what tips memory
# over the edge; a retry alone doesn't help since the pressure that caused
# the first failure is still there milliseconds later.
_SANDBOX_SEMAPHORE = asyncio.Semaphore(1)


async def _run_once(payload: str, timeout_s: float) -> dict[str, Any]:
    proc = None
    try:
        async with _SANDBOX_SEMAPHORE:
            proc = await asyncio.create_subprocess_exec(
                "node", str(_RUN_SCRIPT),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=str(_SANDBOX_DIR),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(payload.encode()), timeout=timeout_s + 2)
    except (asyncio.TimeoutError, RuntimeError) as e:
        # Same defensive shape as graph_agent.py's own subprocess handling --
        # the parent process must never crash because a child hung or died mid-write.
        if proc is not None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        logger.warning("pine sandbox subprocess did not complete: %s", e)
        return {"ok": False, "plots": None, "strategy": None, "error": "sandbox unavailable"}

    if proc.returncode != 0:
        logger.warning("pine sandbox exited %s: %s", proc.returncode, stderr.decode(errors="replace"))
        return {"ok": False, "plots": None, "strategy": None, "error": "sandbox process failed"}

    return json.loads(stdout.decode())


async def run_pine_script(
    source: str,
    bars: list[dict[str, Any]],
    mode: Literal["indicator", "strategy"] = "indicator",
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    payload = json.dumps({"source": source, "bars": bars, "mode": mode, "timeoutMs": int(timeout_s * 1000)})
    result = await _run_once(payload, timeout_s)
    if result.get("error") == "sandbox unavailable":
        # A brief memory-pressure spike (see _SANDBOX_SEMAPHORE above) can
        # still hit an in-flight request even with spawns serialized -- one
        # retry gives it a few seconds to pass, against a failure the
        # frontend previously had no way to recover from (an indicator that
        # silently never rendered). Not guaranteed to help if the box is
        # under sustained pressure rather than a momentary spike.
        logger.info("retrying pine sandbox run after a transient subprocess failure")
        result = await _run_once(payload, timeout_s)
    return result
