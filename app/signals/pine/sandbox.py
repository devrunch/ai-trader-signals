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


async def _run_once(payload: str, timeout_s: float) -> dict[str, Any]:
    proc = None
    try:
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
        # Observed live: an asyncio subprocess-pipe race closes stdin before
        # the payload write finishes, unrelated to the Pine script itself --
        # the exact same source+bars succeeds on retry, or run directly
        # outside asyncio. One retry costs a few seconds and clears most of
        # these, against a failure the frontend previously had no way to
        # recover from (an indicator that silently never rendered).
        logger.info("retrying pine sandbox run after a transient subprocess failure")
        result = await _run_once(payload, timeout_s)
    return result
