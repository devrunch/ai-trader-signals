"""Runs Pine source in a persistent Node sandbox process
(app/pine_sandbox/run_pine_server.mjs) over line-delimited JSON on its
stdin/stdout.

Mirrors graph_agent.py's own diascript-validate subprocess pattern in
spirit -- never trusted in-process -- but the child process is now
long-lived rather than spawned fresh per call. Spawning a full Node
process (V8 init, pinets' module load) on every single indicator attach
was itself the cost that raced badly under the production box's memory
pressure: t4g.small, 2GB RAM, confirmed live at 67MB free with 500MB
already in swap under normal load. A persistent process still isolates
each individual Pine run in its own worker_thread (same resourceLimits
and timeout/terminate as before) -- what's shared now is only the outer
Node process and its already-loaded modules, not execution state between
runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_SANDBOX_DIR = Path(__file__).resolve().parents[2] / "pine_sandbox"
_SERVER_SCRIPT = _SANDBOX_DIR / "run_pine_server.mjs"

# Guards concurrent access to the single persistent process's shared
# stdin/stdout pipes -- two interleaved requests would corrupt each
# other's line-framed JSON, since the server assumes at most one request
# in flight. As a side effect this also caps how many Node processes can
# ever be starting up at once during a respawn, which is the same
# memory-pressure concern that motivated this design in the first place.
_SANDBOX_SEMAPHORE = asyncio.Semaphore(1)

# asyncio's StreamReader defaults to a 64KB line-length limit. A response
# is written as one JSON line -- 1800+ bars across a couple of plots
# routinely clears that, and readline() raises ValueError with the
# oversized data still sitting unread in the buffer, which then re-raises
# on every subsequent readline() until the process is killed. Found live:
# one large response wedged the shared server for every request after it,
# a strictly worse failure than the per-request subprocess it replaced.
# 16MB matches the same headroom already given to Express's body limit
# for this exact class of payload (ai-trader-api's bootstrap.ts).
_STREAM_LIMIT = 16 * 1024 * 1024

_process: asyncio.subprocess.Process | None = None
# asyncio.subprocess.Process is bound to the loop that created it -- a real
# server has exactly one loop for its whole process lifetime, but this is
# tracked defensively anyway: it's what makes the difference between a
# genuinely dead/exited process (safe to await/kill normally) and one whose
# owning loop is gone (touching it through asyncio would raise "Event loop
# is closed" -- caught live running the test suite, where pytest-asyncio
# hands each test function its own loop).
_process_loop: asyncio.AbstractEventLoop | None = None


async def _drain_stderr(proc: asyncio.subprocess.Process) -> None:
    """Logs the sandbox's own stderr as it arrives. A long-lived process's
    stderr pipe fills its OS buffer and eventually blocks the child from
    writing to it at all if nothing ever reads it -- unlike the old
    one-shot process, which only ever wrote a few lines before exiting."""
    assert proc.stderr is not None
    async for line in proc.stderr:
        logger.warning("pine sandbox stderr: %s", line.decode(errors="replace").rstrip())


def _kill_by_pid(pid: int) -> None:
    """Bypasses asyncio entirely -- for a process whose owning loop is
    already gone, asyncio's own subprocess.kill() would try to touch that
    dead loop. A raw signal by pid needs no loop at all. SIGTERM rather
    than SIGKILL: the latter isn't defined on Windows, where this test
    suite also runs, and a defunct-loop orphan doesn't need to die
    instantly, just eventually."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


async def _ensure_process() -> asyncio.subprocess.Process:
    global _process, _process_loop
    current_loop = asyncio.get_running_loop()
    if _process is not None and _process_loop is current_loop and _process.returncode is None:
        return _process
    if _process is not None and _process_loop is not current_loop:
        _kill_by_pid(_process.pid)
    _process = await asyncio.create_subprocess_exec(
        "node", str(_SERVER_SCRIPT),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=str(_SANDBOX_DIR), limit=_STREAM_LIMIT,
    )
    _process_loop = current_loop
    asyncio.create_task(_drain_stderr(_process))
    logger.info("pine sandbox process started (pid %s)", _process.pid)
    return _process


async def _kill_process() -> None:
    global _process, _process_loop
    if _process is None:
        return
    proc, _process = _process, None
    same_loop, _process_loop = _process_loop is asyncio.get_running_loop(), None
    if not same_loop:
        _kill_by_pid(proc.pid)
        return
    try:
        proc.kill()
        await proc.wait()
    except ProcessLookupError:
        pass


async def _run_once(payload: str, timeout_s: float) -> dict[str, Any]:
    async with _SANDBOX_SEMAPHORE:
        proc = await _ensure_process()
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(payload.encode() + b"\n")
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_s + 2)
            if not line:
                raise RuntimeError("sandbox process closed its output")
            return json.loads(line.decode())
        except (asyncio.TimeoutError, RuntimeError, ConnectionResetError, BrokenPipeError, ValueError) as e:
            # Any anomaly here leaves the shared stdin/stdout stream in an
            # indeterminate state -- e.g. a half-written request, a
            # response line still to come from a request we gave up on, or
            # (ValueError) a line that overran the stream's length limit
            # with the overflow still sitting unread in the buffer -- never
            # try to reuse it, or every request after this one fails the
            # same way against a permanently wedged process. The next call
            # pays a fresh process's startup cost once instead.
            logger.warning("pine sandbox did not respond: %s", e)
            await _kill_process()
            return {"ok": False, "plots": None, "strategy": None, "error": "sandbox unavailable"}


async def run_pine_script(
    source: str,
    bars: list[dict[str, Any]],
    mode: Literal["indicator", "strategy"] = "indicator",
    timeout_s: float = 5.0,
    ticker_id: str | None = None,
    timeframe: str | None = None,
    symbol_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = json.dumps({
        "source": source,
        "bars": bars,
        "mode": mode,
        "timeoutMs": int(timeout_s * 1000),
        "tickerId": ticker_id,
        "timeframe": timeframe,
        "symbolInfo": symbol_info,
    })
    result = await _run_once(payload, timeout_s)
    if result.get("error") == "sandbox unavailable":
        # A brief memory-pressure spike can still hit an in-flight request
        # even with spawns serialized -- one retry (against a freshly
        # respawned process by this point) gives it a few seconds to pass,
        # against a failure the frontend previously had no way to recover
        # from (an indicator that silently never rendered).
        logger.info("retrying pine sandbox run after a transient failure")
        result = await _run_once(payload, timeout_s)
    return result


async def shutdown() -> None:
    """Call from the FastAPI app's lifespan shutdown. Unlike the old
    one-shot process (which always exited on its own after each request),
    this one stays running until explicitly killed -- otherwise it
    outlives the container's main process as an orphan until Docker
    force-kills the whole cgroup."""
    await _kill_process()
