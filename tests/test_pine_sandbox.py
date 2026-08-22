import pytest

from app.signals.pine import sandbox as sandbox_module
from app.signals.pine.sandbox import run_pine_script

BARS = [
    {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i, "volume": 1000, "openTime": 1767000900000 + i * 60000}
    for i in range(30)
]


@pytest.mark.asyncio
async def test_run_pine_script_returns_plot_data():
    result = await run_pine_script('//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")', BARS)
    assert result["ok"] is True
    assert isinstance(result["plots"]["SMA5"], list)


@pytest.mark.asyncio
async def test_run_pine_script_reuses_the_same_persistent_process_across_calls():
    # The actual point of this design: no fresh Node process (and its V8 +
    # pinets startup cost) per request.
    await run_pine_script('//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")', BARS)
    pid1 = sandbox_module._process.pid
    await run_pine_script('//@version=5\nindicator("t")\nplot(ta.ema(close, 5), "EMA5")', BARS)
    pid2 = sandbox_module._process.pid
    assert pid1 == pid2


@pytest.mark.asyncio
async def test_shutdown_kills_the_persistent_process():
    await run_pine_script('//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")', BARS)
    assert sandbox_module._process is not None
    await sandbox_module.shutdown()
    assert sandbox_module._process is None


@pytest.mark.asyncio
async def test_run_pine_script_handles_a_production_sized_response():
    # Regression test for a real bug found live: asyncio's StreamReader
    # defaults to a 64KB line-length limit, and a real response -- one
    # JSON line per request, 1800+ bars across a couple of plots -- clears
    # that easily. readline() raised an uncaught ValueError with the
    # oversized line still sitting unread in the buffer, which then
    # wedged the shared server: every request after it failed the same
    # way until the container was restarted.
    big_bars = [
        {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i, "volume": 1000, "openTime": 1767000900000 + i * 60000}
        for i in range(1800)
    ]
    result = await run_pine_script(
        '//@version=5\nindicator("t")\n[m, s, h] = ta.macd(close, 12, 26, 9)\nplot(m, "MACD")\nplot(s, "Signal")\nplot(h, "Histogram")',
        big_bars,
    )
    assert result["ok"] is True
    assert len(result["plots"]["MACD"]) == 1800


@pytest.mark.asyncio
async def test_a_response_over_the_stream_limit_fails_cleanly_and_recovers(monkeypatch):
    # Same bug, forced deterministically: shrink the limit far below any
    # real response so readline() hits it every time, and confirm the
    # failure is a clean structured error (not an uncaught exception
    # crashing the request) and that a later call, once the limit issue no
    # longer applies, still succeeds -- proving the process actually gets
    # killed and respawned rather than staying wedged.
    monkeypatch.setattr(sandbox_module, "_STREAM_LIMIT", 100)
    result = await run_pine_script('//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")', BARS)
    assert result["ok"] is False
    assert result["error"] == "sandbox unavailable"

    monkeypatch.setattr(sandbox_module, "_STREAM_LIMIT", 16 * 1024 * 1024)
    result = await run_pine_script('//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")', BARS)
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_run_pine_script_reports_a_timeout_as_a_structured_error_not_a_hang():
    source = "//@version=5\nindicator(\"t\")\nvar x = 0\nwhile true\n    x := x + 1"
    result = await run_pine_script(source, BARS, timeout_s=0.5)
    assert result["ok"] is False
    assert result["error"]


@pytest.mark.asyncio
async def test_run_pine_script_retries_once_after_a_transient_sandbox_failure(monkeypatch):
    # Observed live: an asyncio subprocess-pipe race closes stdin before the
    # payload write finishes -- the exact same request succeeds moments
    # later. A retry should recover it rather than surface a failure for
    # something that wasn't really wrong with the script.
    calls = []

    async def fake_run_once(payload, timeout_s):
        calls.append(payload)
        if len(calls) == 1:
            return {"ok": False, "plots": None, "strategy": None, "error": "sandbox unavailable"}
        return {"ok": True, "plots": {"SMA5": []}, "strategy": None, "error": None}

    monkeypatch.setattr(sandbox_module, "_run_once", fake_run_once)
    result = await run_pine_script('//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")', BARS)

    assert result["ok"] is True
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_run_pine_script_gives_up_after_one_retry(monkeypatch):
    calls = []

    async def fake_run_once(payload, timeout_s):
        calls.append(payload)
        return {"ok": False, "plots": None, "strategy": None, "error": "sandbox unavailable"}

    monkeypatch.setattr(sandbox_module, "_run_once", fake_run_once)
    result = await run_pine_script('//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")', BARS)

    assert result["ok"] is False
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_run_pine_script_does_not_retry_a_real_script_failure(monkeypatch):
    # "sandbox process failed" (a non-zero exit) and a real ok:False Pine
    # error are not the transient class -- retrying a guaranteed-to-fail
    # script just doubles the latency of an error that was never going away.
    calls = []

    async def fake_run_once(payload, timeout_s):
        calls.append(payload)
        return {"ok": False, "plots": None, "strategy": None, "error": "sandbox process failed"}

    monkeypatch.setattr(sandbox_module, "_run_once", fake_run_once)
    result = await run_pine_script('//@version=5\nindicator("t")\nplot(ta.sma(close, 5), "SMA5")', BARS)

    assert result["ok"] is False
    assert len(calls) == 1
