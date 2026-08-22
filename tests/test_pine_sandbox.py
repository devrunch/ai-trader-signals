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
