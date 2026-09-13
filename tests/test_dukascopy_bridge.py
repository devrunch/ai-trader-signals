"""
dukascopy_bridge — a one-shot Node subprocess wrapping dukascopy-node
(app/dukascopy_bridge/get_ticks.mjs). Mocks asyncio.create_subprocess_exec
the same way deriv_provider's own tests mock websockets.connect -- this
hits a real vendor over the real network, so nothing here spawns the real
process.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.market.providers import dukascopy_bridge


def _mock_process(stdout: bytes, stderr: bytes = b"", returncode: int = 0):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


class TestBridgePlumbing:
    """The subprocess path every bridge call shares (_run_bridge). It used to
    be covered through the tick-count entry point; that went with the
    tick-volume feature, so this exercises the same plumbing through
    fetch_ticks, which is now its only caller."""

    @pytest.mark.asyncio
    async def test_a_nonzero_exit_degrades_to_none_not_a_crash(self):
        proc = _mock_process(b"", stderr=b"instrument not found", returncode=1)
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec",
                   return_value=proc):
            assert await dukascopy_bridge.fetch_ticks("notreal", 1000000, 2000000) is None

    @pytest.mark.asyncio
    async def test_malformed_stdout_degrades_to_none_not_a_crash(self):
        proc = _mock_process(b"this is not json")
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec",
                   return_value=proc):
            assert await dukascopy_bridge.fetch_ticks("xauusd", 1000000, 2000000) is None

    @pytest.mark.asyncio
    async def test_a_hung_subprocess_times_out_and_degrades_to_none(self):
        proc = _mock_process(b"[]")

        async def _never_returns(*_args, **_kwargs):
            await asyncio.sleep(10)

        proc.communicate = AsyncMock(side_effect=_never_returns)
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec",
                   return_value=proc):
            assert await dukascopy_bridge.fetch_ticks(
                "xauusd", 1000000, 2000000, timeout_s=0.05) is None

    @pytest.mark.asyncio
    async def test_spawn_failure_degrades_to_none_not_a_crash(self):
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec",
                   side_effect=OSError("node not on PATH")):
            assert await dukascopy_bridge.fetch_ticks("xauusd", 1000000, 2000000) is None

class TestFetchTicks:
    """fetch_ticks -- Volume Footprint/TPO's own data source, and the only
    caller of this bridge since the delayed tick-count volume was removed.
    The shared error paths are covered above; this is the request and
    response shape that is particular to it."""

    @pytest.mark.asyncio
    async def test_a_real_response_returns_price_and_timestamp_per_tick(self):
        proc = _mock_process(b'[{"t": 1000123, "p": 2650.5}, {"t": 1000456, "p": 2650.6}]')
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec", return_value=proc) as fake:
            result = await dukascopy_bridge.fetch_ticks("xauusd", 1000000, 2000000)

        assert result == [{"t": 1000123, "p": 2650.5}, {"t": 1000456, "p": 2650.6}]
        assert fake.call_args.args == ("node", str(dukascopy_bridge._SCRIPT))
        import json
        sent_payload = json.loads(proc.communicate.call_args.args[0])
        assert sent_payload == {"instrument": "xauusd", "fromMs": 1000000, "toMs": 2000000, "includePrice": True}

    @pytest.mark.asyncio
    async def test_a_vendor_gap_degrades_to_none(self):
        proc = _mock_process(b"", stderr=b"no data", returncode=1)
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec", return_value=proc):
            assert await dukascopy_bridge.fetch_ticks("xauusd", 1000000, 2000000) is None
