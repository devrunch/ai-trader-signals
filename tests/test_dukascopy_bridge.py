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


class TestFetchTickTimestamps:
    @pytest.mark.asyncio
    async def test_a_real_response_returns_the_tick_epochs(self):
        proc = _mock_process(b"[1000123, 1000456, 1000789]")
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec", return_value=proc) as fake:
            result = await dukascopy_bridge.fetch_tick_timestamps("xauusd", 1000000, 2000000)

        assert result == [1000123, 1000456, 1000789]
        # Sent as JSON on stdin, not argv -- matches get_ticks.mjs's own contract.
        sent = fake.call_args
        assert sent.args == ("node", str(dukascopy_bridge._SCRIPT))
        assert sent.kwargs["cwd"] == str(dukascopy_bridge._BRIDGE_DIR)
        proc.communicate.assert_called_once()
        import json
        payload = json.loads(proc.communicate.call_args.args[0])
        assert payload == {"instrument": "xauusd", "fromMs": 1000000, "toMs": 2000000}

    @pytest.mark.asyncio
    async def test_an_empty_range_returns_an_empty_list_not_none(self):
        proc = _mock_process(b"[]")
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec", return_value=proc):
            result = await dukascopy_bridge.fetch_tick_timestamps("xauusd", 1000000, 1000001)

        assert result == []

    @pytest.mark.asyncio
    async def test_a_nonzero_exit_degrades_to_none_not_a_crash(self):
        proc = _mock_process(b"", stderr=b"TypeError: instrument not found", returncode=1)
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec", return_value=proc):
            assert await dukascopy_bridge.fetch_tick_timestamps("notreal", 1000000, 2000000) is None

    @pytest.mark.asyncio
    async def test_malformed_stdout_degrades_to_none_not_a_crash(self):
        proc = _mock_process(b"not json")
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec", return_value=proc):
            assert await dukascopy_bridge.fetch_tick_timestamps("xauusd", 1000000, 2000000) is None

    @pytest.mark.asyncio
    async def test_a_hung_subprocess_times_out_and_degrades_to_none(self):
        proc = AsyncMock()
        proc.communicate = AsyncMock(side_effect=lambda *_: asyncio.sleep(10))
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec", return_value=proc):
            assert await dukascopy_bridge.fetch_tick_timestamps("xauusd", 1000000, 2000000, timeout_s=0.05) is None

    @pytest.mark.asyncio
    async def test_spawn_failure_degrades_to_none_not_a_crash(self):
        with patch("app.market.providers.dukascopy_bridge.asyncio.create_subprocess_exec", side_effect=OSError("node not found")):
            assert await dukascopy_bridge.fetch_tick_timestamps("xauusd", 1000000, 2000000) is None
