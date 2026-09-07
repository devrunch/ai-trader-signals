"""
drift_check -- hourly diff of the global-cue snapshot against the one it
stored an hour ago. Mocks global_cues.collect() and a fake Redis client
the same shape test_macro_events.py's own fake uses.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.market import drift_check


class _FakeRedis:
    def __init__(self, initial: dict | None = None):
        self.store: dict[str, bytes] = dict(initial or {})
        self.closed = False

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value.encode() if isinstance(value, str) else value

    async def aclose(self):
        self.closed = True


def _settings(redis_url="redis://fake", threshold=0.5):
    s = MagicMock()
    s.redis_url = redis_url
    s.drift_check_move_threshold = threshold
    return s


def _cues(values: dict[str, float]) -> dict:
    return {
        "generated_at": "2026-09-07T12:00:00+05:30",
        "cues": [
            {"symbol": sym, "name": sym, "group": "us", "why": f"{sym} matters", "value": val, "change_pct": 0.0, "as_of": "2026-09-07"}
            for sym, val in values.items()
        ],
    }


class TestCheck:
    @pytest.mark.asyncio
    async def test_first_run_ever_stores_a_snapshot_but_alerts_on_nothing(self):
        fake_redis = _FakeRedis()
        with patch("app.market.drift_check.get_settings", return_value=_settings()), \
             patch("app.market.drift_check.redis.from_url", return_value=fake_redis), \
             patch("app.market.global_cues.collect", return_value=_cues({"^NSEI": 100.0})):
            result = await drift_check.check()
        assert result is None
        assert json.loads(fake_redis.store[drift_check._SNAPSHOT_KEY]) == {"^NSEI": 100.0}
        assert fake_redis.closed

    @pytest.mark.asyncio
    async def test_a_move_past_threshold_since_last_check_produces_an_alert(self):
        fake_redis = _FakeRedis({drift_check._SNAPSHOT_KEY: json.dumps({"^NSEI": 100.0}).encode()})
        with patch("app.market.drift_check.get_settings", return_value=_settings(threshold=0.5)), \
             patch("app.market.drift_check.redis.from_url", return_value=fake_redis), \
             patch("app.market.global_cues.collect", return_value=_cues({"^NSEI": 101.0})):  # +1.0%
            result = await drift_check.check()
        assert result is not None
        assert result["type"] == "drift"
        assert "^NSEI" in result["symbols"]
        assert "+1.00%" in result["title"]

    @pytest.mark.asyncio
    async def test_a_move_below_threshold_produces_no_alert(self):
        fake_redis = _FakeRedis({drift_check._SNAPSHOT_KEY: json.dumps({"^NSEI": 100.0}).encode()})
        with patch("app.market.drift_check.get_settings", return_value=_settings(threshold=0.5)), \
             patch("app.market.drift_check.redis.from_url", return_value=fake_redis), \
             patch("app.market.global_cues.collect", return_value=_cues({"^NSEI": 100.1})):  # +0.1%
            result = await drift_check.check()
        assert result is None

    @pytest.mark.asyncio
    async def test_no_cues_fetched_returns_none_without_touching_redis(self):
        fake_redis = _FakeRedis()
        with patch("app.market.drift_check.get_settings", return_value=_settings()), \
             patch("app.market.drift_check.redis.from_url", return_value=fake_redis), \
             patch("app.market.global_cues.collect", return_value=_cues({})):
            result = await drift_check.check()
        assert result is None
        assert fake_redis.store == {}
