"""
Signal publishing.

Behind a Protocol so the backtest runner and the morning brief can pass a
`NullPublisher` instead of threading a `publish: bool` flag down through the
generation path, and so a test never needs a network call.
"""
from __future__ import annotations

import logging
from typing import Protocol

import httpx

from app.config import get_settings
from app.signals.types import GeneratedSignal

logger = logging.getLogger(__name__)


def signal_payload(signal: GeneratedSignal) -> dict:
    """Wire format consumed by the NestJS side.

    snake_case here, camelCase there — the boundary is mapped by hand in
    `ai-trader-api/src/signals/signal.mapper.ts`. Adding a field means editing
    both, or it is silently dropped (Mongoose `create` ignores unknown keys).
    """
    return {
        "symbol": signal.symbol,
        "exchange": signal.exchange,
        "direction": signal.signal_type.value,
        "confidence": signal.confidence,
        "entry_price": signal.entry_price,
        "target_price": signal.target_price,
        "stop_loss": signal.stop_loss,
        "reasoning": signal.reasoning,
        "indicators": signal.indicators,
    }


class SignalPublisher(Protocol):
    async def publish(self, signal: GeneratedSignal) -> None: ...


class NullPublisher:
    """Discards signals.

    Used by the pre-market brief, whose 06:30 signals are priced off the
    PREVIOUS session's close and would otherwise be scored as live intraday
    signals with the entire overnight gap folded into their P&L — and by the
    backtest, which must never touch the live feed.
    """

    async def publish(self, signal: GeneratedSignal) -> None:
        return None


class HttpSignalPublisher:
    """POSTs to the API's internal signals endpoint, which persists and broadcasts.

    Raises on failure; `SignalService.generate` logs it and still returns the signal.
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()

    async def publish(self, signal: GeneratedSignal) -> None:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{self._settings.api_service_url}/api/internal/signals",
                headers={"x-internal-key": self._settings.internal_api_key},
                json=signal_payload(signal),
            )
            r.raise_for_status()
