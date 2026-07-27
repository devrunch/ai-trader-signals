"""
Signal publishing.

Behind a Protocol so the backtest runner and the morning brief can pass a
`NullPublisher` instead of threading a `publish: bool` flag down through the
generation path, and so a test never needs an SQS client.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Protocol

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


class SqsSignalPublisher:
    """Publishes to SQS; the NestJS consumer persists and broadcasts.

    The boto3 client is built on first use, not in `__init__`, so constructing
    this is free — `SignalService` used to build an SQS client unconditionally,
    which meant `brief.py` paid for one on every run just to borrow the LLM.
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._client = None

    def _sqs(self):
        if self._client is None:
            import boto3
            kwargs = {"region_name": self._settings.aws_region}
            if self._settings.aws_access_key_id:
                kwargs["aws_access_key_id"] = self._settings.aws_access_key_id
                kwargs["aws_secret_access_key"] = self._settings.aws_secret_access_key
            self._client = boto3.session.Session(**kwargs).client("sqs")
        return self._client

    async def publish(self, signal: GeneratedSignal) -> None:
        queue_url: str | None = self._settings.sqs_signals_queue_url
        if not queue_url:
            logger.warning("SQS_SIGNALS_QUEUE_URL not set — signal not published")
            return

        payload = json.dumps(signal_payload(signal))
        kwargs = {"QueueUrl": queue_url, "MessageBody": payload}
        if ".fifo" in queue_url:
            kwargs["MessageGroupId"] = "signals"
            kwargs["MessageDeduplicationId"] = (
                f"{signal.symbol}-{signal.signal_type.value}-{int(signal.entry_price)}"
            )

        # boto3 is synchronous — offload so a slow SQS call does not block the
        # event loop.
        await asyncio.to_thread(lambda: self._sqs().send_message(**kwargs))
