"""Delivery. One chat, one bot, no framework.

A brief that arrives hours before an event is worth something; the same brief
sitting in a web panel he has to remember to open is not. This is the only
push channel the product has.

Failure is logged and swallowed: a delivery problem must not take down the job
that produced the content, and the next brief is minutes away, not hours.
"""
from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/{method}"
_TIMEOUT = 15.0
# Telegram rejects anything longer; briefs run well under this, but a runaway
# event list must be truncated visibly rather than silently refused.
MAX_MESSAGE_CHARS = 4096


async def send(text: str, chat_id: str | None = None) -> bool:
    """Push one message. Returns whether Telegram accepted it."""
    settings = get_settings()
    token = getattr(settings, "telegram_bot_token", "")
    chat_id = chat_id or getattr(settings, "telegram_chat_id", "")
    if not token or not chat_id:
        logger.warning("Telegram not configured -- brief not delivered")
        return False

    if len(text) > MAX_MESSAGE_CHARS:
        text = text[:MAX_MESSAGE_CHARS - 20].rstrip() + "\n_(truncated)_"

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                API.format(token=token, method="sendMessage"),
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown",
                      "disable_web_page_preview": True},
            )
        if resp.status_code != 200:
            logger.warning("Telegram rejected the message: %s %s",
                           resp.status_code, resp.text[:200])
            return False
        return True
    except httpx.HTTPError as e:
        logger.warning("Telegram send failed: %s", e)
        return False
