"""Delivery. One bot, a registry of chats, no framework.

A brief that arrives hours before an event is worth something; the same brief
sitting in a web panel he has to remember to open is not. This is the only
push channel the product has.

`send` fans out across the subscriber registry rather than a single configured
chat. Each chat is attempted on its own: one blocked bot or one dead chat must
not cost everyone else the brief, so a failure is logged against its chat id
and the loop continues.

Failure is logged and swallowed: a delivery problem must not take down the job
that produced the content, and the next brief is minutes away, not hours.
"""
from __future__ import annotations

import logging

import httpx

from app.config import get_settings
from app.events import subscribers

logger = logging.getLogger(__name__)

API = "https://api.telegram.org/bot{token}/{method}"
_TIMEOUT = 15.0
# Telegram rejects anything longer; briefs run well under this, but a runaway
# event list must be truncated visibly rather than silently refused.
MAX_MESSAGE_CHARS = 4096

# Telegram's way of saying this chat will never accept another message. The
# user blocking the bot IS an unsubscribe; treating it as a transient error
# means paying for a delivery that can never land, forever.
_GONE_MARKERS = ("bot was blocked", "chat not found", "user is deactivated",
                 "bot was kicked", "group chat was upgraded")


def _truncate(text: str) -> str:
    if len(text) <= MAX_MESSAGE_CHARS:
        return text
    return text[:MAX_MESSAGE_CHARS - 20].rstrip() + "\n_(truncated)_"


def watch_button(token: str) -> dict:
    """The one inline keyboard this product has: watch what actually happens.

    Callback data is capped at 64 bytes by Telegram, which will not hold an
    event title — the token is a handle into `watches`, not the event itself.
    """
    return {"inline_keyboard": [[{"text": "Watch what happens",
                                  "callback_data": f"w:{token}"}]]}


async def send(text: str, buttons: dict | None = None) -> bool:
    """Push one message to every enrolled chat.

    Returns whether it reached at least one of them — "two of three landed" is
    a delivery, and reporting it as a failure would have the agenda job treat
    a working channel as broken.
    """
    chats = await subscribers.all()
    if not chats:
        logger.warning("No Telegram subscribers -- message not delivered")
        return False

    delivered = 0
    for chat_id in chats:
        if await send_to(chat_id, text, buttons):
            delivered += 1
    if delivered < len(chats):
        logger.warning("Telegram: delivered to %d of %d subscribers",
                       delivered, len(chats))
    return delivered > 0


async def send_to(chat_id: str, text: str, buttons: dict | None = None) -> bool:
    """One chat. Used directly for replies, and by the fan-out above."""
    token = getattr(get_settings(), "telegram_bot_token", "")
    if not token or not chat_id:
        logger.warning("Telegram not configured -- message not delivered")
        return False

    body = {"chat_id": str(chat_id), "text": _truncate(text),
            "parse_mode": "Markdown", "disable_web_page_preview": True}
    if buttons:
        body["reply_markup"] = buttons

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                API.format(token=token, method="sendMessage"), json=body,
            )
        if resp.status_code == 200:
            return True

        complaint = (resp.text or "").lower()
        if any(marker in complaint for marker in _GONE_MARKERS):
            logger.info("Chat %s is gone (%s) -- unsubscribing it",
                        chat_id, resp.status_code)
            await subscribers.remove(chat_id)
        else:
            # A 500 from Telegram is Telegram's problem. Dropping a subscriber
            # over it would quietly empty the registry during an outage.
            logger.warning("Telegram rejected the message for %s: %s %s",
                           chat_id, resp.status_code, resp.text[:200])
        return False
    except httpx.HTTPError as e:
        logger.warning("Telegram send to %s failed: %s", chat_id, e)
        return False


async def answer_callback(callback_id: str, text: str) -> bool:
    """Acknowledge a button tap.

    Telegram shows a spinner on the button until this is called, so skipping it
    leaves the user looking at a control that appears stuck even when the work
    succeeded.
    """
    token = getattr(get_settings(), "telegram_bot_token", "")
    if not token or not callback_id:
        return False
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                API.format(token=token, method="answerCallbackQuery"),
                json={"callback_query_id": callback_id, "text": text[:200]},
            )
        return resp.status_code == 200
    except httpx.HTTPError as e:
        logger.warning("Telegram callback answer failed: %s", e)
        return False
