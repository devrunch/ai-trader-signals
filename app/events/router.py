"""The webhook Telegram posts updates to.

Mounted at /telegram in main.py, and exposed publicly by exactly one exact-path
block in the Caddyfile — nothing else on this service's port is reachable from
outside.

The route answers 200 to everything once the header checks out. Telegram
re-delivers any update it did not get a 2xx for, so a handler that throws on
one malformed message would turn into an unbounded retry against a route that
fails it every time. The exception is logged and the update is dropped, which
costs one message; the alternative costs the channel.
"""
from __future__ import annotations

import hmac
import logging

from fastapi import APIRouter, Request, Response

from app.config import get_settings
from app.events import bot, telegram

logger = logging.getLogger(__name__)

router = APIRouter()

SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"


@router.post("/webhook")
async def webhook(request: Request) -> Response:
    expected = str(getattr(get_settings(), "telegram_webhook_secret", "") or "")
    # An unset secret is not "no check" — it would make this world-writable,
    # since an absent header would then compare equal to it.
    if not expected or not hmac.compare_digest(
        request.headers.get(SECRET_HEADER, ""), expected
    ):
        logger.warning("Telegram webhook called without a valid secret token")
        return Response(status_code=401)

    try:
        update = await request.json()
        reply = await bot.handle(update)
        if reply is not None:
            await telegram.send_to(reply.chat_id, reply.text)
    except Exception:
        logger.exception("Dropping a Telegram update this handler could not process")

    return Response(status_code=200)
