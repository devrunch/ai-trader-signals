"""A watch: "tell me what actually happened when this printed."

The brief says what the market has historically done. A watch closes the loop
— it fires after the release and reports the move that really happened,
against the distribution the brief quoted. Without it the desk only ever makes
predictions and never scores them.

Two Redis structures, and the split matters:

  * a hash per offered watch, written when a message carrying the button goes
    out. Telegram caps callback data at 64 bytes, which will not hold an event
    title, so the button carries a short token and the payload lives here.
  * a sorted set of armed watches, scored by when they should be reported.

The queue is deliberately not an APScheduler job. The button is tapped in the
signals process and the report runs in newsd; they share a Redis jobstore, but
a job added from outside a running scheduler is not seen until that scheduler
happens to wake for something else. A sorted set that newsd drains on a fixed
cron has no such timing subtlety, and it is directly testable.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta

import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)

OFFER_PREFIX = "events.watch.offer:"
QUEUE_KEY = "events.watches"

# How long after the release to report. The 30-minute move is the widest thing
# measured, and the minute-bar feed is not instant, so this leaves the window
# time to actually exist before it is asked for.
REPORT_DELAY = timedelta(minutes=35)
# A window that is not downloadable yet is retried rather than dropped: the
# feed is usually a few minutes behind, not broken.
RETRY_AFTER = timedelta(minutes=10)
MAX_ATTEMPTS = 4
# An offered button outlives its event by a day, so a tap on yesterday's
# message says "that one is over" instead of silently doing nothing.
OFFER_TTL_SECONDS = 36 * 3600


def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


def token_for(event_key: str, when: datetime) -> str:
    """A short, stable handle for one release. Stable so the same event offered
    twice — once in the brief, once in the nudge — is one watch, not two."""
    raw = f"{event_key}|{when.isoformat()}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


async def offer(event_key: str, currency: str, title: str, when: datetime) -> str:
    """Record what a button will mean, and return the token it carries."""
    token = token_for(event_key, when)
    payload = json.dumps({"event_key": event_key, "currency": currency,
                          "title": title, "when": when.isoformat()})
    client = _client()
    try:
        await client.set(OFFER_PREFIX + token, payload, ex=OFFER_TTL_SECONDS)
    finally:
        await client.aclose()
    return token


async def arm(token: str, chat_id: str) -> dict | None:
    """Act on a tap. Returns the event watched, or None if the offer expired.

    Idempotent per chat: the member encodes both, so tapping twice arms one
    watch. A second person tapping the same message gets their own.
    """
    client = _client()
    try:
        raw = await client.get(OFFER_PREFIX + token)
        if not raw:
            return None
        event = json.loads(raw)
        when = datetime.fromisoformat(event["when"])
        due = (when + REPORT_DELAY).timestamp()
        await client.zadd(QUEUE_KEY, {_member(token, chat_id, 0): due})
        return event
    finally:
        await client.aclose()


async def due(now: datetime | None = None) -> list[dict]:
    """Watches whose report is owed. Each carries its offer payload, so a
    caller never has to reach back into the offer store itself."""
    now = now or datetime.now(UTC)
    client = _client()
    try:
        members = await client.zrangebyscore(QUEUE_KEY, "-inf", now.timestamp())
        out = []
        for member in members:
            token, chat_id, attempts = _parse(member)
            raw = await client.get(OFFER_PREFIX + token)
            if not raw:
                # The offer expired before the report ran. Nothing can be said
                # about an event we no longer have the details of.
                logger.warning("Dropping watch %s: its offer is gone", token)
                await client.zrem(QUEUE_KEY, member)
                continue
            out.append({**json.loads(raw), "member": member, "token": token,
                        "chat_id": chat_id, "attempts": attempts})
        return out
    finally:
        await client.aclose()


async def done(member: str) -> None:
    client = _client()
    try:
        await client.zrem(QUEUE_KEY, member)
    finally:
        await client.aclose()


async def retry(member: str, now: datetime | None = None) -> bool:
    """Put a watch back for another go. False once it has had enough.

    A watch that can never be priced must leave the queue: an unbounded retry
    would have newsd re-downloading the same missing window every five minutes
    for as long as Redis remembers it.
    """
    now = now or datetime.now(UTC)
    token, chat_id, attempts = _parse(member)
    client = _client()
    try:
        await client.zrem(QUEUE_KEY, member)
        if attempts + 1 >= MAX_ATTEMPTS:
            return False
        await client.zadd(QUEUE_KEY, {
            _member(token, chat_id, attempts + 1): (now + RETRY_AFTER).timestamp(),
        })
        return True
    finally:
        await client.aclose()


def _member(token: str, chat_id: str, attempts: int) -> str:
    return f"{token}|{chat_id}|{attempts}"


def _parse(member: str) -> tuple[str, str, int]:
    token, _, rest = member.partition("|")
    chat_id, _, attempts = rest.partition("|")
    try:
        return token, chat_id, int(attempts)
    except ValueError:
        return token, chat_id, 0
