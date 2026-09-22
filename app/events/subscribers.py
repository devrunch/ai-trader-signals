"""Who receives the desk's messages.

Until now that was one chat id in config. The registry replaces it: chats
enroll themselves with `/start <key>` and leave with `/stop`, and every push
fans out across whoever is in the set.

The seed is the only subtle part. When this module first runs on a box that
has been delivering briefs to a configured chat, an empty set would silently
stop them — so the configured chat is added once. Once, not whenever the set
is empty: an empty set means either "never seeded" or "they asked to leave",
and those are opposites. The flag distinguishes them.
"""
from __future__ import annotations

import logging

import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)

SET_KEY = "events.subscribers"
SEEDED_KEY = "events.subscribers.seeded"


def _client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)


async def _ensure_seeded(client) -> None:
    """Carry the configured chat into the registry, at most once ever.

    `set(nx=True)` is the whole lock: two processes racing here both call it
    and exactly one gets a truthy answer, so the configured chat cannot be
    added twice or added back after a `/stop`.
    """
    if not await client.set(SEEDED_KEY, "1", nx=True):
        return
    chat_id = str(getattr(get_settings(), "telegram_chat_id", "") or "")
    if chat_id:
        await client.sadd(SET_KEY, chat_id)
        logger.info("Subscriber registry seeded from telegram_chat_id")


async def add(chat_id: str) -> None:
    client = _client()
    try:
        await _ensure_seeded(client)
        await client.sadd(SET_KEY, str(chat_id))
    finally:
        await client.aclose()


async def remove(chat_id: str) -> None:
    client = _client()
    try:
        # Seeded first, deliberately: a `/stop` that arrives before anything
        # has read the registry must still be able to remove the seeded chat.
        await _ensure_seeded(client)
        await client.srem(SET_KEY, str(chat_id))
    finally:
        await client.aclose()


async def all() -> list[str]:
    """Every enrolled chat. Sorted so a fan-out is deterministic to read in
    logs, not because order means anything."""
    client = _client()
    try:
        await _ensure_seeded(client)
        return sorted(await client.smembers(SET_KEY))
    finally:
        await client.aclose()


async def contains(chat_id: str) -> bool:
    return str(chat_id) in await all()


async def count() -> int:
    client = _client()
    try:
        await _ensure_seeded(client)
        return int(await client.scard(SET_KEY))
    finally:
        await client.aclose()
