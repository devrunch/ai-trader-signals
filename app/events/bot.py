"""What the bot does with an incoming update.

The decision is taken from a plain dict, so every command is a table test
against a fixture update. The one outbound call is `answerCallbackQuery`: a
button tap has to be acknowledged inside the handler, because Telegram spins
the button until it is and the reply message arrives a moment later.

The refusals are deliberately identical to each other. A bot whose reply
differs between "wrong key", "rate limited" and "unknown command" tells a
stranger exactly how far they got, and the invite secret is the only thing
between a stranger and the feed.
"""
from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from app.config import get_settings
from app.events import subscribers, telegram, watchdog, watches

logger = logging.getLogger(__name__)

ATTEMPTS_PREFIX = "events.start_attempts:"
ATTEMPT_WINDOW_SECONDS = 3600
START_ATTEMPTS_PER_HOUR = 5

NOT_RECOGNISED = "Not recognised."
ENABLE_HINT = "Send /start <key> to enable."
ENROLLED = (
    "*Event desk on.*\n\n"
    "You will get the day's agenda each morning, and a brief before every "
    "high-impact release.\n\n"
    "/status — what is armed\n"
    "/stop — turn it off"
)
UNENROLLED = "Event desk off. Send /start <key> to turn it back on."
HELP = "/status — what is armed\n/stop — turn the desk off"
STATUS_UNAVAILABLE = "Desk state is unreadable right now. The jobs are unaffected."
WATCH_ARMED = "Watching. You will get the real move about 35 minutes after the print."
WATCH_EXPIRED = "That event is over."
WATCH_DENIED = "Send /start <key> to enable."


@dataclass(frozen=True)
class Reply:
    chat_id: str
    text: str


async def handle(update: dict) -> Reply | None:
    """One update in, at most one reply out.

    None means "nothing to say" — a message with no text, or a tap that was
    acknowledged on the button and needs no message of its own.
    """
    if isinstance(update.get("callback_query"), dict):
        return await _tap(update["callback_query"])

    message = update.get("message") or update.get("edited_message")
    if not isinstance(message, dict):
        return None
    text = message.get("text")
    chat_id = str(((message.get("chat") or {}).get("id") or "")).strip()
    if not text or not chat_id:
        return None

    command, _, argument = text.strip().partition(" ")
    # Group chats address a command as `/start@thebot`.
    command = command.split("@", 1)[0].lower()
    argument = argument.strip()

    if command == "/start":
        return Reply(chat_id, await _start(chat_id, argument))

    enrolled = await subscribers.contains(chat_id)
    if not enrolled:
        # Every other command answers the same to a stranger, so the command
        # surface cannot be enumerated by trying words.
        return Reply(chat_id, ENABLE_HINT)

    if command == "/stop":
        await subscribers.remove(chat_id)
        return Reply(chat_id, UNENROLLED)
    if command == "/status":
        return Reply(chat_id, await _status())
    return Reply(chat_id, HELP)


async def _tap(query: dict) -> Reply | None:
    """A button press. The only button is "watch what happens".

    The tap is acknowledged either way — Telegram spins the button until it
    is, so an unacknowledged tap looks broken even when it worked.
    """
    chat_id = str(((query.get("message") or {}).get("chat") or {}).get("id") or "").strip()
    data = str(query.get("data") or "")
    callback_id = str(query.get("id") or "")
    if not chat_id or not data.startswith("w:"):
        await telegram.answer_callback(callback_id, "")
        return None

    if not await subscribers.contains(chat_id):
        # Buttons live in forwardable messages, so a tap can arrive from a
        # chat that was never enrolled.
        await telegram.answer_callback(callback_id, WATCH_DENIED)
        return None

    event = await watches.arm(data[2:], chat_id)
    await telegram.answer_callback(callback_id, WATCH_ARMED if event else WATCH_EXPIRED)
    if event is None:
        return Reply(chat_id, WATCH_EXPIRED)
    logger.info("Chat %s is watching %s", chat_id, event.get("event_key"))
    return Reply(chat_id, f"Watching *{event['currency']} {event['title']}*. "
                          "You will get the realised move once the window exists.")


async def _start(chat_id: str, key: str) -> str:
    secret = str(getattr(get_settings(), "telegram_invite_secret", "") or "")
    # An unset secret must not mean "any key works" — that is a bot open to
    # whoever sends `/start `.
    if not secret:
        logger.warning("telegram_invite_secret is unset — enrolment refused")
        return NOT_RECOGNISED

    if await _over_attempt_limit(chat_id):
        logger.warning("Enrolment attempts from chat %s are rate limited", chat_id)
        return NOT_RECOGNISED

    if not hmac.compare_digest(key, secret):
        return NOT_RECOGNISED

    await subscribers.add(chat_id)
    logger.info("Chat %s enrolled", chat_id)
    return ENROLLED


async def _over_attempt_limit(chat_id: str) -> bool:
    """Counts every attempt, right or wrong, in a rolling hour.

    Keyed by chat id, which an attacker can rotate cheaply — this raises the
    cost of a naive script, not of a determined one. The answer to a
    determined one is rotating the secret.
    """
    key = ATTEMPTS_PREFIX + chat_id
    client = subscribers._client()
    try:
        attempts = int(await client.incr(key))
        if attempts == 1:
            await client.expire(key, ATTEMPT_WINDOW_SECONDS)
        return attempts > START_ATTEMPTS_PER_HOUR
    except Exception:
        # A broken counter must not become a broken door in either direction:
        # refuse, and say so in the log rather than to the sender.
        logger.exception("Could not count enrolment attempts for %s", chat_id)
        return True
    finally:
        await client.aclose()


async def _status() -> str:
    try:
        state = await watchdog.desk_state()
    except Exception:
        logger.exception("Could not read desk state")
        return STATUS_UNAVAILABLE

    lines = [f"*Desk status*\n", f"Armed briefs: {state.get('armed', 0)}"]
    if state.get("next_title"):
        lines.append(f"Next: {state['next_title']} {_when(state.get('next_due'))}")
    lines.append(f"Agenda last ran: {_ago(state.get('last_agenda'))}")
    lines.append(f"Last watchdog sweep: {state.get('last_watchdog') or 'no record'}")
    return "\n".join(lines)


def _when(epoch: float | None) -> str:
    if not epoch:
        return ""
    return f"at {datetime.fromtimestamp(float(epoch), UTC):%H:%M} UTC"


def _ago(epoch: float | None) -> str:
    if not epoch:
        return "no record"
    minutes = int((datetime.now(UTC).timestamp() - float(epoch)) // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    return f"{minutes // 60}h ago"
