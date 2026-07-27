"""
Where a finished turn goes.

Persistence is currently wired into the HTTP layer: `POST /chat` records, the
SSE proxy records, and anything else that runs a turn records nothing. That was
fine while HTTP was the only caller, and stops being fine the moment a Celery
task or a scheduled brief runs one — the cost is incurred either way, and only
the paths someone remembered to wire leave a trace.

So the orchestrator ends every turn by handing it to a `TurnStore`. The default
does nothing, because the signals service does not own a database and should
not grow one; NestJS owns every collection in the product. What this seam buys
is that a new caller inherits persistence instead of having to remember it.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TurnStore(Protocol):
    """Receives a completed turn. Must not raise, and must not block for long."""

    def save(self, turn: dict[str, Any]) -> None: ...


class NullTurnStore:
    """Keeps nothing.

    The default, and correct for the HTTP paths: NestJS already records the turn
    from the response it received, and a second writer here would mean two
    systems owning the same document.
    """

    def save(self, turn: dict[str, Any]) -> None:
        return None


class LoggingTurnStore:
    """Writes one line per turn. For a path with no other record — a scheduled
    run, a worker — where losing the turn entirely is worse than a log line."""

    def save(self, turn: dict[str, Any]) -> None:
        usage = turn.get("usage") or {}
        logger.info(
            "Turn %s on %s finished: %s events, %s tokens, stop_reason=%s",
            turn.get("turn_id"), turn.get("symbol"),
            len(turn.get("events") or []), usage.get("total_tokens"),
            turn.get("stop_reason"),
        )


def save_quietly(store: TurnStore | None, turn: dict[str, Any]) -> None:
    """Hand the turn over, never at the cost of the answer.

    The turn is already complete and already paid for by the time this runs. A
    storage failure must not turn a good answer into an error the user sees —
    it is logged loudly instead, because a silent one would leave us believing
    turns were being recorded when they were not.
    """
    if store is None:
        return
    try:
        store.save(turn)
    except Exception:
        logger.exception("Failed to store turn %s", turn.get("turn_id"))
