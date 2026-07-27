"""
What one turn is allowed to spend.

Three independent ceilings, because they fail in three different ways:

  * **rounds** bound how many times the model may call tools. Without it a model
    that keeps asking one more question never finishes.
  * **wall clock** bounds how long a worker is held. A slow endpoint plus tool
    calls that each fetch market data can pin a worker for minutes, long after
    the caller's own timeout has given up on the response.
  * **tokens** bound what the turn costs. This is the one that was missing: the
    tool schemas serialise to ~3,400 tokens and are resent on every round, so a
    turn running its full round cap spends ~20,000 tokens before a single candle
    is fetched — and nothing counted it.

Exhaustion is never silent. It returns a reason, the turn summarises what it
already has, and the reason is recorded on the turn.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    """Token usage for a turn, as reported by the provider."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "llm_calls": self.calls}


@dataclass
class Budget:
    max_rounds: int
    wall_seconds: float
    max_tokens: int
    usage: Usage = field(default_factory=Usage)
    rounds_used: int = 0
    _t0: float = field(default_factory=time.monotonic)

    @classmethod
    def from_settings(cls, settings) -> Budget:
        return cls(
            max_rounds=settings.agent_max_tool_rounds,
            wall_seconds=settings.chat_turn_budget_seconds,
            max_tokens=getattr(settings, "chat_token_budget", 60_000),
        )

    # -- accounting ---------------------------------------------------------

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._t0

    def start_round(self) -> None:
        self.rounds_used += 1

    def record(self, response: Any) -> None:
        """Fold one LLM response's usage in.

        Tolerant on purpose: an endpoint that omits `usage` must not break a
        turn. It does mean the token budget silently stops binding, so the
        absence is worth noticing in the turn record — hence `calls` is counted
        separately from tokens.
        """
        self.usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.usage.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    # -- the question the loop asks ----------------------------------------

    def exhausted(self) -> str | None:
        """The reason this turn must stop, or None to continue."""
        if self.rounds_used >= self.max_rounds:
            return "rounds"
        if self.elapsed_seconds() >= self.wall_seconds:
            return "time"
        if self.usage.total_tokens >= self.max_tokens:
            return "tokens"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds_used": self.rounds_used,
            "max_rounds": self.max_rounds,
            "elapsed_ms": int(self.elapsed_seconds() * 1000),
            **self.usage.to_dict(),
            "max_tokens": self.max_tokens,
        }
