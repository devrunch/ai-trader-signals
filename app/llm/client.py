"""
LLM client — Bedrock Mantle's OpenAI-compatible endpoint.

Extracted from SignalService, which owned it purely by accident of history and
forced `brief.py` to construct an entire SignalService (including an SQS client
it never used) just to borrow a `_get_llm()`.

Two responsibilities, and only these:
  * acquire an API key — an explicitly configured one, or mint one from IAM
    credentials on first use
  * hand out a live `OpenAI` client, refreshing the minted key before it expires
"""
from __future__ import annotations

import logging
import threading
import time

from openai import OpenAI

from app.config import generate_bedrock_key, get_settings

logger = logging.getLogger(__name__)

# Refresh 1 h before the 12 h Bedrock Mantle expiry.
_LLM_TTL_SECONDS = 11 * 3600


class LlmClient:
    """Lazily-constructed, self-refreshing OpenAI-compatible client.

    Nothing happens in `__init__` — construction is free and safe at import
    time, in a test, or in a process that will never make an LLM call.
    """

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._client: OpenAI | None = None
        self._born: float = 0.0
        self._lock = threading.Lock()

    # -- key acquisition ---------------------------------------------------

    def _acquire_key(self) -> str:
        """Explicit key wins. Otherwise mint one from IAM credentials.

        Minting walks boto3's credential chain, which can take seconds off-EC2 —
        which is exactly why this is here and not in `Settings`.
        """
        if self._settings.bedrock_api_key:
            return self._settings.bedrock_api_key
        key = generate_bedrock_key(
            self._settings.aws_access_key_id,
            self._settings.aws_secret_access_key,
            self._settings.aws_region,
        )
        logger.info("Bedrock API key minted from IAM credentials (valid 12 h)")
        return key

    # -- client ------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._settings.bedrock_model_id

    def client(self) -> OpenAI:
        """Return a live client, refreshing an expiring minted key.

        `time.monotonic()` deliberately, not `time.time()` — immune to wall-clock
        adjustments, which would otherwise either skip a needed refresh or churn.
        """
        with self._lock:
            fresh = self._client is not None and (time.monotonic() - self._born) <= _LLM_TTL_SECONDS
            if fresh:
                return self._client  # type: ignore[return-value]

            try:
                key = self._acquire_key()
                self._client = OpenAI(api_key=key, base_url=self._settings.bedrock_base_url)
                self._born = time.monotonic()
                logger.info("Bedrock client (re)initialised")
            except Exception as exc:
                if self._client is None:
                    # No usable client at all — fail loudly rather than returning
                    # None for a caller to trip over deep inside a tool loop.
                    logger.error("Could not initialise Bedrock client: %s", exc)
                    raise
                logger.warning("Bedrock key refresh failed: %s — using stale key", exc)
            return self._client

    def chat(self, **kwargs):
        """Thin passthrough to `chat.completions.create` with the model defaulted."""
        kwargs.setdefault("model", self.model)
        return self.client().chat.completions.create(**kwargs)


_default: LlmClient | None = None
_default_lock = threading.Lock()


def get_llm() -> LlmClient:
    """Process-wide default client. Callers that need to inject a fake should
    take an `LlmClient` as a constructor argument instead of calling this."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = LlmClient()
    return _default
