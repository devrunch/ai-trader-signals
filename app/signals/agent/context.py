"""
The chat agent's window onto the user's trading account.

Its own module, and injected rather than constructed inline, so the portfolio
tools — position sizing, exposure, aggregate risk — can be tested against a
dict fixture instead of a live NestJS instance. That is the code with the worst
consequences when it is wrong (an earlier version invented a ₹1,00,000 account
value and handed a ₹20,000 account 5x oversize) and it was untestable.
"""
from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class TradingContextClient:
    """Fetches `{portfolio, positions, watchlist}` from the NestJS internal API.

    Memoised per instance — one chat turn asks for the account several times
    (affordability, then sizing, then exposure) and must see one consistent
    snapshot, not three.
    """

    def __init__(self, user_id: str | None, settings=None):
        self.user_id = user_id
        self._settings = settings or get_settings()
        self._cached: dict | None = None

    async def get(self) -> dict:
        if self._cached is not None:
            return self._cached
        if not self.user_id:
            self._cached = {"error": "No user context available."}
            return self._cached
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(
                    f"{self._settings.api_service_url}/api/internal/trading-context/{self.user_id}",
                    headers={"x-internal-key": self._settings.internal_api_key},
                )
                resp.raise_for_status()
                self._cached = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Trading context fetch failed for user %s: %s", self.user_id, e)
            self._cached = {"error": "Could not load the user's portfolio right now."}
        return self._cached


class StaticTradingContext:
    """Test double — returns a fixed dict."""

    def __init__(self, payload: dict):
        self._payload = payload

    async def get(self) -> dict:
        return self._payload
