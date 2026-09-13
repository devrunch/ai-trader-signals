"""
Market data provider contract — every vendor (yFinance today, Dhan/Alpaca/etc.
later) implements this same interface so callers never know or care which
vendor actually answered the request.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from app.market.contract import ProviderCapabilities


class MarketDataProvider(Protocol):
    """A single data vendor for one or more exchanges/asset classes."""

    # What this vendor can actually do. Declared, not inferred: the router
    # pages and clamps from these numbers, so a limit known only inside one
    # provider (or, worse, applied to all of them from a global table) is
    # exactly the shape of bug this replaces.
    capabilities: ProviderCapabilities

    async def get_quote(self, symbol: str, exchange: str) -> dict | None:
        """Current price snapshot: symbol, exchange, ltp, change, change_percent, ..."""
        ...

    async def get_historical_df(
        self, symbol: str, exchange: str, interval: str, days: int
    ) -> pd.DataFrame | None:
        """
        OHLCV history as a DataFrame with a DatetimeIndex and lowercase
        columns: open, high, low, close, volume.
        """
        ...

    async def search(self, query: str, limit: int) -> list[dict]:
        """Company name / symbol -> [{symbol, name, exchange}], exchanges this
        vendor actually covers only — never a result the caller cannot chart."""
        ...
