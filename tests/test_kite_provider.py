"""
KiteProvider — real NSE/BSE data, same shape as YFinanceProvider so the
router and every caller stay unaware which vendor answered.

Mocks the KiteConnect client the same way test_symbol_search.py mocks
yf.Search — nothing here touches the network. The NestJS call for the
current access_token is mocked too, at the httpx level.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.config import Settings
from app.market.providers.kite_provider import KiteProvider

NSE_INSTRUMENTS = [
    {"tradingsymbol": "RELIANCE", "name": "RELIANCE INDUSTRIES", "instrument_token": 738561,
     "instrument_type": "EQ", "exchange": "NSE"},
    {"tradingsymbol": "RELIANCE-FUT", "name": "RELIANCE INDUSTRIES FUT", "instrument_token": 999,
     "instrument_type": "FUT", "exchange": "NSE"},
    {"tradingsymbol": "TCS", "name": "TATA CONSULTANCY SERVICES", "instrument_token": 2953217,
     "instrument_type": "EQ", "exchange": "NSE"},
]
BSE_INSTRUMENTS = [
    {"tradingsymbol": "RELIANCE", "name": "RELIANCE INDUSTRIES", "instrument_token": 500325,
     "instrument_type": "EQ", "exchange": "BSE"},
]


def _settings() -> Settings:
    return Settings(
        zerodha_api_key="key", zerodha_api_secret="secret",
        api_service_url="http://api.test", internal_api_key="ikey",
    )


def _provider_with_token() -> KiteProvider:
    provider = KiteProvider(_settings())
    with patch("app.market.providers.kite_provider.httpx.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"accessToken": "tok_abc", "refreshedAt": "2026-08-16T06:00:00Z"},
        )
        mock_get.return_value.raise_for_status = lambda: None
        provider._ensure_token()
    return provider


class TestGetQuote:
    def test_maps_kite_quote_shape_to_the_shared_contract(self):
        provider = _provider_with_token()
        provider._kite.quote = MagicMock(return_value={
            "NSE:RELIANCE": {
                "last_price": 1292.9,
                "ohlc": {"open": 1279.8, "high": 1297.0, "low": 1275.3, "close": 1278.0},
                "volume": 12158451,
            }
        })

        result = provider._get_quote_sync("RELIANCE", "NSE")

        assert result["symbol"] == "RELIANCE"
        assert result["exchange"] == "NSE"
        assert result["ltp"] == 1292.9
        assert result["prev_close"] == 1278.0
        assert result["change"] == pytest.approx(14.9, abs=0.01)
        assert result["volume"] == 12158451

    def test_a_kite_exception_returns_none_not_a_raise(self):
        from kiteconnect.exceptions import KiteException

        provider = _provider_with_token()
        provider._kite.quote = MagicMock(side_effect=KiteException("rate limited"))

        assert provider._get_quote_sync("RELIANCE", "NSE") is None

    def test_a_network_error_returns_none_not_a_raise(self):
        """Network-layer errors (ConnectionError, TimeoutError, etc.) from the
        underlying requests library are OSError subclasses and must degrade to
        None rather than propagate."""
        provider = _provider_with_token()
        provider._kite.quote = MagicMock(side_effect=ConnectionError("network down"))

        assert provider._get_quote_sync("RELIANCE", "NSE") is None


class TestGetHistoricalDf:
    def test_resolves_symbol_to_instrument_token_and_shapes_the_frame(self):
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {r["tradingsymbol"]: r for r in NSE_INSTRUMENTS if r["instrument_type"] == "EQ"},
            "BSE": {r["tradingsymbol"]: r for r in BSE_INSTRUMENTS},
        }
        provider._instruments_loaded_at = provider._now()
        provider._kite.historical_data = MagicMock(return_value=[
            {"date": pd.Timestamp("2026-08-14"), "open": 1265.0, "high": 1283.4,
             "low": 1249.8, "close": 1278.0, "volume": 9817000},
            {"date": pd.Timestamp("2026-08-15"), "open": 1288.2, "high": 1288.7,
             "low": 1278.0, "close": 1280.0, "volume": 7163132},
        ])

        df = provider._get_historical_df_sync("RELIANCE", "NSE", "1d", 30)

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        called_token = provider._kite.historical_data.call_args[0][0]
        assert called_token == 738561  # RELIANCE's EQ instrument_token, not the FUT one

    def test_an_unmapped_symbol_returns_none(self):
        provider = _provider_with_token()
        provider._instruments = {"NSE": {}, "BSE": {}}
        provider._instruments_loaded_at = provider._now()

        assert provider._get_historical_df_sync("NOTREAL", "NSE", "1d", 30) is None


class TestSearch:
    def test_matches_symbol_or_name_case_insensitively_eq_only(self):
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {r["tradingsymbol"]: r for r in NSE_INSTRUMENTS if r["instrument_type"] == "EQ"},
            "BSE": {r["tradingsymbol"]: r for r in BSE_INSTRUMENTS},
        }
        provider._instruments_loaded_at = provider._now()

        results = provider._search_sync("reliance", limit=8)

        symbols_and_exchanges = {(r["symbol"], r["exchange"]) for r in results}
        # Both the NSE and BSE listing show up — search is not exchange-scoped
        # by the caller, same as yfinance's search.
        assert symbols_and_exchanges == {("RELIANCE", "NSE"), ("RELIANCE", "BSE")}
        # The FUT row must never appear — instrument_type filtering excludes it.
        assert all(r["symbol"] != "RELIANCE-FUT" for r in results)

    def test_respects_the_limit(self):
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {r["tradingsymbol"]: r for r in NSE_INSTRUMENTS if r["instrument_type"] == "EQ"},
            "BSE": {},
        }
        provider._instruments_loaded_at = provider._now()

        results = provider._search_sync("a", limit=1)  # matches RELIANCE and TCS's names
        assert len(results) == 1


@pytest.mark.asyncio
async def test_async_methods_wrap_the_sync_ones():
    """Same executor-wrapping pattern as YFinanceProvider — Kite's SDK is
    synchronous too."""
    provider = _provider_with_token()
    provider._kite.quote = MagicMock(return_value={
        "NSE:TCS": {"last_price": 100.0, "ohlc": {"open": 99, "high": 101, "low": 98, "close": 99},
                     "volume": 1000},
    })

    result = await provider.get_quote("TCS", "NSE")
    assert result["ltp"] == 100.0
