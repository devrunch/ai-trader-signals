"""
TwelveDataProvider — spot gold (XAU/USD) via Twelve Data's REST API.
Mocks httpx.get the same way kite_auth's own tests mock the network layer —
nothing here touches the real vendor.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.config import Settings
from app.market.providers.twelve_data_provider import TwelveDataProvider


def _provider(api_key: str = "test-key") -> TwelveDataProvider:
    return TwelveDataProvider(Settings(twelve_data_api_key=api_key))


def _response(json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    return resp


class TestGetQuote:
    def test_returns_none_for_a_symbol_this_provider_does_not_know(self):
        provider = _provider()
        assert provider._get_quote_sync("EURUSD", "FOREX") is None

    def test_returns_none_when_no_api_key_is_configured(self):
        provider = _provider(api_key="")
        assert provider._get_quote_sync("XAUUSD", "FOREX") is None

    @patch("app.market.providers.twelve_data_provider.httpx.get")
    def test_a_real_quote_maps_every_field(self, mock_get):
        mock_get.return_value = _response({
            "symbol": "XAU/USD", "close": "2650.5", "open": "2640.0",
            "high": "2655.0", "low": "2635.0", "previous_close": "2638.0",
            "change": "12.5", "percent_change": "0.47",
        })
        provider = _provider()

        result = provider._get_quote_sync("XAUUSD", "FOREX")

        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["params"]["symbol"] == "XAU/USD"
        assert result["symbol"] == "XAUUSD"
        assert result["exchange"] == "FOREX"
        assert result["ltp"] == 2650.5
        assert result["change"] == 12.5
        assert result["change_percent"] == 0.47
        assert result["prev_close"] == 2638.0
        # Never fabricated -- this vendor's /quote has no bid/ask/spread for
        # spot metals, and the response above carries none either.
        assert result["bid"] is None
        assert result["ask"] is None
        assert result["spread"] is None

    @patch("app.market.providers.twelve_data_provider.httpx.get")
    def test_a_vendor_error_response_degrades_to_none_not_a_crash(self, mock_get):
        mock_get.return_value = _response({"status": "error", "code": 400, "message": "bad symbol"})
        provider = _provider()

        assert provider._get_quote_sync("XAUUSD", "FOREX") is None

    @patch("app.market.providers.twelve_data_provider.httpx.get")
    def test_a_network_error_degrades_to_none_not_a_crash(self, mock_get):
        import httpx
        mock_get.side_effect = httpx.ConnectError("no route to host")
        provider = _provider()

        assert provider._get_quote_sync("XAUUSD", "FOREX") is None


class TestGetHistoricalDf:
    def test_returns_none_for_a_symbol_this_provider_does_not_know(self):
        provider = _provider()
        assert provider._get_historical_df_sync("EURUSD", "FOREX", "1d", 30) is None

    @patch("app.market.providers.twelve_data_provider.httpx.get")
    def test_a_real_time_series_becomes_a_real_ohlcv_dataframe(self, mock_get):
        mock_get.return_value = _response({
            "status": "ok",
            "values": [
                {"datetime": "2026-08-20", "open": "2600.0", "high": "2610.0", "low": "2590.0", "close": "2605.0", "volume": "0"},
                {"datetime": "2026-08-21", "open": "2605.0", "high": "2620.0", "low": "2600.0", "close": "2615.0", "volume": "0"},
            ],
        })
        provider = _provider()

        df = provider._get_historical_df_sync("XAUUSD", "FOREX", "1d", 10)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        assert df["close"].iloc[-1] == 2615.0
        assert isinstance(df.index, pd.DatetimeIndex)

    @patch("app.market.providers.twelve_data_provider.httpx.get")
    def test_an_empty_values_list_returns_none(self, mock_get):
        mock_get.return_value = _response({"status": "ok", "values": []})
        provider = _provider()

        assert provider._get_historical_df_sync("XAUUSD", "FOREX", "1d", 10) is None

    @patch("app.market.providers.twelve_data_provider.httpx.get")
    def test_a_vendor_error_response_degrades_to_none_not_a_crash(self, mock_get):
        mock_get.return_value = _response({"status": "error", "message": "invalid apikey"})
        provider = _provider()

        assert provider._get_historical_df_sync("XAUUSD", "FOREX", "1d", 10) is None


class TestSearch:
    @pytest.mark.asyncio
    async def test_gold_matches_xauusd(self):
        provider = _provider()
        results = await provider.search("gold", limit=8)
        assert results == [{"symbol": "XAUUSD", "name": "Gold Spot", "exchange": "FOREX"}]

    @pytest.mark.asyncio
    async def test_xauusd_matches_itself(self):
        provider = _provider()
        results = await provider.search("xauusd", limit=8)
        assert len(results) == 1
        assert results[0]["symbol"] == "XAUUSD"

    @pytest.mark.asyncio
    async def test_unrelated_query_matches_nothing(self):
        provider = _provider()
        results = await provider.search("reliance", limit=8)
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_query_matches_nothing(self):
        provider = _provider()
        assert await provider.search("", limit=8) == []
