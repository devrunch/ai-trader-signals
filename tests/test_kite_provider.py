"""
KiteProvider — real NSE/BSE data, same shape as YFinanceProvider so the
router and every caller stay unaware which vendor answered.

Mocks the KiteConnect client the same way test_symbol_search.py mocks
yf.Search — nothing here touches the network. The NestJS call for the
current access_token is mocked too, at the httpx level.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.config import Settings
from app.market.providers.kite_provider import KiteProvider

TODAY = date.today()
MCX_INSTRUMENTS = [
    # Two live GOLD contracts, nearest expiry first once sorted -- the
    # farther one must never be picked over the nearer one.
    {"tradingsymbol": "GOLD25DECFUT", "name": "GOLD", "instrument_token": 111,
     "instrument_type": "FUT", "segment": "MCX", "expiry": TODAY + timedelta(days=30)},
    {"tradingsymbol": "GOLD26FEBFUT", "name": "GOLD", "instrument_token": 112,
     "instrument_type": "FUT", "segment": "MCX", "expiry": TODAY + timedelta(days=90)},
    # An already-expired GOLD contract Kite hasn't purged from the dump yet
    # -- must never be picked as the "nearest" one just because its date is
    # numerically smaller.
    {"tradingsymbol": "GOLD25NOVFUT", "name": "GOLD", "instrument_token": 110,
     "instrument_type": "FUT", "segment": "MCX", "expiry": TODAY - timedelta(days=5)},
    # A different commodity entirely -- must never leak into a GOLD lookup.
    {"tradingsymbol": "SILVER25DECFUT", "name": "SILVER", "instrument_token": 120,
     "instrument_type": "FUT", "segment": "MCX", "expiry": TODAY + timedelta(days=30)},
]

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
        # No "depth" key in this fixture -- must degrade to None, not KeyError.
        assert result["bid"] is None
        assert result["ask"] is None
        assert result["spread"] is None

    def test_extracts_top_of_book_bid_ask_from_depth(self):
        provider = _provider_with_token()
        provider._kite.quote = MagicMock(return_value={
            "NSE:RELIANCE": {
                "last_price": 1292.9,
                "ohlc": {"open": 1279.8, "high": 1297.0, "low": 1275.3, "close": 1278.0},
                "volume": 12158451,
                "depth": {
                    "buy": [{"price": 1292.8, "quantity": 50, "orders": 2}],
                    "sell": [{"price": 1293.1, "quantity": 30, "orders": 1}],
                },
            }
        })

        result = provider._get_quote_sync("RELIANCE", "NSE")

        assert result["bid"] == 1292.8
        assert result["ask"] == 1293.1
        assert result["spread"] == pytest.approx(0.3, abs=0.001)

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

    def test_an_unanticipated_sdk_exception_also_degrades_to_none(self):
        """Anything outside _VENDOR_ERRORS used to propagate — same broad
        fallback as YFinanceProvider so a genuinely unexpected SDK exception
        degrades instead of turning into a raw 500."""
        provider = _provider_with_token()
        provider._kite.quote = MagicMock(side_effect=RuntimeError("something new"))

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

    def test_30m_interval_maps_to_kites_own_30minute_string(self):
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {r["tradingsymbol"]: r for r in NSE_INSTRUMENTS if r["instrument_type"] == "EQ"},
            "BSE": {},
        }
        provider._instruments_loaded_at = provider._now()
        provider._kite.historical_data = MagicMock(return_value=[])

        provider._get_historical_df_sync("RELIANCE", "NSE", "30m", 30)

        called_interval = provider._kite.historical_data.call_args[0][3]
        assert called_interval == "30minute"

    def test_an_unmapped_symbol_returns_none(self):
        provider = _provider_with_token()
        provider._instruments = {"NSE": {}, "BSE": {}}
        provider._instruments_loaded_at = provider._now()

        assert provider._get_historical_df_sync("NOTREAL", "NSE", "1d", 30) is None

    def test_an_unanticipated_sdk_exception_also_degrades_to_none(self):
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {r["tradingsymbol"]: r for r in NSE_INSTRUMENTS if r["instrument_type"] == "EQ"},
            "BSE": {},
        }
        provider._instruments_loaded_at = provider._now()
        provider._kite.historical_data = MagicMock(side_effect=RuntimeError("something new"))

        assert provider._get_historical_df_sync("RELIANCE", "NSE", "1d", 30) is None


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

    def test_relevant_matches_rank_ahead_of_accidental_substring_hits(self):
        """"nifty50" must not rank an unrelated ETF (whose name merely
        starts with the substring "nifty50") ahead of the real tracker,
        whose name has a space where the query has none."""
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {
                "NIFTYBEES": {"tradingsymbol": "NIFTYBEES",
                               "name": "NIPPON INDIA ETF NIFTY 50 BEES"},
                "NIFTY500MOM50": {"tradingsymbol": "NIFTY500MOM50",
                                    "name": "NIFTY500 MOMENTUM 50 ETF"},
            },
            "BSE": {},
        }
        provider._instruments_loaded_at = provider._now()

        results = provider._search_sync("nifty50", limit=8)

        symbols = [r["symbol"] for r in results]
        assert "NIFTYBEES" in symbols
        # The collision candidate is either absent (its only "match" runs
        # into more digits right after) or, if present, ranks strictly
        # behind the real tracker — never ahead of it.
        if "NIFTY500MOM50" in symbols:
            assert symbols.index("NIFTYBEES") < symbols.index("NIFTY500MOM50")

    def test_an_exact_symbol_match_ranks_first(self):
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {
                "TCS": {"tradingsymbol": "TCS", "name": "TATA CONSULTANCY SERVICES"},
                "TCSADV": {"tradingsymbol": "TCSADV", "name": "SOME TCS-ADJACENT THING"},
            },
            "BSE": {},
        }
        provider._instruments_loaded_at = provider._now()

        results = provider._search_sync("tcs", limit=8)

        assert results[0]["symbol"] == "TCS"

    def test_index_listings_tagged_eq_are_excluded_not_just_futures(self):
        """Live bug: Kite tags some index/benchmark listings as instrument_type
        "EQ" too — "NIFTY DIV OPPS 50" is not a real tradeable ticker, and
        unlike every genuine equity its tradingsymbol IS a human-readable name
        with spaces in it, so it can never actually be charted/quoted. This
        exercises the real _ensure_instruments() filtering (not a pre-filtered
        fixture), since that's where the bug actually lived."""
        provider = _provider_with_token()
        provider._kite.instruments = MagicMock(side_effect=[
            [
                {"tradingsymbol": "RELIANCE", "name": "RELIANCE INDUSTRIES",
                 "instrument_token": 738561, "instrument_type": "EQ"},
                {"tradingsymbol": "NIFTY DIV OPPS 50", "name": "NIFTY DIV OPPS 50",
                 "instrument_token": 1, "instrument_type": "EQ"},
            ],
            [],
            [],  # MCX -- empty, not exercised by this test
        ])

        results = provider._search_sync("nifty", limit=8)

        assert results == []
        assert "NIFTY DIV OPPS 50" not in provider._instruments["NSE"]
        assert "RELIANCE" in provider._instruments["NSE"]

    def test_real_indices_are_kept_despite_the_space_in_their_symbol(self):
        """NIFTY 50 itself has a space in its tradingsymbol too, same as the
        junk listings above — but it carries segment "INDICES", which junk
        EQ rows don't, so it must survive the filter that excludes them."""
        provider = _provider_with_token()
        provider._kite.instruments = MagicMock(side_effect=[
            [
                {"tradingsymbol": "NIFTY 50", "name": "NIFTY 50",
                 "instrument_token": 256265, "instrument_type": "EQ", "segment": "INDICES"},
                {"tradingsymbol": "NIFTY DIV OPPS 50", "name": "NIFTY DIV OPPS 50",
                 "instrument_token": 1, "instrument_type": "EQ"},
            ],
            [],
            [],  # MCX -- empty, not exercised by this test
        ])

        results = provider._search_sync("nifty", limit=8)

        symbols = [r["symbol"] for r in results]
        assert "NIFTY 50" in symbols
        assert "NIFTY DIV OPPS 50" not in symbols

    def test_an_unanticipated_sdk_exception_also_degrades_to_empty_list(self):
        provider = _provider_with_token()
        provider._kite.instruments = MagicMock(side_effect=RuntimeError("something new"))

        assert provider._search_sync("reliance", limit=8) == []


class TestMcxContinuousContracts:
    """MCX lists commodity FUTURES under contract-month-specific
    tradingsymbols that expire and roll -- "GOLD1!" (TradingView's own
    continuous-contract convention, confirmed live against tradingview.com)
    resolves to whichever real dated contract is currently nearest to
    expiry without having already expired, recomputed fresh from Kite's own
    instrument dump on every lookup rather than cached as a decision."""

    def _provider_with_mcx(self) -> KiteProvider:
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {}, "BSE": {},
            "MCX": {r["tradingsymbol"]: r for r in MCX_INSTRUMENTS},
        }
        provider._instruments_loaded_at = provider._now()
        return provider

    def test_ensure_instruments_keeps_only_futures_with_a_real_expiry(self):
        provider = _provider_with_token()
        provider._kite.instruments = MagicMock(side_effect=[
            [], [],  # NSE, BSE -- empty, not exercised here
            [
                *MCX_INSTRUMENTS,
                {"tradingsymbol": "GOLD25DEC5000CE", "name": "GOLD", "instrument_token": 200,
                 "instrument_type": "CE", "segment": "MCX-OPT", "expiry": TODAY + timedelta(days=30)},
                {"tradingsymbol": "JUNKNOFUT", "name": "JUNK", "instrument_token": 201,
                 "instrument_type": "FUT", "segment": "MCX", "expiry": None},
            ],
        ])

        provider._ensure_instruments()

        kept = provider._instruments["MCX"]
        assert set(kept) == {r["tradingsymbol"] for r in MCX_INSTRUMENTS}
        assert "GOLD25DEC5000CE" not in kept  # options excluded -- futures only
        assert "JUNKNOFUT" not in kept  # no expiry -- can't resolve a continuous contract from it

    def test_continuous_symbol_resolves_to_the_nearest_unexpired_contract(self):
        provider = self._provider_with_mcx()

        row = provider._resolve_row("GOLD1!", "MCX")

        assert row["tradingsymbol"] == "GOLD25DECFUT"  # nearer than GOLD26FEB, not expired like GOLD25NOV

    def test_continuous_symbol_never_resolves_to_an_already_expired_contract(self):
        provider = self._provider_with_mcx()
        row = provider._resolve_row("GOLD1!", "MCX")
        assert row["tradingsymbol"] != "GOLD25NOVFUT"

    def test_different_commodities_do_not_leak_into_each_other(self):
        provider = self._provider_with_mcx()

        gold = provider._resolve_row("GOLD1!", "MCX")
        silver = provider._resolve_row("SILVER1!", "MCX")

        assert gold["tradingsymbol"] == "GOLD25DECFUT"
        assert silver["tradingsymbol"] == "SILVER25DECFUT"

    def test_an_unknown_commodity_resolves_to_none_not_a_crash(self):
        provider = self._provider_with_mcx()
        assert provider._resolve_row("PLATINUM1!", "MCX") is None

    def test_a_commodity_whose_every_contract_has_expired_resolves_to_none(self):
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {}, "BSE": {},
            "MCX": {"GOLD25NOVFUT": MCX_INSTRUMENTS[2]},  # the expired one, alone
        }
        provider._instruments_loaded_at = provider._now()

        assert provider._resolve_row("GOLD1!", "MCX") is None

    def test_a_real_dated_mcx_symbol_still_resolves_directly_no_1_bang_needed(self):
        """Not every MCX lookup goes through the continuous convention -- a
        caller that already knows the exact contract (from search results,
        or a saved chart) can still ask for it by its real tradingsymbol."""
        provider = self._provider_with_mcx()
        row = provider._resolve_row("GOLD26FEBFUT", "MCX")
        assert row["tradingsymbol"] == "GOLD26FEBFUT"

    def test_non_mcx_lookups_are_completely_unaffected(self):
        """_resolve_row's new branch must never fire outside MCX -- an NSE
        symbol that happened to end in "1!" (it never would in practice)
        stays a plain direct lookup, same as before this existed."""
        provider = _provider_with_token()
        provider._instruments = {
            "NSE": {r["tradingsymbol"]: r for r in NSE_INSTRUMENTS if r["instrument_type"] == "EQ"},
            "BSE": {}, "MCX": {},
        }
        provider._instruments_loaded_at = provider._now()

        assert provider._resolve_row("RELIANCE", "NSE")["tradingsymbol"] == "RELIANCE"

    def test_get_quote_resolves_the_continuous_symbol_but_echoes_it_back(self):
        """The Kite API call must use the real dated contract; the response
        the rest of the app sees must still say "GOLD1!" -- TradingView's
        own convention of the continuous symbol staying stable while the
        real contract underneath rolls."""
        provider = self._provider_with_mcx()
        provider._kite.quote = MagicMock(return_value={
            "MCX:GOLD25DECFUT": {
                "last_price": 72500.0,
                "ohlc": {"open": 72000.0, "high": 72800.0, "low": 71900.0, "close": 72100.0},
                "volume": 12000,
            }
        })

        result = provider._get_quote_sync("GOLD1!", "MCX")

        provider._kite.quote.assert_called_once_with("MCX:GOLD25DECFUT")
        assert result["symbol"] == "GOLD1!"
        assert result["exchange"] == "MCX"
        assert result["ltp"] == 72500.0

    def test_get_quote_for_an_unresolvable_continuous_symbol_returns_none(self):
        provider = self._provider_with_mcx()
        assert provider._get_quote_sync("PLATINUM1!", "MCX") is None

    def test_search_returns_one_result_per_commodity_not_one_per_dated_contract(self):
        """Searching "gold" must not flood results with every live expiry
        month of the same underlying commodity."""
        provider = self._provider_with_mcx()

        results = provider._search_sync("gold", limit=8)

        mcx_results = [r for r in results if r["exchange"] == "MCX"]
        assert len(mcx_results) == 1
        assert mcx_results[0]["symbol"] == "GOLD1!"
        assert mcx_results[0]["name"] == "GOLD"

    def test_search_ranks_and_matches_mcx_the_same_way_as_equities(self):
        provider = self._provider_with_mcx()

        results = provider._search_sync("silver", limit=8)

        symbols = [r["symbol"] for r in results]
        assert "SILVER1!" in symbols
        assert "GOLD1!" not in symbols  # no match for an unrelated commodity


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
