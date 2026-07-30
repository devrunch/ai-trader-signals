"""
Symbol search: company name / ticker -> matches this app can actually chart.

Yahoo's search returns results across every exchange it lists — Frankfurt,
Vienna, Warsaw, ETFs, depositary receipts. The filtering in `_search_sync` is
what stops a suggestion appearing that would 404 (or worse, silently chart the
wrong instrument) the moment it is clicked, so it is worth pinning even though
it never touches the network — the fixture below is the real shape captured
from `yf.Search("apple")` on 2026-07-30.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from app.market.providers.yfinance_provider import YFinanceProvider

APPLE_QUOTES = [
    {"symbol": "AAPL", "shortname": "Apple Inc.", "exchange": "NMS", "quoteType": "EQUITY"},
    {"symbol": "APC.DE", "shortname": "Apple Inc.                    R", "exchange": "GER", "quoteType": "EQUITY"},
    {"symbol": "APLE", "shortname": "Apple Hospitality REIT, Inc.", "exchange": "NYQ", "quoteType": "EQUITY"},
    {"symbol": "APC.F", "shortname": "Apple Inc.                    R", "exchange": "FRA", "quoteType": "EQUITY"},
    {"symbol": "APLE.TO", "shortname": "HARVEST APPLE ENHNCD HIGH INCM ", "exchange": "TOR", "quoteType": "ETF"},
]

RELIANCE_QUOTES = [
    {"symbol": "RS", "shortname": "Reliance, Inc.", "exchange": "NYQ", "quoteType": "EQUITY"},
    {"symbol": "RELIANCE.NS", "shortname": "RELIANCE INDUSTRIES LTD", "exchange": "NSI", "quoteType": "EQUITY"},
    {"symbol": "RELINFRA.BO", "shortname": "RELIANCE INFRASTRUCTURE LTD.", "exchange": "BSE", "quoteType": "EQUITY"},
    {"symbol": "RWC.AX", "shortname": "RWC CORP FPO [RWC]", "exchange": "ASX", "quoteType": "EQUITY"},
]


def _search_returning(quotes: list[dict]):
    return patch(
        "app.market.providers.yfinance_provider.yf.Search",
        return_value=SimpleNamespace(quotes=quotes),
    )


def test_only_exchanges_this_app_can_chart_survive():
    """Frankfurt, Vienna, an ETF wrapper — none of these can be handed to
    _ticker_for and get the right instrument, so none should be suggested."""
    with _search_returning(APPLE_QUOTES):
        results = YFinanceProvider()._search_sync("apple", limit=8)

    symbols = {r["symbol"] for r in results}
    assert symbols == {"AAPL", "APLE"}  # NMS->NASDAQ, NYQ->NYSE kept; DE/FRA/TO dropped


def test_the_ticker_suffix_is_stripped_for_display():
    with _search_returning(RELIANCE_QUOTES):
        results = YFinanceProvider()._search_sync("reliance", limit=8)

    by_symbol = {r["symbol"]: r for r in results}
    assert "RELIANCE" in by_symbol
    assert by_symbol["RELIANCE"]["exchange"] == "NSE"
    assert by_symbol["RELIANCE"]["name"] == "RELIANCE INDUSTRIES LTD"


def test_exchange_names_are_ours_not_yahoos():
    """A frontend that only knows NSE/BSE/NASDAQ/NYSE must never see NSI/NMS/NYQ."""
    with _search_returning(APPLE_QUOTES + RELIANCE_QUOTES):
        results = YFinanceProvider()._search_sync("x", limit=20)

    seen = {r["exchange"] for r in results}
    assert seen <= {"NSE", "BSE", "NASDAQ", "NYSE"}


def test_a_result_with_no_usable_name_falls_back_to_the_symbol():
    with _search_returning([
        {"symbol": "XYZ", "exchange": "NMS", "quoteType": "EQUITY"},
    ]):
        results = YFinanceProvider()._search_sync("xyz", limit=8)

    assert results[0]["name"] == "XYZ"


def test_non_equity_instruments_are_excluded():
    """ETFs and mutual funds pass through the exchange filter fine but are not
    what "search for a stock" means here."""
    with _search_returning([
        {"symbol": "SPY", "shortname": "SPDR S&P 500", "exchange": "NMS", "quoteType": "ETF"},
    ]):
        results = YFinanceProvider()._search_sync("spy", limit=8)

    assert results == []


def test_share_classes_are_kept_distinct_not_collapsed():
    """BRK-A and BRK-B trade at wildly different prices — the "-A"/"-B" marks a
    real share class, unlike the ".NS"/".BO" exchange suffix stripped above."""
    with _search_returning([
        {"symbol": "BRK-A", "shortname": "Berkshire A", "exchange": "NYQ", "quoteType": "EQUITY"},
        {"symbol": "BRK-B", "shortname": "Berkshire B", "exchange": "NYQ", "quoteType": "EQUITY"},
    ]):
        results = YFinanceProvider()._search_sync("berkshire", limit=8)

    assert {r["symbol"] for r in results} == {"BRK-A", "BRK-B"}


def test_the_same_company_listed_twice_on_one_exchange_collapses_to_one():
    """What the dedup actually guards: Yahoo occasionally returns the primary
    listing and a near-duplicate (different data source, same instrument) that
    share a symbol once the exchange suffix is gone."""
    with _search_returning([
        {"symbol": "AAPL", "shortname": "Apple Inc.", "exchange": "NMS", "quoteType": "EQUITY"},
        {"symbol": "AAPL.NS", "shortname": "Apple Inc. (dup)", "exchange": "NSI", "quoteType": "EQUITY"},
    ]):
        results = YFinanceProvider()._search_sync("apple", limit=8)

    assert len(results) == 1
    assert results[0]["symbol"] == "AAPL"


def test_respects_the_limit_after_filtering_not_before():
    quotes = [
        {"symbol": f"SYM{i}", "shortname": f"Company {i}", "exchange": "NMS", "quoteType": "EQUITY"}
        for i in range(5)
    ]
    with _search_returning(quotes):
        results = YFinanceProvider()._search_sync("x", limit=3)

    assert len(results) == 3


def test_a_vendor_error_returns_no_results_rather_than_raising():
    """Yahoo's search is an unofficial, scraped endpoint — the same tolerance
    the quote and historical paths already apply."""
    with patch(
        "app.market.providers.yfinance_provider.yf.Search",
        side_effect=ConnectionError("timed out"),
    ):
        results = YFinanceProvider()._search_sync("apple", limit=8)

    assert results == []
