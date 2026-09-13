"""Resolution, which exists so that a caller's guess cannot pick the vendor.

Three layers default `exchange` to NSE, so "I don't know the venue" and "this
trades on NSE" were the same request. Gold took that path and 404'd.
"""
from app.market import sessions, symbols
from app.market.contract import AssetClass, VolumeSource


class TestForexAndMetals:
    def test_gold_resolves_to_deriv_however_it_was_asked_for(self):
        for hint in (None, "NSE", "NASDAQ", "nonsense"):
            info = symbols.resolve("XAUUSD", hint)
            assert info is not None
            assert info.provider == symbols.DERIV
            assert info.vendor_symbol == "frxXAUUSD"
            assert info.exchange == "FOREX", f"hint {hint!r} leaked into the answer"

    def test_metals_and_currency_pairs_are_different_asset_classes(self):
        assert symbols.resolve("XAUUSD").asset_class is AssetClass.METAL
        assert symbols.resolve("EURUSD").asset_class is AssetClass.FX

    def test_forex_reports_no_volume_at_all(self):
        # Deriv publishes none for spot/CFD forex. A Dukascopy tick count used
        # to stand in for it and was dropped: ~15 minutes behind real time,
        # which on a 1m chart reads as live and is not.
        assert symbols.resolve("EURUSD").volume_source is VolumeSource.NONE

    def test_they_carry_the_forex_session(self):
        assert symbols.resolve("XAUUSD").session is sessions.FX


class TestIndianEquity:
    def test_an_equity_follows_its_hint(self):
        info = symbols.resolve("RELIANCE", "NSE")
        assert info.provider == symbols.KITE
        assert info.exchange == "NSE"
        assert info.session is sessions.NSE
        assert info.volume_source is VolumeSource.EXCHANGE

    def test_the_hint_still_disambiguates_a_dual_listing(self):
        assert symbols.resolve("RELIANCE", "BSE").exchange == "BSE"

    def test_an_index_is_not_an_equity(self):
        assert symbols.resolve("^NSEI", "NSE").asset_class is AssetClass.INDEX

    def test_mcx_gets_its_own_session_not_the_equity_one(self):
        info = symbols.resolve("GOLD1!", "MCX")
        assert info.asset_class is AssetClass.COMMODITY
        assert info.session is sessions.MCX


class TestFallback:
    def test_an_unhinted_symbol_lands_on_the_fallback_vendor(self):
        info = symbols.resolve("AAPL")
        assert info.provider == symbols.YFINANCE
        assert info.session is sessions.US_EQUITY

    def test_an_unmodelled_venue_is_treated_as_always_open(self):
        # Better to report "the vendor had nothing" than to claim confidently
        # that a market we have never modelled is closed.
        assert symbols.resolve("XYZ", "LSE").session is sessions.ALWAYS_OPEN

    def test_an_empty_symbol_resolves_to_nothing(self):
        assert symbols.resolve("") is None
        assert symbols.resolve("   ") is None


def test_resolution_is_case_and_whitespace_insensitive():
    assert symbols.resolve(" xauusd ").vendor_symbol == "frxXAUUSD"
    assert symbols.resolve("reliance", "nse").exchange == "NSE"
