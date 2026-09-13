"""One suite, run against every provider.

Each vendor used to be tested on its own terms, which is how four providers
ended up with four different ideas of what an empty result means and one
global limits table describing only one of them. Anything a caller is allowed
to assume about *any* provider is asserted here, for all of them.

Grows with the contract: the bars/status assertions arrive with the router
rewrite (see the spec's sequencing). What is here now is what the
capability declarations already promise.
"""
import pytest

from app.market.contract import ProviderCapabilities, VolumeSource
from app.market.providers.deriv_provider import DerivProvider
from app.market.providers.kite_provider import KiteProvider
from app.market.providers.yfinance_provider import YFinanceProvider
from app.market.symbols import DERIV, KITE, YFINANCE

# The provider classes, not instances: KiteProvider's constructor wants live
# settings, and none of this needs an instance.
PROVIDERS = [
    pytest.param(DerivProvider, DERIV, id="deriv"),
    pytest.param(KiteProvider, KITE, id="kite"),
    pytest.param(YFinanceProvider, YFINANCE, id="yfinance"),
]


@pytest.mark.parametrize(("provider", "_key"), PROVIDERS)
def test_every_provider_declares_its_limits(provider, _key):
    assert isinstance(provider.capabilities, ProviderCapabilities)


@pytest.mark.parametrize(("provider", "_key"), PROVIDERS)
def test_the_intervals_it_claims_all_have_a_reachable_window(provider, _key):
    # A provider that lists an interval but no window for it would be clamped
    # to the fallback ceiling, i.e. asked for history no vendor serves.
    caps = provider.capabilities
    intraday = {i for i in caps.intervals if i != "1d"}
    missing = intraday - set(caps.max_days_by_interval)
    assert not missing, f"intervals with no declared window: {sorted(missing)}"


@pytest.mark.parametrize(("provider", "_key"), PROVIDERS)
def test_every_declared_window_is_positive(provider, _key):
    assert all(days > 0 for days in provider.capabilities.max_days_by_interval.values())


@pytest.mark.parametrize(("provider", "_key"), PROVIDERS)
def test_a_finer_interval_never_reaches_further_back_than_a_coarser_one(provider, _key):
    # Vendors keep less of the fine stuff, never more. A table that says
    # otherwise is a typo, and typos here become silent truncation.
    caps = provider.capabilities
    order = [i for i in ("1m", "5m", "15m", "30m", "1h", "1d") if i in caps.max_days_by_interval]
    windows = [caps.max_days_by_interval[i] for i in order]
    assert windows == sorted(windows), dict(zip(order, windows, strict=True))


@pytest.mark.parametrize(("provider", "_key"), PROVIDERS)
def test_a_paging_vendor_declares_the_page_size_it_pages_by(provider, _key):
    # anchors_window_on_end means the router must walk `end` backwards in
    # whole windows; a wrong or missing page size there silently skips bars.
    caps = provider.capabilities
    if caps.anchors_window_on_end:
        assert 0 < caps.max_bars_per_request < 100_000


@pytest.mark.parametrize(("provider", "_key"), PROVIDERS)
def test_volume_provenance_is_stated(provider, _key):
    # "Which kind of volume is this" has to be answerable without reading the
    # provider: a tick count and an exchange quantity are not comparable, and
    # a vendor with neither must not be allowed to imply zero.
    assert provider.capabilities.volume_source in set(VolumeSource)


def test_the_resolver_only_names_providers_that_exist():
    from app.market import symbols

    known = {DERIV, KITE, YFINANCE}
    for symbol, hint in (("XAUUSD", None), ("RELIANCE", "NSE"), ("AAPL", None), ("XYZ", "LSE")):
        assert symbols.resolve(symbol, hint).provider in known
