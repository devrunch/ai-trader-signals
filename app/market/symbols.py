"""Symbol -> the vendor that can actually answer for it.

Callers used to pass an `exchange` alongside every symbol, and all three layers
above this one default it to ``NSE``. A caller that simply does not know one --
a saved chart layout, a chat turn, a watchlist row -- therefore asked for gold
on the Indian equity exchange, which 404s. The venue is a property of the
symbol, so it is resolved here instead of guessed there.

`hint_exchange` stays useful for the genuinely ambiguous case (a ticker listed
on both NSE and BSE) and is ignored everywhere else.
"""
from __future__ import annotations

from functools import lru_cache

from app.market import sessions
from app.market.contract import AssetClass, SymbolInfo, VolumeSource
from app.market.providers.deriv_provider import KNOWN_PAIRS, deriv_symbol_for

# Registry keys, matching MarketDataRouter.providers.
DERIV = "deriv"
KITE = "kite"
YFINANCE = "yfinance"

_KITE_EXCHANGES = frozenset({"NSE", "BSE", "MCX"})
_METALS = frozenset({"XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"})

_INTRADAY_AND_DAILY = frozenset({"1m", "5m", "15m", "30m", "1h", "1d"})


def _index_or_equity(symbol: str) -> AssetClass:
    # Yahoo's own convention, which this app already follows for ^NSEI etc.
    return AssetClass.INDEX if symbol.startswith("^") else AssetClass.EQUITY


@lru_cache(maxsize=2048)
def resolve(symbol: str, hint_exchange: str | None = None) -> SymbolInfo | None:
    """The one way a request reaches a provider. None means we cannot chart it.

    Cached: a symbol does not change venue intraday, and this is on the path of
    every quote, bar and tick request.
    """
    symbol = symbol.upper().strip()
    if not symbol:
        return None
    hint = (hint_exchange or "").upper().strip()

    pair = deriv_symbol_for(symbol)
    if pair:
        # The symbol decides, not the hint: Deriv's table is the authority for
        # its 29 instruments, and no equity exchange lists any of them.
        return SymbolInfo(
            symbol=symbol,
            vendor_symbol=pair,
            provider=DERIV,
            asset_class=AssetClass.METAL if symbol in _METALS else AssetClass.FX,
            exchange="FOREX",
            session=sessions.FX,
            # Deriv publishes no volume for spot/CFD forex, and a delayed
            # tick count is worse than none on a 1m chart.
            volume_source=VolumeSource.NONE,
            intervals=_INTRADAY_AND_DAILY,
        )

    if hint in _KITE_EXCHANGES:
        return SymbolInfo(
            symbol=symbol,
            vendor_symbol=f"{hint}:{symbol}",
            provider=KITE,
            asset_class=AssetClass.COMMODITY if hint == "MCX" else _index_or_equity(symbol),
            exchange=hint,
            session=sessions.MCX if hint == "MCX" else sessions.NSE,
            volume_source=VolumeSource.EXCHANGE,
            intervals=_INTRADAY_AND_DAILY,
            # The hint named a venue; whether that venue lists this symbol is
            # Kite's instrument dump to answer, not ours.
            authoritative=False,
        )

    # Everything else is the fallback vendor's: US listings, and any symbol a
    # caller named without a venue we recognise.
    exchange = hint or "NASDAQ"
    return SymbolInfo(
        symbol=symbol,
        vendor_symbol=symbol,
        provider=YFINANCE,
        asset_class=_index_or_equity(symbol),
        exchange=exchange,
        session=sessions.US_EQUITY if exchange in {"NASDAQ", "NYSE"} else sessions.ALWAYS_OPEN,
        volume_source=VolumeSource.EXCHANGE,
        intervals=_INTRADAY_AND_DAILY,
        authoritative=False,
    )


def known_forex_symbols() -> frozenset[str]:
    """The pairs the resolver will route to Deriv regardless of hint."""
    return frozenset(KNOWN_PAIRS)
