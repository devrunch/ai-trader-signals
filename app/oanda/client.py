"""
OANDA v20 adapter — Forex paper trading via OANDA demo account.

Wraps oandapyV20 to provide:
  - Account summary (balance, NAV, unrealised P&L)
  - Live streaming bid/ask prices
  - Place market/limit orders
  - List open trades and positions
  - Close a trade

Requires env vars:
    OANDA_API_KEY   — generated at https://www.oanda.com/demo-account/tpa/personal_token
    OANDA_ACCOUNT_ID — e.g. 101-001-12345678-001
    OANDA_ENV       — "practice" (demo) or "live"
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

try:
    import oandapyV20
    import oandapyV20.endpoints.accounts as accounts_ep
    import oandapyV20.endpoints.orders as orders_ep
    import oandapyV20.endpoints.trades as trades_ep
    import oandapyV20.endpoints.positions as positions_ep
    import oandapyV20.endpoints.pricing as pricing_ep
    from oandapyV20.contrib.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitDetails, StopLossDetails
    OANDA_AVAILABLE = True
except ImportError:
    OANDA_AVAILABLE = False
    logger.warning("oandapyV20 not installed — OANDA endpoints will return errors")


def _get_client() -> "oandapyV20.API":
    if not OANDA_AVAILABLE:
        raise RuntimeError("oandapyV20 is not installed. Run: pip install oandapyV20")
    if not settings.oanda_api_key:
        raise RuntimeError("OANDA_API_KEY is not configured")
    return oandapyV20.API(
        access_token=settings.oanda_api_key,
        environment=settings.oanda_env,
    )


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

def get_account_summary() -> dict:
    client = _get_client()
    r = accounts_ep.AccountSummary(settings.oanda_account_id)
    client.request(r)
    acc = r.response["account"]
    return {
        "account_id": acc["id"],
        "currency": acc["currency"],
        "balance": float(acc["balance"]),
        "nav": float(acc["NAV"]),
        "unrealised_pl": float(acc["unrealizedPL"]),
        "margin_used": float(acc["marginUsed"]),
        "open_trade_count": int(acc["openTradeCount"]),
    }


# ---------------------------------------------------------------------------
# Pricing (bid/ask snapshot)
# ---------------------------------------------------------------------------

def get_price(instrument: str) -> dict:
    """instrument e.g. EUR_USD, GBP_USD, USD_JPY"""
    instrument = _normalise_instrument(instrument)
    client = _get_client()
    params = {"instruments": instrument}
    r = pricing_ep.PricingInfo(settings.oanda_account_id, params=params)
    client.request(r)
    price = r.response["prices"][0]
    bid = float(price["bids"][0]["price"])
    ask = float(price["asks"][0]["price"])
    return {
        "instrument": instrument,
        "bid": bid,
        "ask": ask,
        "mid": round((bid + ask) / 2, 5),
        "spread": round(ask - bid, 5),
        "tradeable": price.get("tradeable", True),
        "time": price.get("time"),
    }


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def place_market_order(
    instrument: str,
    units: float,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
) -> dict:
    """
    units > 0 = BUY, units < 0 = SELL.
    Automatically rounds units to nearest integer (OANDA requires integers for most pairs).
    """
    instrument = _normalise_instrument(instrument)
    client = _get_client()

    tp = TakeProfitDetails(price=take_profit) if take_profit else None
    sl = StopLossDetails(price=stop_loss) if stop_loss else None

    order_data = MarketOrderRequest(
        instrument=instrument,
        units=int(units),
        takeProfitOnFill=tp,
        stopLossOnFill=sl,
    )
    r = orders_ep.OrderCreate(settings.oanda_account_id, data=order_data.data)
    client.request(r)

    fill = r.response.get("orderFillTransaction", {})
    return {
        "order_id": r.response.get("relatedTransactionIDs", [None])[0],
        "trade_id": fill.get("tradeOpened", {}).get("tradeID"),
        "instrument": instrument,
        "units": int(units),
        "price": float(fill.get("price", 0)),
        "pl": float(fill.get("pl", 0)),
        "status": "FILLED" if fill else "PENDING",
    }


def place_limit_order(
    instrument: str,
    units: float,
    price: float,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
) -> dict:
    instrument = _normalise_instrument(instrument)
    client = _get_client()

    tp = TakeProfitDetails(price=take_profit) if take_profit else None
    sl = StopLossDetails(price=stop_loss) if stop_loss else None

    order_data = LimitOrderRequest(
        instrument=instrument,
        units=int(units),
        price=str(round(price, 5)),
        takeProfitOnFill=tp,
        stopLossOnFill=sl,
    )
    r = orders_ep.OrderCreate(settings.oanda_account_id, data=order_data.data)
    client.request(r)
    return {
        "order_id": r.response.get("relatedTransactionIDs", [None])[0],
        "instrument": instrument,
        "units": int(units),
        "limit_price": price,
        "status": "PENDING",
    }


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def list_open_trades() -> list[dict]:
    client = _get_client()
    r = trades_ep.OpenTrades(settings.oanda_account_id)
    client.request(r)
    return [
        {
            "trade_id": t["id"],
            "instrument": t["instrument"],
            "units": float(t["currentUnits"]),
            "open_price": float(t["price"]),
            "unrealised_pl": float(t["unrealizedPL"]),
            "open_time": t["openTime"],
        }
        for t in r.response.get("trades", [])
    ]


def close_trade(trade_id: str, units: Optional[float] = None) -> dict:
    """Close a trade fully (default) or partially by specifying units."""
    client = _get_client()
    data = {"units": str(int(units))} if units else {"units": "ALL"}
    r = trades_ep.TradeClose(settings.oanda_account_id, tradeID=trade_id, data=data)
    client.request(r)
    fill = r.response.get("orderFillTransaction", {})
    return {
        "trade_id": trade_id,
        "price": float(fill.get("price", 0)),
        "pl": float(fill.get("pl", 0)),
        "status": "CLOSED",
    }


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def list_open_positions() -> list[dict]:
    client = _get_client()
    r = positions_ep.OpenPositions(settings.oanda_account_id)
    client.request(r)
    result = []
    for pos in r.response.get("positions", []):
        long_units = float(pos["long"].get("units", 0))
        short_units = float(pos["short"].get("units", 0))
        result.append({
            "instrument": pos["instrument"],
            "long_units": long_units,
            "short_units": short_units,
            "long_avg_price": float(pos["long"].get("averagePrice", 0)) if long_units else None,
            "short_avg_price": float(pos["short"].get("averagePrice", 0)) if short_units else None,
            "unrealised_pl": float(pos.get("unrealizedPL", 0)),
        })
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_instrument(pair: str) -> str:
    """EURUSD → EUR_USD, EUR/USD → EUR_USD, EUR_USD → EUR_USD"""
    pair = pair.upper().replace("/", "_").replace("-", "_")
    if "_" not in pair and len(pair) == 6:
        pair = f"{pair[:3]}_{pair[3:]}"
    return pair
