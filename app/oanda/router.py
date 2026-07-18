"""
OANDA Forex paper trading endpoints — called by NestJS API.

Base path: /oanda (registered in main.py)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.oanda import client as oanda

router = APIRouter()


class MarketOrderRequest(BaseModel):
    instrument: str
    units: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


class LimitOrderRequest(BaseModel):
    instrument: str
    units: float
    price: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


class CloseTradeRequest(BaseModel):
    units: Optional[float] = None


def _handle(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OANDA error: {e}")


@router.get("/account")
def account_summary():
    """OANDA demo account summary — balance, NAV, unrealised P&L."""
    return _handle(oanda.get_account_summary)


@router.get("/price/{instrument}")
def price(instrument: str):
    """Live bid/ask snapshot for a Forex pair (e.g. EUR_USD)."""
    return _handle(oanda.get_price, instrument)


@router.post("/order/market")
def market_order(req: MarketOrderRequest):
    """Place a market order. units > 0 = BUY, units < 0 = SELL."""
    return _handle(
        oanda.place_market_order,
        req.instrument, req.units, req.take_profit, req.stop_loss
    )


@router.post("/order/limit")
def limit_order(req: LimitOrderRequest):
    """Place a limit order."""
    return _handle(
        oanda.place_limit_order,
        req.instrument, req.units, req.price, req.take_profit, req.stop_loss
    )


@router.get("/trades")
def open_trades():
    """List all open OANDA trades on the demo account."""
    return _handle(oanda.list_open_trades)


@router.post("/trades/{trade_id}/close")
def close_trade(trade_id: str, req: CloseTradeRequest = CloseTradeRequest()):
    """Close a trade fully or partially."""
    return _handle(oanda.close_trade, trade_id, req.units)


@router.get("/positions")
def open_positions():
    """List all open Forex positions."""
    return _handle(oanda.list_open_positions)
