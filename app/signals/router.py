from fastapi import APIRouter
from app.signals.service import SignalService

router = APIRouter()
_service = SignalService()


@router.post("/generate/{symbol}")
async def generate_signal(symbol: str, exchange: str = "NSE"):
    """Manually trigger signal generation for a symbol (dev/testing)."""
    signal = await _service.generate_signal(symbol.upper(), exchange.upper())
    if signal is None:
        return {"signal": None, "message": "No signal generated (below threshold or insufficient data)"}
    return {"signal": signal.__dict__}
