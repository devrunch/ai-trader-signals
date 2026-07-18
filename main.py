from fastapi import FastAPI
from app.config import settings
from app.signals.router import router as signals_router
from app.market.router import router as market_router
from app.oanda.router import router as oanda_router

app = FastAPI(title="AI Trader Signals Service", version="0.1.0")

app.include_router(signals_router, prefix="/signals", tags=["signals"])
app.include_router(market_router, prefix="/market", tags=["market"])
app.include_router(oanda_router, prefix="/oanda", tags=["oanda"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "signals"}
