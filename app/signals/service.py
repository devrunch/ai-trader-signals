"""
SignalService — single entry point for all signal generation logic.

Flow per symbol:
  1. Fetch OHLCV data (yfinance / NSE)
  2. Compute technical indicators (RSI, MACD, EMA, VWAP, SuperTrend, ADX)
  3. Run FinBERT sentiment on recent headlines
  4. Call Claude with full context → structured reasoning + signal
  5. Validate confidence threshold and R:R ratio
  6. Publish to Redis → NestJS picks up → stores in MongoDB → WebSocket push
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class GeneratedSignal:
    symbol: str
    exchange: str
    signal_type: SignalType
    confidence: float
    entry_price: float
    target_price: float
    stop_loss: float
    reasoning: str
    indicators: dict


class SignalService:
    def __init__(self):
        self._redis = aioredis.from_url(settings.redis_url)
        self._finbert = None   # loaded lazily on first use
        self._claude = None    # anthropic client, loaded lazily

    # ------------------------------------------------------------------
    # Public entry point — Celery tasks call only this
    # ------------------------------------------------------------------
    async def generate_signal(self, symbol: str, exchange: str = "NSE") -> Optional[GeneratedSignal]:
        try:
            df = await self._fetch_ohlcv(symbol, exchange)
            if df is None or len(df) < 50:
                return None

            indicators = self._compute_indicators(df)
            sentiment = await self._run_finbert(symbol)
            signal = await self._call_claude(symbol, exchange, df, indicators, sentiment)

            if signal is None:
                return None
            if signal.confidence < settings.confidence_threshold:
                logger.info("Signal for %s below threshold (%.2f)", symbol, signal.confidence)
                return None

            await self._publish(signal)
            return signal
        except Exception:
            logger.exception("Signal generation failed for %s", symbol)
            return None

    # ------------------------------------------------------------------
    # Step 1: OHLCV fetch
    # ------------------------------------------------------------------
    async def _fetch_ohlcv(self, symbol: str, exchange: str) -> Optional[pd.DataFrame]:
        import yfinance as yf
        ticker = f"{symbol}.NS" if exchange == "NSE" else f"{symbol}.BO"
        df = yf.download(ticker, period="60d", interval="15m", progress=False, auto_adjust=True)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df

    # ------------------------------------------------------------------
    # Step 2: Indicators (RSI, MACD, EMA20/50, VWAP, SuperTrend, ADX)
    # ------------------------------------------------------------------
    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        import pandas_ta as ta

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        rsi = ta.rsi(close, length=14)
        macd = ta.macd(close)
        ema20 = ta.ema(close, length=20)
        ema50 = ta.ema(close, length=50)
        adx = ta.adx(high, low, close, length=14)
        supertrend = ta.supertrend(high, low, close, length=10, multiplier=3)

        # VWAP (intraday — rolling 390-min proxy)
        typical = (high + low + close) / 3
        vwap = (typical * volume).cumsum() / volume.cumsum()

        last = -1
        return {
            "rsi": round(float(rsi.iloc[last]), 2) if rsi is not None else None,
            "macd": round(float(macd["MACD_12_26_9"].iloc[last]), 4) if macd is not None else None,
            "macd_signal": round(float(macd["MACDs_12_26_9"].iloc[last]), 4) if macd is not None else None,
            "ema20": round(float(ema20.iloc[last]), 2) if ema20 is not None else None,
            "ema50": round(float(ema50.iloc[last]), 2) if ema50 is not None else None,
            "adx": round(float(adx["ADX_14"].iloc[last]), 2) if adx is not None else None,
            "supertrend_dir": int(supertrend["SUPERTd_10_3.0"].iloc[last]) if supertrend is not None else None,
            "vwap": round(float(vwap.iloc[last]), 2),
            "ltp": round(float(df["close"].iloc[last]), 2),
        }

    # ------------------------------------------------------------------
    # Step 3: FinBERT sentiment
    # ------------------------------------------------------------------
    async def _run_finbert(self, symbol: str) -> dict:
        headlines = await self._fetch_headlines(symbol)
        if not headlines:
            return {"label": "neutral", "score": 0.5, "headlines_count": 0}

        if self._finbert is None:
            from transformers import pipeline
            self._finbert = pipeline(
                "text-classification",
                model=settings.finbert_model,
                truncation=True,
                max_length=512,
            )

        results = self._finbert(headlines[:10])
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for r in results:
            scores[r["label"]] += r["score"]

        dominant = max(scores, key=scores.__getitem__)
        return {
            "label": dominant,
            "score": round(scores[dominant] / len(results), 3),
            "headlines_count": len(headlines),
        }

    async def _fetch_headlines(self, symbol: str) -> list[str]:
        """Fetch recent news headlines for the symbol."""
        if not settings.news_api_key:
            return []
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": symbol,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 15,
                    "apiKey": settings.news_api_key,
                },
            )
            if resp.status_code != 200:
                return []
            articles = resp.json().get("articles", [])
            return [a["title"] for a in articles if a.get("title")]

    # ------------------------------------------------------------------
    # Step 4: Claude reasoning
    # ------------------------------------------------------------------
    async def _call_claude(
        self,
        symbol: str,
        exchange: str,
        df: pd.DataFrame,
        indicators: dict,
        sentiment: dict,
    ) -> Optional[GeneratedSignal]:
        if self._claude is None:
            import anthropic
            self._claude = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        ltp = indicators["ltp"]
        prompt = f"""You are an expert Indian intraday equity trader analysing {symbol} ({exchange}).

CURRENT PRICE: ₹{ltp}

TECHNICAL INDICATORS:
- RSI(14): {indicators['rsi']}
- MACD: {indicators['macd']} | Signal: {indicators['macd_signal']}
- EMA20: {indicators['ema20']} | EMA50: {indicators['ema50']}
- ADX(14): {indicators['adx']}
- SuperTrend direction: {'Bullish' if indicators['supertrend_dir'] == 1 else 'Bearish'}
- VWAP: {indicators['vwap']}

NEWS SENTIMENT: {sentiment['label'].upper()} (score: {sentiment['score']}, from {sentiment['headlines_count']} headlines)

RECENT OHLCV (last 5 candles, 15m):
{df[['open','high','low','close','volume']].tail(5).to_string()}

Analyse and respond with ONLY valid JSON:
{{
  "signal_type": "BUY" | "SELL" | "HOLD",
  "confidence": <0.0-1.0>,
  "entry_price": <float>,
  "target_price": <float>,
  "stop_loss": <float>,
  "reasoning": "<2-3 sentences explaining the signal>"
}}

Rules: Only BUY/SELL if confidence >= 0.65. Minimum R:R = 1.5. For intraday only."""

        response = await self._claude.messages.create(
            model=settings.anthropic_model,
            max_tokens=512,
            system=[{
                "type": "text",
                "text": "You are an expert Indian equity trader. Respond ONLY with valid JSON.",
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            data = json.loads(response.content[0].text)
            rr = (data["target_price"] - data["entry_price"]) / max(
                data["entry_price"] - data["stop_loss"], 0.01
            )
            if rr < settings.min_reward_risk:
                return None

            return GeneratedSignal(
                symbol=symbol,
                exchange=exchange,
                signal_type=SignalType(data["signal_type"]),
                confidence=float(data["confidence"]),
                entry_price=float(data["entry_price"]),
                target_price=float(data["target_price"]),
                stop_loss=float(data["stop_loss"]),
                reasoning=data["reasoning"],
                indicators=indicators,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Claude response parse error for %s: %s", symbol, e)
            return None

    # ------------------------------------------------------------------
    # Step 5: Publish to Redis → NestJS picks up
    # ------------------------------------------------------------------
    async def _publish(self, signal: GeneratedSignal) -> None:
        payload = json.dumps({
            "symbol": signal.symbol,
            "exchange": signal.exchange,
            "signal_type": signal.signal_type.value,
            "confidence": signal.confidence,
            "entry_price": signal.entry_price,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "reasoning": signal.reasoning,
            "indicators": signal.indicators,
        })
        await self._redis.publish("signals:new", payload)
        logger.info("Published %s signal for %s (conf=%.2f)", signal.signal_type, signal.symbol, signal.confidence)
