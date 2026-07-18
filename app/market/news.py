"""
News feed with FinBERT sentiment scoring.
Falls back to NewsAPI.org if FINBERT is unavailable.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"

INDIA_MARKET_QUERY = (
    "NSE OR BSE OR Nifty OR Sensex OR \"Indian stock\" OR SEBI OR "
    "\"Dalal Street\" OR RBI OR \"equity market\""
)


async def _fetch_newsapi(query: str, page_size: int = 20) -> list[dict]:
    if not settings.news_api_key:
        return []
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": settings.news_api_key,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(NEWSAPI_URL, params=params)
        r.raise_for_status()
        return r.json().get("articles", [])


def _finbert_sentiment(text: str) -> tuple[str, float]:
    """Returns (label, score). Lazy-loads FinBERT; falls back to NEUTRAL on error."""
    try:
        from transformers import pipeline
        _pipe = getattr(_finbert_sentiment, "_pipe", None)
        if _pipe is None:
            _pipe = pipeline(
                "text-classification",
                model=settings.finbert_model,
                truncation=True,
                max_length=512,
            )
            _finbert_sentiment._pipe = _pipe  # type: ignore[attr-defined]
        result = _pipe(text[:512])[0]
        label = result["label"].upper()   # positive | negative | neutral
        score = float(result["score"])
        return label, score if label == "POSITIVE" else (-score if label == "NEGATIVE" else 0.0)
    except Exception as e:
        logger.debug("FinBERT skipped: %s", e)
        return "NEUTRAL", 0.0


def _extract_symbols(text: str) -> list[str]:
    """Naive keyword match for common NSE tickers mentioned in a headline."""
    KNOWN = [
        "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "WIPRO",
        "TATAMOTORS", "TATASTEEL", "AXISBANK", "SBIN", "BAJFINANCE",
        "MARUTI", "HINDUNILVR", "ASIANPAINT", "LT", "SUNPHARMA",
        "NIFTY", "SENSEX", "BANKNIFTY",
    ]
    upper = text.upper()
    return [s for s in KNOWN if s in upper]


async def get_market_news(symbols: Optional[list[str]] = None, page_size: int = 15) -> list[dict]:
    """
    Fetch market news and score sentiment.
    symbols: optional list to narrow query (e.g. ["RELIANCE", "INFY"])
    """
    if symbols:
        query = " OR ".join(symbols[:5])
    else:
        query = INDIA_MARKET_QUERY

    try:
        articles = await _fetch_newsapi(query, page_size)
    except Exception as e:
        logger.warning("NewsAPI fetch failed: %s", e)
        return []

    results = []
    for a in articles:
        headline = a.get("title") or ""
        description = a.get("description") or ""
        text = f"{headline}. {description}"

        label, score = _finbert_sentiment(text)

        published = a.get("publishedAt", "")
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            published_iso = dt.astimezone(timezone.utc).isoformat()
        except Exception:
            published_iso = published

        results.append({
            "id": a.get("url", "")[-32:],
            "headline": headline,
            "description": description,
            "source": (a.get("source") or {}).get("name", ""),
            "url": a.get("url", ""),
            "publishedAt": published_iso,
            "sentiment": label,
            "sentimentScore": round(score, 4),
            "symbols": _extract_symbols(text),
        })

    return results
