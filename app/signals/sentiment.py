"""
Per-symbol news sentiment for the signal engine.

Headlines and scoring both come from `market.news`; this module only keeps the
aggregation the signal path needs — the dominant label across recent headlines
for one symbol.
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.llm.client import LlmClient, get_llm
from app.market import news

logger = logging.getLogger(__name__)

NEUTRAL: dict = {"label": "neutral", "score": 0.5, "headlines_count": 0}


async def symbol_sentiment(symbol: str, llm: LlmClient | None = None) -> dict:
    """Dominant sentiment label across recent headlines for one symbol.

    Returns `{label, score, headlines_count}`. Never raises — sentiment is an
    input to the signal, not a precondition for it, so a news outage degrades to
    neutral rather than killing the signal. The degradation is logged.
    """
    settings = get_settings()
    try:
        articles = await news._fetch_newsapi(symbol, page_size=settings.sentiment_headline_limit)
    except Exception as e:
        logger.warning("Headline fetch failed for %s: %s", symbol, e)
        return dict(NEUTRAL)

    headlines = [a["title"] for a in articles if a.get("title")]
    if not headlines:
        return dict(NEUTRAL)

    labels = await news.score_headlines(llm or get_llm(), headlines)
    if not labels:
        return {**NEUTRAL, "headlines_count": len(headlines)}

    counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
    for label in labels:
        counts[label.lower()] += 1

    dominant = max(counts, key=counts.__getitem__)
    return {
        "label": dominant,
        # Share of headlines carrying the dominant label -- how one-sided the
        # coverage is, which is what the signal prompt reads it as.
        "score": round(counts[dominant] / len(labels), 3),
        "headlines_count": len(headlines),
    }
