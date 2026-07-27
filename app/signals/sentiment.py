"""
Per-symbol news sentiment for the signal engine.

`SignalService` used to carry its own NewsAPI client and its own HF FinBERT
client, duplicating `app/market/news.py` down to the copy-pasted comment about
HF retiring the old inference endpoint — with the two copies disagreeing on
page size and on how per-article scores were aggregated. This module keeps the
aggregation the signal path needs and delegates all I/O to `market.news`.
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.market import news

logger = logging.getLogger(__name__)

NEUTRAL: dict = {"label": "neutral", "score": 0.5, "headlines_count": 0}


async def symbol_sentiment(symbol: str) -> dict:
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

    scored = await news._hf_sentiment_batch(headlines)
    if not scored:
        return {**NEUTRAL, "headlines_count": len(headlines)}

    # `_hf_sentiment_batch` returns signed scores (+ for POSITIVE, - for
    # NEGATIVE, 0 for NEUTRAL). Aggregate per label and take the dominant one,
    # which is the shape `generate_signal` and the prompt expect.
    counts: dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for label, score in scored:
        counts[label.lower()] = counts.get(label.lower(), 0.0) + abs(score)

    if not any(counts.values()):
        return {**NEUTRAL, "headlines_count": len(headlines)}

    dominant = max(counts, key=counts.__getitem__)
    return {
        "label": dominant,
        "score": round(counts[dominant] / len(scored), 3),
        "headlines_count": len(headlines),
    }
