"""
News feed with FinBERT sentiment scoring via HF Inference API.
No local torch/transformers — calls the free HF hosted inference endpoint.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Failures we expect from a third-party HTTP API: the network, a non-2xx, a
# body that is not the JSON shape documented. Anything else is our bug.
_API_ERRORS = (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError)

NEWSAPI_URL  = "https://newsapi.org/v2/everything"
# HF retired api-inference.huggingface.co — inference now routes through router.huggingface.co
HF_INFER_URL = "https://router.huggingface.co/hf-inference/models"

INDIA_MARKET_QUERY = (
    "NSE OR BSE OR Nifty OR Sensex OR \"Indian stock\" OR SEBI OR "
    "\"Dalal Street\" OR RBI OR \"equity market\""
)


async def _fetch_newsapi(query: str, page_size: int = 20) -> list[dict]:
    settings = get_settings()
    if not settings.news_api_key:
        return []
    params: dict[str, str | int] = {
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


async def _hf_sentiment_batch(texts: list[str]) -> list[tuple[str, float]] | None:
    """
    Call HF Inference API in one batch request.
    Returns list of (label, score) where label is POSITIVE/NEGATIVE/NEUTRAL.

    Returns **None** when scoring did not happen — an unset token, a network
    failure, HF rate-limiting us. It used to return a full list of
    ("NEUTRAL", 0.0), which is indistinguishable from genuinely neutral news:
    a dead sentiment pipeline read to every caller as "the market feels fine".
    Callers must decide what unavailable sentiment means for them.
    """
    if not texts:
        return []
    settings = get_settings()
    url = f"{HF_INFER_URL}/{settings.finbert_model}"
    headers = {"Authorization": f"Bearer {settings.hf_api_token}"} if settings.hf_api_token else {}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json={"inputs": texts}, headers=headers)
            resp.raise_for_status()
            raw = resp.json()
            # raw is [[{label, score}, ...], ...] — one list per input text
            results = []
            for item in raw:
                best = max(item, key=lambda x: x["score"])
                label = best["label"].upper()
                score = float(best["score"])
                results.append((label, score if label == "POSITIVE" else (-score if label == "NEGATIVE" else 0.0)))
            if len(results) != len(texts):
                logger.warning(
                    "HF sentiment returned %d scores for %d texts — discarding",
                    len(results), len(texts),
                )
                return None
            return results
    except _API_ERRORS as e:
        logger.warning("HF sentiment batch failed: %s", e)
        return None
    except Exception:
        logger.exception("Unexpected error scoring sentiment for %d texts", len(texts))
        return None


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


async def get_market_news_result(
    symbols: list[str] | None = None, page_size: int = 15
) -> dict:
    """
    Fetch market news and score sentiment, reporting what failed.

    symbols: optional list to narrow query (e.g. ["RELIANCE", "INFY"])

    Returns `{articles, count, degraded, degraded_reason}`. `degraded` is the
    point of this function: both failure modes below used to return a plain
    list — empty for a news outage, NEUTRAL-scored for a sentiment outage — and
    the caller could not tell either apart from a quiet news day.
    """
    if symbols:
        query = " OR ".join(symbols[:5])
    else:
        query = INDIA_MARKET_QUERY

    try:
        articles = await _fetch_newsapi(query, page_size)
    except _API_ERRORS as e:
        logger.warning("NewsAPI fetch failed: %s", e)
        return {"articles": [], "count": 0, "degraded": True, "degraded_reason": "news_unavailable"}
    except Exception:
        logger.exception("Unexpected error fetching news for query %r", query)
        return {"articles": [], "count": 0, "degraded": True, "degraded_reason": "news_unavailable"}

    texts = [f"{a.get('title') or ''}. {a.get('description') or ''}" for a in articles]
    scored = await _hf_sentiment_batch(texts)
    sentiment_ok = scored is not None
    sentiments: list[tuple[str, float]] = scored if scored is not None else [("NEUTRAL", 0.0)] * len(texts)

    results = []
    # strict=True: `_hf_sentiment_batch` already rejects a length mismatch, and
    # the neutral fallback is built from `texts`, so the two are the same length
    # by construction. If that ever stops being true the scores would be
    # silently attached to the wrong headlines.
    for a, (label, score) in zip(articles, sentiments, strict=True):
        headline    = a.get("title") or ""
        description = a.get("description") or ""
        text        = f"{headline}. {description}"

        published = a.get("publishedAt", "")
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            published_iso = dt.astimezone(UTC).isoformat()
        except (ValueError, AttributeError, TypeError):
            # NewsAPI has been seen returning a non-ISO string; passing it
            # through unparsed is better than dropping the article.
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
            # False means the NEUTRAL above is "we could not score it", not
            # "FinBERT read it as neutral".
            "sentimentAvailable": sentiment_ok,
            "symbols": _extract_symbols(text),
        })

    return {
        "articles": results,
        "count": len(results),
        "degraded": not sentiment_ok,
        "degraded_reason": None if sentiment_ok else "sentiment_unavailable",
    }


async def get_market_news(symbols: list[str] | None = None, page_size: int = 15) -> list[dict]:
    """Articles only. Prefer `get_market_news_result` — it says what degraded."""
    return (await get_market_news_result(symbols, page_size))["articles"]
