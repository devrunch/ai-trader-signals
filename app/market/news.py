"""
News feed with FinBERT sentiment scoring (HF Inference API) and real
per-headline stock-impact analysis (LLM, one batched call for the whole
page) -- see _analyze_impacts' own docstring for why this replaced a naive
keyword match against a fixed ticker shortlist.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import httpx

from app.config import get_settings
from app.llm.client import LlmClient, get_llm
from app.signals.prompts import extract_json_text

logger = logging.getLogger(__name__)

# Failures we expect from a third-party HTTP API: the network, a non-2xx, a
# body that is not the JSON shape documented. Anything else is our bug.
_API_ERRORS = (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError)

NEWSAPI_URL  = "https://newsapi.org/v2/everything"
# HF retired api-inference.huggingface.co — inference now routes through router.huggingface.co
HF_INFER_URL = "https://router.huggingface.co/hf-inference/models"


# The India-only version of this query meant a real, forex-moving headline
# (e.g. "Dollar falls ahead of US inflation data") never matched at all --
# this app charts FOREX/metals (XAUUSD etc.) too, so the default feed can't
# stay India-equity-only.
DEFAULT_MARKET_QUERY = (
    "NSE OR BSE OR Nifty OR Sensex OR \"Indian stock\" OR SEBI OR "
    "\"Dalal Street\" OR RBI OR \"equity market\" OR "
    "\"Federal Reserve\" OR \"interest rate\" OR inflation OR CPI OR "
    "dollar OR gold OR crude OR forex OR \"currency market\""
)

NEWS_IMPACT_SYSTEM = (
    "You are a markets analyst. Never invent a connection that isn't really "
    "there -- most headlines affect no tradeable stock or instrument at all, "
    "and an empty list is the correct, honest answer for those, not a guess."
)

# This app's own real exchange/asset-class set (market.controller.ts's
# EXCHANGES) plus CRYPTO (informational only -- no crypto trading/price
# integration exists anywhere in this app) and OTHER, the honest fallback
# for anything that doesn't fit rather than a forced wrong guess.
ASSET_CLASSES = frozenset({"NSE", "BSE", "NASDAQ", "NYSE", "FOREX", "MCX", "CRYPTO", "OTHER"})

# One entry per article costs real output tokens (symbol + direction +
# assetClass + a reason each). 2500 was sized for a mostly-India-equity feed
# where few headlines had any real impact; broadening the default query to
# include global macro/forex made real (multi-symbol) impacts common enough
# per page that a 25-article batch routinely got truncated mid-array --
# confirmed live ("News impact analysis returned 24 entries for 25
# articles"), which discards the WHOLE batch since a short array can't be
# trusted to still be in headline order. Not raised further than this
# without also raising page_size, since the call still needs a real cap.
IMPACT_MAX_TOKENS = 4096


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


def _parse_impact_response(raw_text: str, n: int) -> list[list[dict]] | None:
    """Parses the LLM's JSON array into one clean impacts list per article,
    in order. Any malformed entry (missing symbol, a direction other than
    up/down, wrong shape) is dropped rather than guessed at -- a partially-
    wrong impact list is worse than a shorter, honest one. Returns None
    (not a partial result) if the response isn't even the right SHAPE
    (not a JSON array, or the wrong length) -- that means the whole batch
    failed to analyze, not that zero articles had impacts."""
    try:
        parsed = json.loads(extract_json_text(raw_text))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list) or len(parsed) != n:
        logger.warning(
            "News impact analysis returned %s entries for %d articles — discarding",
            len(parsed) if isinstance(parsed, list) else type(parsed).__name__, n,
        )
        return None

    results: list[list[dict]] = []
    for entry in parsed:
        affected = entry.get("affected") if isinstance(entry, dict) else None
        clean: list[dict] = []
        if isinstance(affected, list):
            for item in affected:
                if not isinstance(item, dict):
                    continue
                symbol = item.get("symbol")
                direction = item.get("direction")
                if not symbol or direction not in ("up", "down"):
                    continue
                asset_class = item.get("assetClass")
                clean.append({
                    "symbol": str(symbol).strip().upper(),
                    "direction": direction,
                    "reason": str(item.get("reason") or "").strip()[:200],
                    "assetClass": asset_class if asset_class in ASSET_CLASSES else "OTHER",
                })
        results.append(clean)
    return results


async def _analyze_impacts(llm: LlmClient, articles: list[dict]) -> list[list[dict]] | None:
    """Real per-headline stock/instrument impact, one batched LLM call for
    the whole page -- not the keyword match this replaced (`_extract_symbols`,
    a plain substring check against a hardcoded 17-ticker shortlist: blind to
    anything outside that list, and "mentions the ticker" is not the same
    question as "this headline plausibly moves this instrument"). The model
    names real symbols freely, not constrained to any fixed universe -- a
    hardcoded shortlist here would just be the same arbitrariness one level
    up.

    Returns None (not a list of empty lists) when the WHOLE batch could not
    be analyzed -- no LLM configured, the call failed, the response was the
    wrong shape. Callers must not read that as "no headline had a real
    impact today"; see get_market_news_result's own docs for how it's
    surfaced.
    """
    if not articles:
        return []
    numbered = "\n".join(
        f"{i}. {a.get('title') or ''} -- {a.get('description') or ''}"
        for i, a in enumerate(articles)
    )
    n = len(articles)
    prompt = (
        "For each numbered headline below, name the real, tradeable stocks or "
        "instruments it plausibly affects (their real ticker or a clear, "
        "specific name -- not limited to any fixed list), the direction each "
        "would plausibly move (\"up\" or \"down\"), which market it trades on "
        "(assetClass: one of NSE, BSE, NASDAQ, NYSE, FOREX, MCX, CRYPTO -- use "
        "OTHER only if truly none fit), and one short reason grounded in the "
        "headline itself. Most headlines affect nothing tradeable -- return an "
        "empty \"affected\" list for those rather than forcing a connection. "
        "A headline with several affected instruments still gets ONE object, "
        "with all of them in that one \"affected\" array -- never split one "
        f"headline across two objects.\n\n{numbered}\n\n"
        f"Respond with ONLY a JSON array of EXACTLY {n} objects -- one per "
        "headline above, in the same order, never fewer or more -- no other "
        "text:\n"
        "[{\"affected\": [{\"symbol\": \"RELIANCE\", \"direction\": \"down\", "
        "\"assetClass\": \"NSE\", \"reason\": \"...\"}]}, "
        "{\"affected\": [{\"symbol\": \"BZ=F\", \"direction\": \"up\", "
        "\"assetClass\": \"MCX\", \"reason\": \"...\"}, {\"symbol\": \"USO\", "
        "\"direction\": \"up\", \"assetClass\": \"NYSE\", \"reason\": \"...\"}]}, "
        "...]"
    )
    messages = [
        {"role": "system", "content": NEWS_IMPACT_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    # One retry, not a loop: the model occasionally drifts off the exact
    # count (splits a multi-impact headline into two objects, or drops one)
    # -- worth one more real attempt before reporting the whole batch
    # unavailable, but this must stay bounded, not become a silent retry
    # storm against a model that's reliably getting it wrong.
    for attempt in range(2):
        try:
            resp = await asyncio.to_thread(
                llm.chat, temperature=0, max_tokens=IMPACT_MAX_TOKENS, messages=messages,
            )
        except Exception as e:
            logger.warning("News impact analysis failed: %s", e)
            return None
        parsed = _parse_impact_response(resp.choices[0].message.content or "", n)
        if parsed is not None:
            return parsed
        if attempt == 0:
            logger.info("News impact analysis retrying once after a malformed response")
    return None


async def get_market_news_result(
    symbols: list[str] | None = None, page_size: int = 15, llm: LlmClient | None = None,
) -> dict:
    """
    Fetch market news, score sentiment, and analyze real per-headline stock
    impact -- reporting what failed at each stage rather than silently
    degrading to something that looks like a normal result.

    symbols: optional list to narrow query (e.g. ["RELIANCE", "INFY"])

    Returns `{articles, count, degraded, degraded_reason}`. Both the
    sentiment and impact pipelines can fail independently of the news fetch
    itself and of each other -- each article carries its own
    `sentimentAvailable` (already existed) and `impacts: null` (new) so a
    caller can tell "analyzed, found nothing" from "couldn't analyze",
    which used to look identical for both sentiment (NEUTRAL either way)
    and impact (empty list either way).
    """
    if symbols:
        query = " OR ".join(symbols[:5])
    else:
        query = DEFAULT_MARKET_QUERY

    try:
        articles = await _fetch_newsapi(query, page_size)
    except _API_ERRORS as e:
        logger.warning("NewsAPI fetch failed: %s", e)
        return {"articles": [], "count": 0, "degraded": True, "degraded_reason": "news_unavailable"}
    except Exception:
        logger.exception("Unexpected error fetching news for query %r", query)
        return {"articles": [], "count": 0, "degraded": True, "degraded_reason": "news_unavailable"}

    texts = [f"{a.get('title') or ''}. {a.get('description') or ''}" for a in articles]
    llm = llm or get_llm()
    sentiment_task = _hf_sentiment_batch(texts)
    impacts_task = _analyze_impacts(llm, articles)
    scored, impacts_by_article = await asyncio.gather(sentiment_task, impacts_task)

    sentiment_ok = scored is not None
    sentiments: list[tuple[str, float]] = scored if scored is not None else [("NEUTRAL", 0.0)] * len(texts)
    impacts_ok = impacts_by_article is not None
    impacts_list: list[list[dict] | None] = (
        list(impacts_by_article) if impacts_by_article is not None else [None] * len(articles)
    )

    results = []
    # strict=True: both fallback lists above are built to match `articles`'
    # own length by construction (or `_hf_sentiment_batch`/_analyze_impacts
    # already rejected a mismatched batch and returned None). If that ever
    # stops being true, scores/impacts would be silently attached to the
    # wrong headline.
    for a, (label, score), impacts in zip(articles, sentiments, impacts_list, strict=True):
        headline    = a.get("title") or ""
        description = a.get("description") or ""

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
            # Real per-symbol impact from the LLM analysis above -- null
            # (not []) means the batch couldn't be analyzed at all, same
            # "unavailable, not empty" contract sentimentAvailable already
            # gives sentiment.
            "impacts": impacts,
        })

    degraded = not sentiment_ok or not impacts_ok
    if not sentiment_ok and not impacts_ok:
        degraded_reason = "sentiment_and_impact_unavailable"
    elif not sentiment_ok:
        degraded_reason = "sentiment_unavailable"
    elif not impacts_ok:
        degraded_reason = "impact_unavailable"
    else:
        degraded_reason = None

    return {
        "articles": results,
        "count": len(results),
        "degraded": degraded,
        "degraded_reason": degraded_reason,
    }


async def get_market_news(symbols: list[str] | None = None, page_size: int = 15) -> list[dict]:
    """Articles only. Prefer `get_market_news_result` — it says what degraded."""
    return (await get_market_news_result(symbols, page_size))["articles"]
