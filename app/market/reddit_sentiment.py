"""
Odd-hour Reddit-flavored crowd-sentiment check.

Runs at 1, 3, 5, ... 23 IST (see celery_app.py's own schedule), not every
hour, to keep Tavily's free-credit usage down -- two focused
site:reddit.com searches per run via macro_events.reddit_chatter (India
equity chatter, global macro/forex/crypto chatter), one LLM call
summarizes whatever real snippets came back into a short sentiment note
per topic. Same discipline as news.py's own impact analysis: the LLM only
writes the read on real fetched snippets, it never invents a sentiment for
a topic nothing came back for.
"""
from __future__ import annotations

import asyncio
import json
import logging

from app.llm.client import LlmClient, get_llm
from app.market import macro_events
from app.signals.prompts import extract_json_text

logger = logging.getLogger(__name__)

# Fixed, small, and split by asset class rather than one per-symbol query --
# a query per major would multiply Tavily calls past what the odd-hour
# throttle is for. topic key -> search query.
_QUERIES: dict[str, str] = {
    "india_equity": "Nifty Sensex Indian stock market",
    "global_macro": "gold XAUUSD crude oil Bitcoin dollar forex",
}

SENTIMENT_SYSTEM = (
    "You read real Reddit discussion snippets and report the crowd's mood. "
    "Never invent a reading for a topic the snippets don't actually cover "
    "-- skip it, don't guess."
)

MAX_TOKENS = 1024


def _parse_response(raw_text: str) -> list[dict] | None:
    """Malformed individual entries are dropped, not guessed at -- same
    rule as news.py's _parse_impact_response. Unlike that function this one
    does NOT enforce an exact count: the whole point of this prompt is that
    a topic gets skipped when the snippets don't cover it, so a short array
    is the expected, correct shape, not a sign something broke."""
    try:
        parsed = json.loads(extract_json_text(raw_text))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        topic = item.get("topic")
        sentiment = item.get("sentiment")
        if not topic or sentiment not in ("bullish", "bearish", "mixed"):
            continue
        out.append({
            "topic": str(topic).strip(),
            "sentiment": sentiment,
            "reason": str(item.get("reason") or "").strip()[:200],
        })
    return out


async def check(llm: LlmClient | None = None) -> dict | None:
    """Returns an alert dict (`type: "reddit_sentiment"`) when real Reddit
    chatter was found and summarized. Returns None when there was nothing
    real to report -- Tavily not configured, both searches came back
    empty, or the summarization failed -- the caller (the Celery task)
    skips on None either way."""
    results = await asyncio.gather(*(
        macro_events.reddit_chatter(query, max_results=5) for query in _QUERIES.values()
    ))

    snippets: list[str] = []
    for (key, _query), result in zip(_QUERIES.items(), results, strict=True):
        if result.get("error"):
            logger.info("Reddit sentiment search skipped for %s: %s", key, result["error"])
            continue
        for r in result.get("results", []):
            if r.get("snippet"):
                snippets.append(f"[{key}] {r.get('title', '')}: {r['snippet'][:300]}")

    if not snippets:
        return None

    prompt = (
        "Real Reddit discussion snippets below, each tagged by topic area "
        "(india_equity or global_macro). For each topic area that has REAL "
        "coverage below, report the crowd's overall mood as one of "
        "\"bullish\", \"bearish\", or \"mixed\", with one short reason "
        "grounded in the actual snippets. Skip a topic area entirely if "
        "nothing below actually covers it.\n\n"
        + "\n".join(snippets)
        + "\n\nRespond with ONLY a JSON array, no other text:\n"
        "[{\"topic\": \"india_equity\", \"sentiment\": \"bullish\", \"reason\": \"...\"}]"
    )
    llm = llm or get_llm()
    try:
        resp = await asyncio.to_thread(
            llm.chat, temperature=0, max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SENTIMENT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        topics = _parse_response(resp.choices[0].message.content or "")
    except Exception as e:
        logger.warning("Reddit sentiment summarization failed: %s", e)
        return None

    if not topics:
        return None

    return {
        "type": "reddit_sentiment",
        "title": f"Reddit sentiment: {', '.join(t['topic'] for t in topics)}",
        "body": " | ".join(f"{t['topic']}: {t['sentiment']} -- {t['reason']}" for t in topics),
        "symbols": [t["topic"] for t in topics],
        "data": {"topics": topics},
    }
