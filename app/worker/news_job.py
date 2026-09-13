"""The hourly news pipeline, importable without the terminal's stack.

`newsd` runs this job and nothing else. It does not live in jobs.py because
that module imports SignalService, and pandas with it: the news process is
memory-capped precisely so the LLM work cannot starve the charts, which only
holds while this import stays light (tests/test_news_import_weight.py).
"""
from __future__ import annotations

import logging

from app.worker import heartbeat
from app.worker.runner import run_async

logger = logging.getLogger(__name__)


@heartbeat.monitored("news-analysis")
def run_news_analysis():
    """News + sentiment + real per-headline stock impact -- hourly, all
    day (NewsAPI's free tier caps at 100 requests/day, so hourly rather
    than every 15 min). Moves the real NewsAPI/HF/LLM work out of the
    request path:
    the frontend now reads the stored latest result from ai-trader-api
    instead of triggering this analysis live on every page load, the same
    pipeline-then-read pattern the brief and the two alert tasks already
    use."""
    from app.market import news

    async def run() -> tuple[dict, bool]:
        result = await news.get_market_news_result(page_size=25)
        return result, await news.publish(result)

    try:
        result, ok = run_async(run())
        logger.info("News analysis done: %d articles, degraded=%s, published=%s",
                    result["count"], result["degraded"], ok)
        return {"count": result["count"], "degraded": result["degraded"], "published": ok}
    except Exception:
        logger.exception("News analysis failed")
        return {"error": True}
