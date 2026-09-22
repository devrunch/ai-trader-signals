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

    async def run() -> tuple[dict, bool, int]:
        result = await news.get_market_news_result(page_size=25)
        published = await news.publish(result)
        # After publishing, so a Telegram outage cannot cost the terminal its
        # feed -- the web panel reads the published result either way.
        pushed = await push_shocks(result.get("articles") or [])
        return result, published, pushed

    try:
        result, ok, pushed = run_async(run())
        logger.info("News analysis done: %d articles, degraded=%s, published=%s, pushed=%d",
                    result["count"], result["degraded"], ok, pushed)
        return {"count": result["count"], "degraded": result["degraded"],
                "published": ok, "pushed": pushed}
    except Exception:
        logger.exception("News analysis failed")
        return {"error": True}


async def push_shocks(articles: list[dict]) -> int:
    """Send the headlines that move this desk, and nothing else.

    A sink on the pipeline above rather than a pipeline of its own: selection
    reads the sentiment and impact analysis that run already paid for, so the
    only new cost is the Telegram call. The write-up costs a model call and is
    therefore behind the button, not in this message.

    Never raises. A push problem must not fail the news run that produced the
    articles, which the terminal reads from whether or not Telegram is up.
    """
    from app.events import shocks, telegram

    try:
        found = [s for s in (shocks.select(a) for a in articles) if s is not None]
        fresh = await shocks.unseen(shocks.rank(found))
        sent = 0
        for shock in fresh[:shocks.MAX_PER_RUN]:
            token = await shocks.offer(shock)
            if await telegram.send(shocks.render(shock),
                                   buttons=telegram.brief_button(token)):
                sent += 1
        if found:
            logger.info("Shocks: %d matched, %d new, %d sent",
                        len(found), len(fresh), sent)
        return sent
    except Exception:
        logger.exception("Could not push news shocks")
        return 0
