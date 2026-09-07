"""
reddit_sentiment -- odd-hour crowd-sentiment check. Mocks
macro_events.reddit_chatter (already covers Tavily's own HTTP shape in
test_macro_events.py) and the LLM client, same FakeLlm pattern
test_news.py established.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.market import reddit_sentiment


def _response(content: str | None = None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeLlm:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _tavily_result(snippets: list[tuple[str, str]]) -> dict:
    return {
        "answer": None,
        "count": len(snippets),
        "results": [{"title": t, "url": "https://reddit.com/x", "snippet": s} for t, s in snippets],
    }


class TestCheck:
    @pytest.mark.asyncio
    async def test_no_real_snippets_returns_none_without_calling_the_llm(self):
        llm = FakeLlm(_response("should never be reached"))
        with patch("app.market.macro_events.reddit_chatter", new=AsyncMock(return_value={"error": "not configured"})):
            result = await reddit_sentiment.check(llm)
        assert result is None
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_real_snippets_produce_a_grounded_alert(self):
        llm = FakeLlm(_response(json.dumps([
            {"topic": "india_equity", "sentiment": "bullish", "reason": "Threads expect a rate cut to lift banks."},
        ])))
        with patch("app.market.macro_events.reddit_chatter", new=AsyncMock(
            side_effect=[_tavily_result([("Nifty rally?", "Many expect a breakout")]), _tavily_result([])],
        )):
            result = await reddit_sentiment.check(llm)
        assert result is not None
        assert result["type"] == "reddit_sentiment"
        assert result["data"]["topics"] == [
            {"topic": "india_equity", "sentiment": "bullish", "reason": "Threads expect a rate cut to lift banks."},
        ]
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_a_topic_with_no_real_coverage_is_skipped_not_guessed(self):
        # The model correctly reports only one topic even though the prompt
        # asks about two -- that's the honest answer, not a malformed response.
        llm = FakeLlm(_response(json.dumps([
            {"topic": "global_macro", "sentiment": "mixed", "reason": "Split views on gold."},
        ])))
        with patch("app.market.macro_events.reddit_chatter", new=AsyncMock(
            side_effect=[_tavily_result([]), _tavily_result([("Gold talk", "Mixed takes on gold")])],
        )):
            result = await reddit_sentiment.check(llm)
        assert result is not None
        assert len(result["data"]["topics"]) == 1
        assert result["data"]["topics"][0]["topic"] == "global_macro"

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_to_none_not_a_crash(self):
        class BrokenLlm:
            def chat(self, **kwargs):
                raise RuntimeError("upstream down")
        with patch("app.market.macro_events.reddit_chatter", new=AsyncMock(
            return_value=_tavily_result([("x", "y")]),
        )):
            result = await reddit_sentiment.check(BrokenLlm())
        assert result is None

    @pytest.mark.asyncio
    async def test_a_non_json_response_returns_none(self):
        llm = FakeLlm(_response("I cannot help with that."))
        with patch("app.market.macro_events.reddit_chatter", new=AsyncMock(
            return_value=_tavily_result([("x", "y")]),
        )):
            result = await reddit_sentiment.check(llm)
        assert result is None
