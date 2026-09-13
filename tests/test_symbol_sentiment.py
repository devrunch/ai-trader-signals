"""Per-symbol headline sentiment for the signal path, now that FinBERT is gone.

The honesty rule that matters here: "we could not read the news" must never be
reported as "the news is neutral" with a confident score.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.market import news
from app.signals import sentiment as sentiment_mod


class FakeLlm:
    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        body = self.bodies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])


def _articles(*titles):
    return [{"title": t} for t in titles]


class TestScoreHeadlines:
    async def test_labels_come_back_in_order(self):
        llm = FakeLlm(json.dumps(["positive", "NEGATIVE", "Neutral"]))
        assert await news.score_headlines(llm, ["a", "b", "c"]) == ["POSITIVE", "NEGATIVE", "NEUTRAL"]

    async def test_no_headlines_costs_no_call(self):
        llm = FakeLlm()
        assert await news.score_headlines(llm, []) == []
        assert llm.calls == []

    async def test_a_wrong_length_answer_is_discarded_not_misaligned(self):
        llm = FakeLlm(json.dumps(["POSITIVE"]))
        assert await news.score_headlines(llm, ["a", "b"]) is None

    async def test_an_unrecognised_label_discards_the_batch(self):
        llm = FakeLlm(json.dumps(["POSITIVE", "VAGUELY UPBEAT"]))
        assert await news.score_headlines(llm, ["a", "b"]) is None

    async def test_a_failing_model_returns_none_not_neutral(self):
        class Broken:
            def chat(self, **kwargs):
                raise RuntimeError("upstream down")

        assert await news.score_headlines(Broken(), ["a"]) is None


class TestSymbolSentiment:
    async def test_the_dominant_label_and_its_share_are_reported(self):
        llm = FakeLlm(json.dumps(["POSITIVE", "POSITIVE", "NEGATIVE", "NEUTRAL"]))
        with patch.object(news, "_fetch_newsapi", new=AsyncMock(return_value=_articles("a", "b", "c", "d"))):
            result = await sentiment_mod.symbol_sentiment("RELIANCE", llm=llm)

        assert result == {"label": "positive", "score": 0.5, "headlines_count": 4}

    async def test_unreadable_sentiment_is_neutral_with_no_confidence(self):
        llm = FakeLlm("not json at all")
        with patch.object(news, "_fetch_newsapi", new=AsyncMock(return_value=_articles("a", "b"))):
            result = await sentiment_mod.symbol_sentiment("RELIANCE", llm=llm)

        assert result["label"] == "neutral"
        assert result["score"] == 0.5          # the NEUTRAL default, not a computed share
        assert result["headlines_count"] == 2  # headlines were found, they just could not be read

    async def test_a_news_outage_degrades_instead_of_raising(self):
        with patch.object(news, "_fetch_newsapi", new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await sentiment_mod.symbol_sentiment("RELIANCE", llm=FakeLlm())

        assert result == {"label": "neutral", "score": 0.5, "headlines_count": 0}

    @pytest.mark.parametrize("articles", [[], [{"title": ""}]])
    async def test_no_usable_headlines_costs_no_llm_call(self, articles):
        llm = FakeLlm()
        with patch.object(news, "_fetch_newsapi", new=AsyncMock(return_value=articles)):
            result = await sentiment_mod.symbol_sentiment("RELIANCE", llm=llm)

        assert result["headlines_count"] == 0
        assert llm.calls == []
