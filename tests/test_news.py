"""
News feed: sentiment (existing, unaffected here) + real per-headline stock
impact analysis (new -- replaces the old `_extract_symbols` keyword match).
Mocks NewsAPI/HF's own HTTP calls and the LLM client, same fake shapes
test_orchestrator.py already established for LlmClient.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.market import news


def _response(content: str | None = None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeLlm:
    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]


def _article(title="Headline", description="Description"):
    return {"title": title, "description": description, "url": "https://example.com/a", "publishedAt": "2026-01-01T00:00:00Z",
            "source": {"name": "Reuters"}}


class _FakeRedis:
    """Same minimal shape as test_macro_events.py's own fake -- get/set/aclose
    only. Always a cache miss unless the test seeds `store` itself, so these
    tests exercise the real fetch/analyze path, not a leftover cached page."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value.encode() if isinstance(value, str) else value

    async def aclose(self):
        pass


class TestParseImpactResponse:
    def test_a_clean_valid_response_parses_as_is(self):
        raw = json.dumps([
            {"affected": [{"symbol": "reliance", "direction": "down", "assetClass": "NSE", "reason": "Crude spike squeezes refining margins."}]},
            {"affected": []},
        ])
        result = news._parse_impact_response(raw, n=2)
        assert result == [
            [{"symbol": "RELIANCE", "direction": "down", "reason": "Crude spike squeezes refining margins.", "assetClass": "NSE"}],
            [],
        ]

    def test_a_missing_or_invalid_asset_class_defaults_to_other_not_dropped(self):
        raw = json.dumps([{"affected": [
            {"symbol": "TCS", "direction": "up", "reason": "No assetClass given."},
            {"symbol": "XAUUSD", "direction": "up", "assetClass": "MADE_UP", "reason": "Bogus class."},
        ]}])
        result = news._parse_impact_response(raw, n=1)
        assert result[0][0]["assetClass"] == "OTHER"
        assert result[0][1]["assetClass"] == "OTHER"

    def test_every_real_asset_class_passes_through(self):
        classes = ["NSE", "BSE", "NASDAQ", "NYSE", "FOREX", "MCX", "CRYPTO"]
        raw = json.dumps([{"affected": [
            {"symbol": "X", "direction": "up", "assetClass": c, "reason": "r"} for c in classes
        ]}])
        result = news._parse_impact_response(raw, n=1)
        assert [item["assetClass"] for item in result[0]] == classes

    def test_a_markdown_fenced_response_is_unwrapped_first(self):
        raw = "```json\n" + json.dumps([{"affected": []}]) + "\n```"
        assert news._parse_impact_response(raw, n=1) == [[]]

    def test_malformed_individual_entries_are_dropped_not_guessed_at(self):
        raw = json.dumps([{
            "affected": [
                {"symbol": "TCS", "direction": "up", "reason": "Real one."},
                {"symbol": "", "direction": "up", "reason": "No symbol -- dropped."},
                {"symbol": "INFY", "direction": "sideways", "reason": "Not up/down -- dropped."},
                "not even an object",
            ],
        }])
        result = news._parse_impact_response(raw, n=1)
        assert result == [[{"symbol": "TCS", "direction": "up", "reason": "Real one.", "assetClass": "OTHER"}]]

    def test_non_json_text_returns_none_not_an_empty_list(self):
        assert news._parse_impact_response("I cannot help with that.", n=3) is None

    def test_wrong_length_array_returns_none(self):
        # The model dropped or merged an entry -- can't trust the alignment
        # to the original headline order, so the whole batch is discarded
        # rather than silently misattributed.
        raw = json.dumps([{"affected": []}, {"affected": []}])
        assert news._parse_impact_response(raw, n=3) is None

    def test_a_reason_longer_than_200_chars_is_truncated_not_dropped(self):
        long_reason = "x" * 500
        raw = json.dumps([{"affected": [{"symbol": "TCS", "direction": "up", "reason": long_reason}]}])
        result = news._parse_impact_response(raw, n=1)
        assert len(result[0][0]["reason"]) == 200


class TestAnalyzeImpacts:
    @pytest.mark.asyncio
    async def test_no_articles_short_circuits_without_calling_the_llm(self):
        llm = FakeLlm()
        result = await news._analyze_impacts(llm, [])
        assert result == []
        assert llm.calls == []

    @pytest.mark.asyncio
    async def test_a_real_batched_call_covers_every_article_in_order(self):
        articles = [_article("CPI hotter than expected"), _article("Local bakery wins award")]
        llm = FakeLlm(_response(json.dumps([
            {"affected": [{"symbol": "XAUUSD", "direction": "down", "assetClass": "FOREX", "reason": "Hot CPI strengthens USD, pressuring gold."}]},
            {"affected": []},
        ])))
        result = await news._analyze_impacts(llm, articles)

        assert result == [
            [{"symbol": "XAUUSD", "direction": "down", "reason": "Hot CPI strengthens USD, pressuring gold.", "assetClass": "FOREX"}],
            [],
        ]
        # One call for the WHOLE batch, not one per article.
        assert len(llm.calls) == 1
        prompt = llm.calls[0]["messages"][1]["content"]
        assert "CPI hotter than expected" in prompt
        assert "Local bakery wins award" in prompt

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_to_none_not_a_crash(self):
        class BrokenLlm:
            def chat(self, **kwargs):
                raise RuntimeError("upstream down")
        result = await news._analyze_impacts(BrokenLlm(), [_article()])
        assert result is None

    @pytest.mark.asyncio
    async def test_a_wrong_length_response_is_retried_once_and_can_succeed(self):
        articles = [_article("A"), _article("B")]
        # First response drops an entry (wrong length); second is clean.
        llm = FakeLlm(
            _response(json.dumps([{"affected": []}])),
            _response(json.dumps([{"affected": []}, {"affected": []}])),
        )
        result = await news._analyze_impacts(llm, articles)
        assert result == [[], []]
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_two_wrong_length_responses_in_a_row_gives_up_as_none(self):
        articles = [_article("A"), _article("B")]
        llm = FakeLlm(
            _response(json.dumps([{"affected": []}])),
            _response(json.dumps([{"affected": []}])),
        )
        result = await news._analyze_impacts(llm, articles)
        assert result is None
        assert len(llm.calls) == 2


class _ContentAwareLlm:
    """Picks its response by matching a substring in the PROMPT rather than
    call order -- chunks run concurrently via asyncio.gather, so which
    chunk's chat() call actually lands first is not deterministic the way
    FakeLlm's plain queue assumes."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][1]["content"]
        for key, content in self.mapping.items():
            if key in prompt:
                return _response(content)
        raise AssertionError(f"No mapped response for prompt containing: {prompt[:120]!r}")


class _EchoLlm:
    """Returns a syntactically valid response sized to match whatever the
    prompt actually asked for -- avoids hardcoding one specific chunk
    split, since the point of the balanced-chunking test is the invariant
    (every article covered, no call ever sees exactly 1 headline unless
    there's truly only 1 article total), not one particular boundary."""

    def __init__(self):
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        n = len(re.findall(r"(?m)^\d+\. ", kwargs["messages"][1]["content"]))
        return _response(json.dumps([{"affected": []}] * n))


class TestAnalyzeImpactsChunking:
    @pytest.mark.asyncio
    async def test_more_than_chunk_size_splits_into_concurrent_calls_combined_in_order(self):
        # IMPACT_CHUNK_SIZE is 8 -- 10 articles balance into two 5-article
        # chunks (ceil(10/8)=2 chunks, split as evenly as possible), not
        # one call for all 10 and not a greedy 8+2.
        articles = [_article(f"Headline {i}") for i in range(10)]
        llm = _ContentAwareLlm({
            "0. Headline 0": json.dumps([{"affected": []}] * 5),
            "0. Headline 5": json.dumps([
                {"affected": []}, {"affected": []}, {"affected": []},
                {"affected": [{"symbol": "TCS", "direction": "up", "assetClass": "NSE", "reason": "r"}]},
                {"affected": []},
            ]),
        })
        result = await news._analyze_impacts(llm, articles)
        assert len(result) == 10
        assert result[8] == [{"symbol": "TCS", "direction": "up", "assetClass": "NSE", "reason": "r"}]
        assert result[9] == []
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_one_failing_chunk_fails_the_whole_batch(self):
        articles = [_article(f"Headline {i}") for i in range(10)]
        llm = _ContentAwareLlm({
            "0. Headline 0": json.dumps([{"affected": []}] * 5),
            "0. Headline 5": "not json at all",  # both the attempt and its retry return this
        })
        result = await news._analyze_impacts(llm, articles)
        assert result is None

    @pytest.mark.asyncio
    async def test_25_articles_split_evenly_with_no_size_one_straggler(self):
        # Confirmed live: a fixed-size greedy split of 25 articles at chunk
        # size 8 gives 8,8,8,1 -- and that lone leftover article was a
        # harder case for the model, sometimes answering with a bare `[]`
        # instead of the required one-object-per-headline response.
        articles = [_article(f"Headline {i}") for i in range(25)]
        llm = _EchoLlm()
        result = await news._analyze_impacts(llm, articles)
        assert result == [[] for _ in range(25)]
        assert len(llm.calls) == 4  # ceil(25/8)
        sizes = sorted(len(re.findall(r"(?m)^\d+\. ", c["messages"][1]["content"])) for c in llm.calls)
        assert sizes == [6, 6, 6, 7]
        assert 1 not in sizes


class TestGetMarketNewsResult:
    @pytest.mark.asyncio
    async def test_real_impacts_and_sentiment_both_land_on_the_right_article(self):
        articles = [_article("Rate cut expected"), _article("Nothing special")]
        llm = FakeLlm(_response(json.dumps([
            {"affected": [{"symbol": "NIFTY", "direction": "up", "assetClass": "NSE", "reason": "Cheaper credit lifts equities."}]},
            {"affected": []},
        ])))
        with patch("app.market.news._fetch_newsapi", new=AsyncMock(return_value=articles)), \
             patch("app.market.news._hf_sentiment_batch", new=AsyncMock(return_value=[("POSITIVE", 0.9), ("NEUTRAL", 0.0)])), \
             patch("app.market.news.redis.from_url", return_value=_FakeRedis()):
            result = await news.get_market_news_result(llm=llm)

        assert result["degraded"] is False
        assert result["degraded_reason"] is None
        a0, a1 = result["articles"]
        assert a0["sentiment"] == "POSITIVE"
        assert a0["impacts"] == [{"symbol": "NIFTY", "direction": "up", "reason": "Cheaper credit lifts equities.", "assetClass": "NSE"}]
        assert a1["impacts"] == []

    @pytest.mark.asyncio
    async def test_impact_analysis_failing_alone_is_reported_distinctly_from_sentiment_failing(self):
        articles = [_article()]
        broken_llm = FakeLlm()  # never queued a response -- IndexError inside chat()
        with patch("app.market.news._fetch_newsapi", new=AsyncMock(return_value=articles)), \
             patch("app.market.news._hf_sentiment_batch", new=AsyncMock(return_value=[("POSITIVE", 0.5)])), \
             patch("app.market.news.redis.from_url", return_value=_FakeRedis()):
            result = await news.get_market_news_result(llm=broken_llm)

        assert result["degraded"] is True
        assert result["degraded_reason"] == "impact_unavailable"
        assert result["articles"][0]["impacts"] is None  # not [] -- "couldn't check", not "nothing found"
        assert result["articles"][0]["sentimentAvailable"] is True

    @pytest.mark.asyncio
    async def test_both_pipelines_failing_is_reported_as_both(self):
        articles = [_article()]
        broken_llm = FakeLlm()
        with patch("app.market.news._fetch_newsapi", new=AsyncMock(return_value=articles)), \
             patch("app.market.news._hf_sentiment_batch", new=AsyncMock(return_value=None)), \
             patch("app.market.news.redis.from_url", return_value=_FakeRedis()):
            result = await news.get_market_news_result(llm=broken_llm)

        assert result["degraded_reason"] == "sentiment_and_impact_unavailable"
        assert result["articles"][0]["impacts"] is None
        assert result["articles"][0]["sentimentAvailable"] is False

    @pytest.mark.asyncio
    async def test_a_news_fetch_failure_still_returns_a_well_shaped_empty_result(self):
        with patch("app.market.news._fetch_newsapi", new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch("app.market.news.redis.from_url", return_value=_FakeRedis()):
            result = await news.get_market_news_result()
        assert result == {"articles": [], "count": 0, "degraded": True, "degraded_reason": "news_unavailable"}

    @pytest.mark.asyncio
    async def test_a_cached_result_is_returned_without_hitting_newsapi_again(self):
        fake_redis = _FakeRedis()
        articles = [_article("Cached headline")]
        llm = FakeLlm(_response(json.dumps([{"affected": []}])))
        fetch = AsyncMock(return_value=articles)
        with patch("app.market.news._fetch_newsapi", new=fetch), \
             patch("app.market.news._hf_sentiment_batch", new=AsyncMock(return_value=[("NEUTRAL", 0.0)])), \
             patch("app.market.news.redis.from_url", return_value=fake_redis):
            first = await news.get_market_news_result(llm=llm)
            second = await news.get_market_news_result(llm=llm)

        assert first == second
        # The second call served the cached result -- NewsAPI was hit once, not twice.
        assert fetch.await_count == 1

    @pytest.mark.asyncio
    async def test_a_total_fetch_failure_is_never_cached(self):
        fake_redis = _FakeRedis()
        fetch = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("app.market.news._fetch_newsapi", new=fetch), \
             patch("app.market.news.redis.from_url", return_value=fake_redis):
            await news.get_market_news_result()
            await news.get_market_news_result()

        # Retried on the very next call rather than replaying a cached failure.
        assert fetch.await_count == 2
