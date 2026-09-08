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
from unittest.mock import AsyncMock, MagicMock, patch

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
    # A url per title -- _merge_sources dedupes on url, so a shared one
    # would silently collapse every multi-article fixture down to one.
    slug = title.lower().replace(" ", "-")
    return {"title": title, "description": description, "url": f"https://example.com/{slug}",
            "publishedAt": "2026-01-01T00:00:00Z", "source": {"name": "Reuters"}}


class _FakeRedis:
    """Same minimal shape as test_macro_events.py's own fake -- get/set/aclose
    only. Always a cache miss unless the test seeds `store` itself, so these
    tests exercise the real fetch/analyze path, not a leftover cached page."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    async def get(self, key):
        return self.store.get(key)

    async def mget(self, keys):
        return [self.store.get(k) for k in keys]

    async def set(self, key, value, ex=None):
        self.store[key] = value.encode() if isinstance(value, str) else value

    async def aclose(self):
        pass


class TestDedupeArticles:
    def test_the_same_url_twice_is_kept_once(self):
        articles = [
            {"title": "A", "url": "https://x.com/1"},
            {"title": "B", "url": "https://x.com/1"},
        ]
        assert news._dedupe_articles(articles) == [{"title": "A", "url": "https://x.com/1"}]

    def test_a_syndicated_story_under_two_urls_is_kept_once(self):
        # Confirmed live: NewsAPI served "US Stock Market: Equity funds see
        # second straight week of outflows" twice in one page.
        articles = [
            {"title": "Equity funds see outflows", "url": "https://a.com/1"},
            {"title": "equity funds see outflows", "url": "https://b.com/2"},
        ]
        result = news._dedupe_articles(articles)
        assert len(result) == 1
        assert result[0]["url"] == "https://a.com/1"

    def test_genuinely_different_articles_all_survive(self):
        articles = [
            {"title": "A", "url": "https://x.com/1"},
            {"title": "B", "url": "https://x.com/2"},
        ]
        assert len(news._dedupe_articles(articles)) == 2

    def test_articles_missing_a_url_or_title_are_not_collapsed_together(self):
        # Two untitled, unlinked articles are not evidence of a duplicate --
        # dropping one would lose a real headline on a technicality.
        articles = [{"description": "one"}, {"description": "two"}]
        assert len(news._dedupe_articles(articles)) == 2


class TestParseHfSentiment:
    """Shapes B and C below are verbatim from the live endpoint -- the
    router served B while the old parser only understood A, which is why
    every article read `sentimentAvailable: false` until it was caught."""

    def test_shape_a_one_list_of_all_labels_per_input(self):
        raw = [
            [{"label": "positive", "score": 0.8}, {"label": "negative", "score": 0.1}],
            [{"label": "positive", "score": 0.2}, {"label": "negative", "score": 0.9}],
        ]
        assert news._parse_hf_sentiment(raw, 2) == [("POSITIVE", 0.8), ("NEGATIVE", -0.9)]

    def test_shape_b_single_wrapper_with_one_top_result_per_input(self):
        # Captured live: 3 inputs -> one inner list of 3 top results.
        raw = [[
            {"label": "positive", "score": 0.8221865892410278},
            {"label": "negative", "score": 0.7716991305351257},
            {"label": "neutral", "score": 0.611751914024353},
        ]]
        assert news._parse_hf_sentiment(raw, 3) == [
            ("POSITIVE", 0.8221865892410278),
            ("NEGATIVE", -0.7716991305351257),
            ("NEUTRAL", 0.0),
        ]

    def test_shape_c_already_flat(self):
        raw = [{"label": "positive", "score": 0.7}, {"label": "neutral", "score": 0.6}]
        assert news._parse_hf_sentiment(raw, 2) == [("POSITIVE", 0.7), ("NEUTRAL", 0.0)]

    def test_a_single_input_is_read_the_same_under_a_or_b(self):
        # With n == 1 the two shapes are indistinguishable; both must
        # reduce to the argmax of that one input's labels.
        assert news._parse_hf_sentiment([[{"label": "positive", "score": 0.82}]], 1) == [("POSITIVE", 0.82)]

    def test_neutral_is_zero_regardless_of_confidence(self):
        raw = [{"label": "neutral", "score": 0.99}]
        assert news._parse_hf_sentiment(raw, 1) == [("NEUTRAL", 0.0)]

    def test_a_length_mismatch_is_rejected_rather_than_misaligned(self):
        # Two scores for three inputs cannot be attached to the right
        # headlines, so the whole batch is discarded.
        assert news._parse_hf_sentiment([{"label": "positive", "score": 0.5}] * 2, 3) is None

    def test_an_unrecognisable_body_returns_none(self):
        assert news._parse_hf_sentiment({"error": "model loading"}, 2) is None
        assert news._parse_hf_sentiment(["nonsense"], 1) is None


class TestFinanceDomains:
    def test_no_duplicates(self):
        # A hand-maintained grouped list makes it easy to add the same
        # outlet under two headings (cnbc.com was in both "markets" and
        # "wires" on the first draft of the expanded list).
        dupes = {d for d in news._FINANCE_DOMAIN_LIST if news._FINANCE_DOMAIN_LIST.count(d) > 1}
        assert dupes == set()

    def test_entries_are_bare_hostnames(self):
        # NewsAPI's `domains` wants hostnames, not URLs -- a scheme or a
        # path silently matches nothing rather than erroring.
        for d in news._FINANCE_DOMAIN_LIST:
            assert "/" not in d and ":" not in d, d
            assert "." in d, d


class TestFetchNewsapi:
    @pytest.mark.asyncio
    async def test_it_restricts_to_finance_domains_and_dedupes_to_page_size(self):
        raw = {"articles": [
            {"title": "A", "url": "https://x.com/1"},
            {"title": "A", "url": "https://x.com/1"},  # dupe
            {"title": "B", "url": "https://x.com/2"},
            {"title": "C", "url": "https://x.com/3"},
        ]}
        captured: dict = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return raw

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, params=None):
                captured.update(params or {})
                return _Resp()

        settings = MagicMock()
        settings.news_api_key = "k"
        with patch("app.market.news.get_settings", return_value=settings), \
             patch("app.market.news.httpx.AsyncClient", return_value=_Client()):
            result = await news._fetch_newsapi("q", page_size=2)

        assert [a["title"] for a in result] == ["A", "B"]  # deduped, then capped at page_size
        assert "reuters.com" in captured["domains"]
        assert captured["pageSize"] == 4  # over-fetched (page_size * 2) before deduping

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty_without_a_request(self):
        settings = MagicMock()
        settings.news_api_key = ""
        with patch("app.market.news.get_settings", return_value=settings):
            assert await news._fetch_newsapi("q") == []


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
    @pytest.fixture(autouse=True)
    def _only_newsapi(self):
        """These cases are about the NewsAPI half and the assembly around
        it. Without this the real yfinance source would run -- a live
        network call inside a unit test. newsdata is pinned explicitly
        too rather than relying on the test env happening to have no key.
        The merge itself has its own cases in TestMergeSources /
        TestSourceIndependence below."""
        with patch("app.market.news._fetch_yfinance", new=AsyncMock(return_value=[])), \
             patch("app.market.news._fetch_newsdata_safe", new=AsyncMock(return_value=[])), \
             patch("app.market.news._fetch_alphavantage_safe", new=AsyncMock(return_value=[])):
            yield

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


class TestYfToArticle:
    def test_it_maps_onto_newsapi_shape(self):
        result = news._yf_to_article({
            "title": "Gold rallies", "summary": "On Fed bets", "url": "https://y.com/1",
            "published_at": "2026-09-08T01:47:00Z", "publisher": "Reuters",
        })
        assert result == {
            "title": "Gold rallies", "description": "On Fed bets", "url": "https://y.com/1",
            "publishedAt": "2026-09-08T01:47:00Z", "source": {"name": "Reuters"},
        }

    def test_the_older_payload_shape_epoch_seconds_becomes_an_iso_string(self):
        result = news._yf_to_article({"title": "T", "published_at": 1757300820})
        assert result["publishedAt"].startswith("2025-") or result["publishedAt"].startswith("2026-")
        assert "T" in result["publishedAt"]

    def test_a_missing_publisher_falls_back_rather_than_reading_as_blank(self):
        assert news._yf_to_article({"title": "T"})["source"]["name"] == "Yahoo Finance"


class TestMergeSources:
    def _at(self, title, when):
        return {"title": title, "url": f"https://x.com/{title}", "publishedAt": when}

    def test_newest_first_across_all_sources(self):
        result = news._merge_sources([
            [self._at("old", "2026-09-06T00:00:00Z")],       # NewsAPI, 24h lag
            [self._at("new", "2026-09-08T06:00:00Z")],       # yfinance, ~1h lag
            [self._at("middle", "2026-09-07T18:00:00Z")],    # newsdata, 12h lag
        ], page_size=10)
        assert [a["title"] for a in result] == ["new", "middle", "old"]

    def test_the_same_story_from_two_sources_appears_once(self):
        shared = {"title": "Fed holds", "url": "https://x.com/fed", "publishedAt": "2026-09-08T00:00:00Z"}
        result = news._merge_sources([[shared], [dict(shared)], []], page_size=10)
        assert len(result) == 1

    def test_an_unparseable_timestamp_sorts_last_but_is_not_dropped(self):
        result = news._merge_sources([
            [self._at("broken", "not-a-date")],
            [self._at("fine", "2026-09-08T00:00:00Z")],
        ], page_size=10)
        assert [a["title"] for a in result] == ["fine", "broken"]

    def test_it_trims_to_page_size(self):
        articles = [self._at(str(i), "2026-09-08T00:00:00Z") for i in range(10)]
        assert len(news._merge_sources([articles, []], page_size=3)) == 3


class TestNewsdata:
    def test_it_maps_onto_newsapi_shape_and_normalises_the_timestamp(self):
        result = news._newsdata_to_article({
            "title": "Oil rises", "description": "Iran tensions", "link": "https://n.io/1",
            "pubDate": "2026-09-07 16:00:00", "source_name": "Reuters",
        })
        assert result == {
            "title": "Oil rises", "description": "Iran tensions", "url": "https://n.io/1",
            "publishedAt": "2026-09-07T16:00:00+00:00", "source": {"name": "Reuters"},
        }

    def test_an_unexpected_date_format_is_passed_through_not_dropped(self):
        result = news._newsdata_to_article({"title": "T", "pubDate": "07/09/2026"})
        assert result["publishedAt"] == "07/09/2026"

    @pytest.mark.asyncio
    async def test_no_key_means_the_source_is_simply_absent(self):
        settings = MagicMock()
        settings.newsdata_api_key = ""
        with patch("app.market.news.get_settings", return_value=settings):
            assert await news._fetch_newsdata() == []

    @pytest.mark.asyncio
    async def test_an_in_band_error_object_is_not_treated_as_articles(self):
        # newsdata reports some failures with HTTP 200 and `results` as an
        # error OBJECT rather than a list.
        payload = {"status": "error", "results": {"message": "bad domain", "code": "UnsupportedFilter"}}

        class _Resp:
            def raise_for_status(self): pass
            def json(self): return payload

        class _Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, url, params=None): return _Resp()

        settings = MagicMock()
        settings.newsdata_api_key = "k"
        with patch("app.market.news.get_settings", return_value=settings), \
             patch("app.market.news.httpx.AsyncClient", return_value=_Client()):
            assert await news._fetch_newsdata() == []

    @pytest.mark.asyncio
    async def test_a_failure_never_reaches_the_other_sources(self):
        with patch("app.market.news._fetch_newsdata", new=AsyncMock(side_effect=RuntimeError("boom"))):
            assert await news._fetch_newsdata_safe(10) == []

    def test_the_query_fits_the_free_tier_100_char_cap(self):
        assert len(news.NEWSDATA_QUERY) <= 100

    def test_no_more_than_five_domains(self):
        assert len(news.NEWSDATA_DOMAINS.split(",")) <= 5


class TestAlphaVantage:
    def _payload(self, n=2):
        return {"feed": [
            {"title": f"AV {i}", "summary": "s", "url": f"https://av.co/{i}",
             "time_published": "20260908T042705", "source": "CNBC"}
            for i in range(n)
        ]}

    def _client(self, payload, captured=None):
        class _Resp:
            def raise_for_status(self): pass
            def json(self): return payload

        class _Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, url, params=None):
                if captured is not None:
                    captured.update(params or {})
                return _Resp()
        return _Client()

    def test_it_maps_onto_newsapi_shape_and_normalises_the_timestamp(self):
        result = news._alphavantage_to_article({
            "title": "Oil up", "summary": "Iran", "url": "https://av.co/1",
            "time_published": "20260908T042705", "source": "CNBC",
        })
        assert result == {
            "title": "Oil up", "description": "Iran", "url": "https://av.co/1",
            "publishedAt": "2026-09-08T04:27:05+00:00", "source": {"name": "CNBC"},
        }

    @pytest.mark.asyncio
    async def test_no_key_means_the_source_is_simply_absent(self):
        settings = MagicMock()
        settings.alphavantage_api_key = ""
        with patch("app.market.news.get_settings", return_value=settings):
            assert await news._fetch_alphavantage() == []

    @pytest.mark.asyncio
    async def test_a_second_call_is_served_from_cache_not_the_api(self):
        # The whole point: the free tier is 25 requests/DAY, and an hourly
        # pipeline plus any manual run would blow straight through it.
        fake_redis = _FakeRedis()
        settings = MagicMock()
        settings.alphavantage_api_key = "k"
        settings.redis_url = "redis://fake"
        calls = {"n": 0}

        class _CountingClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, url, params=None):
                calls["n"] += 1
                class _R:
                    def raise_for_status(self): pass
                    def json(_self): return {"feed": [{"title": "AV", "url": "https://av.co/1",
                                                       "time_published": "20260908T042705"}]}
                return _R()

        with patch("app.market.news.get_settings", return_value=settings), \
             patch("app.market.news.redis.from_url", return_value=fake_redis), \
             patch("app.market.news.httpx.AsyncClient", side_effect=lambda **kw: _CountingClient()):
            first = await news._fetch_alphavantage()
            second = await news._fetch_alphavantage()

        assert first == second
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_an_in_band_throttle_note_is_not_treated_as_articles(self):
        # Alpha Vantage reports throttling with HTTP 200 and a "Note" key.
        settings = MagicMock()
        settings.alphavantage_api_key = "k"
        settings.redis_url = "redis://fake"
        payload = {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."}
        with patch("app.market.news.get_settings", return_value=settings), \
             patch("app.market.news.redis.from_url", return_value=_FakeRedis()), \
             patch("app.market.news.httpx.AsyncClient", return_value=self._client(payload)):
            assert await news._fetch_alphavantage() == []

    @pytest.mark.asyncio
    async def test_a_throttled_response_is_never_cached(self):
        fake_redis = _FakeRedis()
        settings = MagicMock()
        settings.alphavantage_api_key = "k"
        settings.redis_url = "redis://fake"
        with patch("app.market.news.get_settings", return_value=settings), \
             patch("app.market.news.redis.from_url", return_value=fake_redis), \
             patch("app.market.news.httpx.AsyncClient", return_value=self._client({"Note": "limit"})):
            await news._fetch_alphavantage()
        assert fake_redis.store == {}

    @pytest.mark.asyncio
    async def test_a_failure_never_reaches_the_other_sources(self):
        with patch("app.market.news._fetch_alphavantage", new=AsyncMock(side_effect=RuntimeError("boom"))):
            assert await news._fetch_alphavantage_safe(10) == []


class TestSourceIndependence:
    @pytest.mark.asyncio
    async def test_one_surviving_source_still_produces_a_feed(self):
        yf = [_article("Yahoo only")]
        llm = FakeLlm(_response(json.dumps([{"affected": []}])))
        with patch("app.market.news._fetch_newsapi", new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch("app.market.news._fetch_yfinance", new=AsyncMock(return_value=yf)), \
             patch("app.market.news._fetch_newsdata_safe", new=AsyncMock(return_value=[])), \
             patch("app.market.news._hf_sentiment_batch", new=AsyncMock(return_value=[("NEUTRAL", 0.0)])), \
             patch("app.market.news.redis.from_url", return_value=_FakeRedis()):
            result = await news.get_market_news_result(llm=llm)

        assert result["count"] == 1
        assert result["articles"][0]["headline"] == "Yahoo only"
        assert result["degraded_reason"] != "news_unavailable"

    @pytest.mark.asyncio
    async def test_newsdata_alone_still_produces_a_feed(self):
        nd = [_article("Reuters via newsdata")]
        llm = FakeLlm(_response(json.dumps([{"affected": []}])))
        with patch("app.market.news._fetch_newsapi", new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch("app.market.news._fetch_yfinance", new=AsyncMock(return_value=[])), \
             patch("app.market.news._fetch_newsdata_safe", new=AsyncMock(return_value=nd)), \
             patch("app.market.news._hf_sentiment_batch", new=AsyncMock(return_value=[("NEUTRAL", 0.0)])), \
             patch("app.market.news.redis.from_url", return_value=_FakeRedis()):
            result = await news.get_market_news_result(llm=llm)

        assert result["count"] == 1
        assert result["articles"][0]["headline"] == "Reuters via newsdata"

    @pytest.mark.asyncio
    async def test_only_all_four_failing_is_a_real_news_outage(self):
        with patch("app.market.news._fetch_newsapi", new=AsyncMock(side_effect=RuntimeError("boom"))), \
             patch("app.market.news._fetch_yfinance", new=AsyncMock(return_value=[])), \
             patch("app.market.news._fetch_newsdata_safe", new=AsyncMock(return_value=[])), \
             patch("app.market.news._fetch_alphavantage_safe", new=AsyncMock(return_value=[])), \
             patch("app.market.news.redis.from_url", return_value=_FakeRedis()):
            result = await news.get_market_news_result()

        assert result["degraded_reason"] == "news_unavailable"


class TestImpactCacheAcrossRuns:
    @pytest.mark.asyncio
    async def test_an_already_analyzed_url_is_not_sent_to_the_llm_again(self):
        articles = [_article("Seen before"), _article("Brand new")]
        fake_redis = _FakeRedis()
        fake_redis.store["news:impact:https://example.com/seen-before"] = json.dumps(
            [{"symbol": "TCS", "direction": "up", "assetClass": "NSE", "reason": "cached"}]
        ).encode()

        llm = FakeLlm(_response(json.dumps([{"affected": []}])))
        with patch("app.market.news.redis.from_url", return_value=fake_redis):
            impacts, ok = await news._analyze_impacts_cached(llm, articles)

        assert ok is True
        assert impacts[0] == [{"symbol": "TCS", "direction": "up", "assetClass": "NSE", "reason": "cached"}]
        assert impacts[1] == []
        # Only the uncached article reached the model.
        assert len(llm.calls) == 1
        assert "Brand new" in llm.calls[0]["messages"][1]["content"]
        assert "Seen before" not in llm.calls[0]["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_a_fresh_result_is_written_back_for_the_next_run(self):
        articles = [_article("First run")]
        fake_redis = _FakeRedis()
        llm = FakeLlm(_response(json.dumps([
            {"affected": [{"symbol": "INFY", "direction": "down", "assetClass": "NSE", "reason": "r"}]},
        ])))
        with patch("app.market.news.redis.from_url", return_value=fake_redis):
            await news._analyze_impacts_cached(llm, articles)

        stored = json.loads(fake_redis.store["news:impact:https://example.com/first-run"])
        assert stored == [{"symbol": "INFY", "direction": "down", "assetClass": "NSE", "reason": "r"}]

    @pytest.mark.asyncio
    async def test_a_failed_analysis_is_never_cached(self):
        articles = [_article("Doomed")]
        fake_redis = _FakeRedis()
        broken = FakeLlm(_response("not json"))
        with patch("app.market.news.redis.from_url", return_value=fake_redis):
            impacts, ok = await news._analyze_impacts_cached(broken, articles)

        assert ok is False
        assert impacts == [None]
        # Nothing written -- a transient failure must not be frozen in for the TTL.
        assert fake_redis.store == {}

    @pytest.mark.asyncio
    async def test_a_dead_cache_degrades_to_analyzing_everything(self):
        articles = [_article("Anything")]
        llm = FakeLlm(_response(json.dumps([{"affected": []}])))

        def _boom(*a, **k):
            raise RuntimeError("redis down")

        with patch("app.market.news.redis.from_url", new=_boom):
            impacts, ok = await news._analyze_impacts_cached(llm, articles)

        assert ok is True
        assert impacts == [[]]
        assert len(llm.calls) == 1
