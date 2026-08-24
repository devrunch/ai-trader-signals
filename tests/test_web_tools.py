"""fetch_url / web_search (app/signals/agent/tools/web.py).

fetch_url is handed URLs the model chose (from user text or a search
result), unlike every other tool's arguments -- these tests exist mainly to
prove the SSRF guard actually blocks the addresses that matter on an EC2
box (169.254.169.254, the AWS instance-metadata endpoint, and the usual
private ranges), on the first hop and on a redirect.

httpx.MockTransport stands in for the network everywhere below -- nothing
here makes a real request, including no real DNS lookup: every URL used is
an IP literal, which `_safe_url` classifies without a getaddrinfo call.
"""
from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.signals.agent.tools import web
from app.signals.agent.tools.base import ToolContext


def _ctx(tavily_api_key: str = "") -> ToolContext:
    settings = get_settings()
    settings = settings.model_copy(update={"tavily_api_key": tavily_api_key})
    return ToolContext(symbol="RELIANCE", exchange="NSE", base_df=None, settings=settings)


def _mock_client(monkeypatch, handler):
    monkeypatch.setattr(web, "_client", lambda **kw: httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _mock_tavily(monkeypatch, handler):
    """web_search builds its own client inline rather than through `_client` --
    patching `httpx.AsyncClient` at the module httpx (shared with web.py's own
    `import httpx`) covers it the same way. Captures the real class first --
    the replacement must not call the now-patched `httpx.AsyncClient`, or every
    call recurses into itself."""
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async_client(transport=httpx.MockTransport(handler)))


class TestSafeUrl:
    @pytest.mark.asyncio
    async def test_public_ip_is_safe(self):
        ok, _ = await web._safe_url("http://8.8.8.8/page")
        assert ok is True

    @pytest.mark.asyncio
    async def test_aws_instance_metadata_endpoint_is_blocked(self):
        ok, reason = await web._safe_url("http://169.254.169.254/latest/meta-data/")
        assert ok is False
        assert "private or internal" in reason

    @pytest.mark.asyncio
    async def test_loopback_is_blocked(self):
        ok, _ = await web._safe_url("http://127.0.0.1/admin")
        assert ok is False

    @pytest.mark.asyncio
    async def test_rfc1918_private_range_is_blocked(self):
        ok, _ = await web._safe_url("http://10.0.0.5/internal")
        assert ok is False

    @pytest.mark.asyncio
    async def test_non_http_scheme_is_rejected(self):
        ok, reason = await web._safe_url("file:///etc/passwd")
        assert ok is False
        assert "http" in reason


class TestFetchUrl:
    @pytest.mark.asyncio
    async def test_extracts_readable_text_from_html(self, monkeypatch):
        def handler(request):
            html = "<html><head><style>.x{}</style></head><body><script>evil()</script><p>Hello world</p></body></html>"
            return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

        _mock_client(monkeypatch, handler)
        result = await web.fetch_url(_ctx(), {"url": "http://8.8.8.8/page"})
        assert "Hello world" in result["text"]
        assert "evil()" not in result["text"]
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_rejects_a_request_to_the_metadata_endpoint_without_any_network_call(self, monkeypatch):
        def handler(request):
            raise AssertionError("must never actually request a blocked URL")

        _mock_client(monkeypatch, handler)
        result = await web.fetch_url(_ctx(), {"url": "http://169.254.169.254/latest/meta-data/"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_a_redirect_to_a_private_address_is_blocked_not_followed(self, monkeypatch):
        def handler(request):
            if str(request.url) == "http://8.8.8.8/start":
                return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
            raise AssertionError("must never follow the redirect to a blocked URL")

        _mock_client(monkeypatch, handler)
        result = await web.fetch_url(_ctx(), {"url": "http://8.8.8.8/start"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_a_redirect_to_a_public_address_is_followed(self, monkeypatch):
        def handler(request):
            if str(request.url) == "http://8.8.8.8/start":
                return httpx.Response(302, headers={"location": "http://1.1.1.1/final"})
            return httpx.Response(200, headers={"content-type": "text/html"}, text="<p>Landed</p>")

        _mock_client(monkeypatch, handler)
        result = await web.fetch_url(_ctx(), {"url": "http://8.8.8.8/start"})
        assert "Landed" in result["text"]
        assert result["url"] == "http://1.1.1.1/final"

    @pytest.mark.asyncio
    async def test_unreadable_content_type_is_rejected(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4")

        _mock_client(monkeypatch, handler)
        result = await web.fetch_url(_ctx(), {"url": "http://8.8.8.8/file.pdf"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_non_2xx_status_is_reported_as_an_error(self, monkeypatch):
        def handler(request):
            return httpx.Response(404, headers={"content-type": "text/html"}, text="not found")

        _mock_client(monkeypatch, handler)
        result = await web.fetch_url(_ctx(), {"url": "http://8.8.8.8/missing"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_url_raises_for_the_tool_runner_to_report(self):
        with pytest.raises(ValueError):
            await web.fetch_url(_ctx(), {})


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_missing_api_key_returns_a_clean_error_not_a_crash(self):
        result = await web.web_search(_ctx(tavily_api_key=""), {"query": "nifty 50 outlook"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_successful_search_returns_answer_and_results(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={
                "answer": "Nifty 50 closed higher today.",
                "results": [{"title": "Market wrap", "url": "http://1.1.1.1/wrap", "content": "Nifty gained 1%"}],
            })

        _mock_tavily(monkeypatch, handler)
        result = await web.web_search(_ctx(tavily_api_key="test-key"), {"query": "nifty 50 outlook"})
        assert result["answer"] == "Nifty 50 closed higher today."
        assert result["count"] == 1
        assert result["results"][0]["url"] == "http://1.1.1.1/wrap"

    @pytest.mark.asyncio
    async def test_api_failure_returns_a_clean_error_not_a_crash(self, monkeypatch):
        def handler(request):
            return httpx.Response(500, json={"error": "upstream down"})

        _mock_tavily(monkeypatch, handler)
        result = await web.web_search(_ctx(tavily_api_key="test-key"), {"query": "nifty 50 outlook"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_query_raises_for_the_tool_runner_to_report(self):
        with pytest.raises(ValueError):
            await web.web_search(_ctx(tavily_api_key="test-key"), {})


class TestRegistryWiring:
    def test_both_tools_are_registered(self):
        from app.signals.agent import tools as tool_registry
        assert tool_registry.get("fetch_url") is web.fetch_url
        assert tool_registry.get("web_search") is web.web_search
        assert tool_registry.GROUP_OF["fetch_url"] == "web"
        assert tool_registry.GROUP_OF["web_search"] == "web"
