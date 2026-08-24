"""
Open-web tools: read a specific page (fetch_url), or search the web for one
(web_search). Unlike every other tool group, these leave the app's own data
and hit arbitrary third-party hosts -- fetch_url in particular is handed a
URL the model chose (from user text or a search result), so it is treated as
untrusted input, not a caller-supplied constant.

SSRF guard: this service runs on an EC2 box, where a request to
169.254.169.254 (AWS's instance metadata endpoint, itself link-local) or to
any RFC1918 address can reach infrastructure the model was never meant to
touch. _safe_url resolves the hostname and rejects private/loopback/
link-local/reserved ranges -- checked again on every redirect hop, since a
public URL that 302s to an internal address is the standard bypass for a
same-request-only check.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings
from app.signals.agent.tools.base import Handler, ToolContext

logger = logging.getLogger(__name__)

_API_ERRORS = (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError)

_HEADERS = {"User-Agent": "ai-trader-agent/1.0 (+https://github.com/devrunch/ai-trader)"}
_READABLE_TYPES = ("html", "text/plain", "json")

MAX_REDIRECTS = 5
MAX_BYTES = 1_000_000
MAX_TEXT_CHARS = 8_000
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_SEARCH_RESULTS = 10


def _is_unsafe(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


async def _safe_url(url: str) -> tuple[bool, str]:
    """Whether `url` is safe to request right now -- (ok, reason-if-not).

    A literal IP in the URL (or a redirect target) needs no DNS lookup, which
    also keeps this network-free and deterministic in tests. A real hostname
    is resolved and EVERY returned address must be safe -- a resolver that
    returns one public and one private address is a known SSRF technique,
    and accepting the first "looks fine" answer would defeat the guard.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, "only http/https URLs are supported"
    host = parsed.hostname
    if not host:
        return False, "URL has no hostname"

    try:
        ips = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            loop = asyncio.get_running_loop()
            infos = await loop.getaddrinfo(host, None)
        except OSError:
            return False, "could not resolve hostname"
        ips = [ipaddress.ip_address(info[4][0]) for info in infos]

    if not ips or any(_is_unsafe(ip) for ip in ips):
        return False, "URL resolves to a private or internal address"
    return True, ""


def _extract_text(html: str) -> str:
    """Readable text only -- no script/style/nav noise, no markup."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text(separator="\n").splitlines())
    return "\n".join(line for line in lines if line)


def _client(**kwargs) -> httpx.AsyncClient:
    """The real HTTP client, as a factory so tests can swap in a MockTransport
    without changing fetch_url's own logic."""
    return httpx.AsyncClient(timeout=10, follow_redirects=False, **kwargs)


async def _fetch_hop(client: httpx.AsyncClient, url: str) -> dict:
    """One hop: GET `url` and report a redirect, an error, or the body.

    Never follows a redirect itself -- fetch_url re-runs the SSRF guard on
    whatever hop this returns before it is requested.
    """
    try:
        async with client.stream("GET", url, headers=_HEADERS) as resp:
            if resp.is_redirect:
                location = resp.headers.get("location")
                if not location:
                    return {"error": "redirect with no Location header"}
                return {"redirect": str(resp.url.join(location))}
            if resp.status_code >= 400:
                return {"error": f"server returned {resp.status_code}"}
            content_type = resp.headers.get("content-type", "")
            if not any(t in content_type for t in _READABLE_TYPES):
                return {"error": f"cannot read content type '{content_type or 'unknown'}'"}
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body += chunk
                if len(body) >= MAX_BYTES:
                    break
            return {"content_type": content_type, "body": bytes(body), "final_url": str(resp.url)}
    except httpx.HTTPError as e:
        return {"error": f"request failed: {e}"}


async def fetch_url(ctx: ToolContext, args: dict) -> dict:
    url = str(args.get("url") or "").strip()
    if not url:
        raise ValueError("url is required")

    async with _client() as client:
        hops = 0
        while True:
            ok, reason = await _safe_url(url)
            if not ok:
                return {"error": f"cannot fetch this URL: {reason}"}

            result = await _fetch_hop(client, url)
            if "error" in result:
                return result
            if "redirect" not in result:
                break

            hops += 1
            if hops > MAX_REDIRECTS:
                return {"error": "too many redirects"}
            url = result["redirect"]

    text = result["body"].decode("utf-8", errors="replace")
    if "html" in result["content_type"]:
        text = _extract_text(text)
    truncated = len(text) > MAX_TEXT_CHARS
    return {"url": result["final_url"], "text": text[:MAX_TEXT_CHARS], "truncated": truncated}


async def web_search(ctx: ToolContext, args: dict) -> dict:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    max_results = min(int(args.get("max_results") or 5), MAX_SEARCH_RESULTS)

    settings = ctx.settings or get_settings()
    api_key = getattr(settings, "tavily_api_key", "")
    if not api_key:
        return {"error": "Web search is not configured (no Tavily API key set)."}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(TAVILY_SEARCH_URL, json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
            })
            resp.raise_for_status()
            data = resp.json()
    except _API_ERRORS as e:
        logger.warning("Tavily search failed: %s", e)
        return {"error": f"web search failed: {e}"}

    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]
    return {"answer": data.get("answer"), "results": results, "count": len(results)}


TOOLS: dict[str, Handler] = {
    "fetch_url": fetch_url,
    "web_search": web_search,
}
