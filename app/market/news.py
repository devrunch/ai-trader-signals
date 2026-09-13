"""
News feed with per-headline sentiment and real
per-headline stock-impact analysis (LLM, chunked into small concurrent
batches -- see _analyze_impacts' own docstring) -- see _analyze_chunk's own
docstring for why this replaced a naive keyword match against a fixed
ticker shortlist.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import httpx
import redis.asyncio as redis

from app.config import get_settings
from app.llm.client import LlmClient, get_llm
from app.market import macro_events
from app.signals.prompts import extract_json_text

logger = logging.getLogger(__name__)

# Failures we expect from a third-party HTTP API: the network, a non-2xx, a
# body that is not the JSON shape documented. Anything else is our bug.
_API_ERRORS = (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError)

NEWSAPI_URL  = "https://newsapi.org/v2/everything"
# HF retired api-inference.huggingface.co — inference now routes through router.huggingface.co


# The India-only version of this query meant a real, forex-moving headline
# (e.g. "Dollar falls ahead of US inflation data") never matched at all --
# this app charts FOREX/metals (XAUUSD etc.) too. The first broadened
# version kept 7 India-specific terms (NSE, BSE, "Indian stock", SEBI,
# "Dalal Street", RBI, "equity market") against only 6 global ones,
# skewing the default feed India-heavy in practice -- Indian financial
# outlets use those particular words constantly, flooding the match. Down
# to Nifty/Sensex only for India (still real coverage of this app's own
# NSE/BSE symbols) and weighted toward global macro/forex/equity terms,
# which is what the product actually needs front and center.
DEFAULT_MARKET_QUERY = (
    "\"Federal Reserve\" OR \"interest rate\" OR inflation OR CPI OR "
    "dollar OR gold OR crude oil OR Bitcoin OR forex OR \"stock market\" OR "
    "Nasdaq OR \"S&P 500\" OR \"Wall Street\" OR Nifty OR Sensex"
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

# How long one article's analyzed impact is remembered, keyed by its URL.
# Comfortably longer than a story stays in a "latest headlines" pull, so a
# multi-hour story is analyzed once across all the hourly pipeline ticks it
# survives, not once per tick. Bounded rather than permanent because the
# analysis prompt itself changes over time, and a stale prompt's answers
# shouldn't outlive it by much.
IMPACT_CACHE_TTL_SECONDS = 48 * 60 * 60


# Restricting to real finance/markets outlets. Without this, an unrestricted
# `q` match pulled roughly a quarter junk -- a Bigg Boss episode recap, a
# phone launch, a road accident -- all matching incidentally on a common word
# like "dollar", "gold" or "prices". That junk is invisible on the Home page
# (which shows only headlines with a real impact) but it still burns an LLM
# impact-analysis slot each, and clutters the News tab's own "All" list.
#
# Deliberately no India-only outlets (Business Standard, BusinessLine,
# Livemint, Moneycontrol, Economic Times). Including them buried the feed
# in Indian coverage -- confirmed live, 21 of 25 headlines from three
# Indian outlets in one pull -- because NewsAPI's index covers them far
# more densely than Reuters/Bloomberg, both of which restrict their
# NewsAPI availability. India still reaches the feed through global desks
# covering it, plus Nifty/Sensex in DEFAULT_MARKET_QUERY.
#
# Deliberately finance-dedicated outlets rather than general-news ones.
# General desks (a BBC, a Guardian) carry huge non-finance sections that
# the query's own common words -- "dollar", "gold", "prices" -- match
# constantly, which is exactly where the junk came from before. The
# general-interest names still in this list (Business Insider, Forbes,
# Fortune) are the remaining source of it.
_FINANCE_DOMAIN_LIST = [
    # Markets / equities desks
    "cnbc.com", "marketwatch.com", "investing.com", "finance.yahoo.com",
    "barrons.com", "seekingalpha.com", "benzinga.com", "thestreet.com",
    "fool.com", "nasdaq.com", "investors.com", "morningstar.com",
    "barchart.com", "stocktwits.com", "insidermonkey.com", "zacks.com",
    "247wallst.com", "schaeffersresearch.com", "tipranks.com",
    "marketbeat.com", "streetinsider.com", "finbold.com",
    # Wires / global business
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "economist.com",
    "businessinsider.com", "forbes.com", "fortune.com", "axios.com", "qz.com",
    # Crypto
    "coindesk.com", "cointelegraph.com", "decrypt.co", "theblock.co",
    "cryptoslate.com", "bitcoinist.com", "cryptobriefing.com",
    # FX, commodities, energy, metals
    "oilprice.com", "kitco.com", "fxstreet.com", "fxempire.com",
    "dailyfx.com", "mining.com", "rigzone.com", "agweb.com",
]

FINANCE_DOMAINS = ",".join(_FINANCE_DOMAIN_LIST)


def _dedupe_articles(articles: list[dict]) -> list[dict]:
    """Same story, syndicated to two outlets (or served twice by NewsAPI
    itself -- confirmed live), is one headline to a reader and one wasted
    impact-analysis slot to us. Keyed on url first, then on the headline
    itself for the syndicated case where the urls differ."""
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    out: list[dict] = []
    for a in articles:
        url = (a.get("url") or "").strip()
        title = (a.get("title") or "").strip().lower()
        if (url and url in seen_urls) or (title and title in seen_titles):
            continue
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        out.append(a)
    return out


async def _fetch_newsapi(query: str, page_size: int = 20) -> list[dict]:
    settings = get_settings()
    if not settings.news_api_key:
        return []
    params: dict[str, str | int] = {
        "q": query,
        "domains": FINANCE_DOMAINS,
        "language": "en",
        "sortBy": "publishedAt",
        # Over-fetch, since deduping below only ever removes articles and a
        # short page is worse than a slightly wider net -- NewsAPI charges
        # one request either way, and pageSize is not itself metered.
        "pageSize": min(page_size * 2, 100),
        "apiKey": settings.news_api_key,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(NEWSAPI_URL, params=params)
        r.raise_for_status()
        return _dedupe_articles(r.json().get("articles", []))[:page_size]


# Yahoo attaches news to a symbol, so this ticker list IS the topic
# selection -- one per asset class this app actually charts, so the feed
# isn't skewed to any single market the way the NewsAPI query alone was.
YF_NEWS_TICKERS = [
    "^GSPC",      # US broad equities
    "^IXIC",      # NASDAQ
    "GC=F",       # gold
    "BZ=F",       # Brent crude
    "BTC-USD",    # crypto
    "DX-Y.NYB",   # US dollar index
    "^NSEI",      # Nifty 50
    "USDINR=X",   # rupee
]


def _yf_to_article(item: dict) -> dict:
    """Maps one yfinance news item onto NewsAPI's own article shape, so the
    two sources merge into one list nothing downstream has to special-case.
    `published_at` is already an ISO string on the current payload shape;
    the older flat shape's epoch seconds are converted here."""
    published = item.get("published_at")
    if isinstance(published, (int, float)):
        published = datetime.fromtimestamp(published, tz=UTC).isoformat()
    return {
        "title": item.get("title") or "",
        "description": item.get("summary") or "",
        "url": item.get("url") or "",
        "publishedAt": published or "",
        "source": {"name": item.get("publisher") or "Yahoo Finance"},
    }


async def _fetch_yfinance(page_size: int = 20) -> list[dict]:
    """Yahoo Finance headlines, NewsAPI-shaped. Free, no key, and measured
    at roughly an hour old at the freshest -- against NewsAPI's free tier,
    which withholds every article for a full 24h, this is the fresher of
    the two sources by a wide margin, which is the whole reason it's here."""
    try:
        items = await macro_events.yfinance_headlines(YF_NEWS_TICKERS)
    except Exception:
        # yfinance scrapes an undocumented endpoint; a shape change there
        # must degrade to "no yfinance articles this run", never take the
        # whole feed down with it -- NewsAPI's half still stands alone.
        logger.exception("yfinance news fetch failed")
        return []
    return [a for a in (_yf_to_article(i) for i in items) if a["title"]][:page_size]


async def _fetch_newsapi_safe(query: str, page_size: int) -> list[dict] | None:
    """`_fetch_newsapi`, but returns None instead of raising, so one dead
    source can't take the other's articles down with it. None means
    "couldn't fetch"; an empty list means "fetched, nothing matched"."""
    try:
        return await _fetch_newsapi(query, page_size)
    except _API_ERRORS as e:
        logger.warning("NewsAPI fetch failed: %s", e)
        return None
    except Exception:
        logger.exception("Unexpected error fetching news for query %r", query)
        return None


def _merge_sources(sources: list[list[dict]], page_size: int) -> list[dict]:
    """Newest first across every source, deduped, trimmed to page_size.

    Sorted on the real publish timestamp rather than interleaving by
    source: the three sources run at very different lags (yfinance ~1h,
    newsdata 12h, NewsAPI 24h), so a plain interleave would put day-old
    articles above hour-old ones on the page. An unparseable or missing
    timestamp sorts last rather than being dropped -- a real headline is
    worth more than its position.
    """
    def sort_key(a: dict) -> tuple[int, float]:
        raw = a.get("publishedAt") or ""
        try:
            return (0, -datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
        except (ValueError, AttributeError, TypeError):
            return (1, 0.0)

    merged = _dedupe_articles([a for source in sources for a in source])
    merged.sort(key=sort_key)
    return merged[:page_size]


# newsdata.io. Worth a third source specifically because Reuters, Bloomberg
# and Barron's ARE in its index and NewsAPI restricts all three -- confirmed
# by probing its own database. Free tier: 200 credits/day (this pipeline
# uses 24), 12h delay, and two hard caps that shape the config below --
# max 5 domains, and a 100-character `q`.
NEWSDATA_URL = "https://newsdata.io/api/1/latest"
NEWSDATA_DOMAINS = "reuters.com,bloomberg.com,barrons.com,marketwatch.com,cnbc.com"
# Deliberately short: the free tier rejects a `q` over 100 characters, so
# this cannot be DEFAULT_MARKET_QUERY. Domain-filtering alone is NOT enough
# here -- confirmed live that Reuters' own feed, unqueried, is mostly
# sport and general news.
NEWSDATA_QUERY = 'stocks OR inflation OR "Federal Reserve" OR gold OR crude OR bitcoin OR forex'


def _newsdata_to_article(item: dict) -> dict:
    """Maps one newsdata.io result onto NewsAPI's own article shape.
    Its `pubDate` is "YYYY-MM-DD HH:MM:SS" in the zone named by
    `pubDateTZ` (UTC in practice), not an ISO string -- normalised here so
    _merge_sources can sort every source on one comparable timestamp."""
    published = (item.get("pubDate") or "").strip()
    if published:
        try:
            dt = datetime.strptime(published, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            published = dt.isoformat()
        except ValueError:
            # Keep whatever it sent rather than dropping a real article over
            # a format change; _merge_sources sorts unparseable dates last.
            pass
    return {
        "title": item.get("title") or "",
        "description": item.get("description") or "",
        "url": item.get("link") or "",
        "publishedAt": published,
        "source": {"name": item.get("source_name") or "newsdata.io"},
    }


async def _fetch_newsdata(page_size: int = 20) -> list[dict]:
    """newsdata.io headlines, NewsAPI-shaped. Returns [] when no key is
    configured, so the source is simply absent rather than an error."""
    settings = get_settings()
    if not settings.newsdata_api_key:
        return []
    params = {
        "apikey": settings.newsdata_api_key,
        "language": "en",
        "domainurl": NEWSDATA_DOMAINS,
        "q": NEWSDATA_QUERY,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(NEWSDATA_URL, params=params)
        r.raise_for_status()
        payload = r.json()
    # newsdata reports its own failures in-band with HTTP 200 sometimes, and
    # `results` is an error OBJECT rather than a list in that case.
    results = payload.get("results")
    if not isinstance(results, list):
        logger.warning("newsdata.io returned no usable results: %s", str(results)[:200])
        return []
    return [a for a in (_newsdata_to_article(i) for i in results) if a["title"]][:page_size]


# Alpha Vantage. The freshest keyed source measured (~1h old, comparable to
# yfinance) and it ships its own sentiment and per-ticker relevance. Its
# free tier is the tightest of the lot though -- 25 requests/DAY, 5/min --
# so an hourly pipeline (24/day) would sit one request from the ceiling
# with nothing left for a retry or a manual run. Hence the cache below.
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
# One topic, not three. A/B'd against the live API: `financial_markets`
# alone returns ~1h-old items from MarketBeat/Yahoo/CNBC/Benzinga, while
# adding economy_macro,economy_monetary returned ~8h-old items that were
# entirely Kalkine Media content-farm pieces. More topics is strictly
# worse here on both freshness and source quality.
ALPHAVANTAGE_TOPICS = "financial_markets"
ALPHAVANTAGE_CACHE_KEY = "news:alphavantage:latest"
# Halves the hourly pipeline's usage to ~12/day, and -- the real point --
# means a manual or debug run reuses the last response instead of eating
# into a 25/day budget that has no slack.
ALPHAVANTAGE_CACHE_TTL_SECONDS = 2 * 60 * 60


def _alphavantage_to_article(item: dict) -> dict:
    """Maps one Alpha Vantage feed item onto NewsAPI's article shape.

    Its own `overall_sentiment_label`/`ticker_sentiment` are deliberately
    NOT carried through: one analysis scores every article in this feed, and
    mixing a second scorer's labels in for one source's articles only
    would make the sentiment column mean two different things depending
    on where the headline came from.
    """
    published = (item.get("time_published") or "").strip()
    if published:
        try:
            published = datetime.strptime(published, "%Y%m%dT%H%M%S").replace(tzinfo=UTC).isoformat()
        except ValueError:
            pass  # keep as-is; _merge_sources sorts unparseable dates last
    return {
        "title": item.get("title") or "",
        "description": item.get("summary") or "",
        "url": item.get("url") or "",
        "publishedAt": published,
        "source": {"name": item.get("source") or "Alpha Vantage"},
    }


async def _fetch_alphavantage(page_size: int = 20) -> list[dict]:
    """Alpha Vantage market news, NewsAPI-shaped, behind a 2h Redis cache.

    Returns [] when no key is configured. A cache failure degrades to a
    real fetch rather than to no news -- but note that makes the 25/day
    budget the thing at risk, so the cache is load-bearing here in a way
    the others are not.
    """
    settings = get_settings()
    if not settings.alphavantage_api_key:
        return []

    try:
        r = redis.from_url(settings.redis_url)
    except Exception as e:
        logger.warning("Alpha Vantage cache unavailable: %s", e)
        r = None

    try:
        if r is not None:
            try:
                cached = await r.get(ALPHAVANTAGE_CACHE_KEY)
                if cached:
                    return json.loads(cached)[:page_size]
            except Exception as e:
                logger.warning("Alpha Vantage cache read failed: %s", e)

        params = {
            "function": "NEWS_SENTIMENT",
            "topics": ALPHAVANTAGE_TOPICS,
            "sort": "LATEST",
            "apikey": settings.alphavantage_api_key,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(ALPHAVANTAGE_URL, params=params)
            resp.raise_for_status()
            payload = resp.json()

        feed = payload.get("feed")
        if not isinstance(feed, list):
            # Alpha Vantage reports throttling and key problems in-band with
            # HTTP 200, under "Note" / "Information" / "Error Message".
            detail = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
            logger.warning("Alpha Vantage returned no feed: %s", str(detail or payload)[:200])
            return []

        articles = [a for a in (_alphavantage_to_article(i) for i in feed) if a["title"]]
        if r is not None and articles:
            try:
                await r.set(ALPHAVANTAGE_CACHE_KEY, json.dumps(articles), ex=ALPHAVANTAGE_CACHE_TTL_SECONDS)
            except Exception as e:
                logger.warning("Alpha Vantage cache write failed: %s", e)
        return articles[:page_size]
    finally:
        if r is not None:
            try:
                await r.aclose()
            except Exception:
                logger.debug("Alpha Vantage cache close failed", exc_info=True)


async def _fetch_alphavantage_safe(page_size: int) -> list[dict]:
    """Never lets an Alpha Vantage failure reach the other sources."""
    try:
        return await _fetch_alphavantage(page_size)
    except Exception:
        logger.exception("Alpha Vantage fetch failed")
        return []


async def _fetch_newsdata_safe(page_size: int) -> list[dict]:
    """Never lets a newsdata failure reach the other two sources."""
    try:
        return await _fetch_newsdata(page_size)
    except Exception:
        logger.exception("newsdata.io fetch failed")
        return []


SENTIMENT_LABELS = ("POSITIVE", "NEGATIVE", "NEUTRAL")


def _parse_analysis_response(raw_text: str, n: int) -> list[dict] | None:
    """Parses the LLM's JSON array into one `{sentiment, impacts}` entry per
    article, in order. Any malformed entry (missing symbol, a direction other than
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
            "News analysis returned %s entries for %d articles — discarding",
            len(parsed) if isinstance(parsed, list) else type(parsed).__name__, n,
        )
        return None

    results: list[dict] = []
    for entry in parsed:
        affected = entry.get("affected") if isinstance(entry, dict) else None
        # An unrecognised label means "we don't know", never a defaulted
        # NEUTRAL -- that is the distinction sentimentAvailable exists for.
        raw_sentiment = entry.get("sentiment") if isinstance(entry, dict) else None
        sentiment = str(raw_sentiment).strip().upper() if raw_sentiment else None
        if sentiment not in SENTIMENT_LABELS:
            sentiment = None
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
        results.append({"sentiment": sentiment, "impacts": clean})
    return results


# Articles per LLM call. Splitting into small chunks (run concurrently, see
# _analyze_impacts) rather than one call for the whole page keeps each
# call's real output comfortably under IMPACT_MAX_TOKENS regardless of how
# many real, multi-symbol impacts land in a given page -- a single 25-
# article call was getting cut off mid-array once the broadened news query
# (see DEFAULT_MARKET_QUERY) made real impacts common, and a fixed token
# ceiling can't scale with "how much real content happened to be in this
# batch." Running chunks concurrently also means splitting the page doesn't
# cost extra wall-clock time.
IMPACT_CHUNK_SIZE = 8


async def _analyze_chunk(llm: LlmClient, chunk: list[dict]) -> list[dict] | None:
    """One LLM call (with one retry) for a single chunk of articles --
    see _analyze_impacts for how chunks are split and combined, and for
    the "why not keyword-matching" rationale this used to carry.

    Returns None (not a list of empty lists) when this CHUNK could not be
    analyzed -- no LLM configured, the call failed, the response was the
    wrong shape. Callers must not read that as "no headline in this chunk
    had a real impact"; see get_market_news_result's own docs for how it's
    surfaced.
    """
    numbered = "\n".join(
        f"{i}. {a.get('title') or ''} -- {a.get('description') or ''}"
        for i, a in enumerate(chunk)
    )
    n = len(chunk)
    prompt = (
        "For each numbered headline below, give two things.\n\n"
        "\"sentiment\": the likely effect on the price of the main asset the "
        "headline is about -- POSITIVE (up), NEGATIVE (down) or NEUTRAL (no clear "
        "effect). Judge the price effect, not the tone of the words: an "
        "inflation print that came in hotter than expected reads upbeat but is "
        "NEGATIVE for equities.\n\n"
        "\"affected\": name the real, tradeable stocks or "
        "instruments it plausibly affects (their real ticker or a clear, "
        "specific name -- not limited to any fixed list), the direction each "
        "would plausibly move (\"up\" or \"down\"), which market it trades on "
        "(assetClass: one of NSE, BSE, NASDAQ, NYSE, FOREX, MCX, CRYPTO -- use "
        "OTHER only if truly none fit), and one short reason grounded in the "
        "headline itself. Most headlines affect nothing tradeable -- return an "
        "empty \"affected\" list for those rather than forcing a connection -- "
        "even a headline with nothing tradeable still gets its OWN object "
        "with an empty \"affected\" array, never omitted from the response. "
        "A headline with several affected instruments still gets ONE object, "
        "with all of them in that one \"affected\" array -- never split one "
        f"headline across two objects.\n\n{numbered}\n\n"
        f"Respond with ONLY a JSON array of EXACTLY {n} object"
        f"{'s' if n != 1 else ''} -- one per "
        "headline above, in the same order, never fewer or more -- no other "
        "text:\n"
        "[{\"sentiment\": \"NEGATIVE\", \"affected\": [{\"symbol\": \"RELIANCE\", \"direction\": \"down\", "
        "\"assetClass\": \"NSE\", \"reason\": \"...\"}]}, "
        "{\"sentiment\": \"POSITIVE\", \"affected\": [{\"symbol\": \"BZ=F\", \"direction\": \"up\", "
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
    # -- worth one more real attempt before reporting this chunk unavailable,
    # but this must stay bounded, not become a silent retry storm against a
    # model that's reliably getting it wrong. The retry runs at a nonzero
    # temperature deliberately: at temperature=0 a retry would just replay
    # the exact same (wrong) output for the exact same prompt.
    for attempt, temperature in enumerate((0, 0.3)):
        try:
            resp = await asyncio.to_thread(
                llm.chat, temperature=temperature, max_tokens=IMPACT_MAX_TOKENS, messages=messages,
            )
        except Exception as e:
            logger.warning("News impact analysis failed: %s", e)
            return None
        parsed = _parse_analysis_response(resp.choices[0].message.content or "", n)
        if parsed is not None:
            return parsed
        if attempt == 0:
            logger.info("News impact analysis retrying once after a malformed response")
    return None


async def _analyze_articles(llm: LlmClient, articles: list[dict]) -> list[dict] | None:
    """Real per-headline stock/instrument impact -- not the keyword match
    this replaced (`_extract_symbols`, a plain substring check against a
    hardcoded 17-ticker shortlist: blind to anything outside that list, and
    "mentions the ticker" is not the same question as "this headline
    plausibly moves this instrument"). The model names real symbols freely,
    not constrained to any fixed universe -- a hardcoded shortlist here
    would just be the same arbitrariness one level up.

    Splits `articles` into roughly-EQUAL-sized chunks around
    IMPACT_CHUNK_SIZE and analyzes them concurrently (see _analyze_chunk)
    rather than one call for the whole page -- keeps latency roughly flat
    as the page grows instead of one long serial call, and keeps each
    call's real output safely under the token cap regardless of how many
    real impacts land in a given page. Deliberately balanced rather than
    greedy fixed-size slicing (25 articles at size 8 greedily gives
    8,8,8,1): confirmed live that a 1-article leftover chunk is a HARDER
    case for the model to shape correctly than a normal-sized one -- it
    would sometimes answer a single irrelevant headline with a bare `[]`
    instead of the required `[{"affected": []}]`, tripping the exact-length
    check for no real reason.

    Returns None (not a list of empty lists) when ANY chunk could not be
    analyzed -- no LLM configured, a call failed, a response was the wrong
    shape even after its retry. Callers must not read that as "no headline
    had a real impact today"; see get_market_news_result's own docs for how
    it's surfaced.
    """
    if not articles:
        return []
    num_chunks = -(-len(articles) // IMPACT_CHUNK_SIZE)  # ceil division
    base, extra = divmod(len(articles), num_chunks)
    chunks, i = [], 0
    for c in range(num_chunks):
        size = base + (1 if c < extra else 0)
        chunks.append(articles[i:i + size])
        i += size
    results = await asyncio.gather(*(_analyze_chunk(llm, c) for c in chunks))
    if any(r is None for r in results):
        return None
    combined: list[dict] = []
    for r in results:
        combined.extend(r)
    return combined


async def _analyze_articles_cached(
    llm: LlmClient, articles: list[dict],
) -> tuple[list[dict | None], bool]:
    """`_analyze_articles`, but each article's result is remembered by URL
    for IMPACT_CACHE_TTL_SECONDS.

    The pipeline re-fetches the same still-running story every hour, and
    the same headline text cannot honestly produce a different answer an
    hour later -- so re-analyzing it was pure repeat spend, the single
    biggest avoidable LLM cost here. A story already seen still SHOWS
    (it's still current news); it just isn't paid for twice.

    Returns `(per-article impacts, analysis_ok)`. An article whose entry is
    None could not be analyzed; `analysis_ok` is False when any article
    needed fresh analysis and did not get it. A failed analysis is never
    cached -- only a real result, so a transient failure doesn't get
    frozen in for two days.
    """
    settings = get_settings()
    analyses: list[dict | None] = [None] * len(articles)
    # news:analysis:, not the old news:impact: -- the cached value now carries
    # sentiment too, so old entries must not be read back as the new shape.
    keys = [f"news:analysis:{a.get('url')}" if a.get("url") else None for a in articles]

    r = None
    try:
        r = redis.from_url(settings.redis_url)
        real_keys = [k for k in keys if k]
        cached_values = await r.mget(real_keys) if real_keys else []
        by_key = dict(zip(real_keys, cached_values, strict=True))
        for i, key in enumerate(keys):
            raw = by_key.get(key) if key else None
            if raw:
                try:
                    analyses[i] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    analyses[i] = None
    except Exception as e:
        # A cache read failing must degrade to "analyze everything fresh",
        # never to a dead feed -- this is a cost saving, not a dependency.
        logger.warning("Impact cache read failed: %s", e)

    pending = [i for i, v in enumerate(analyses) if v is None]
    analysis_ok = True
    if pending:
        fresh = await _analyze_articles(llm, [articles[i] for i in pending])
        if fresh is None:
            analysis_ok = False
        else:
            for i, value in zip(pending, fresh, strict=True):
                analyses[i] = value
            if r is not None:
                try:
                    for i in pending:
                        if keys[i]:
                            await r.set(keys[i], json.dumps(analyses[i]), ex=IMPACT_CACHE_TTL_SECONDS)
                except Exception as e:
                    logger.warning("Impact cache write failed: %s", e)

    if r is not None:
        try:
            await r.aclose()
        except Exception:
            logger.debug("Impact cache close failed", exc_info=True)

    logger.info("News analysis: %d cached, %d freshly analyzed", len(articles) - len(pending), len(pending))
    return analyses, analysis_ok


async def score_headlines(llm: LlmClient, headlines: list[str]) -> list[str] | None:
    """One label per headline: POSITIVE, NEGATIVE or NEUTRAL, in order.

    The likely price effect on the asset the headline is about, not the tone
    of its words. Returns None when the batch could not be scored at all --
    callers must not read that as "the news is neutral".
    """
    if not headlines:
        return []
    n = len(headlines)
    numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    prompt = (
        "For each numbered headline, give the likely effect on the price of the "
        "asset it is about: POSITIVE (up), NEGATIVE (down) or NEUTRAL (no clear "
        "effect). Judge the price effect, not the tone of the words.\n\n"
        f"{numbered}\n\n"
        f"Respond with ONLY a JSON array of EXACTLY {n} labels, in the same "
        'order, no other text: ["POSITIVE", "NEUTRAL", ...]'
    )
    try:
        resp = await asyncio.to_thread(
            llm.chat, temperature=0, max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = json.loads(extract_json_text(resp.choices[0].message.content or ""))
    except Exception as e:
        logger.warning("Headline sentiment scoring failed: %s", e)
        return None
    if not isinstance(parsed, list) or len(parsed) != n:
        logger.warning("Headline sentiment returned %s labels for %d headlines — discarding",
                       len(parsed) if isinstance(parsed, list) else type(parsed).__name__, n)
        return None
    labels = [str(x).strip().upper() for x in parsed]
    if any(label not in SENTIMENT_LABELS for label in labels):
        logger.warning("Headline sentiment returned an unrecognised label — discarding")
        return None
    return labels


async def get_market_news_result(
    symbols: list[str] | None = None, page_size: int = 15, llm: LlmClient | None = None,
) -> dict:
    """
    Fetch market news and analyze each headline: its sentiment (the likely
    price effect on the asset it is about) and the real stocks or instruments
    it moves -- reporting what failed rather than silently degrading into
    something that looks like a normal result.

    News comes from four independent sources merged newest-first
    (yfinance, Alpha Vantage, NewsAPI, newsdata.io); any one failing
    leaves the rest standing. Impact analysis is cached per article URL, so a story that
    survives several hourly pipeline ticks is analyzed once -- see
    _analyze_impacts_cached.

    symbols: optional list to narrow query (e.g. ["RELIANCE", "INFY"])

    Returns `{articles, count, degraded, degraded_reason}`. The analysis can
    fail independently of the news fetch, so each article carries its own
    `sentimentAvailable` and `impacts: null` -- a caller can tell "analyzed,
    found nothing" from "couldn't analyze", which would otherwise look
    identical for sentiment (NEUTRAL either way) and impact (empty list).
    """
    if symbols:
        query = " OR ".join(symbols[:5])
    else:
        query = DEFAULT_MARKET_QUERY

    # Four independent sources, fetched concurrently, each covering the
    # others' gaps: yfinance and Alpha Vantage (freshest, ~1h), NewsAPI
    # (broadest domain list, but its free tier withholds everything 24h),
    # and newsdata.io (12h, and the only one whose index actually carries
    # Reuters/Bloomberg/Barron's). Any one failing leaves the rest
    # standing -- only all four coming back empty is a real "no news".
    newsapi_articles, yf_articles, newsdata_articles, av_articles = await asyncio.gather(
        _fetch_newsapi_safe(query, page_size),
        _fetch_yfinance(page_size),
        _fetch_newsdata_safe(page_size),
        _fetch_alphavantage_safe(page_size),
    )
    if newsapi_articles is None and not yf_articles and not newsdata_articles and not av_articles:
        return {"articles": [], "count": 0, "degraded": True, "degraded_reason": "news_unavailable"}

    articles = _merge_sources(
        [newsapi_articles or [], yf_articles, newsdata_articles, av_articles], page_size,
    )
    if not articles:
        return {"articles": [], "count": 0, "degraded": True, "degraded_reason": "news_unavailable"}

    llm = llm or get_llm()
    analyses, analysis_ok = await _analyze_articles_cached(llm, articles)

    results = []
    # strict=True: the analysis list is built to match `articles`' own length
    # by construction (or _analyze_articles rejected a mismatched batch and
    # returned None). If that ever stops being true, sentiment and impacts
    # would be silently attached to the wrong headline.
    for a, analysis in zip(articles, analyses, strict=True):
        sentiment = (analysis or {}).get("sentiment")
        impacts = (analysis or {}).get("impacts") if analysis else None
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
            # NEUTRAL with sentimentAvailable false means "we could not read
            # it", not "the model read it as neutral".
            "sentiment": sentiment or "NEUTRAL",
            "sentimentAvailable": sentiment is not None,
            # Real per-symbol impact from the same analysis -- null (not [])
            # means this article could not be analyzed at all, the same
            # "unavailable, not empty" contract sentimentAvailable gives.
            "impacts": impacts,
        })

    # One call now produces both sentiment and impacts, so they can no longer
    # fail independently of each other.
    degraded = not analysis_ok
    degraded_reason = None if analysis_ok else "impact_unavailable"

    final = {
        "articles": results,
        "count": len(results),
        "degraded": degraded,
        "degraded_reason": degraded_reason,
    }

    return final


async def publish(result: dict) -> bool:
    """Store the latest news analysis via the NestJS internal endpoint --
    mirrors app/signals/brief.py's own `publish`. Called by the
    run_news_analysis Celery task, not by the live HTTP route: this is what
    moves the real NewsAPI/HF/LLM work out of the request path, the same
    way the twice-daily brief and the drift-check/reddit-sentiment alerts
    already run on a schedule instead of live per page load."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                f"{settings.api_service_url}/api/internal/news",
                headers={"x-internal-key": settings.internal_api_key},
                json=result,
            )
            r.raise_for_status()
        logger.info("News analysis published: %d articles", result.get("count", 0))
        return True
    except httpx.HTTPError as e:
        logger.warning("News analysis publish failed: %s", e)
        return False
