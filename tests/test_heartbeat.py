from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.worker import heartbeat

CHECKS = {"checks": [
    {"slug": "news-analysis", "ping_url": "https://hc-ping.com/uuid-news"},
    {"slug": "market-overview-morning", "ping_url": "https://hc-ping.com/uuid-am"},
    {"slug": "market-overview-evening", "ping_url": "https://hc-ping.com/uuid-pm"},
]}


class FakeHttp:
    def __init__(self, fail_ping=False, fail_api=False):
        self.gets, self.posts = [], []
        self.fail_ping, self.fail_api = fail_ping, fail_api

    def get(self, url, headers=None, timeout=None):
        self.gets.append((url, headers))
        if self.fail_api:
            raise httpx.ConnectError("down")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: CHECKS)

    def post(self, url, content=None, timeout=None):
        self.posts.append((url, content))
        if self.fail_ping:
            raise httpx.ConnectError("down")
        return SimpleNamespace(status_code=200)


@pytest.fixture
def http(monkeypatch):
    fake = FakeHttp()
    monkeypatch.setattr(heartbeat.httpx, "get", fake.get)
    monkeypatch.setattr(heartbeat.httpx, "post", fake.post)
    monkeypatch.setattr(heartbeat, "get_settings", lambda: SimpleNamespace(healthchecks_api_key="hcw_test"))
    monkeypatch.setattr(heartbeat, "_urls", None)
    return fake


def test_a_successful_run_pings_its_check(http):
    @heartbeat.monitored("news-analysis")
    def job():
        return {"count": 25, "degraded": True, "published": True}

    assert job() == {"count": 25, "degraded": True, "published": True}
    assert [url for url, _ in http.posts] == ["https://hc-ping.com/uuid-news"]


@pytest.mark.parametrize("result", [{"error": True}, {"ok": False, "error": "login"}, {"published": False}])
def test_a_run_that_did_not_do_its_job_pings_fail(http, result):
    @heartbeat.monitored("news-analysis")
    def job():
        return result

    job()
    assert [url for url, _ in http.posts] == ["https://hc-ping.com/uuid-news/fail"]


def test_a_skipped_run_counts_as_success(http):
    @heartbeat.monitored("news-analysis")
    def job():
        return {"skipped": "not_a_trading_day"}

    job()
    assert [url for url, _ in http.posts] == ["https://hc-ping.com/uuid-news"]


def test_an_exception_pings_fail_and_still_raises(http):
    @heartbeat.monitored("news-analysis")
    def job():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        job()
    assert [url for url, _ in http.posts] == ["https://hc-ping.com/uuid-news/fail"]


def test_ping_urls_are_fetched_once_per_process(http):
    @heartbeat.monitored("news-analysis")
    def job():
        return {"published": True}

    for _ in range(3):
        job()
    assert len(http.gets) == 1
    assert len(http.posts) == 3


def test_monitoring_failures_never_break_the_job(http):
    http.fail_ping = True

    @heartbeat.monitored("news-analysis")
    def job():
        return {"published": True}

    assert job() == {"published": True}


def test_an_unreachable_api_is_retried_on_the_next_run(http):
    http.fail_api = True

    @heartbeat.monitored("news-analysis")
    def job():
        return {"published": True}

    job()
    assert http.posts == []
    http.fail_api = False
    job()
    assert [url for url, _ in http.posts] == ["https://hc-ping.com/uuid-news"]


def test_without_an_api_key_nothing_is_sent(http, monkeypatch):
    monkeypatch.setattr(heartbeat, "get_settings", lambda: SimpleNamespace(healthchecks_api_key=""))

    @heartbeat.monitored("news-analysis")
    def job():
        return {"published": True}

    assert job() == {"published": True}
    assert http.gets == [] and http.posts == []


def test_an_unknown_check_is_skipped(http):
    @heartbeat.monitored("not-a-check")
    def job():
        return {"published": True}

    job()
    assert http.posts == []


def test_market_overview_pings_the_check_for_its_time_of_day():
    ist = ZoneInfo("Asia/Kolkata")
    assert heartbeat.market_overview_slug(datetime(2026, 9, 14, 6, 30, tzinfo=ist)) == "market-overview-morning"
    assert heartbeat.market_overview_slug(datetime(2026, 9, 14, 18, 0, tzinfo=ist)) == "market-overview-evening"
