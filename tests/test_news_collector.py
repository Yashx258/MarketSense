from datetime import datetime, timedelta, timezone

import pytest
import requests
from sqlalchemy import func, select

import scripts.collect_news as cli
from marketsense.collectors.base import NewsFetchBlocked, NewsFetchError, robots_allows
from marketsense.collectors.news_collector import (REGISTRY, BusinessRecorderCollector, DawnBusinessCollector,
                                                   ProfitPakistanTodayCollector, RssNewsCollector, html_to_text)
from marketsense.config import Settings
from marketsense.db.models import NewsArticle
from tests.conftest import utc

NOW = utc(2026, 10, 12, 0, 0)
FEED = "https://www.dawn.com/feeds/business"
ROBOTS = "https://www.dawn.com/robots.txt"


class Resp:
    def __init__(self, status=200, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}
        self.text = content.decode("utf-8", "replace") if isinstance(content, bytes) else content


class FakeHttp:
    """Scripted responses per URL (a list is consumed in order; an Exception instance is raised). Records calls."""
    def __init__(self, routes):
        self.routes, self.calls = {k: (list(v) if isinstance(v, list) else [v]) for k, v in routes.items()}, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers, timeout))
        seq = self.routes.get(url, [Resp(404)])
        r = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(r, Exception):
            raise r
        return r


def collector(http=None, **kw):
    sleeps = []
    c = DawnBusinessCollector(settings=Settings(news_http_timeout=7.0), http=http, sleep=sleeps.append, now=lambda: NOW, **kw)
    c.sleeps = sleeps
    return c


# ------------------------------------------------------------------------------------------ parsing
def test_parse_fixture_counts_and_order(dawn_xml):
    c = collector(store_article_text=False)
    items = c.parse(dawn_xml)
    assert c.stats == dict(entries=9, items=8, no_title=1, no_timestamp=3, future_dated=1, older_than_since=0,
                           with_body=7, link_only_body=1)
    stamped = [i.published_at for i in items if i.published_at]
    assert stamped == sorted(stamped, reverse=True)                           # newest first
    assert [i.published_at is None for i in items] == [False] * 5 + [True] * 3   # timestamp-less items last


def test_timestamps_are_correct_utc_instants(dawn_xml):
    by_title = {i.headline: i for i in collector().parse(dawn_xml)}
    assert by_title["Lucky Cement lifts quarterly profit"].published_at == utc(2026, 10, 11, 2, 39, 40)   # 07:39:40 +0500
    assert by_title["Cotton arrivals rise"].published_at == utc(2026, 10, 1, 4, 0)
    assert all(i.published_at.utcoffset() == timedelta(0) for i in by_title.values() if i.published_at)


def test_bad_missing_and_future_timestamps_become_none_not_guesses(dawn_xml):
    by_title = {i.headline: i for i in collector().parse(dawn_xml)}
    assert by_title["Oil prices steady"].published_at is None            # no pubDate
    assert by_title["Remittances edge up"].published_at is None          # unparseable pubDate
    assert by_title["Budget preview from the future"].published_at is None   # 2030 > now + 10 min


def test_headline_whitespace_cleaned_and_empty_title_skipped(dawn_xml):
    titles = [i.headline for i in collector().parse(dawn_xml)]
    assert "Lucky Cement lifts quarterly profit" in titles               # trailing newline removed
    assert all(t == t.strip() and t for t in titles) and len(titles) == 8


def test_canonical_url_is_the_stable_guid_so_edits_dedupe(dawn_xml):
    items = collector().parse(dawn_xml)
    luck = [i for i in items if i.headline.startswith("Lucky Cement")]
    assert len(luck) == 2 and {i.url for i in luck} == {"https://www.dawn.com/news/9000001"}   # not the slug links


def test_source_language_and_name(dawn_xml):
    assert {(i.source, i.language) for i in collector().parse(dawn_xml)} == {("dawn_business", "en")}


def test_article_text_is_opt_in(dawn_xml):
    assert all(i.article_text is None for i in collector(store_article_text=False).parse(dawn_xml))
    items = {i.headline: i for i in collector(store_article_text=True).parse(dawn_xml)}
    body = items["Lucky Cement lifts quarterly profit"].article_text
    assert "cement despatches & prices" in body and "alert" not in body and "<" not in body   # entities decoded, script dropped
    assert items["Global growth set to slow, UN says"].article_text is None                   # link-only stub


def test_default_follows_settings(dawn_xml):
    c = DawnBusinessCollector(settings=Settings(news_store_article_text=True), now=lambda: NOW)
    assert any(i.article_text for i in c.parse(dawn_xml))
    c = DawnBusinessCollector(settings=Settings(news_store_article_text=False), now=lambda: NOW)
    assert not any(i.article_text for i in c.parse(dawn_xml))


def test_since_filters_old_but_keeps_undated_and_does_not_stop_early(dawn_xml):
    c = collector()
    items = c.parse(dawn_xml, since=utc(2026, 10, 10))
    assert "Cotton arrivals rise" not in [i.headline for i in items] and c.stats["older_than_since"] == 1
    assert sum(i.published_at is None for i in items) == 3 and len(items) == 7


def test_malformed_and_non_feed_content_raise():
    c = collector()
    for junk in (b"<html><body>Access denied</body></html>", b"", b"\x00\x01 not xml"):
        with pytest.raises(NewsFetchError):
            c.parse(junk)


def test_valid_empty_feed_is_not_an_error():
    xml = b'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title><link>https://x</link><description>d</description></channel></rss>'
    assert collector().parse(xml) == []


def test_html_to_text():
    assert html_to_text("<p>A &amp; B</p><script>x()</script><p>C<br>D</p><iframe src=a>no</iframe>") == "A & B\nC\nD"
    assert html_to_text(None) == "" and html_to_text("") == ""


# ------------------------------------------------------------------------------------------ HTTP behaviour
def test_fetch_happy_path_uses_ua_timeout_and_checks_robots_first(dawn_xml):
    http = FakeHttp({ROBOTS: Resp(404), FEED: Resp(200, dawn_xml)})
    items = collector(http).fetch()
    assert len(items) == 8
    assert [u for u, *_ in http.calls] == [ROBOTS, FEED]
    assert all(h["User-Agent"].startswith("MarketSenseAcademicBot") and t == 7.0 for _, h, t in http.calls)


@pytest.mark.parametrize("robots", [Resp(200, b"User-agent: *\nDisallow: /feeds/\n"), Resp(200, b"User-agent: *\nDisallow: /\n"),
                                    Resp(403), Resp(401), Resp(500), Resp(503), requests.ConnectionError("dns")])
def test_robots_disallow_or_unreadable_blocks_before_feed_is_requested(robots, dawn_xml):
    http = FakeHttp({ROBOTS: robots, FEED: Resp(200, dawn_xml)})
    with pytest.raises(NewsFetchBlocked):
        collector(http).fetch()
    assert FEED not in [u for u, *_ in http.calls]          # never touched the feed


def test_missing_robots_txt_means_allowed(dawn_xml):
    http = FakeHttp({ROBOTS: Resp(404), FEED: Resp(200, dawn_xml)})
    assert collector(http).fetch()


def test_robots_rules_unit():
    rules = b"User-agent: MarketSenseAcademicBot\nDisallow: /feeds/\n\nUser-agent: *\nDisallow: /private\n"
    http = FakeHttp({ROBOTS: Resp(200, rules)})
    assert robots_allows(FEED, "MarketSenseAcademicBot/0.1", http=http) is False
    assert robots_allows(FEED, "OtherBot/1.0", http=http) is True
    assert robots_allows("https://www.dawn.com/private/x", "OtherBot/1.0", http=http) is False


@pytest.mark.parametrize("status", [403, 429, 401])
def test_blocking_statuses_stop_immediately_without_retry(status):
    http = FakeHttp({ROBOTS: Resp(404), FEED: Resp(status, headers={"Retry-After": "120"})})
    c = collector(http)
    with pytest.raises(NewsFetchBlocked, match=str(status)):
        c.fetch()
    assert [u for u, *_ in http.calls].count(FEED) == 1 and c.sleeps == []


def test_5xx_is_retried_with_backoff_then_gives_up():
    http = FakeHttp({ROBOTS: Resp(404), FEED: Resp(503)})
    c = collector(http)
    with pytest.raises(NewsFetchError, match="3 attempts"):
        c.fetch()
    assert [u for u, *_ in http.calls].count(FEED) == 3 and c.sleeps == [1, 2]


def test_transient_failure_then_success(dawn_xml):
    http = FakeHttp({ROBOTS: Resp(404), FEED: [requests.Timeout("slow"), Resp(502), Resp(200, dawn_xml)]})
    c = collector(http)
    assert len(c.fetch()) == 8 and c.sleeps == [1, 2]


def test_404_is_an_error_without_retry():
    http = FakeHttp({ROBOTS: Resp(404), FEED: Resp(404)})
    with pytest.raises(NewsFetchError, match="404"):
        collector(http).fetch()
    assert [u for u, *_ in http.calls].count(FEED) == 1


def test_html_error_page_with_200_is_rejected():
    http = FakeHttp({ROBOTS: Resp(404), FEED: Resp(200, b"<html><body>Please enable JavaScript</body></html>")})
    with pytest.raises(NewsFetchError):
        collector(http).fetch()


# ------------------------------------------------------------------------------------------ registry / placeholders
def test_unverified_sources_refuse_honestly():
    for cls in (BusinessRecorderCollector, ProfitPakistanTodayCollector):
        with pytest.raises(NotImplementedError, match="NOT been verified"):
            cls().fetch()
        assert cls.enabled is False
    assert DawnBusinessCollector.enabled is True and set(REGISTRY) == {"dawn_business", "business_recorder", "profit_pakistan_today"}


# ------------------------------------------------------------------------------------------ CLI
@pytest.fixture()
def feed_bytes(monkeypatch, dawn_xml):
    monkeypatch.setattr(RssNewsCollector, "fetch_bytes", lambda self: dawn_xml)
    monkeypatch.setattr(RssNewsCollector, "__init__", _fixed_now_init(RssNewsCollector.__init__))


def _fixed_now_init(orig):
    def init(self, *a, **kw):
        kw.setdefault("now", lambda: NOW)
        orig(self, *a, **kw)
    return init


def test_cli_refuses_without_enable_flag(capsys):
    assert cli.main(["--source", "dawn_business"]) == 2
    assert "NEWS_COLLECTION_ENABLED" in capsys.readouterr().err


def test_cli_refuses_unverified_source(capsys):
    assert cli.main(["--source", "business_recorder", "--dry-run"]) == 2
    assert "NOT been verified" in capsys.readouterr().err


def test_cli_check_needs_no_database_or_flag(feed_bytes, capsys):
    assert cli.main(["--source", "dawn_business", "--check"]) == 0
    out = capsys.readouterr().out
    assert "robots.txt: allowed" in out and "items parsed: 8" in out and "newest: 2026-10-11T03:10:00+00:00" in out
    assert "article bodies saved: False" in out


def test_cli_dry_run_writes_nothing(feed_bytes, capsys, monkeypatch):
    monkeypatch.setattr(cli, "session_scope", lambda *a, **k: (_ for _ in ()).throw(AssertionError("DB must not be touched")))
    monkeypatch.setattr(cli, "_since", lambda a: utc(2026, 10, 10))
    assert cli.main(["--source", "dawn_business", "--dry-run", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "2 item(s) would be stored" in out and "Lucky Cement" in out


def test_cli_blocked_returns_2(monkeypatch, capsys):
    def boom(self): raise NewsFetchBlocked("HTTP 403")
    monkeypatch.setattr(RssNewsCollector, "fetch_bytes", boom)
    assert cli.main(["--source", "dawn_business", "--check"]) == 2
    assert "REFUSED: HTTP 403" in capsys.readouterr().err


def test_cli_network_error_returns_1(monkeypatch, capsys):
    def boom(self): raise NewsFetchError("giving up after 3 attempts")
    monkeypatch.setattr(RssNewsCollector, "fetch_bytes", boom)
    assert cli.main(["--source", "dawn_business", "--dry-run"]) == 1


def test_cli_store_path_end_to_end(feed_bytes, news_session, monkeypatch, capsys):
    from contextlib import contextmanager

    @contextmanager
    def scope(*a, **k):
        yield news_session
        news_session.commit()
    monkeypatch.setattr(cli, "session_scope", scope)
    monkeypatch.setattr(cli, "Settings", lambda: Settings(news_enabled=True))
    monkeypatch.setattr(cli, "_since", lambda a: utc(2026, 10, 10))
    assert cli.main(["--source", "dawn_business"]) == 0
    out = capsys.readouterr().out
    # 7 fetched since 10 Oct: 4 dated + 3 undated; the edited Lucky Cement item repeats one URL -> 6 inserted, 1 in-batch duplicate
    assert "fetched=7" in out and "inserted=6" in out and "duplicates_in_batch=1" in out
    assert "articles_with_stocks=2" in out and "stock_links=3" in out and "stored_without_timestamp=3" in out
    n1 = news_session.scalar(select(func.count()).select_from(NewsArticle))
    assert cli.main(["--source", "dawn_business"]) == 0                  # second run: nothing new
    assert news_session.scalar(select(func.count()).select_from(NewsArticle)) == n1
    assert "inserted=0" in capsys.readouterr().out
