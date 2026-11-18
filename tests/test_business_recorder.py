"""Business Recorder RSS: parsing, wire-copy protection, HTTP behaviour, storage with company aliases."""
import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from marketsense.collectors.base import NewsFetchBlocked
from marketsense.collectors.news_collector import BusinessRecorderCollector
from marketsense.collectors.news_store import build_stock_index, store_news
from marketsense.config import Settings
from marketsense.db.models import NewsArticle, NewsArticleStock, Stock
from tests.conftest import FIXTURES, as_utc, utc
from tests.test_news_collector import FakeHttp, Resp

NOW = utc(2026, 10, 12, 0, 0)
FEED, ROBOTS = "https://www.brecorder.com/feeds/latest-news", "https://www.brecorder.com/robots.txt"


@pytest.fixture()
def br_xml() -> bytes:
    with open(os.path.join(FIXTURES, "brecorder_latest_news_sample.xml"), "rb") as f:
        return f.read()


def br(http=None, **kw):
    return BusinessRecorderCollector(settings=Settings(news_http_timeout=5.0), http=http, sleep=lambda s: None, now=lambda: NOW, **kw)


def test_feed_identity_is_the_verified_first_party_url():
    assert BusinessRecorderCollector.feed_url == "https://www.brecorder.com/feeds/latest-news"
    assert BusinessRecorderCollector.name == "business_recorder" and BusinessRecorderCollector.enabled


def test_parse_counts_timestamps_and_stable_urls(br_xml):
    c = br()
    items = c.parse(br_xml)
    assert c.stats["entries"] == 6 and c.stats["items"] == 6 and c.stats["no_timestamp"] == 0
    by = {i.headline: i for i in items}
    assert by["OGDCL initiates production from new well"].published_at == utc(2026, 10, 11, 4, 0)   # 09:00 +0500
    assert by["Oil prices rise on supply fears"].published_at == utc(2026, 10, 11, 3, 30)
    urls = [i.url for i in items]
    assert "https://www.brecorder.com/news/4000001" in urls and urls.count("https://www.brecorder.com/news/4000001") == 2
    assert all(i.source == "business_recorder" and i.doc_type == "news" and i.language == "en" for i in items)


def test_wire_copy_body_is_never_stored_even_when_bodies_are_enabled(br_xml):
    c = br(store_article_text=True)
    by = {i.headline: i for i in c.parse(br_xml)}
    assert by["Oil prices rise on supply fears"].article_text is None       # Reuters
    assert by["Global markets slip"].article_text is None                    # AFP
    assert by["OGDCL initiates production from new well"].article_text.startswith("KARACHI")   # local desk
    assert c.stats["body_withheld"] == 2


def test_bodies_are_off_by_default(br_xml):
    assert all(i.article_text is None for i in br(store_article_text=False).parse(br_xml))


def test_fetch_checks_robots_then_downloads(br_xml):
    http = FakeHttp({ROBOTS: Resp(404), FEED: Resp(200, br_xml)})
    assert len(br(http).fetch()) == 6
    assert [u for u, *_ in http.calls] == [ROBOTS, FEED]


def test_robots_disallow_blocks(br_xml):
    http = FakeHttp({ROBOTS: Resp(200, b"User-agent: *\nDisallow: /feeds/\n"), FEED: Resp(200, br_xml)})
    with pytest.raises(NewsFetchBlocked):
        br(http).fetch()
    assert FEED not in [u for u, *_ in http.calls]


def test_bot_detection_403_is_a_stop_not_a_retry():
    http = FakeHttp({ROBOTS: Resp(404), FEED: Resp(403)})
    with pytest.raises(NewsFetchBlocked, match="403"):
        br(http).fetch()
    assert [u for u, *_ in http.calls].count(FEED) == 1


def test_since_filter_keeps_newer_only(br_xml):
    items = br().parse(br_xml, since=utc(2026, 10, 11, 4, 30))          # 09:30 PKT
    assert {i.headline for i in items} >= {"KSE-100 gains as investors cheer rate outlook"} and len(items) == 2


def test_real_relevance_is_annotated_without_discarding_articles(news_session, br_xml):
    c = br()
    items = c.parse(br_xml)
    assert c.stats["parsed"] == len(items) == 6
    summary = store_news(news_session, items)
    assert (summary.inserted, summary.duplicates_in_batch) == (5, 1)
    articles = {a.headline: a for a in news_session.scalars(select(NewsArticle))}
    assert articles["KSE-100 gains as investors cheer rate outlook"].relevance_status == "include"
    assert articles["Global markets slip"].relevance_status == "reject"
    assert all(a.relevance_topic and a.relevance_reason and a.relevance_version != "unclassified"
               for a in articles.values())

# ------------------------------------------------------------------------------------------ storage + company mapping
def test_store_maps_alias_tickers_and_leaves_market_news_unlinked(news_session, br_xml):
    s = news_session
    items = br().parse(br_xml)
    rep = store_news(s, items, build_stock_index(s, aliases={"OGDCL": "OGDC"}))
    s.commit()
    assert (rep.inserted, rep.duplicates_in_batch) == (5, 1)              # edited headline shares the stable guid URL
    links = {(a.headline[:12], sym, how) for a, sym, how in s.execute(
        select(NewsArticle, Stock.symbol, NewsArticleStock.matched_by).join(NewsArticleStock, NewsArticleStock.news_id == NewsArticle.news_id)
        .join(Stock, Stock.stock_id == NewsArticleStock.stock_id)).all()}
    assert links == {("OGDCL initia", "OGDC", "alias"), ("HBL, LUCK am", "HBL", "symbol"), ("HBL, LUCK am", "LUCK", "symbol")}
    market = s.scalars(select(NewsArticle).where(NewsArticle.headline.like("KSE-100%"))).one()
    assert market.stock_id is None                                         # legitimate market-wide news keeps no stock
    first = s.scalars(select(NewsArticle).where(NewsArticle.url == "https://www.brecorder.com/news/4000001")).one()
    assert as_utc(first.published_at) == utc(2026, 10, 11, 5, 0) and first.headline == "KSE-100 gains as investors cheer rate outlook"   # EARLIEST kept


def test_rerun_is_idempotent(news_session, br_xml):
    s = news_session
    idx = build_stock_index(s, aliases={})
    store_news(s, br().parse(br_xml), idx); s.commit()
    again = store_news(s, br().parse(br_xml), idx)
    assert (again.inserted, again.duplicates) == (0, 5)
