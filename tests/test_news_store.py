from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from marketsense.collectors.base import NewsItem
from marketsense.collectors.news_store import (GENERIC_ACRONYMS, StockIndex, build_stock_index, match_stocks,
                                               normalize_url, store_news)
from marketsense.db.models import NewsArticle, NewsArticleStock, Stock
from tests.conftest import as_utc, utc

PKT = timezone(timedelta(hours=5))


def item(headline="Lucky Cement lifts profit", url="https://www.dawn.com/news/1", published=utc(2026, 10, 11, 2, 0), **kw):
    return NewsItem(headline=headline, source="dawn_business", url=url, published_at=published, **kw)


def count(s, model):
    return s.scalar(select(func.count()).select_from(model))


# ------------------------------------------------------------------------------------------ normalize_url
@pytest.mark.parametrize("raw,expected", [
    ("HTTPS://WWW.Dawn.com:443/news/1/?utm_source=x&b=2&a=1#frag", "https://www.dawn.com/news/1?a=1&b=2"),
    ("https://www.dawn.com/news/1/", "https://www.dawn.com/news/1"),
    ("https://www.dawn.com", "https://www.dawn.com/"),
    ("http://example.com:80/a", "http://example.com/a"),
    ("http://example.com:8080/a", "http://example.com:8080/a"),
    ("https://x.com/p?fbclid=abc&id=7&utm_campaign=z", "https://x.com/p?id=7"),
    ("  https://x.com/a  ", "https://x.com/a"),
    ("https://x.com/p?a=1&a=2", "https://x.com/p?a=1&a=2"),
])
def test_normalize_url(raw, expected):
    assert normalize_url(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "mailto:a@b.c", "ftp://x.com/a", "not a url", "https:///nohost", "/relative/path"])
def test_normalize_url_rejects(raw):
    assert normalize_url(raw) is None


# ------------------------------------------------------------------------------------------ matching
def test_match_stocks(news_session):
    idx = build_stock_index(news_session)
    ids = idx.by_symbol
    assert match_stocks("HBL and LUCK lead gains", idx) == {ids["HBL"]: "symbol", ids["LUCK"]: "symbol"}
    assert match_stocks("Lucky Cement raised prices", idx) == {ids["LUCK"]: "name"}
    assert match_stocks("hbl and luck", idx) == {}                      # lower-case is not a ticker
    assert match_stocks("Lucky Cementation plant", idx) == {}           # word boundary
    assert match_stocks("Engro posts profit", idx) == {}                # single short word: too ambiguous for a name match
    assert match_stocks("ENGRO posts profit", idx) == {ids["ENGRO"]: "symbol"}
    assert match_stocks("Oil & Gas Development reports output", idx) == {ids["OGDC"]: "name"}
    assert match_stocks("", idx) == {} and match_stocks(None, idx) == {}


def test_generic_acronyms_never_match(news_session):
    news_session.add(Stock(symbol="SBP", company_name="Spurious Co"))   # even if a symbol collides with a generic acronym
    news_session.commit()
    assert "SBP" in GENERIC_ACRONYMS
    assert match_stocks("SBP holds rates", build_stock_index(news_session)) == {}


def test_ambiguous_company_name_is_never_guessed(news_session):
    news_session.add_all([Stock(symbol="AAA", company_name="Pak Textile Mills Limited"),
                          Stock(symbol="AAB", company_name="Pak Textile Mills Ltd")])
    news_session.commit()
    assert match_stocks("Pak Textile Mills expands", build_stock_index(news_session)) == {}


# ------------------------------------------------------------------------------------------ store_news
def test_single_stock_sets_stock_id_and_link(news_session):
    rep = store_news(news_session, [item()])
    news_session.commit()
    art = news_session.scalars(select(NewsArticle)).one()
    luck = news_session.scalar(select(Stock.stock_id).where(Stock.symbol == "LUCK"))
    assert (rep.inserted, rep.articles_with_stocks, rep.stock_links) == (1, 1, 1)
    assert art.stock_id == luck
    assert news_session.execute(select(NewsArticleStock.stock_id, NewsArticleStock.matched_by)).all() == [(luck, "name")]


def test_multi_stock_article_has_null_stock_id_and_all_links(news_session):
    store_news(news_session, [item("Banks and cement lead gains; HBL, LUCK top", url="https://www.dawn.com/news/2")])
    art = news_session.scalars(select(NewsArticle)).one()
    assert art.stock_id is None
    assert count(news_session, NewsArticleStock) == 2


def test_no_stock_is_market_wide(news_session):
    rep = store_news(news_session, [item("IMF reviews Pakistan programme", url="https://www.dawn.com/news/3")])
    assert news_session.scalars(select(NewsArticle)).one().stock_id is None
    assert (rep.inserted, rep.articles_with_stocks, count(news_session, NewsArticleStock)) == (1, 0, 0)


def test_rerun_is_idempotent(news_session):
    batch = [item(), item("HBL profit", url="https://www.dawn.com/news/2")]
    r1 = store_news(news_session, batch); news_session.commit()
    r2 = store_news(news_session, batch); news_session.commit()
    assert (r1.inserted, r1.duplicates) == (2, 0)
    assert (r2.inserted, r2.duplicates) == (0, 2)
    assert count(news_session, NewsArticle) == 2 and count(news_session, NewsArticleStock) == 2


def test_url_variants_are_the_same_article(news_session):
    batch = [item(url="https://www.dawn.com/news/1"),
             item(url="HTTPS://www.dawn.com/news/1/?utm_source=tw#x"),
             item(url="https://www.dawn.com:443/news/1/")]
    rep = store_news(news_session, batch)
    assert (rep.inserted, rep.duplicates_in_batch) == (1, 2)
    again = store_news(news_session, [item(url="https://www.dawn.com/news/1/?fbclid=zz")])
    assert (again.inserted, again.duplicates) == (0, 1)
    assert news_session.scalars(select(NewsArticle.url)).one() == "https://www.dawn.com/news/1"


def test_repeat_in_batch_keeps_earliest_timestamp(news_session):
    later = item("Lucky Cement lifts profit (updated)", published=utc(2026, 10, 11, 3, 10))
    earlier = item("Lucky Cement lifts profit", published=utc(2026, 10, 11, 2, 39))
    store_news(news_session, [later, earlier])          # later one listed first on purpose
    art = news_session.scalars(select(NewsArticle)).one()
    assert as_utc(art.published_at) == utc(2026, 10, 11, 2, 39) and art.headline == "Lucky Cement lifts profit"


def test_missing_url_is_deduplicated_on_source_headline_time(news_session):
    a = item(url=None, headline="Oil steady", published=utc(2026, 10, 10, 5, 0))
    r1 = store_news(news_session, [a]); r2 = store_news(news_session, [a])
    other_time = item(url=None, headline="Oil steady", published=utc(2026, 10, 10, 6, 0))
    r3 = store_news(news_session, [other_time])
    assert (r1.inserted, r2.inserted, r2.duplicates, r3.inserted) == (1, 0, 1, 1)
    assert count(news_session, NewsArticle) == 2


def test_rejects_unusable_items(news_session):
    rep = store_news(news_session, [item(url=None, published=None),               # nothing to dedup or time-gate on
                                    item(headline="   "),                          # empty headline
                                    item(url="mailto:x@y.z", published=None)])     # unusable url and no timestamp
    assert (rep.received, rep.rejected, rep.inserted) == (3, 3, 0) and count(news_session, NewsArticle) == 0


def test_item_with_url_but_no_timestamp_is_stored_and_flagged(news_session):
    rep = store_news(news_session, [item(published=None)])
    assert (rep.inserted, rep.stored_without_timestamp) == (1, 1)
    assert news_session.scalars(select(NewsArticle.published_at)).one() is None


def test_naive_datetime_is_not_guessed(news_session):
    rep = store_news(news_session, [item(published=datetime(2026, 10, 11, 7, 39)),                # naive + url -> NULL
                                    item(url=None, headline="x", published=datetime(2026, 10, 11, 7, 39))])  # naive, no url -> reject
    assert (rep.inserted, rep.rejected, rep.stored_without_timestamp) == (1, 1, 1)


def test_timestamps_stored_as_utc_instants(news_session):
    store_news(news_session, [item(published=datetime(2026, 10, 11, 7, 39, 40, tzinfo=PKT))])   # 07:39:40 +05:00
    news_session.commit()
    assert as_utc(news_session.scalars(select(NewsArticle.published_at)).one()) == utc(2026, 10, 11, 2, 39, 40)


def test_explicit_source_tickers_known_and_unknown(news_session):
    rep = store_news(news_session, [item("Results season", url="https://x.com/1", symbols=["luck", "ZZZZ"]),
                                    item("More results", url="https://x.com/2", symbols=["ZZZZ"])])
    assert rep.unknown_symbols == {"ZZZZ": 2}
    luck = news_session.scalar(select(Stock.stock_id).where(Stock.symbol == "LUCK"))
    rows = news_session.execute(select(NewsArticleStock.stock_id, NewsArticleStock.matched_by)).all()
    assert rows == [(luck, "source")]


def test_explicit_and_text_match_of_same_stock_make_one_link(news_session):
    rep = store_news(news_session, [item("Lucky Cement and LUCK shares rise", symbols=["LUCK"])])
    assert (rep.stock_links, count(news_session, NewsArticleStock)) == (1, 1)


def test_unicode_and_language_roundtrip(news_session):
    store_news(news_session, [item("روپیہ مستحکم — rupee steady", url="https://x.com/u", language="en-US-very-long-tag")])
    art = news_session.scalars(select(NewsArticle)).one()
    assert art.headline == "روپیہ مستحکم — rupee steady" and len(art.language) <= 10


def test_mixed_batch_counts(news_session):
    batch = [item(url="https://x.com/1"), item(url="https://x.com/1"), item(url="https://x.com/2", headline="HBL up"),
             item(url=None, published=None), item(url="https://x.com/3", headline="IMF talks", published=None)]
    rep = store_news(news_session, batch)
    assert (rep.received, rep.inserted, rep.duplicates_in_batch, rep.rejected, rep.stored_without_timestamp) == (5, 3, 1, 1, 1)
    assert "inserted=3" in rep.summary()
