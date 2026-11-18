"""Leakage safety for DATE-ONLY historical rows and imports (see sentiment/features.py::availability_condition)."""
from datetime import datetime, timedelta, timezone, date

import pytest
from sqlalchemy import select

from marketsense.db.models import NewsArticle, NewsArticleStock, SentimentFeature, Stock
from marketsense.sentiment.features import articles_available_by, sentiment_available_by
from tests.conftest import utc

CUTOFF = utc(2026, 10, 9, 12, 0)             # Friday 17:00 PKT; the PKT calendar date is 2026-10-09


def sid(s, sym):
    return s.scalar(select(Stock.stock_id).where(Stock.symbol == sym))


def add(s, headline, *, published=None, pdate=None, precision="exact", doc_type="news", syms=(), retrieved=utc(2026, 10, 1), scored=False):
    a = NewsArticle(headline=headline, source="t", published_at=published, published_date=pdate, time_precision=precision,
                    doc_type=doc_type, retrieved_at=retrieved, stock_id=sid(s, syms[0]) if len(syms) == 1 else None)
    s.add(a); s.flush()
    for y in syms:
        s.add(NewsArticleStock(news_id=a.news_id, stock_id=sid(s, y)))
    if scored:
        s.add(SentimentFeature(news_id=a.news_id, stock_id=a.stock_id, positive_score=.6, neutral_score=.3, negative_score=.1,
                               sentiment_label="positive", model_name="ProsusAI/finbert"))
    s.flush()
    return a.news_id


def ids(df):
    return set(df.news_id)


def test_date_only_rows_become_available_the_day_after(news_session):
    s = news_session
    same_day = add(s, "dated today", pdate=date(2026, 10, 9), precision="date", syms=["LUCK"])       # may have appeared after 17:00
    prev_day = add(s, "dated yesterday", pdate=date(2026, 10, 8), precision="date", syms=["LUCK"])
    future = add(s, "dated tomorrow", pdate=date(2026, 10, 10), precision="date", syms=["LUCK"])
    assert ids(articles_available_by(s, CUTOFF)) == {prev_day}
    assert same_day not in ids(articles_available_by(s, CUTOFF)) and future not in ids(articles_available_by(s, CUTOFF))
    # one day later the same-day article is knowable
    assert same_day in ids(articles_available_by(s, utc(2026, 10, 10, 12, 0)))


def test_exact_and_date_only_mix_and_precision_none_is_never_available(news_session):
    s = news_session
    exact_ok = add(s, "exact before", published=CUTOFF - timedelta(minutes=1), pdate=date(2026, 10, 9), syms=["LUCK"])
    exact_late = add(s, "exact after", published=CUTOFF + timedelta(minutes=1), pdate=date(2026, 10, 9), syms=["LUCK"])
    none_ = add(s, "no time at all", precision="none", syms=["LUCK"])
    broken = add(s, "claims exact but has no timestamp", published=None, precision="exact", syms=["LUCK"])
    assert ids(articles_available_by(s, CUTOFF)) == {exact_ok}


def test_lookback_applies_to_date_only_rows_too(news_session):
    s = news_session
    old = add(s, "old", pdate=date(2026, 10, 5), precision="date", syms=["LUCK"])
    recent = add(s, "recent", pdate=date(2026, 10, 7), precision="date", syms=["LUCK"])
    assert ids(articles_available_by(s, CUTOFF, lookback=timedelta(days=3))) == {recent}
    assert ids(articles_available_by(s, CUTOFF, lookback=timedelta(days=10))) == {old, recent}


def test_strict_retrieval_excludes_historical_imports(news_session):
    s = news_session
    live = add(s, "collected live", published=utc(2026, 10, 9, 6, 0), retrieved=utc(2026, 10, 9, 6, 5), syms=["LUCK"])
    imported = add(s, "imported later", published=utc(2026, 10, 9, 6, 0), retrieved=utc(2026, 10, 24), syms=["LUCK"])
    assert ids(articles_available_by(s, CUTOFF)) == {live, imported}                       # published-time view
    assert ids(articles_available_by(s, CUTOFF, strict_retrieval=True)) == {live}          # "could we have HAD it then?"


def test_multi_stock_rows_and_market_wide_opt_in(news_session):
    s = news_session
    both = add(s, "HBL and LUCK", published=CUTOFF - timedelta(hours=2), syms=["HBL", "LUCK"])
    market = add(s, "IMF talks", published=CUTOFF - timedelta(hours=3))
    df = articles_available_by(s, CUTOFF)
    assert sorted(zip(df.news_id, df.stock_id)) == sorted([(both, sid(s, "HBL")), (both, sid(s, "LUCK"))])
    mw = articles_available_by(s, CUTOFF, include_market_wide=True)
    assert ids(mw) == {both, market} and mw.loc[mw.news_id == market, "stock_id"].isna().all()
    assert list(articles_available_by(s, utc(2000, 1, 1)).columns) == list(df.columns)


def test_sentiment_query_applies_the_same_date_only_rule(news_session):
    s = news_session
    ok = add(s, "yesterday scored", pdate=date(2026, 10, 8), precision="date", syms=["LUCK"], scored=True)
    same = add(s, "today scored", pdate=date(2026, 10, 9), precision="date", syms=["LUCK"], scored=True)
    add(s, "date-only with contradictory future timestamp", pdate=date(2026, 10, 8), precision="date",
        published=CUTOFF + timedelta(hours=1), syms=["LUCK"], scored=True)
    assert ids(sentiment_available_by(s, CUTOFF)) == {ok}


def test_naive_cutoff_rejected(news_session):
    with pytest.raises(ValueError, match="timezone-aware"):
        articles_available_by(news_session, datetime(2026, 10, 9, 12, 0))
