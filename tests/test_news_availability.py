"""Leakage-safety of news/sentiment for future Model B features."""
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from marketsense.db.models import NewsArticle, NewsArticleStock, SentimentFeature, Stock
from marketsense.sentiment.features import COLUMNS, cutoff_for_trading_date, sentiment_available_by
from tests.conftest import utc

PKT = timezone(timedelta(hours=5))
# Friday 2026-10-09, close 17:00 PKT == 12:00 UTC
CUTOFF = utc(2026, 10, 9, 12, 0)


def sid(s, sym):
    return s.scalar(select(Stock.stock_id).where(Stock.symbol == sym))


def article(s, headline, published, symbols=(), retrieved=None, model="ProsusAI/finbert", scored=True, label="positive"):
    a = NewsArticle(headline=headline, source="t", published_at=published, retrieved_at=retrieved or utc(2026, 10, 1),
                    stock_id=sid(s, symbols[0]) if len(symbols) == 1 else None)
    s.add(a); s.flush()
    for sym in symbols:
        s.add(NewsArticleStock(news_id=a.news_id, stock_id=sid(s, sym)))
    if scored:
        s.add(SentimentFeature(news_id=a.news_id, stock_id=a.stock_id, positive_score=.7, neutral_score=.2,
                               negative_score=.1, sentiment_label=label, model_name=model))
    s.flush()
    return a.news_id


# ------------------------------------------------------------------------------------------ cutoff helper
def test_cutoff_is_post_close_pakistan_time_in_utc():
    assert cutoff_for_trading_date(date(2026, 10, 9)) == utc(2026, 10, 9, 12, 0)                   # 17:00 PKT
    assert cutoff_for_trading_date(date(2026, 10, 9), "15:45") == utc(2026, 10, 9, 10, 45)
    assert cutoff_for_trading_date(date(2026, 10, 9)).utcoffset() == timedelta(0)


# ------------------------------------------------------------------------------------------ the leakage rules
def test_nothing_after_the_cutoff_is_ever_returned(news_session):
    s = news_session
    before = article(s, "before", CUTOFF - timedelta(seconds=1), ["LUCK"])
    at = article(s, "exactly at cutoff", CUTOFF, ["LUCK"])
    after = article(s, "one second after", CUTOFF + timedelta(seconds=1), ["LUCK"])
    much_after = article(s, "next week", CUTOFF + timedelta(days=7), ["LUCK"])
    df = sentiment_available_by(s, CUTOFF)
    assert set(df.news_id) == {before, at}                       # inclusive at the cutoff, nothing later
    assert after not in set(df.news_id) and much_after not in set(df.news_id)
    assert df.published_at.max() <= CUTOFF


def test_cutoff_is_expressed_correctly_whatever_timezone_the_caller_uses(news_session):
    a = article(news_session, "x", utc(2026, 10, 9, 11, 59), ["LUCK"])
    late = article(news_session, "late", utc(2026, 10, 9, 12, 1), ["LUCK"])
    pkt_cutoff = datetime(2026, 10, 9, 17, 0, tzinfo=PKT)       # same instant as CUTOFF
    assert set(sentiment_available_by(news_session, pkt_cutoff).news_id) == {a}
    assert late not in set(sentiment_available_by(news_session, pkt_cutoff).news_id)


def test_articles_without_a_timestamp_are_never_returned(news_session):
    article(news_session, "undated", None, ["LUCK"])
    dated = article(news_session, "dated", utc(2026, 10, 8), ["LUCK"])
    assert list(sentiment_available_by(news_session, CUTOFF, include_market_wide=True).news_id) == [dated]


def test_naive_cutoff_is_rejected(news_session):
    with pytest.raises(ValueError, match="timezone-aware"):
        sentiment_available_by(news_session, datetime(2026, 10, 9, 12, 0))


def test_late_scoring_does_not_change_availability(news_session):
    """processed_at (when WE scored it) must never decide availability; published_at does."""
    s = news_session
    old = article(s, "old news scored today", utc(2026, 10, 1), ["LUCK"])
    s.execute(SentimentFeature.__table__.update().where(SentimentFeature.news_id == old)
              .values(processed_at=utc(2026, 12, 25)))
    assert old in set(sentiment_available_by(s, CUTOFF).news_id)


def test_strict_mode_also_requires_retrieval_before_the_cutoff(news_session):
    s = news_session
    live = article(s, "seen live", utc(2026, 10, 9, 6, 0), ["LUCK"], retrieved=utc(2026, 10, 9, 6, 5))
    backfilled = article(s, "backfilled later", utc(2026, 10, 9, 6, 0), ["LUCK"], retrieved=utc(2026, 10, 20))
    assert set(sentiment_available_by(s, CUTOFF).news_id) == {live, backfilled}                 # published-time view
    assert set(sentiment_available_by(s, CUTOFF, strict_retrieval=True).news_id) == {live}      # could we have HAD it then?


def test_lookback_window(news_session):
    s = news_session
    new = article(s, "new", CUTOFF - timedelta(hours=5), ["LUCK"])
    old = article(s, "old", CUTOFF - timedelta(days=3), ["LUCK"])
    assert set(sentiment_available_by(s, CUTOFF, lookback=timedelta(days=1)).news_id) == {new}
    assert set(sentiment_available_by(s, CUTOFF, lookback=timedelta(days=5)).news_id) == {new, old}


# ------------------------------------------------------------------------------------------ shape / joins
def test_multi_stock_article_yields_one_row_per_stock_and_filters_work(news_session):
    s = news_session
    both = article(s, "HBL and LUCK", utc(2026, 10, 8), ["HBL", "LUCK"])
    luck_only = article(s, "LUCK only", utc(2026, 10, 8, 1), ["LUCK"])
    df = sentiment_available_by(s, CUTOFF)
    assert sorted(zip(df.news_id, df.stock_id)) == sorted([(both, sid(s, "HBL")), (both, sid(s, "LUCK")), (luck_only, sid(s, "LUCK"))])
    only_hbl = sentiment_available_by(s, CUTOFF, stock_ids=[sid(s, "HBL")])
    assert list(only_hbl.news_id) == [both]


def test_market_wide_articles_are_opt_in(news_session):
    s = news_session
    market = article(s, "IMF talks", utc(2026, 10, 8), [])
    stock = article(s, "LUCK", utc(2026, 10, 8, 1), ["LUCK"])
    assert list(sentiment_available_by(s, CUTOFF).news_id) == [stock]
    df = sentiment_available_by(s, CUTOFF, include_market_wide=True)
    assert set(df.news_id) == {market, stock} and df.loc[df.news_id == market, "stock_id"].isna().all()


def test_unscored_articles_and_other_models_are_excluded(news_session):
    s = news_session
    article(s, "unscored", utc(2026, 10, 8), ["LUCK"], scored=False)
    article(s, "other model", utc(2026, 10, 8), ["LUCK"], model="other/model")
    mine = article(s, "mine", utc(2026, 10, 8), ["LUCK"])
    assert list(sentiment_available_by(s, CUTOFF).news_id) == [mine]
    assert len(sentiment_available_by(s, CUTOFF, model_name="other/model")) == 1


def test_output_schema_and_dtypes(news_session):
    article(news_session, "x", utc(2026, 10, 8), ["LUCK"])
    df = sentiment_available_by(news_session, CUTOFF)
    assert list(df.columns) == COLUMNS
    assert str(df.published_at.dtype) == "datetime64[us, UTC]" or "UTC" in str(df.published_at.dtype)
    assert df.positive_score.dtype == float
    empty = sentiment_available_by(news_session, utc(2000, 1, 1))
    assert empty.empty and list(empty.columns) == COLUMNS
