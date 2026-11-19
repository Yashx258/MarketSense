"""Leakage-safe access to sentiment for future Model B features.

RULE: a prediction made after the close of `trading_date` may only use articles with
    published_at <= cutoff_for_trading_date(trading_date)          (default 17:00 Asia/Karachi)
and, for STRICT live-style backtests, also retrieved_at <= cutoff (we could not have had it earlier).

Date-only articles become available on the following PKT day. Undated articles are excluded.
Scoring time (`sentiment_features.processed_at`) is independent of
availability time: a score is a pure function of the text, so scoring an old article today is harmless as long as
the article itself was published before the cutoff.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import and_, or_, select

from marketsense.config import Settings
from marketsense.db.models import NewsArticle, NewsArticleStock, SentimentFeature
from marketsense.sentiment.processor import LEGACY_VARIANT, MODEL_NAME

COLUMNS = ["news_id", "stock_id", "published_at", "retrieved_at",
           "positive_score", "neutral_score", "negative_score", "sentiment_label"]


def cutoff_for_trading_date(trading_date: date, close_time: str | None = None, tz: str = "Asia/Karachi") -> datetime:
    """Post-close cutoff for `trading_date` as an aware UTC datetime. News after it belongs to the NEXT prediction."""
    hh, mm = (close_time or Settings().news_cutoff_time).split(":")
    local = datetime.combine(trading_date, time(int(hh), int(mm)), tzinfo=ZoneInfo(tz))
    return local.astimezone(timezone.utc)


PKT = ZoneInfo("Asia/Karachi")


def availability_condition(cutoff):
    """
    SQL condition for whether an article could have been available by cutoff.

    Exact timestamps:
        published_at <= cutoff

    Date-only timestamps:
        published_date < the cutoff's PKT calendar date

    A date-only article is deliberately not available on its recorded date,
    because its actual publication time is unknown.
    """
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("cutoff must be timezone-aware")

    cutoff_utc = cutoff.astimezone(timezone.utc)
    cutoff_pkt_date = cutoff_utc.astimezone(PKT).date()

    return or_(
        and_(
            NewsArticle.time_precision == "exact",
            NewsArticle.published_at.is_not(None),
            NewsArticle.published_at <= cutoff_utc,
        ),
        and_(
            NewsArticle.time_precision == "date",
            NewsArticle.published_at.is_(None),
            NewsArticle.published_date.is_not(None),
            NewsArticle.published_date < cutoff_pkt_date,
        ),
    )


def _lookback_condition(cutoff, lookback):
    """Use an exclusive instant bound for exact timestamps and a PKT calendar bound for date-only rows."""
    if lookback is None:
        return None

    if not isinstance(lookback, timedelta):
        raise TypeError("lookback must be a datetime.timedelta or None")

    if lookback.total_seconds() < 0:
        raise ValueError("lookback must not be negative")

    cutoff_utc = cutoff.astimezone(timezone.utc)
    cutoff_pkt_date = cutoff_utc.astimezone(PKT).date()
    lower_utc = cutoff_utc - lookback
    lower_pkt_date = lower_utc.astimezone(PKT).date()

    return or_(
        and_(
            NewsArticle.time_precision == "exact",
            NewsArticle.published_at.is_not(None),
            NewsArticle.published_at > lower_utc,
        ),
        and_(
            NewsArticle.time_precision == "date",
            NewsArticle.published_at.is_(None),
            NewsArticle.published_date.is_not(None),
            NewsArticle.published_date >= lower_pkt_date,
            NewsArticle.published_date < cutoff_pkt_date,
        ),
    )


ARTICLE_COLUMNS = ["news_id", "stock_id", "published_at", "retrieved_at"]


def _available_articles(cutoff, lookback, strict_retrieval, stock_ids, include_market_wide):
    available = availability_condition(cutoff)
    cutoff_utc = cutoff.astimezone(timezone.utc)
    stmt = (
        select(
            NewsArticle.news_id.label("news_id"),
            NewsArticleStock.stock_id.label("stock_id"),
            NewsArticle.published_at.label("published_at"),
            NewsArticle.retrieved_at.label("retrieved_at"),
        )
        .outerjoin(NewsArticleStock, NewsArticleStock.news_id == NewsArticle.news_id)
        .where(available, NewsArticle.relevance_status == "include")
    )

    stock_id = NewsArticleStock.stock_id
    if stock_ids is not None:
        selected = stock_id.in_(list(stock_ids))
        stmt = stmt.where(or_(selected, stock_id.is_(None)) if include_market_wide else selected)
    elif not include_market_wide:
        stmt = stmt.where(stock_id.is_not(None))

    if strict_retrieval:
        stmt = stmt.where(NewsArticle.retrieved_at <= cutoff_utc)
    lower_bound = _lookback_condition(cutoff_utc, lookback)
    if lower_bound is not None:
        stmt = stmt.where(lower_bound)
    return stmt


def _availability_frame(session, stmt, columns):
    df = pd.DataFrame(session.execute(stmt).mappings().all(), columns=columns)
    for column in ("published_at", "retrieved_at"):
        df[column] = pd.to_datetime(df[column], utc=True)
    return df


def articles_available_by(
    session,
    cutoff,
    *,
    lookback=None,
    strict_retrieval=False,
    stock_ids=None,
    include_market_wide=False,
):
    """Return one row per available article and linked stock."""
    stmt = _available_articles(cutoff, lookback, strict_retrieval, stock_ids, include_market_wide)
    stmt = stmt.order_by(NewsArticle.news_id, NewsArticleStock.stock_id)
    return _availability_frame(session, stmt, ARTICLE_COLUMNS)


def sentiment_available_by(
    session,
    cutoff,
    *,
    model_name=MODEL_NAME,
    text_variant=LEGACY_VARIANT,
    lookback=None,
    strict_retrieval=False,
    stock_ids=None,
    include_market_wide=False,
):
    """Return one score per available article and linked stock for the selected model."""
    stmt = (
        _available_articles(cutoff, lookback, strict_retrieval, stock_ids, include_market_wide)
        .add_columns(
            SentimentFeature.positive_score,
            SentimentFeature.neutral_score,
            SentimentFeature.negative_score,
            SentimentFeature.sentiment_label,
        )
        .join(SentimentFeature, SentimentFeature.news_id == NewsArticle.news_id)
        .where(SentimentFeature.model_name == model_name, SentimentFeature.text_variant == text_variant)
        .order_by(NewsArticle.published_at.asc().nulls_last(), NewsArticle.news_id, NewsArticleStock.stock_id)
    )
    df = _availability_frame(session, stmt, COLUMNS)
    for column in ("positive_score", "neutral_score", "negative_score"):
        df[column] = df[column].astype(float)
    return df
