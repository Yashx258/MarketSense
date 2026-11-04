"""Leakage-safe access to sentiment for future Model B features.

RULE: a prediction made after the close of `trading_date` may only use articles with
    published_at <= cutoff_for_trading_date(trading_date)          (default 17:00 Asia/Karachi)
and, for STRICT live-style backtests, also retrieved_at <= cutoff (we could not have had it earlier).

Articles with NULL published_at are NEVER returned. Scoring time (`sentiment_features.processed_at`) is NOT an
availability time: a score is a pure function of the text, so scoring an old article today is harmless as long as
the article itself was published before the cutoff.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, select

from marketsense.config import Settings
from marketsense.db.models import NewsArticle, NewsArticleStock, SentimentFeature
from marketsense.sentiment.processor import MODEL_NAME

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
            NewsArticle.published_date.is_not(None),
            NewsArticle.published_date < cutoff_pkt_date,
        ),
    )


def _lookback_condition(cutoff, lookback):
    """Return a lower-bound condition appropriate to each precision type."""
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
            NewsArticle.published_at >= lower_utc,
        ),
        and_(
            NewsArticle.time_precision == "date",
            NewsArticle.published_date.is_not(None),
            NewsArticle.published_date >= lower_pkt_date,
            NewsArticle.published_date < cutoff_pkt_date,
        ),
    )


def articles_available_by(
    session,
    cutoff,
    *,
    lookback=None,
    strict_retrieval=False,
    stock_ids=None,
    include_market_wide=False,
):
    """
    Return article-stock rows that were available by cutoff.

    Date-only rows become available on the day after published_date.
    Rows with unknown or inconsistent time precision are excluded.

    Market-wide articles have a NULL stock_id and are excluded unless
    include_market_wide=True.
    """
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("cutoff must be timezone-aware")

    cutoff_utc = cutoff.astimezone(timezone.utc)

    stmt = (
        select(
            NewsArticle.news_id.label("news_id"),
            NewsArticleStock.stock_id.label("stock_id"),
            NewsArticle.published_at.label("published_at"),
            NewsArticle.retrieved_at.label("retrieved_at"),
        )
        .select_from(NewsArticle)
        .outerjoin(
            NewsArticleStock,
            NewsArticleStock.news_id == NewsArticle.news_id,
        )
        .where(availability_condition(cutoff_utc))
    )

    if not include_market_wide:
        stmt = stmt.where(NewsArticleStock.stock_id.is_not(None))

    if stock_ids is not None:
        stock_ids = list(stock_ids)
        if not stock_ids:
            return pd.DataFrame(
                columns=["news_id", "stock_id", "published_at", "retrieved_at"]
            )
        stmt = stmt.where(NewsArticleStock.stock_id.in_(stock_ids))

    if strict_retrieval:
        stmt = stmt.where(
            NewsArticle.retrieved_at.is_not(None),
            NewsArticle.retrieved_at <= cutoff_utc,
        )

    lookback_condition = _lookback_condition(cutoff_utc, lookback)
    if lookback_condition is not None:
        stmt = stmt.where(lookback_condition)

    stmt = stmt.distinct().order_by(
        NewsArticle.news_id,
        NewsArticleStock.stock_id,
    )

    columns = ["news_id", "stock_id", "published_at", "retrieved_at"]
    rows = session.execute(stmt).mappings().all()
    return pd.DataFrame(rows, columns=columns)


def sentiment_available_by(
    session,
    cutoff,
    *,
    model_name=MODEL_NAME,
    lookback=None,
    strict_retrieval=False,
    stock_ids=None,
):
    """
    Return sentiment scores only for articles available by cutoff.

    Availability is determined from the original article's publication
    precision and timestamp/date, not from sentiment processed_at.
    """
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("cutoff must be timezone-aware")

    cutoff_utc = cutoff.astimezone(timezone.utc)

    stmt = (
        select(
            SentimentFeature.news_id.label("news_id"),
            SentimentFeature.stock_id.label("stock_id"),
            NewsArticle.published_at.label("published_at"),
            NewsArticle.retrieved_at.label("retrieved_at"),
            SentimentFeature.positive_score.label("positive_score"),
            SentimentFeature.neutral_score.label("neutral_score"),
            SentimentFeature.negative_score.label("negative_score"),
            SentimentFeature.sentiment_label.label("sentiment_label"),
        )
        .join(
            NewsArticle,
            NewsArticle.news_id == SentimentFeature.news_id,
        )
        .where(
            availability_condition(cutoff_utc),
            SentimentFeature.model_name == model_name,
        )
    )

    if strict_retrieval:
        stmt = stmt.where(
            NewsArticle.retrieved_at.is_not(None),
            NewsArticle.retrieved_at <= cutoff_utc,
        )

    if stock_ids is not None:
        stock_ids = list(stock_ids)
        if not stock_ids:
            return pd.DataFrame(
                columns=[
                    "news_id", "stock_id", "published_at", "retrieved_at",
                    "positive_score", "neutral_score", "negative_score",
                    "sentiment_label",
                ]
            )
        stmt = stmt.where(SentimentFeature.stock_id.in_(stock_ids))

    lookback_condition = _lookback_condition(cutoff_utc, lookback)
    if lookback_condition is not None:
        stmt = stmt.where(lookback_condition)

    stmt = stmt.order_by(
        NewsArticle.news_id,
        SentimentFeature.stock_id,
    )

    columns = [
    "news_id",
    "stock_id",
    "published_at",
    "retrieved_at",
    "positive_score",
    "neutral_score",
    "negative_score",
    "sentiment_label",
    ]
    rows = session.execute(stmt).mappings().all()
    return pd.DataFrame(rows, columns=columns)
