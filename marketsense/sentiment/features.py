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


def sentiment_available_by(session: Session, cutoff: datetime, *, model_name: str = MODEL_NAME,
                           lookback: timedelta | None = None, strict_retrieval: bool = False,
                           stock_ids: list[int] | None = None, include_market_wide: bool = False) -> pd.DataFrame:
    """Sentiment rows an observer standing at `cutoff` could have known. One row per (article, stock mentioned).

    include_market_wide=True also returns articles that mention no stock (stock_id = NaN).
    """
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    cutoff = cutoff.astimezone(timezone.utc)
    sf, na, nas = SentimentFeature, NewsArticle, NewsArticleStock

    conds = [sf.model_name == model_name, na.published_at.is_not(None), na.published_at <= cutoff]
    if strict_retrieval:
        conds.append(na.retrieved_at <= cutoff)
    if lookback is not None:
        conds.append(na.published_at > cutoff - lookback)

    cols = [sf.news_id, na.published_at, na.retrieved_at, sf.positive_score, sf.neutral_score, sf.negative_score,
            sf.sentiment_label]
    q = (select(sf.news_id, nas.stock_id, *cols[1:]).select_from(sf)
         .join(na, na.news_id == sf.news_id).join(nas, nas.news_id == na.news_id).where(and_(*conds)))
    if stock_ids is not None:
        q = q.where(nas.stock_id.in_(stock_ids))
    frames = [pd.DataFrame(session.execute(q).all(), columns=COLUMNS)]

    if include_market_wide:
        q2 = (select(sf.news_id, *cols[1:]).select_from(sf).join(na, na.news_id == sf.news_id)
              .where(and_(*conds), ~exists().where(nas.news_id == na.news_id)))
        mw = pd.DataFrame(session.execute(q2).all(), columns=[c for c in COLUMNS if c != "stock_id"])
        mw.insert(1, "stock_id", pd.NA)
        frames.append(mw)

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    for c in ("published_at", "retrieved_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    for c in ("positive_score", "neutral_score", "negative_score"):
        df[c] = df[c].astype(float)
    return df.sort_values(["published_at", "news_id"]).reset_index(drop=True)
