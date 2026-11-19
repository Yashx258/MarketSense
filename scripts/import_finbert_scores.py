"""Import and verify the Colab scores with python -m scripts.import_finbert_scores."""
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from marketsense.db.connection import get_engine, insert_ignore_duplicates
from marketsense.db.models import NewsArticle, SentimentFeature
from marketsense.features.model_b import VARIANTS
from marketsense.sentiment.processor import MODEL_NAME, SentimentScores
from scripts.export_news_for_finbert import EXPECTED_ROWS, OUTPUT

SCORED = OUTPUT.with_name("finbert_scored_8149.parquet")
MIGRATION = Path(__file__).resolve().parents[1] / "sql/migrations/004_finbert_variants.sql"
SCORE_COLUMNS = ["positive_score", "neutral_score", "negative_score"]
KEY = ["news_id", "model_name", "text_variant"]


def validate_scores(frame: pd.DataFrame) -> None:
    ids = frame["news_id"]
    if (len(frame) != EXPECTED_ROWS or ids.isna().any() or not ids.is_unique
            or not pd.api.types.is_integer_dtype(ids)):
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows with unique integer news IDs.")
    for prefix in VARIANTS.values():
        pos, neu, neg, net = frame[
            [f"{prefix}_{suffix}" for suffix in ("pos", "neu", "neg", "net_sentiment")]
        ].to_numpy(dtype=float).T
        probabilities = np.array([pos, neu, neg])
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=0), 1, atol=1e-6)
                or not np.allclose(net, pos - neg, atol=1e-6, rtol=0)):
            raise ValueError(f"Invalid probabilities or net sentiment for {prefix}.")


def persist_scores(session: Session, frame: pd.DataFrame) -> int:
    articles = pd.read_sql(
        select(NewsArticle.news_id, NewsArticle.stock_id, NewsArticle.source, NewsArticle.published_at)
        .where(NewsArticle.relevance_status == "include").with_for_update(read=True),
        session.connection(),
    ).set_index("news_id").sort_index()
    frame = frame.set_index("news_id").sort_index()
    if (not articles.index.equals(frame.index) or
            not (articles["source"] == frame["source"]).all() or
            not (pd.to_datetime(articles["published_at"], utc=True)
                 == pd.to_datetime(frame["published_at_utc"], utc=True)).all()):
        raise ValueError("Scored file does not match the live included news cohort.")

    rows = []
    for variant, prefix in VARIANTS.items():
        values = frame[[f"{prefix}_{suffix}" for suffix in ("pos", "neu", "neg")]]
        for news_id, pos, neu, neg in values.itertuples():
            stock_id = articles.at[news_id, "stock_id"]
            score = SentimentScores(pos, neu, neg)
            rows.append(dict(news_id=int(news_id), stock_id=None if pd.isna(stock_id) else int(stock_id),
                             model_name=MODEL_NAME, text_variant=variant, sentiment_label=score.label,
                             **dict(zip(SCORE_COLUMNS, (pos, neu, neg), strict=True))))

    inserted = insert_ignore_duplicates(session, SentimentFeature, rows, KEY)
    stored = pd.read_sql(
        select(*(getattr(SentimentFeature, name) for name in KEY + SCORE_COLUMNS + ["sentiment_label"]))
        .where(SentimentFeature.model_name == MODEL_NAME, SentimentFeature.text_variant.in_(VARIANTS)),
        session.connection(),
    ).set_index(KEY).sort_index()
    expected = pd.DataFrame(rows).set_index(KEY).sort_index()
    # NUMERIC(6,5) rounds the persisted probabilities. Net sentiment is derived from them.
    if (not stored.index.equals(expected.index) or stored.isna().any().any()
            or not np.allclose(stored[SCORE_COLUMNS], expected[SCORE_COLUMNS], atol=5.00001e-6, rtol=0)
            or not (stored["sentiment_label"] == expected["sentiment_label"]).all()):
        raise RuntimeError("Persisted score rows do not match the scored file at database precision.")
    return inserted


def main() -> None:
    frame = pd.read_parquet(SCORED)
    validate_scores(frame)
    with get_engine().begin() as connection:
        applied = connection.scalar(text("""
            SELECT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conrelid = 'sentiment_features'::regclass AND conname = 'uq_sent_news_model_variant'
            )
        """))
        if not applied:
            connection.exec_driver_sql(MIGRATION.read_text(encoding="utf-8"))
        with Session(bind=connection) as session:
            inserted = persist_scores(session, frame)
    print(f"Verified {EXPECTED_ROWS:,} rows per variant in PostgreSQL "
          f"({inserted:,} newly inserted, zero null score fields).")


if __name__ == "__main__":
    main()
