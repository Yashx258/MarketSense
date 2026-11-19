"""Extend the exact Model A artifact with news features: python -m scripts.build_model_b_dataset."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sqlalchemy import text

from marketsense.db.connection import get_engine
from marketsense.features.model_a import FEATURES, HORIZON, NON_FEATURE_COLS, build_calendar
from marketsense.features.model_b import WINDOWS, add_sentiment_features
from marketsense.sentiment.processor import MODEL_NAME
from scripts.build_model_a_dataset import drop_possible_intraday_rows

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/model_a/model_a_dataset.parquet"
OUT = ROOT / "data/model_b"

PRICE_DATES = text("""
    SELECT p.stock_id, p.trading_date, p.created_at
    FROM price_history p JOIN stocks s ON p.stock_id = s.stock_id
    WHERE p.trading_date <= :last_date AND (NOT :only_kse100 OR s.is_kse100)
""")
SENTIMENT = text("""
    SELECT n.news_id, n.published_at, f.text_variant,
           f.positive_score AS pos, f.negative_score AS neg, f.neutral_score AS neu,
           ARRAY(
               SELECT s.symbol FROM news_article_stocks ns JOIN stocks s USING (stock_id)
               WHERE ns.news_id = n.news_id
               UNION
               SELECT s.symbol FROM stocks s WHERE s.stock_id = n.stock_id
               ORDER BY symbol
           ) AS matched_tickers
    FROM sentiment_features f JOIN news_articles n USING (news_id)
    WHERE f.model_name = :model_name AND f.text_variant IN ('headline', 'headline_lead')
      AND n.relevance_status = 'include'
      AND n.time_precision = 'exact' AND n.published_at IS NOT NULL
""")


def verified_calendar(base, prices, filters):
    prices, _ = drop_possible_intraday_rows(prices, filters["intraday_cutoff_pkt"])
    prices["trading_date"] = pd.to_datetime(prices["trading_date"])
    calendar, _ = build_calendar(prices, filters["min_session_coverage"])
    positions = calendar.get_indexer(base["trading_date"])
    if ((positions < 0).any() or (positions + HORIZON >= len(calendar)).any()
            or not (calendar.take(positions + HORIZON) == base["label_end_date"]).all()):
        raise RuntimeError("PostgreSQL session calendar disagrees with the Model A label horizon.")
    return calendar


def main():
    base = pd.read_parquet(BASE)
    meta = json.loads(BASE.with_name("model_a_meta.json").read_text(encoding="utf-8"))
    if (len(base) != 132_899 or base["symbol"].nunique() != 94
            or base.duplicated(["symbol", "trading_date"]).any()
            or list(base.columns) != NON_FEATURE_COLS + FEATURES
            or meta["features"] != FEATURES or meta["horizon_sessions"] != HORIZON):
        raise RuntimeError("Unexpected Model A cohort or schema.")
    filters = meta["filters"]
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        prices = pd.read_sql(PRICE_DATES, connection, params={
            "last_date": base["label_end_date"].max().date(), "only_kse100": filters["only_kse100"],
        })
        articles = pd.read_sql(SENTIMENT, connection, params={"model_name": MODEL_NAME})

    calendar = verified_calendar(base, prices, filters)
    for column in ("pos", "neg", "neu"):
        articles[column] = articles[column].astype(float)
    if set(articles["text_variant"]) != {"headline", "headline_lead"}:
        raise RuntimeError("Both FinBERT variants must be imported before building Model B.")
    result, groups = add_sentiment_features(base, articles, calendar)
    meta.update({
        "model_a_sha256": hashlib.sha256(BASE.read_bytes()).hexdigest(),
        "sentiment_model": MODEL_NAME,
        "sentiment_features": groups,
        "sentiment_windows_sessions": list(WINDOWS),
        "prediction_cutoff": "17:00 Asia/Karachi on trading_date",
        "sentiment_window": "(previous session cutoff, current session cutoff], rolled by market session",
        "sentiment_market_scope": "All scored included articles, counted once per article",
        "sentiment_company_scope": "All matched tickers for each article, counted once per ticker",
        "sentiment_availability": "Exact publication timestamps only",
        "exact_timestamp_score_rows_by_variant": articles["text_variant"].value_counts().to_dict(),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUT) as temporary:
        staged = Path(temporary)
        dataset = staged / "model_b_dataset.parquet"
        metadata = staged / "model_b_meta.json"
        result.to_parquet(dataset, index=False)
        metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        pd.testing.assert_frame_equal(pd.read_parquet(dataset, columns=list(base.columns)), base)
        dataset.replace(OUT / dataset.name)
        metadata.replace(OUT / metadata.name)
    print(f"Verified {len(result):,} Model A-aligned rows across {base['symbol'].nunique()} stocks "
          f"with {sum(map(len, groups.values()))} sentiment features in {OUT}")


if __name__ == "__main__":
    main()
