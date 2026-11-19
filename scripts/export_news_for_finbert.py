"""Export the included PostgreSQL news cohort for FinBERT scoring.

Run from the repository root with `python -m scripts.export_news_for_finbert`.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pyarrow import parquet
from sqlalchemy import text

from marketsense.db.connection import get_engine
from marketsense.sentiment.text_hygiene import clean_article_text

EXPECTED_ROWS = 8_149
OUTPUT = Path(__file__).resolve().parents[1] / "data/finbert/finbert_input_8149.parquet"

QUERY = text("""
    SELECT n.news_id,
           COALESCE(NULLIF(n.source_record_id, ''), n.news_id::text) AS article_id,
           n.source,
           n.published_at AS published_at_utc,
           COALESCE(n.published_date, (n.published_at AT TIME ZONE 'Asia/Karachi')::date)
               AS published_date_pkt,
           COALESCE(NULLIF(n.category, ''), n.relevance_topic) AS relevance_bucket,
           ARRAY(
               SELECT s.symbol
               FROM news_article_stocks AS ns
               JOIN stocks AS s ON s.stock_id = ns.stock_id
               WHERE ns.news_id = n.news_id
               UNION
               SELECT s.symbol FROM stocks AS s WHERE s.stock_id = n.stock_id
               ORDER BY symbol
           ) AS matched_tickers,
           n.headline,
           n.article_text
    FROM news_articles AS n
    WHERE n.relevance_status = 'include'
    ORDER BY n.news_id
""")


def clean_scoring_text(headline: str, body: str | None) -> tuple[str, str]:
    headline = " ".join(clean_article_text(headline.strip()).split())
    body = clean_article_text((body or "").strip())
    # Stored HTML paragraphs are newline-separated. A flattened body has one
    # paragraph available. Clean the whole body first to remove multi-line footers.
    for paragraph in body.splitlines():
        lead = " ".join(clean_article_text(paragraph.strip()).split())
        if lead:
            return headline, " ".join(filter(None, (headline, lead)))
    return headline, headline


def main() -> None:
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        articles = pd.read_sql_query(QUERY, connection)

    if len(articles) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} included articles, found {len(articles):,}.")
    articles["published_at_utc"] = pd.to_datetime(articles["published_at_utc"], utc=True)
    bodies = articles.pop("article_text").fillna("")
    texts = [
        clean_scoring_text(headline, body)
        for headline, body in zip(articles.pop("headline"), bodies, strict=True)
    ]
    articles["text_headline"], articles["text_headline_lead"] = zip(*texts, strict=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT.parent) as directory:
        staged = Path(directory) / OUTPUT.name
        articles.to_parquet(staged, engine="pyarrow", index=False)
        if parquet.read_metadata(staged).num_rows != EXPECTED_ROWS:
            raise RuntimeError("Unexpected Parquet row count.")
        staged.replace(OUTPUT)
    print(f"Verified {len(articles):,} rows in {OUTPUT}")


if __name__ == "__main__":
    main()
