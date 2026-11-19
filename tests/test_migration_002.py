"""PostgreSQL only: apply 002 to a database that is on the CURRENT GitHub schema, with data in it."""
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from scripts import init_db

URL = os.getenv("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL.startswith("postgresql") or "test" not in URL.rsplit("/", 1)[-1],
                                reason="needs TEST_DATABASE_URL=postgresql+psycopg2://.../<db with 'test' in its name> (the schema is dropped!)")
ROOT = Path(__file__).resolve().parent.parent
V1 = (ROOT / "tests/fixtures/schema_v1_before_002.sql").read_text()
NEW = (ROOT / "sql/schema.sql").read_text()
MIG = (ROOT / "sql/migrations/002_news_imports.sql").read_text()


@pytest.fixture()
def eng():
    e = create_engine(URL, future=True)
    with e.begin() as c:
        c.exec_driver_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        c.exec_driver_sql(V1)
        c.exec_driver_sql("""
            INSERT INTO stocks(symbol, is_kse100) VALUES ('LUCK', true), ('HBL', true);
            INSERT INTO news_articles(stock_id, headline, source, url, published_at) VALUES
                (1, 'legacy dated', 'dawn_business', 'https://www.dawn.com/news/1', '2026-02-05 20:30:00+00'),
                (NULL, 'legacy undated', 'dawn_business', 'https://www.dawn.com/news/2', NULL);
            INSERT INTO news_article_stocks(news_id, stock_id, matched_by) VALUES (1, 1, 'symbol');
            INSERT INTO sentiment_features(news_id, stock_id, positive_score, neutral_score, negative_score, sentiment_label, model_name)
                VALUES (1, 1, .7, .2, .1, 'positive', 'ProsusAI/finbert'), (2, NULL, .3, .4, .3, 'neutral', 'ProsusAI/finbert');""")
    yield e
    e.dispose()


def q(e, sql):
    with e.connect() as c:
        return c.execute(text(sql)).all()


@pytest.mark.parametrize("script", ["schema", "migration"])
def test_migration_preserves_data_and_derives_precision(eng, script):
    sql = NEW if script == "schema" else MIG
    with eng.begin() as c:
        c.exec_driver_sql(sql)
        c.exec_driver_sql(sql)                                                  # idempotent: a second run changes nothing and does not fail
    rows = q(eng, "SELECT headline, doc_type, time_precision, published_date, category, import_batch_id FROM news_articles ORDER BY news_id")
    assert [tuple(r)[:3] for r in rows] == [("legacy dated", "news", "exact"), ("legacy undated", "news", "none")]
    assert str(rows[0][3]) == "2026-02-06"          # 20:30Z on the 5th is already the 6th in Pakistan: DERIVED from the stored timestamp
    assert rows[1][3] is None and rows[0][4:] == (None, None)              # nothing invented for the undated row
    assert q(eng, "SELECT count(*) FROM stocks")[0][0] == 2 and q(eng, "SELECT count(*) FROM sentiment_features")[0][0] == 2
    assert q(eng, "SELECT matched_by FROM news_article_stocks") == [("symbol",)]


def test_new_constraints_and_tables_work_after_migration(eng):
    with eng.begin() as c:
        c.exec_driver_sql(NEW)
        c.exec_driver_sql("INSERT INTO import_batches(source_name, file_name, file_sha256, status) VALUES ('x','f.csv', repeat('a',64), 'completed')")
        c.exec_driver_sql("INSERT INTO news_article_stocks(news_id, stock_id, matched_by) VALUES (1, 2, 'alias')")      # new evidence value
    for bad in ["INSERT INTO news_article_stocks(news_id, stock_id, matched_by) VALUES (2, 1, 'guess')",
                "UPDATE news_articles SET doc_type = 'rumour' WHERE news_id = 1",
                "UPDATE news_articles SET time_precision = 'approx' WHERE news_id = 1",
                "INSERT INTO import_batches(source_name, file_name, file_sha256, status) VALUES ('x','g.csv', repeat('a',64), 'completed')"]:
        with pytest.raises(IntegrityError):
            with eng.begin() as c:
                c.exec_driver_sql(bad)
    with eng.begin() as c:                                                      # a SUPERSEDED duplicate is allowed (forced re-import)
        c.exec_driver_sql("INSERT INTO import_batches(source_name, file_name, file_sha256, status) VALUES ('x','g.csv', repeat('a',64), 'superseded')")


def test_schema_sql_and_orm_models_agree_on_news_columns(eng):
    from marketsense.db.models import ImportBatch, NewsArticle
    with eng.begin() as c:
        c.exec_driver_sql(NEW)
    insp = inspect(eng)
    for model in (NewsArticle, ImportBatch):
        db_cols = {c["name"] for c in insp.get_columns(model.__tablename__)}
        assert db_cols == {c.name for c in model.__table__.columns}, model.__tablename__


def test_fresh_init_matches_upgrade_with_existing_data(eng, monkeypatch):
    before = {
        table: q(eng, f"SELECT * FROM {table} ORDER BY 1")
        for table in ("stocks", "news_article_stocks", "sentiment_features")
    }
    with eng.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA fresh_init")
    fresh = create_engine(URL, connect_args={"options": "-csearch_path=fresh_init"})
    try:
        for engine in (fresh, eng, eng):
            monkeypatch.setattr(init_db, "get_engine", lambda: engine)
            init_db.main()

        def signature(engine):
            inspector = inspect(engine)
            return {
                table: {
                    "columns": [(c["name"], str(c["type"]), c["nullable"], c["default"])
                                for c in inspector.get_columns(table)],
                    "checks": inspector.get_check_constraints(table),
                    "foreign_keys": inspector.get_foreign_keys(table),
                    "primary_key": inspector.get_pk_constraint(table),
                    "unique": inspector.get_unique_constraints(table),
                    "indexes": inspector.get_indexes(table),
                }
                for table in inspector.get_table_names()
            }

        assert signature(fresh) == signature(eng)
        assert q(eng, "SELECT relevance_status, relevance_version FROM news_articles ORDER BY news_id") == [
            ("reject", "psx-rules-v1"), ("reject", "psx-rules-v1"),
        ]
        for table in ("stocks", "news_article_stocks"):
            assert q(eng, f"SELECT * FROM {table} ORDER BY 1") == before[table]
        actual = q(eng, "SELECT * FROM sentiment_features ORDER BY sentiment_id")
        assert actual == [(*row, "legacy_full_text") for row in before["sentiment_features"]]
        assert q(eng, "SELECT headline, time_precision, published_date::text FROM news_articles ORDER BY news_id") == [
            ("legacy dated", "exact", "2026-02-06"),
            ("legacy undated", "none", None),
        ]
    finally:
        fresh.dispose()
        with eng.begin() as conn:
            conn.exec_driver_sql("DROP SCHEMA fresh_init CASCADE")


def test_migration_003_backfills_review_and_preserves_annotations(eng):
    migration = (ROOT / "sql/migrations/003_news_relevance.sql").read_text()
    with eng.begin() as conn:
        conn.exec_driver_sql(MIG)
        conn.exec_driver_sql(migration)
    assert q(eng, "SELECT relevance_status, relevance_version FROM news_articles ORDER BY news_id") == [
        ("review", "unclassified"), ("review", "unclassified"),
    ]
    with pytest.raises(IntegrityError):
        with eng.begin() as conn:
            conn.exec_driver_sql("UPDATE news_articles SET relevance_status = 'unknown' WHERE news_id = 1")
    with eng.begin() as conn:
        conn.exec_driver_sql("UPDATE news_articles SET relevance_status = 'include', relevance_version = 'manual' WHERE news_id = 1")
        conn.exec_driver_sql(migration)
    assert q(eng, "SELECT relevance_status, relevance_version FROM news_articles ORDER BY news_id") == [
        ("include", "manual"), ("review", "unclassified"),
    ]


def test_migration_004_preserves_legacy_and_allows_both_colab_variants(eng):
    migration = (ROOT / "sql/migrations/004_finbert_variants.sql").read_text(encoding="utf-8")
    insert = """
        INSERT INTO sentiment_features(
            news_id, positive_score, neutral_score, negative_score, sentiment_label, model_name, text_variant
        ) VALUES (1, .7, .2, .1, 'positive', 'ProsusAI/finbert', %s)
    """
    with eng.begin() as conn:
        conn.exec_driver_sql(migration)
        for variant in ("headline", "headline_lead"):
            conn.exec_driver_sql(insert, (variant,))
        conn.exec_driver_sql(migration)
    assert q(eng, "SELECT text_variant FROM sentiment_features WHERE news_id = 1 ORDER BY 1") == [
        ("headline",), ("headline_lead",), ("legacy_full_text",),
    ]
    with pytest.raises(IntegrityError), eng.begin() as conn:
        conn.exec_driver_sql(insert, ("headline",))
