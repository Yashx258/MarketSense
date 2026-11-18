"""Initialize the schema and apply migrations. Usage: python -m scripts.init_db."""
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from marketsense.collectors.news_store import backfill_relevance
from marketsense.config import setup_logging
from marketsense.db.connection import get_engine

def main() -> None:
    setup_logging()
    sql_dir = Path(__file__).resolve().parent.parent / "sql"
    migrations = sorted((sql_dir / "migrations").glob("*.sql"))
    with get_engine().begin() as conn:
        conn.exec_driver_sql((sql_dir / "schema.sql").read_text(encoding="utf-8"))
        for migration in migrations:
            conn.exec_driver_sql(migration.read_text(encoding="utf-8"))
        with Session(bind=conn) as session:
            annotated = backfill_relevance(session)
            session.commit()
    logging.getLogger("init_db").info("Schema applied. Annotated %d legacy news articles.", annotated)

if __name__ == "__main__":
    main()
