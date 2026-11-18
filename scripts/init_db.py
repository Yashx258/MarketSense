"""Initialize the schema and apply migrations. Usage: python -m scripts.init_db."""
import logging
from pathlib import Path

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
    logging.getLogger("init_db").info("Schema applied.")

if __name__ == "__main__":
    main()
