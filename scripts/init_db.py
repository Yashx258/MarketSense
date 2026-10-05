"""Create all tables from sql/schema.sql.   Usage: python -m scripts.init_db"""
from pathlib import Path
from sqlalchemy import text
from marketsense.config import setup_logging
from marketsense.db.connection import get_engine
import logging

def main() -> None:
    setup_logging()
    sql = (Path(__file__).resolve().parent.parent / "sql" / "schema.sql").read_text()
    with get_engine().begin() as conn:
        conn.exec_driver_sql(sql)          # psycopg2 executes the multi-statement script (incl. $$ function body)
    logging.getLogger("init_db").info("Schema applied.")

if __name__ == "__main__":
    main()
