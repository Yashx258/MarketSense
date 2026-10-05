"""Central configuration. All secrets come from environment variables / .env."""
from __future__ import annotations
import os
from dataclasses import dataclass
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name} (see .env.example)")
    return val


def database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    user, pw = _require("DB_USER"), _require("DB_PASSWORD")
    host, port = os.getenv("DB_HOST", "localhost"), os.getenv("DB_PORT", "5432")
    name = _require("DB_NAME")
    return f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(pw)}@{host}:{port}/{name}"


@dataclass(frozen=True)
class Settings:
    universe_source: str = os.getenv("STOCK_UNIVERSE_SOURCE", "psxdata")
    universe_index: str = os.getenv("STOCK_UNIVERSE_INDEX", "KSE100")
    universe_file: str = os.getenv("STOCK_UNIVERSE_FILE", "config/stock_universe.csv")
    price_start_date: str = os.getenv("PRICE_START_DATE", "2020-01-01")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    news_enabled: bool = os.getenv("NEWS_COLLECTION_ENABLED", "false").lower() == "true"
    yahoo_suffix: str = os.getenv("YAHOO_SYMBOL_SUFFIX", ".KA")


def setup_logging() -> None:
    import logging
    logging.basicConfig(
        level=getattr(logging, Settings().log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
