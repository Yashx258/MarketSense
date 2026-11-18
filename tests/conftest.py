"""Shared fixtures. SQLite in memory by default; set TEST_DATABASE_URL=postgresql+psycopg2://... to run on PostgreSQL."""
import os
from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from marketsense.db.connection import get_engine
from marketsense.db.models import Base, Stock

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture()
def news_session():
    engine = get_engine(os.getenv("TEST_DATABASE_URL", "sqlite://"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all([Stock(symbol="LUCK", company_name="Lucky Cement Limited", is_kse100=True),
                   Stock(symbol="HBL", company_name="Habib Bank Limited", is_kse100=True),
                   Stock(symbol="ENGRO", company_name="Engro Corporation Limited", is_kse100=True),
                   Stock(symbol="OGDC", company_name="Oil & Gas Development Company Limited", is_kse100=True)])
        s.commit()
        yield s


@pytest.fixture()
def dawn_xml() -> bytes:
    with open(os.path.join(FIXTURES, "dawn_business_sample.xml"), "rb") as f:
        return f.read()


def utc(*a) -> datetime:
    return datetime(*a, tzinfo=timezone.utc)


def as_utc(dt):
    """Normalise a value read back from the DB (SQLite returns naive UTC, PostgreSQL returns aware)."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
