"""Engine/session factory + dialect-aware idempotent insert helpers."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from marketsense.config import database_url

_engine: Engine | None = None


def get_engine(url: str | None = None) -> Engine:
    """Return the shared engine (built from env vars), or a fresh engine for an explicit url (tests)."""
    global _engine
    if url is not None:
        return create_engine(url, future=True)
    if _engine is None:
        _engine = create_engine(database_url(), pool_pre_ping=True, future=True)
    return _engine


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    engine = engine or get_engine()
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _dialect_insert(session: Session):
    name = session.get_bind().dialect.name
    if name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif name == "sqlite":                    # used only by unit tests
        from sqlalchemy.dialects.sqlite import insert
    else:
        raise NotImplementedError(f"Unsupported dialect: {name}")
    return insert


def insert_ignore_duplicates(session: Session, model, rows: Sequence[dict], conflict_cols: Iterable[str],
                             chunk: int = 1000) -> int:
    """INSERT ... ON CONFLICT DO NOTHING. Returns number of rows actually inserted."""
    inserted = 0
    for i in range(0, len(rows), chunk):
        stmt = (_dialect_insert(session)(model).values(list(rows[i:i + chunk]))
                .on_conflict_do_nothing(index_elements=list(conflict_cols)))
        inserted += max(session.execute(stmt).rowcount or 0, 0)
    return inserted


def upsert(session: Session, model, rows: Sequence[dict], conflict_cols: Iterable[str],
           update_cols: Iterable[str], chunk: int = 1000) -> int:
    """INSERT ... ON CONFLICT DO UPDATE (for *derived* tables that are safe to recompute)."""
    total = 0
    for i in range(0, len(rows), chunk):
        ins = _dialect_insert(session)(model).values(list(rows[i:i + chunk]))
        stmt = ins.on_conflict_do_update(index_elements=list(conflict_cols),
                                         set_={c: getattr(ins.excluded, c) for c in update_cols})
        total += max(session.execute(stmt).rowcount or 0, 0)
    return total
