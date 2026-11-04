"""Data-quality and coverage report for news already in the database. Read-only.

  python -m scripts.news_data_report [--months 12]
Per source/doc_type: rows, date range, timestamp precision, stock-mapping coverage; plus a duplicate-URL check,
monthly volume and import batches. Run it BEFORE any sentiment scoring.
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import String, case, cast, exists, func, select

from marketsense.config import setup_logging
from marketsense.db.connection import session_scope
from marketsense.db.models import ImportBatch, NewsArticle as N, NewsArticleStock

_pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "  n/a"          # noqa: E731


def _count_if(cond):
    return func.sum(case((cond, 1), else_=0))


def report(session, months: int = 12) -> str:
    L = [f"articles total: {session.scalar(select(func.count()).select_from(N))}"]
    linked = exists().where(NewsArticleStock.news_id == N.news_id)
    q = (select(N.source, N.doc_type, func.count(), func.min(N.published_date), func.max(N.published_date),
                _count_if(N.time_precision == "exact"), _count_if(N.time_precision == "date"),
                _count_if(N.time_precision == "none"), _count_if(linked))
         .group_by(N.source, N.doc_type).order_by(N.source))
    L.append(f"{'source':<22}{'type':<13}{'rows':>8}  {'first':<11}{'last':<11}{'exact':>7}{'date-only':>10}{'no-time':>8}  stock-linked")
    for src, dt, n, d0, d1, ex, do, no, ln in session.execute(q).all():
        ln = int(ln or 0)
        L.append(f"{src:<22}{dt:<13}{n:>8}  {str(d0):<11}{str(d1):<11}{int(ex or 0):>7}{int(do or 0):>10}{int(no or 0):>8}  {ln} ({_pct(ln, n)})")
    dup = session.execute(select(N.url).where(N.url.is_not(None)).group_by(N.url).having(func.count() > 1)).all()
    L.append(f"duplicate URLs: {len(dup)} (expect 0)")
    month = func.substr(cast(N.published_date, String), 1, 7)
    rows = session.execute(select(month, N.doc_type, func.count()).where(N.published_date.is_not(None))
                           .group_by(month, N.doc_type).order_by(month.desc()).limit(months * 3)).all()
    if rows:
        L.append("monthly volume (latest first): " + ", ".join(f"{m}/{t[:3]}={c}" for m, t, c in rows))
    for b in session.scalars(select(ImportBatch).order_by(ImportBatch.batch_id)):
        L.append(f"import batch {b.batch_id}: {b.source_name} file={b.file_name} status={b.status} read={b.rows_read} "
                 f"inserted={b.rows_inserted} dup={b.rows_duplicate} rejected={b.rows_rejected}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    a = ap.parse_args(argv)
    setup_logging()
    with session_scope() as s:
        print(report(s, a.months))
    return 0


if __name__ == "__main__":
    sys.exit(main())
