"""Measure include precision and recall against curated news labels."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from sqlalchemy import select

from marketsense.collectors.base import NewsItem
from marketsense.collectors.news_store import StockIndex, annotate_news, build_stock_index
from marketsense.db.connection import session_scope
from marketsense.db.models import ImportBatch, NewsArticle

LABELS = {
    "include": True, "true": True, "1": True, "yes": True,
    "review": False, "reject": False, "false": False, "0": False, "no": False,
}


def calibrate(path: Path, index: StockIndex, headline_column: str, label_column: str, category_column: str | None):
    counts = Counter()
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        required = {headline_column, label_column}
        if category_column:
            required.add(category_column)
        missing = required.difference(rows.fieldnames or ())
        if missing:
            raise ValueError(f"missing CSV column(s): {', '.join(sorted(missing))}")
        for row in rows:
            gold = LABELS.get((row[label_column] or "").strip().casefold())
            headline = (row[headline_column] or "").strip()
            if gold is None or not headline:
                raise ValueError(f"line {rows.line_num} has an invalid label or empty headline")
            item = NewsItem(headline, "calibration", category=row.get(category_column) if category_column else None)
            predicted = annotate_news(item, index).status == "include"
            counts[gold, predicted] += 1
    if not counts:
        raise ValueError("no labeled articles found")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path)
    parser.add_argument("--positive-batch", metavar="SOURCE_NAME",
                        help="evaluate recall on a completed import containing curated positives")
    parser.add_argument("--headline-column", default="title")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--category-column", help="optional original publisher category column")
    parser.add_argument("--expected-rows", type=int, help="require this many labeled rows")
    args = parser.parse_args(argv)
    if bool(args.csv) == bool(args.positive_batch):
        parser.error("provide either a labeled CSV or --positive-batch")
    if args.label_column in (args.headline_column, args.category_column):
        parser.error("the gold label column must be distinct from the classifier inputs")
    try:
        with session_scope() as session:
            conn = session.connection()
            if conn.dialect.name == "postgresql":
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            index = build_stock_index(session)
            if args.positive_batch:
                rows = session.execute(
                    select(NewsArticle.headline, NewsArticle.source)
                    .join(ImportBatch, ImportBatch.batch_id == NewsArticle.import_batch_id)
                    .where(ImportBatch.source_name == args.positive_batch, ImportBatch.status == "completed")
                )
                counts = Counter(
                    (True, annotate_news(NewsItem(headline, source), index).status == "include")
                    for headline, source in rows
                )
            else:
                counts = calibrate(args.csv, index, args.headline_column, args.label_column, args.category_column)
        total = sum(counts.values())
        if not total:
            raise ValueError("no labeled articles found")
        if args.expected_rows is not None and total != args.expected_rows:
            raise ValueError(f"expected {args.expected_rows} labels, found {total}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    tp, fp, fn, tn = (counts[True, True], counts[False, True], counts[True, False], counts[False, False])
    precision = f"{tp / (tp + fp):.2%}" if (tp + fp and fp + tn) else "n/a"
    recall = f"{tp / (tp + fn):.2%}" if tp + fn else "n/a"
    print(f"labeled={total} TP={tp} FP={fp} FN={fn} TN={tn} precision={precision} recall={recall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
