"""Import historical news from CSV or Parquet files into PostgreSQL.

  python -m scripts.import_news --inspect FILE [--profile generic] [--tz Asia/Karachi]        # look BEFORE importing (no DB)
  python -m scripts.import_news FILE --profile generic --source-name kaggle_psx_news_macro --doc-type news \\
        --tz Asia/Karachi --map headline=Title --map published=Date                           # any other dataset

Profiles: psx_news_primary_candidates and generic (auto-detect + --map). Article text is only stored with
--store-text (the same opt-in as NEWS_STORE_ARTICLE_TEXT). Reports + reject CSVs go to reports/imports/.
Exit codes: 0 ok | 1 error | 2 refused (bad schema/arguments)
"""
from __future__ import annotations

import argparse
import json
import sys

from marketsense.config import Settings, setup_logging
from marketsense.db.connection import session_scope
from marketsense.importers.pipeline import import_file, inspect_file
from marketsense.importers.profiles import PROFILES
from marketsense.importers.reader import ImportSchemaError


def _kv(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--map expects field=Column, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import historical news from CSV or Parquet.")
    ap.add_argument("files", nargs="+", help="one or more .csv/.tsv/.parquet files")
    ap.add_argument("--profile", default="generic", choices=sorted(PROFILES))
    ap.add_argument("--inspect", action="store_true", help="describe the file(s); no database")
    ap.add_argument("--dry-run", action="store_true", help="run everything, then roll back")
    ap.add_argument("--source-name", help="value stored in news_articles.source (required for generic)")
    ap.add_argument("--dataset", help="provenance label for import_batches (default: profile name)")
    ap.add_argument("--doc-type", choices=["news", "other"])
    ap.add_argument("--tz", help="IANA timezone of naive times in the file, e.g. Asia/Karachi (default: none = drop times)")
    ap.add_argument("--dayfirst", choices=["yes", "no"], help="order of numeric dates like 03/04/2026 (no default: ambiguous rows are rejected)")
    ap.add_argument("--map", action="append", help="override a column, e.g. --map headline=Title (repeatable)")
    ap.add_argument("--limit", type=int, help="only the first N rows (smoke test)")
    ap.add_argument("--store-text", action="store_true", help="also store article text (opt-in; mind the source's terms)")
    ap.add_argument("--force", action="store_true", help="re-import a file whose sha256 was already imported")
    ap.add_argument("--report-dir", default=None)
    a = ap.parse_args(argv)
    setup_logging()
    settings = Settings()
    profile = PROFILES[a.profile]
    dayfirst = None if a.dayfirst is None else (a.dayfirst == "yes")
    overrides = _kv(a.map)

    try:
        if a.inspect:
            for f in a.files:
                print(json.dumps(inspect_file(f, profile, overrides=overrides, tz=a.tz, dayfirst=dayfirst), indent=2, default=str))
            return 0
        rc = 0
        for f in a.files:
            with session_scope() as s:
                rep = import_file(s, f, profile, source_name=a.source_name, dataset=a.dataset, doc_type=a.doc_type, tz=a.tz,
                                  dayfirst=dayfirst, overrides=overrides, dry_run=a.dry_run, limit=a.limit,
                                  store_text=a.store_text or settings.news_store_article_text, force=a.force,
                                  report_dir=a.report_dir or settings.import_report_dir)
            print(rep.summary())
            print()
        return rc
    except ImportSchemaError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                                   # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
