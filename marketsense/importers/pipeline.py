"""Historical import pipeline: inspect -> map columns -> normalise -> store (dedup, company mapping) -> report.

Guarantees: nothing is fabricated (a missing/ambiguous time stays missing, see timeparse.py); every rejected row has a
reason; each file is recorded in `import_batches` (sha256, counts, full report); re-importing an identical file is a
no-op unless --force; --dry-run runs the whole pipeline inside a transaction that is rolled back, so its duplicate and
company-mapping numbers are exact but nothing is written.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from marketsense.collectors.base import NewsItem
from marketsense.collectors.news_store import StoreReport, build_stock_index, store_news
from marketsense.db.models import ImportBatch
from marketsense.importers.profiles import GENERIC_COLUMNS, PROFILES, ImportProfile
from marketsense.importers.reader import ImportSchemaError, read_table
from marketsense.timeparse import ParsedTime, is_missing, parse_datetime

log = logging.getLogger(__name__)
CHUNK = 5000
_SYMBOL_SPLIT = re.compile(r"[,;/|]")


def _norm_col(c) -> str:
    return re.sub(r"[\s_\-]+", " ", str(c).strip().lower())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_columns(columns: list[str], profile: ImportProfile, overrides: dict[str, str] | None = None):
    """-> (mapping canonical->actual column | None, ignored_present, ambiguities). Raises ImportSchemaError if required missing."""
    by_norm = {_norm_col(c): c for c in columns}
    mapping: dict[str, str | None] = {}
    ambiguous: dict[str, list[str]] = {}
    for canon, candidates in (profile.columns or GENERIC_COLUMNS).items():
        present = [by_norm[_norm_col(c)] for c in candidates if _norm_col(c) in by_norm]
        mapping[canon] = present[0] if present else None
        if len(present) > 1:
            ambiguous[canon] = present
    for canon, actual in (overrides or {}).items():
        if canon not in GENERIC_COLUMNS:
            raise ImportSchemaError(f"--map: unknown field '{canon}' (valid: {', '.join(GENERIC_COLUMNS)})")
        if _norm_col(actual) not in by_norm:
            raise ImportSchemaError(f"--map {canon}={actual}: no such column. Columns are: {columns}")
        mapping[canon] = by_norm[_norm_col(actual)]
        ambiguous.pop(canon, None)
    if not mapping.get("headline") or not (mapping.get("published") or mapping.get("url")):
        raise ImportSchemaError(
            "cannot import: need a headline column and a date or url column. Detected columns: "
            f"{columns}. Use --map headline=<col> --map published=<col> (run --inspect first).")
    ignored = [c for c in columns if _norm_col(c) in {_norm_col(i) for i in profile.ignore}]
    return mapping, ignored, ambiguous


@dataclass
class ImportReport:
    file: str = ""
    sha256: str = ""
    profile: str = ""
    dataset: str = ""
    source: str = ""
    dry_run: bool = False
    already_imported: bool = False
    batch_id: int | None = None
    encoding: str | None = None
    rows_read: int = 0
    rows_parsed: int = 0                                    # became NewsItems
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    rejected_samples: list[dict] = field(default_factory=list)
    malformed_lines_skipped: int = 0
    columns_used: dict[str, str | None] = field(default_factory=dict)
    columns_ignored: list[str] = field(default_factory=list)
    ambiguous_columns: dict[str, list[str]] = field(default_factory=dict)
    notes: dict[str, int] = field(default_factory=dict)
    precision: dict[str, int] = field(default_factory=dict)
    date_min: str | None = None
    date_max: str | None = None
    rows_by_year: dict[str, int] = field(default_factory=dict)
    categories: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    store: dict = field(default_factory=dict)
    rows_with_text: int = 0
    text_stored: bool = False

    @property
    def rows_rejected(self) -> int:
        return sum(self.rejected_by_reason.values())

    def summary(self) -> str:
        s = self.store
        L = [f"{'DRY RUN ' if self.dry_run else ''}import: {self.file}  [profile={self.profile} dataset={self.dataset} source={self.source}]"]
        if self.already_imported:
            return L[0] + "\n  SKIPPED: this exact file (sha256) was already imported; use --force to re-run (rows are de-duplicated anyway)."
        L += [f"  rows read={self.rows_read} parsed={self.rows_parsed} rejected={self.rows_rejected} "
              f"malformed_lines_skipped={self.malformed_lines_skipped}",
              f"  stored: inserted={s.get('inserted', 0)} duplicates_in_db={s.get('duplicates', 0)} "
              f"duplicates_in_file={s.get('duplicates_in_batch', 0)} store_rejected={s.get('rejected', 0)}",
              f"  time precision: {self.precision}   date range: {self.date_min} -> {self.date_max}",
              f"  company mapping: articles_with_stocks={s.get('articles_with_stocks', 0)} stock_links={s.get('stock_links', 0)}"
              f" (market-wide/unmapped articles keep no stock)"]
        if self.rejected_by_reason:
            L.append(f"  rejected by reason: {self.rejected_by_reason}")
        if self.notes:
            L.append(f"  notes: {self.notes}")
        if s.get("unknown_symbols"):
            top = sorted(s["unknown_symbols"].items(), key=lambda kv: -kv[1])[:10]
            L.append(f"  tickers not in `stocks` (top 10): {top}")
        if self.by_source and len(self.by_source) > 1:
            L.append(f"  parsed rows by source: {self.by_source}")
        if self.categories:
            L.append(f"  categories (dataset labels, stored as-is): {dict(sorted(self.categories.items(), key=lambda kv: -kv[1]))}")
        L.append(f"  article text: {'stored' if self.text_stored else 'NOT stored (opt-in)'}; rows that had text in the file: {self.rows_with_text}")
        return "\n".join(L)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _clean(v) -> str:
    return "" if is_missing(v) else " ".join(str(v).split())


def _symbols(v) -> list[str]:
    out = []
    for tok in _SYMBOL_SPLIT.split("" if is_missing(v) else str(v)):
        t = tok.strip().upper()
        t = re.sub(r"^(PSX|KSE)\s*:\s*", "", t)
        t = re.sub(r"\.KA$", "", t)
        if t and t not in ("NAN", "NONE", "NULL", "-"):
            out.append(t)
    return out


def build_items(table, mapping, profile: ImportProfile, *, source: str | None, doc_type: str, tz: str | None,
                dayfirst: bool | None, store_text: bool, now: datetime, rep: ImportReport):
    """Rows -> NewsItems. Fills rep counters/samples. Never raises on a bad row."""
    df = table.df
    cols = {k: v for k, v in mapping.items() if v}
    idx = {k: list(df.columns).index(v) for k, v in cols.items()}
    reasons, notes, prec, years, cats, srcs = (Counter() for _ in range(6))
    dmin = dmax = None
    items: list[NewsItem] = []
    all_rejects: list[dict] = []
    for n, row in enumerate(df.itertuples(index=False, name=None), start=1):
        get = lambda k: row[idx[k]] if k in idx else None            # noqa: E731
        reason = None
        headline = _clean(get("headline"))
        row_source = source or _clean(get("source"))              # explicit --source-name wins; else the file's own column
        if not headline:
            reason = "empty_headline"
        elif not row_source:
            reason = "missing_source"
        raw_url = _clean(get("url"))
        if reason is None:
            if "published" in idx:
                pt = parse_datetime(get("published"), get("time") if "time" in idx else None, tz=tz, dayfirst=dayfirst, now=now)
            else:
                pt = ParsedTime(error="missing_date")
            if pt.error == "missing_date" and raw_url:
                pt = ParsedTime(precision="none", note="no_date_kept_because_url")        # stored, excluded from features
            elif pt.error:
                reason = pt.error
        if reason is None and not raw_url and pt.precision == "none":
            reason = "no_url_and_no_date"
        if reason:
            reasons[reason] += 1
            rec = {"row": n, "reason": reason, "headline": headline[:80]}
            if len(rep.rejected_samples) < 50:
                rep.rejected_samples.append(rec)
            if len(all_rejects) < 200_000:
                all_rejects.append(rec)
            continue
        if pt.note:
            notes[pt.note] += 1
        url = raw_url or None
        rec_id = _clean(get("record_id")) or None
        text = _clean(get("text"))
        rep.rows_with_text += bool(text)
        syms = _symbols(get("symbol")) if "symbol" in idx else []
        d = pt.published_at
        if d:
            dmin, dmax = (d if dmin is None or d < dmin else dmin), (d if dmax is None or d > dmax else dmax)
            years[str(d.year)] += 1
        prec[pt.precision] += 1
        category = _clean(get("category")) or None
        cats[category] += category is not None
        srcs[row_source] += 1
        items.append(NewsItem(headline=headline, source=row_source, url=url, category=category, article_text=text if (store_text and text) else None,
                              published_at=pt.published_at, time_precision=pt.precision,
                              language=(_clean(get("language")) or profile.language)[:10], symbols=syms,
                              doc_type=doc_type, source_record_id=rec_id, published_date=pt.published_date))
    rep._all_rejects = all_rejects                       # full list for the reject CSV (not part of the JSON report)
    rep.rows_read = len(df)
    rep.rows_parsed = len(items)
    rep.rejected_by_reason = dict(reasons)
    rep.notes = {**rep.notes, **dict(notes)}
    rep.precision = dict(prec)
    rep.date_min, rep.date_max = (dmin.isoformat() if dmin else None), (dmax.isoformat() if dmax else None)
    rep.rows_by_year = dict(sorted(years.items()))
    rep.categories, rep.by_source = dict(cats), dict(srcs)
    return items


def inspect_file(path, profile: ImportProfile, *, overrides=None, tz=None, dayfirst=None, sample_rows: int = 2000) -> dict:
    """Describe a file BEFORE importing: columns, detected mapping, date parseability, problems. No database."""
    table = read_table(path)
    df = table.df
    mapping, ignored, ambiguous = resolve_columns(list(df.columns), profile, overrides)
    now = datetime.now(timezone.utc)
    sample = df.head(sample_rows)
    errors, precision = Counter(), Counter()
    dropped = 0
    if mapping.get("published"):
        i, t = list(df.columns).index(mapping["published"]), (list(df.columns).index(mapping["time"]) if mapping.get("time") else None)
        for row in sample.itertuples(index=False, name=None):
            pt = parse_datetime(row[i], row[t] if t is not None else None, tz=tz or profile.tz, dayfirst=dayfirst if dayfirst is not None else profile.dayfirst, now=now)
            errors[pt.error or "ok"] += 1
            if not pt.error:
                precision[pt.precision] += 1
            dropped += pt.note == "time_dropped_no_timezone"
    cols = []
    for c in df.columns:
        s = df[c].map(_clean)
        nonempty = s[s != ""]
        cols.append({"column": c, "non_empty_pct": round(100 * len(nonempty) / max(len(df), 1), 1),
                     "examples": list(nonempty.head(3).str.slice(0, 60))})
    return {"file": str(path), "rows": len(df), "encoding": table.encoding, "delimiter": table.delimiter,
            "malformed_lines": table.malformed_count, "columns": cols, "mapping": mapping, "ignored": ignored,
            "ambiguous": ambiguous, "date_parse_sample": dict(errors), "precision_sample": dict(precision),
            "timezone_used": tz or profile.tz,
            "warning": (None if not dropped else
                        f"{dropped} of the sampled rows have a time-of-day but no timezone (none declared, no offset in the file): "
                        "those times will be DROPPED to date-only (never guessed). Pass --tz Asia/Karachi (or the source's zone) if you know it.")}


def import_file(session: Session, path, profile: ImportProfile, *, source_name: str | None = None,
                dataset: str | None = None, doc_type: str | None = None, tz: str | None = None,
                dayfirst: bool | None = None, overrides: dict[str, str] | None = None, dry_run: bool = False,
                limit: int | None = None, store_text: bool = False, force: bool = False,
                report_dir: str | Path | None = None, now: datetime | None = None, index=None) -> ImportReport:
    p = Path(path)
    now = now or datetime.now(timezone.utc)
    source = source_name or profile.source                     # None => taken per row from the mapped `source` column
    dataset = dataset or profile.name
    doc_type = doc_type or profile.doc_type
    tz = tz or profile.tz
    dayfirst = dayfirst if dayfirst is not None else profile.dayfirst
    rep = ImportReport(file=p.name, profile=profile.name, dataset=dataset, source=source, dry_run=dry_run,
                       text_stored=store_text)
    rep.sha256 = sha256_file(p)

    if not dry_run and not force:
        prior = session.scalar(select(ImportBatch).where(ImportBatch.source_name == dataset,
                                                         ImportBatch.file_sha256 == rep.sha256,
                                                         ImportBatch.status == "completed"))
        if prior is not None:
            rep.already_imported, rep.batch_id = True, prior.batch_id
            return rep

    table = read_table(p)
    rep.encoding, rep.malformed_lines_skipped = table.encoding, table.malformed_count
    if table.short_rows:
        rep.notes["short_rows_padded_with_empty_cells"] = table.short_rows
    if limit:
        table.df = table.df.head(limit)
    mapping, ignored, ambiguous = resolve_columns(list(table.df.columns), profile, overrides)
    rep.columns_used, rep.columns_ignored, rep.ambiguous_columns = mapping, ignored, ambiguous
    if not source and not mapping.get("source"):
        raise ImportSchemaError("--source-name is required for this file (it labels the provenance of every row), "
                                "or map a per-row publisher column with --map source=<column>")
    if not source:
        rep.source = f"per row (column '{mapping['source']}')"

    batch = None
    if not dry_run:
        batch = ImportBatch(source_name=dataset, file_name=p.name, file_sha256=rep.sha256, status="running")
        session.add(batch)
        session.flush()
        rep.batch_id = batch.batch_id
        session.commit()
    try:
        items = build_items(table, mapping, profile, source=source, doc_type=doc_type, tz=tz, dayfirst=dayfirst,
                            store_text=store_text, now=now, rep=rep)
        index = index or build_stock_index(session)
        total = StoreReport()
        for i in range(0, len(items), CHUNK):
            total.merge(store_news(session, items[i:i + CHUNK], index, batch_id=rep.batch_id))
            if not dry_run:
                session.commit()
        rep.store = asdict(total)
        if dry_run:
            session.rollback()                                    # nothing persisted; counts above are still exact
        else:
            # a forced re-import of an identical file supersedes the earlier completed batch (keeps the unique "already
            # imported" guard valid while preserving the history of both runs)
            session.execute(update(ImportBatch).where(
                ImportBatch.source_name == dataset, ImportBatch.file_sha256 == rep.sha256,
                ImportBatch.status == "completed", ImportBatch.batch_id != rep.batch_id).values(status="superseded"))
            batch = session.get(ImportBatch, rep.batch_id)
            batch.status, batch.finished_at = "completed", datetime.now(timezone.utc)
            batch.rows_read, batch.rows_inserted = rep.rows_read, total.inserted
            batch.rows_duplicate = total.duplicates + total.duplicates_in_batch
            batch.rows_rejected = rep.rows_rejected + total.rejected
            batch.report = rep.to_json()
            session.commit()
    except Exception as exc:
        session.rollback()
        if batch is not None:
            b = session.get(ImportBatch, rep.batch_id)
            if b is not None:
                b.status, b.finished_at, b.report = "failed", datetime.now(timezone.utc), f"{type(exc).__name__}: {exc}"
                session.commit()
        raise

    if report_dir and not dry_run:
        d = Path(report_dir)
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{rep.batch_id}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', dataset)}"
        (d / f"{stem}_report.json").write_text(rep.to_json(), encoding="utf-8")
        with open(d / f"{stem}_rejects.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["row", "reason", "headline"])
            w.writeheader()
            w.writerows(getattr(rep, "_all_rejects", rep.rejected_samples))
    return rep
