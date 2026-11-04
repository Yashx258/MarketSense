"""Tolerant CSV/Parquet reader. Never silently drops malformed lines: they are counted and sampled."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class ImportSchemaError(RuntimeError):
    """The file cannot be imported as given (unsupported type, missing required columns...)."""


@dataclass
class TableRead:
    df: pd.DataFrame
    encoding: str | None = None
    delimiter: str | None = None
    malformed: list[str] = field(default_factory=list)    # lines skipped because they could not be parsed (sampled)
    malformed_count: int = 0
    short_rows: int = 0                                    # rows with fewer fields than the header (padded with empty cells)


def _sniff_encoding(path: Path) -> str:
    raw = path.read_bytes()[:2_000_000]
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def read_table(path: str | Path) -> TableRead:
    p = Path(path)
    if not p.is_file():
        raise ImportSchemaError(f"file not found: {p}")
    ext = p.suffix.lower()
    if ext == ".parquet":
        try:
            df = pd.read_parquet(p)
        except Exception as exc:                                    # noqa: BLE001
            raise ImportSchemaError(f"cannot read parquet {p.name}: {exc}") from exc
        return TableRead(df.astype(object).where(df.notna(), None))
    if ext not in (".csv", ".tsv", ".txt"):
        raise ImportSchemaError(f"unsupported file type '{ext}' (use .csv, .tsv or .parquet; export Excel to CSV first)")
    if p.stat().st_size == 0:
        raise ImportSchemaError(f"{p.name} is empty")
    enc = _sniff_encoding(p)
    sample = p.read_bytes()[:65536].decode(enc, errors="replace")
    if ext == ".tsv":
        sep = "\t"
    else:
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
    too_long, too_short, samples = _scan_field_counts(p, enc, sep)
    # index_col=False: a row with extra fields must NEVER make pandas silently promote the first column to the index
    kw = dict(sep=sep, dtype=str, encoding=enc, keep_default_na=False, index_col=False)
    if too_long:                                                    # deterministic path: we skip and report them ourselves
        df = _read_with_csv_module(p, enc, sep)
        return TableRead(df, enc, sep, samples, too_long, too_short)
    try:
        return TableRead(pd.read_csv(p, **kw), enc, sep, short_rows=too_short)
    except pd.errors.ParserError as exc:                            # e.g. unbalanced quotes: fail loudly rather than guess
        raise ImportSchemaError(f"{p.name} is not parseable as CSV: {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ImportSchemaError(f"{p.name} has no columns/rows") from exc


def _read_with_csv_module(path: Path, enc: str, sep: str) -> pd.DataFrame:
    """Rows with MORE fields than the header are dropped (their column alignment cannot be trusted); shorter rows are
    padded with empty cells. Everything is read as text."""
    with open(path, newline="", encoding=enc) as fh:
        rd = csv.reader(fh, delimiter=sep)
        header = next(rd)
        rows = [row + [""] * (len(header) - len(row)) for row in rd if len(row) <= len(header)]
    return pd.DataFrame(rows, columns=header, dtype=str)


def _scan_field_counts(path: Path, enc: str, sep: str):
    """(rows_with_too_many_fields, rows_with_too_few, samples_of_too_many). Cheap O(n) pass over the raw file."""
    too_long = too_short = 0
    samples: list[str] = []
    with open(path, newline="", encoding=enc) as fh:
        rd = csv.reader(fh, delimiter=sep)
        header = next(rd, None)
        if header is None:
            return 0, 0, samples
        for row in rd:
            if len(row) > len(header):
                too_long += 1
                if len(samples) < 20:
                    samples.append(" | ".join(row)[:200])
            elif len(row) < len(header) and any(c.strip() for c in row):
                too_short += 1
    return too_long, too_short, samples
