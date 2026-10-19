"""Build the Model A dataset from PostgreSQL and write Parquet + metadata JSON.

Usage (from repo root, venv active, .env pointing at the Docker Postgres):
    python -m scripts.build_model_a_dataset
    python -m scripts.build_model_a_dataset --out data/model_a --max-daily-move 0.10

Outputs (in --out, default data/model_a):
    model_a_dataset.parquet   one row per (stock, session) with complete features and a known 5-session label
    model_a_meta.json         feature list, target definition, filters, counts, class balance
Nothing is sent over the network. Upload the two files to Google Colab yourself.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from marketsense.config import setup_logging
from marketsense.db.connection import get_engine
from marketsense.features.model_a import FEATURES, HORIZON, RETURN_COL, TARGET_COL, build_dataset

log = logging.getLogger("build_model_a")

QUERY = text("""
    SELECT s.stock_id, s.symbol, p.trading_date, p.open, p.high, p.low, p.close, p.volume, p.created_at,
           t.rsi, t.macd, t.macd_signal, t.macd_histogram
    FROM price_history p
    JOIN stocks s ON s.stock_id = p.stock_id
    LEFT JOIN technical_features t
           ON t.stock_id = p.stock_id AND t.trading_date = p.trading_date
    ORDER BY s.symbol, p.trading_date
""")


def drop_possible_intraday_rows(prices: pd.DataFrame, cutoff_hhmm: str, tz: str = "Asia/Karachi") -> tuple[pd.DataFrame, int]:
    """price_history inserts use ON CONFLICT DO NOTHING, so a bar saved DURING the session is never corrected.
    Drop rows whose insertion (local PSX time) was on the same date as the bar and before `cutoff_hhmm`.
    Such a bar would corrupt features on day d and the label of the row 5 sessions earlier."""
    created = pd.to_datetime(prices["created_at"], utc=True).dt.tz_convert(tz)
    same_day = created.dt.strftime("%Y-%m-%d") == pd.to_datetime(prices["trading_date"]).dt.strftime("%Y-%m-%d")
    before_cutoff = created.dt.strftime("%H:%M") < cutoff_hhmm
    suspect = same_day & before_cutoff
    return prices[~suspect].drop(columns="created_at"), int(suspect.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/model_a")
    ap.add_argument("--max-daily-move", type=float, default=0.10,
                    help="flag label windows containing a raw 1-day |return| above this (possible bonus/split; default 0.10)")
    ap.add_argument("--min-session-coverage", type=float, default=0.5,
                    help="a date is a market session if >= this x median daily stock count traded (default 0.5)")
    ap.add_argument("--min-stock-coverage", type=float, default=0.8,
                    help="exclude stocks present on fewer than this share of sessions since listing (default 0.8)")
    ap.add_argument("--intraday-cutoff", default="17:00",
                    help="drop bars inserted same-day before this PKT time HH:MM (default 17:00)")
    ap.add_argument("--only-kse100", action="store_true", help="restrict to stocks currently flagged is_kse100")
    a = ap.parse_args()
    setup_logging()

    engine = get_engine()
    with engine.connect() as conn:
        prices = pd.read_sql(QUERY, conn)
        if a.only_kse100:
            ids = pd.read_sql(text("SELECT stock_id FROM stocks WHERE is_kse100"), conn)["stock_id"]
            prices = prices[prices["stock_id"].isin(ids)]
    if prices.empty:
        log.error("price_history returned 0 rows (wrong database? check DB_HOST/DB_PORT/DB_NAME in .env "
                  "against db_setup/.env.db, and that Docker is running).")
        return 1
    log.info("Read %d price rows for %d stocks", len(prices), prices["stock_id"].nunique())

    prices, n_intraday = drop_possible_intraday_rows(prices, a.intraday_cutoff)
    if n_intraday:
        log.warning("Dropped %d bars that were inserted before %s PKT on their own trading date (possibly partial-day)",
                    n_intraday, a.intraday_cutoff)

    ds, rep = build_dataset(prices, HORIZON, a.min_session_coverage, a.min_stock_coverage, a.max_daily_move)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out / "model_a_dataset.parquet", index=False)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_sessions": HORIZON,
        "target_col": TARGET_COL,
        "return_col": RETURN_COL,
        "features": FEATURES,
        "prediction_cutoff": "post-market-close of trading_date; features use sessions <= trading_date",
        "price_basis": "raw (unadjusted) close; no adjusted close or corporate-action data exists in the database",
        "filters": {"max_daily_move_flag": a.max_daily_move, "min_session_coverage": a.min_session_coverage,
                    "min_stock_coverage": a.min_stock_coverage, "intraday_cutoff_pkt": a.intraday_cutoff,
                    "only_kse100": a.only_kse100},
        "counts": {
            "price_rows_read": rep.n_price_rows, "intraday_bars_dropped": n_intraday,
            "market_sessions": rep.n_sessions, "stocks_in_dataset": int(ds["symbol"].nunique()),
            "rows_dataset": len(ds), "rows_dropped_missing_label": rep.n_dropped_missing_label,
            "rows_dropped_missing_features": rep.n_dropped_missing_features,
            "rows_flagged_corp_action_suspect": rep.n_suspect_corp_action,
            "rows_with_exactly_zero_return": rep.n_zero_return_labels,
        },
        "date_range": [str(ds["trading_date"].min().date()), str(ds["trading_date"].max().date())],
        "class_balance": {str(k): int(v) for k, v in ds[TARGET_COL].value_counts().sort_index().items()},
        "excluded_stocks": rep.excluded_stocks,
        "dropped_sparse_dates": rep.dropped_sparse_dates,
        "calendar_gaps_over_5_days": rep.calendar_gaps_over_5_days,
    }
    (out / "model_a_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== MODEL A DATASET ===")
    print(f"rows={len(ds):,}  stocks={ds['symbol'].nunique()}  dates={meta['date_range'][0]} -> {meta['date_range'][1]}")
    print(f"class balance (0=down/flat, 1=up): {meta['class_balance']}  | share up = {ds[TARGET_COL].mean():.3f}")
    print(f"exact-zero 5d returns (labelled 0): {rep.n_zero_return_labels:,}")
    print(f"rows flagged corp-action-suspect (|1d ret| > {a.max_daily_move:.0%} in window): {rep.n_suspect_corp_action:,}")
    if rep.excluded_stocks:
        print("EXCLUDED STOCKS:", "; ".join(f"{s} ({r})" for s, r in sorted(rep.excluded_stocks.items())))
    if rep.dropped_sparse_dates:
        print(f"non-session dates dropped ({len(rep.dropped_sparse_dates)}): {rep.dropped_sparse_dates[:10]}")
    if rep.calendar_gaps_over_5_days:
        print("calendar gaps > 5 days (check for missing data):", rep.calendar_gaps_over_5_days)
    print(f"wrote {out/'model_a_dataset.parquet'} and {out/'model_a_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
