"""Sync KSE-100 universe + collect OHLCV.  Usage: python -m scripts.collect_prices [--symbols ENGRO LUCK] [--start 2020-01-01]"""
import argparse, sys
from marketsense.config import setup_logging
from marketsense.db.connection import session_scope
from marketsense.collectors.psx_collector import PSXCollector

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--skip-universe-sync", action="store_true")
    a = ap.parse_args()
    setup_logging()
    with session_scope() as s:
        c = PSXCollector(s)
        if not a.skip_universe_sync:
            c.sync_stock_universe(); s.commit()
        report = c.collect_prices(a.symbols, a.start, a.end)
    print(report.summary())
    return 1 if report.failed or report.no_data else 0

if __name__ == "__main__":
    sys.exit(main())
