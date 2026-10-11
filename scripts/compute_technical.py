
"""Compute RSI/MACD.

Examples:
    python -m scripts.compute_technical
    python -m scripts.compute_technical --symbols AGP
    python -m scripts.compute_technical --symbols AGP ATRL BAFL
"""

import argparse

from marketsense.config import setup_logging
from marketsense.db.connection import session_scope
from marketsense.features.technical import calculate_and_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RSI and MACD from price_history."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Optional list of stock symbols. Defaults to active KSE100 stocks.",
    )
    args = parser.parse_args()

    setup_logging()

    with session_scope() as session:
        result = calculate_and_store(session, args.symbols)

    print(result)


if __name__ == "__main__":
    main()
