"""PSX collector built on the installed psxdata API (v1.1.1).

psxdata.indices("KSE100") -> DataFrame containing live KSE100 constituents
psxdata.symbols()         -> DataFrame[symbol, name, sector_name, ...]
psxdata.stocks(sym)       -> DataFrame[date, open, high, low, close, volume, ...]

Important:
    PSX can expose temporary trading symbols such as LUCKXD.
    The database uses the canonical/base symbol (LUCK) as the security identity.
    A temporary symbol is normalized only when its base symbol exists in
    psxdata.symbols().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from marketsense.config import Settings
from marketsense.db.connection import insert_ignore_duplicates
from marketsense.db.models import PriceHistory, Stock
from marketsense.preprocessing.clean import clean_price_frame

log = logging.getLogger(__name__)


# These are temporary PSX trading suffixes.
# We do NOT blindly strip them. A suffix is removed only when the
# resulting base symbol exists in psxdata.symbols().
TEMPORARY_SUFFIXES = ("XD", "XB", "XR")


@dataclass
class CollectionReport:
    requested: int = 0

    # symbol -> new rows inserted
    succeeded: dict[str, int] = field(default_factory=dict)

    # valid API call but zero rows returned
    no_data: list[str] = field(default_factory=list)

    # symbol -> error message
    failed: dict[str, str] = field(default_factory=dict)

    # symbol -> rejection reasons
    rejected_rows: dict[str, dict[str, int]] = field(default_factory=dict)

    # raw symbol -> canonical symbol
    normalized_symbols: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Requested: {self.requested} | "
            f"OK: {len(self.succeeded)} | "
            f"No data: {len(self.no_data)} | "
            f"Failed: {len(self.failed)}",
            f"New price rows inserted: {sum(self.succeeded.values())}",
        ]

        if self.normalized_symbols:
            lines.append(
                "NORMALIZED: "
                + ", ".join(
                    f"{raw}->{canonical}"
                    for raw, canonical in sorted(self.normalized_symbols.items())
                )
            )

        if self.no_data:
            lines.append("NO DATA: " + ", ".join(sorted(self.no_data)))

        for symbol, error in sorted(self.failed.items()):
            lines.append(f"FAILED {symbol}: {error}")

        for symbol, reasons in sorted(self.rejected_rows.items()):
            lines.append(f"REJECTED ROWS {symbol}: {reasons}")

        return "\n".join(lines)


class PSXCollector:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        psx=None,
    ):
        self.session = session
        self.settings = settings or Settings()

        if psx is None:
            import psxdata as psx  # injectable for tests / offline use

        self.psx = psx

        # Cache canonical symbols for this collector instance.
        self._canonical_symbols: set[str] | None = None

    # ------------------------------------------------------------------
    # symbol normalization
    # ------------------------------------------------------------------

    def _load_canonical_symbols(self) -> set[str]:
        """Load the canonical PSX symbol universe from psxdata.symbols()."""
        if self._canonical_symbols is not None:
            return self._canonical_symbols

        meta = self.psx.symbols()

        if meta is None or meta.empty:
            raise RuntimeError("psxdata.symbols() returned no symbol metadata")

        if "symbol" not in meta.columns:
            raise RuntimeError(
                "psxdata.symbols() does not contain required 'symbol' column"
            )

        self._canonical_symbols = {
            str(symbol).strip().upper()
            for symbol in meta["symbol"].dropna()
        }

        if not self._canonical_symbols:
            raise RuntimeError("No canonical symbols found in psxdata.symbols()")

        log.info(
            "Loaded %d canonical PSX symbols",
            len(self._canonical_symbols),
        )

        return self._canonical_symbols

    def _normalize_symbol(self, symbol: str) -> str:
        """Return the canonical/base PSX symbol.

        Exact canonical symbols are preserved.

        Temporary suffixes such as XD/XB/XR are removed only when the
        resulting base symbol exists in psxdata.symbols().

        Examples:
            LUCKXD -> LUCK
            COLGXD -> COLG
            ABL    -> ABL
            UNKNOWNXD -> UNKNOWNXD   # no unsafe normalization
        """
        raw = str(symbol).strip().upper()

        if not raw:
            raise ValueError("Empty stock symbol")

        canonical_symbols = self._load_canonical_symbols()

        # Already canonical.
        if raw in canonical_symbols:
            return raw

        for suffix in TEMPORARY_SUFFIXES:
            if raw.endswith(suffix):
                base = raw[: -len(suffix)]

                if base and base in canonical_symbols:
                    log.info(
                        "Normalized temporary PSX symbol %s -> %s",
                        raw,
                        base,
                    )
                    return base

        # Unknown/non-canonical symbol: leave unchanged.
        return raw

    def _normalize_symbol_list(
        self,
        symbols: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        """Normalize a list while preserving order and removing duplicates."""
        normalized: list[str] = []
        mapping: dict[str, str] = {}
        seen: set[str] = set()

        for symbol in symbols:
            raw = str(symbol).strip().upper()
            canonical = self._normalize_symbol(raw)

            mapping[raw] = canonical

            if canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)

        return normalized, mapping

    # ------------------------------------------------------------------
    # stock universe
    # ------------------------------------------------------------------

    def _load_universe(self) -> pd.DataFrame:
        """Return DataFrame[symbol, company_name, sector].

        Symbols are normalized to canonical PSX symbols.
        """

        if self.settings.universe_source == "file":
            df = pd.read_csv(
                Path(self.settings.universe_file),
                dtype=str,
            )

            if "symbol" not in df.columns:
                raise ValueError(
                    f"Universe file {self.settings.universe_file} "
                    "must contain a 'symbol' column"
                )

            for column in ("company_name", "sector"):
                if column not in df.columns:
                    df[column] = None

            df = df[
                ["symbol", "company_name", "sector"]
            ].copy()

        elif self.settings.universe_source == "psxdata":
            # psxdata v1.1.1 exposes indices() directly.
            index_df = self.psx.indices(
                self.settings.universe_index
            )

            if index_df is None or index_df.empty:
                raise RuntimeError(
                    f"psxdata returned no constituents for "
                    f"{self.settings.universe_index}"
                )

            if "symbol" not in index_df.columns:
                raise RuntimeError(
                    "psxdata.indices() does not contain a 'symbol' column"
                )

            df = pd.DataFrame(
                {"symbol": index_df["symbol"]}
            )

            meta = self.psx.symbols()

            if meta is None or meta.empty:
                raise RuntimeError(
                    "psxdata.symbols() returned no metadata"
                )

            meta = meta.rename(
                columns={
                    "name": "company_name",
                    "sector_name": "sector",
                }
            )

            # Make sure expected metadata columns exist.
            for column in ("company_name", "sector"):
                if column not in meta.columns:
                    meta[column] = None

            meta = meta[
                ["symbol", "company_name", "sector"]
            ].copy()

            # Normalize metadata symbols as well.
            meta["symbol"] = meta["symbol"].map(
                self._normalize_symbol
            )

            # If normalization creates duplicate metadata rows,
            # keep the first canonical record.
            meta = meta.drop_duplicates(
                subset="symbol",
                keep="first",
            )

            df = df.merge(
                meta,
                on="symbol",
                how="left",
            )

        else:
            raise ValueError(
                f"Unknown STOCK_UNIVERSE_SOURCE: "
                f"{self.settings.universe_source}"
            )

        # Normalize universe symbols.
        df["symbol"] = (
            df["symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(self._normalize_symbol)
        )

        # Remove empty/invalid rows.
        df = df[
            df["symbol"].notna()
            & (df["symbol"] != "")
            & (df["symbol"] != "NAN")
        ].copy()

        # Canonical symbol must be unique.
        df = df.drop_duplicates(
            subset="symbol",
            keep="first",
        )

        return df[
            ["symbol", "company_name", "sector"]
        ].reset_index(drop=True)

    def sync_stock_universe(self) -> list[str]:
        """Upsert the current universe into stocks.

        Canonical symbols are stored in stocks.symbol.

        Symbols that leave the current KSE100 universe are NOT deleted;
        they are simply marked is_kse100=False.
        """

        uni = self._load_universe()

        current = set(uni["symbol"])

        for row in uni.itertuples(index=False):
            stock = self.session.scalar(
                select(Stock).where(
                    Stock.symbol == row.symbol
                )
            )

            name = (
                row.company_name
                if pd.notna(row.company_name)
                else None
            )

            sector = (
                row.sector
                if pd.notna(row.sector)
                else None
            )

            if stock is None:
                self.session.add(
                    Stock(
                        symbol=row.symbol,
                        company_name=name,
                        sector=sector,
                        is_kse100=True,
                        status="active",
                    )
                )
            else:
                stock.is_kse100 = True
                stock.status = "active"

                if name:
                    stock.company_name = name

                if sector:
                    stock.sector = sector

        # Existing symbols that are no longer in the current universe
        # remain in the database but are no longer marked as KSE100.
        self.session.execute(
            update(Stock)
            .where(
                Stock.is_kse100,
                Stock.symbol.notin_(current),
            )
            .values(is_kse100=False)
        )

        self.session.flush()

        log.info(
            "Stock universe synced: %d canonical symbols "
            "(source=%s, index=%s)",
            len(current),
            self.settings.universe_source,
            self.settings.universe_index,
        )

        return sorted(current)

    # ------------------------------------------------------------------
    # prices
    # ------------------------------------------------------------------

    def collect_prices(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> CollectionReport:
        """Fetch OHLCV and insert new rows.

        Re-running is safe because price_history has:

            UNIQUE(stock_id, trading_date)

        and insertion uses ON CONFLICT DO NOTHING.
        """

        # Important:
        # If symbols were explicitly supplied, normalize them first.
        requested_mapping: dict[str, str] = {}

        if symbols:
            normalized_symbols, requested_mapping = (
                self._normalize_symbol_list(symbols)
            )

            q = select(Stock).where(
                Stock.symbol.in_(normalized_symbols)
            )
        else:
            q = select(Stock).where(
                Stock.is_kse100
            )

        stocks = list(self.session.scalars(q))

        report = CollectionReport(
            requested=len(stocks),
            normalized_symbols={
                raw: canonical
                for raw, canonical in requested_mapping.items()
                if raw != canonical
            },
        )

        start = start or self.settings.price_start_date

        from psxdata.exceptions import PSXDataError

        for stock in stocks:
            try:
                last = self.session.scalar(
                    select(
                        func.max(
                            PriceHistory.trading_date
                        )
                    ).where(
                        PriceHistory.stock_id
                        == stock.stock_id
                    )
                )

                # Fetch using the canonical symbol.
                raw = self.psx.stocks(
                    stock.symbol,
                    start=start,
                    end=end,
                )

                if raw is None or raw.empty:
                    report.no_data.append(stock.symbol)

                    log.warning(
                        "%s: no data returned",
                        stock.symbol,
                    )
                    continue

                result = clean_price_frame(raw)

                if result.rejected:
                    report.rejected_rows[
                        stock.symbol
                    ] = result.reasons

                    log.warning(
                        "%s: rejected %d rows %s",
                        stock.symbol,
                        result.rejected,
                        result.reasons,
                    )

                rows = [
                    {
                        "stock_id": stock.stock_id,
                        "trading_date": row.date.date(),
                        "open": _num(row.open),
                        "high": _num(row.high),
                        "low": _num(row.low),
                        "close": float(row.close),
                        "volume": (
                            None
                            if pd.isna(row.volume)
                            else int(row.volume)
                        ),
                        "source": "psxdata",
                    }
                    for row in result.clean.itertuples(
                        index=False
                    )
                ]

                if not rows:
                    report.no_data.append(stock.symbol)

                    log.warning(
                        "%s: all fetched rows were rejected",
                        stock.symbol,
                    )
                    continue

                inserted = insert_ignore_duplicates(
                    self.session,
                    PriceHistory,
                    rows,
                    ["stock_id", "trading_date"],
                )

                self.session.commit()

                report.succeeded[
                    stock.symbol
                ] = inserted

                log.info(
                    "%s: %d clean rows, %d new "
                    "(previous last date: %s)",
                    stock.symbol,
                    len(rows),
                    inserted,
                    last,
                )

            except PSXDataError as exc:
                self.session.rollback()

                report.failed[
                    stock.symbol
                ] = f"{type(exc).__name__}: {exc}"

                log.error(
                    "%s: %s",
                    stock.symbol,
                    exc,
                )

            except Exception as exc:
                self.session.rollback()

                report.failed[
                    stock.symbol
                ] = f"{type(exc).__name__}: {exc}"

                log.exception(
                    "%s: unexpected error",
                    stock.symbol,
                )

        return report


def _num(value):
    """Convert pandas numeric values to Python float/None."""
    return None if pd.isna(value) else float(value)