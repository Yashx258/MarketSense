"""RSI + MACD from price_history -> technical_features (derived; safe to recompute)."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from marketsense.db.connection import upsert
from marketsense.db.models import PriceHistory, Stock, TechnicalFeature

log = logging.getLogger(__name__)



def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI; the first value is at index `period`."""
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = pd.Series(np.nan, index=close.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=close.index, dtype=float)

    # Need `period` actual price changes to seed the first averages.
    if len(close) <= period:
        return avg_gain.rename("rsi")

    avg_gain.iloc[period] = gain.iloc[1:period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1:period + 1].mean()

    # Wilder's smoothing.
    for i in range(period + 1, len(close)):
        avg_gain.iloc[i] = (
            avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]
        ) / period

        avg_loss.iloc[i] = (
            avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]
        ) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases explicitly.
    no_losses = (avg_loss == 0) & (avg_gain > 0)
    no_changes = (avg_loss == 0) & (avg_gain == 0)

    rsi.loc[no_losses] = 100.0
    rsi.loc[no_changes] = 50.0

    return rsi.rename("rsi")



def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Standard MACD with initial EMA warm-up values masked."""
    ema_f = close.ewm(
        span=fast, adjust=False, min_periods=fast
    ).mean()

    ema_s = close.ewm(
        span=slow, adjust=False, min_periods=slow
    ).mean()

    line = ema_f - ema_s
    sig = line.ewm(
        span=signal, adjust=False, min_periods=signal
    ).mean()

    return pd.DataFrame({
        "macd": line,
        "macd_signal": sig,
        "macd_histogram": line - sig,
    })



def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators for one stock's ascending price history."""
    p = prices.sort_values("trading_date").reset_index(drop=True)
    close = p["close"].astype(float)

    out = pd.concat(
        [
            p[["trading_date"]],
            rsi_wilder(close, period=14),
            macd(close, fast=12, slow=26, signal=9),
        ],
        axis=1,
    )

    # Keep warm-up dates. Indicators are NULL until they are available.
    return out



def calculate_and_store(session: Session, symbols: list[str] | None = None) -> dict[str, int]:
    q = select(Stock.stock_id, Stock.symbol)
    q = q.where(Stock.symbol.in_([s.upper() for s in symbols])) if symbols else q.where(Stock.is_kse100)
    result: dict[str, int] = {}
    for stock_id, symbol in session.execute(q).all():
        rows = session.execute(select(PriceHistory.trading_date, PriceHistory.close)
                               .where(PriceHistory.stock_id == stock_id)
                               .order_by(PriceHistory.trading_date)).all()
        if len(rows) < 34:
            log.warning(
                "%s: only %d price rows; insufficient history "
                "for fully initialized RSI/MACD signal features.",
                symbol,
                len(rows),
            )
            result[symbol] = 0
            continue
        feats = compute_indicators(pd.DataFrame(rows, columns=["trading_date", "close"]))
        recs = [dict(stock_id=stock_id, trading_date=r.trading_date,
                     rsi=_f(r.rsi), macd=_f(r.macd), macd_signal=_f(r.macd_signal), macd_histogram=_f(r.macd_histogram))
                for r in feats.itertuples(index=False)]
        upsert(session, TechnicalFeature, recs, ["stock_id", "trading_date"],
               ["rsi", "macd", "macd_signal", "macd_histogram"])
        session.commit()
        result[symbol] = len(recs)
        log.info("%s: %d technical rows written", symbol, len(recs))
    return result


def _f(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else float(v)
