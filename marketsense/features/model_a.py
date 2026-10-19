"""Model A dataset builder: causal OHLCV + RSI/MACD features and a 5-session direction target.

Pure pandas (no database access) so it can be unit-tested.

Timing convention (post-market-close cutoff)
    Row (stock, t) = "the market has closed on session t; predict the next 5 sessions".
    Features use data from sessions <= t only.
    forward_return_5d = close[t+5 sessions] / close[t] - 1
    "Sessions" are defined by a MARKET-WIDE trading calendar, NOT by row position inside one
    stock's history (a stock with missing rows would otherwise silently look >5 sessions ahead).
    If the stock has no close on session t+5, the label is missing (never filled or interpolated).

Corporate actions
    price_history holds RAW closes: there is no adjusted close and no corporate-action table.
    Large discontinuities (bonus issues, splits) can be flagged; small cash-dividend drops cannot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HORIZON = 5
TARGET_COL = "target_direction_5d"
RETURN_COL = "forward_return_5d"

# Columns the model may use. Everything else in the dataset is metadata / outcome / flag.
FEATURES: list[str] = [
    # returns / momentum (all backward-looking)
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    # candle shape / gap on day t
    "oc_ret", "hl_range", "gap_open",
    # trend position and volatility
    "close_sma5_ratio", "close_sma10_ratio", "close_sma20_ratio", "vol_20d",
    # volume
    "log_volume", "vol_ratio_5_20",
    # indicators from technical_features (MACD scaled by price so stocks are comparable)
    "rsi", "macd_pct", "macd_signal_pct", "macd_hist_pct",
]

# Never allowed in the feature matrix.
NON_FEATURE_COLS = ["stock_id", "symbol", "trading_date", "label_end_date",
                    RETURN_COL, TARGET_COL, "corp_action_suspect_5d"]


@dataclass
class BuildReport:
    n_price_rows: int = 0
    n_sessions: int = 0
    dropped_sparse_dates: list[str] = field(default_factory=list)
    excluded_stocks: dict[str, str] = field(default_factory=dict)       # symbol -> reason
    n_rows_before_filters: int = 0
    n_dropped_missing_label: int = 0
    n_dropped_missing_features: int = 0
    n_suspect_corp_action: int = 0
    n_zero_return_labels: int = 0
    calendar_gaps_over_5_days: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
def build_calendar(prices: pd.DataFrame, min_coverage: float = 0.5) -> tuple[pd.DatetimeIndex, list[str]]:
    """Market sessions = dates on which at least `min_coverage` x (median daily row count) stocks traded.

    Filters out stray dates that exist for only a handful of stocks (partial / erroneous loads).
    """
    counts = prices.groupby("trading_date")["stock_id"].nunique()
    threshold = min_coverage * counts.median()
    keep = counts[counts >= threshold].index
    dropped = [d.strftime("%Y-%m-%d") for d in counts.index.difference(keep)]
    return pd.DatetimeIndex(sorted(keep)), dropped


def _features_for_stock(g: pd.DataFrame, cal: pd.DatetimeIndex, horizon: int, max_daily_move: float) -> pd.DataFrame:
    """g: one stock's rows (index = trading_date). Returns calendar-aligned frame with features + target."""
    g = g[g.index.isin(cal)]
    gi = g.reindex(cal[cal >= g.index.min()])           # one row per market session from first listing onward
    c, o, h, l, v = gi["close"], gi["open"], gi["high"], gi["low"], gi["volume"]
    out = pd.DataFrame(index=gi.index)
    out["present"] = c.notna()

    # --- backward-looking features (NaN whenever a needed session is missing: strict, no filling)
    for k in (1, 3, 5, 10, 20):
        out[f"ret_{k}d"] = c / c.shift(k) - 1
    out["oc_ret"] = c / o - 1
    out["hl_range"] = (h - l) / c
    out["gap_open"] = o / c.shift(1) - 1
    for k in (5, 10, 20):
        out[f"close_sma{k}_ratio"] = c / c.rolling(k, min_periods=k).mean() - 1
    out["vol_20d"] = out["ret_1d"].rolling(20, min_periods=20).std()
    out["log_volume"] = np.log1p(v)
    v5, v20 = v.rolling(5, min_periods=5).mean(), v.rolling(20, min_periods=20).mean()
    out["vol_ratio_5_20"] = (v5 / v20).where(v20 > 0)
    out["rsi"] = gi["rsi"]
    out["macd_pct"] = gi["macd"] / c
    out["macd_signal_pct"] = gi["macd_signal"] / c
    out["macd_hist_pct"] = gi["macd_histogram"] / c

    # --- label: uses the close `horizon` MARKET sessions ahead (OUTCOME, never a feature)
    fwd_close = c.shift(-horizon)
    out[RETURN_COL] = fwd_close / c - 1
    out[TARGET_COL] = (out[RETURN_COL] > 0).astype("Int8").where(out[RETURN_COL].notna())
    out["label_end_date"] = pd.Series(gi.index, index=gi.index).shift(-horizon)

    # --- raw-price discontinuity flag over the label window (t, t+horizon]
    absr = out["ret_1d"].abs()
    fut = pd.concat([absr.shift(-k) for k in range(1, horizon + 1)], axis=1).max(axis=1)
    out["corp_action_suspect_5d"] = fut > max_daily_move
    return out[out["present"]].drop(columns="present")


def build_dataset(
    prices: pd.DataFrame,
    horizon: int = HORIZON,
    min_session_coverage: float = 0.5,
    min_stock_coverage: float = 0.8,
    max_daily_move: float = 0.10,
) -> tuple[pd.DataFrame, BuildReport]:
    """prices columns: stock_id, symbol, trading_date, open, high, low, close, volume,
    rsi, macd, macd_signal, macd_histogram. Returns (dataset, report)."""
    rep = BuildReport(n_price_rows=len(prices))
    p = prices.copy()
    p["trading_date"] = pd.to_datetime(p["trading_date"])
    for col in ("open", "high", "low", "close", "volume", "rsi", "macd", "macd_signal", "macd_histogram"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p = p.drop_duplicates(["stock_id", "trading_date"]).sort_values(["stock_id", "trading_date"])

    cal, rep.dropped_sparse_dates = build_calendar(p, min_session_coverage)
    rep.n_sessions = len(cal)
    gaps = pd.Series(cal[1:] - cal[:-1], index=cal[1:])
    rep.calendar_gaps_over_5_days = [f"{d.date()} (+{g.days}d)" for d, g in gaps[gaps > pd.Timedelta(days=5)].items()]

    frames = []
    for sid, g in p.groupby("stock_id", sort=False):
        symbol = g["symbol"].iloc[0]
        g = g.set_index("trading_date")
        in_cal = g.index.isin(cal)
        # coverage = share of market sessions (from its first listing in the calendar) on which the stock has a close
        first = g.index[in_cal].min() if in_cal.any() else None
        if first is None:
            rep.excluded_stocks[symbol] = "no rows on market sessions"
            continue
        span = int((cal >= first).sum())
        coverage = int((in_cal & g["close"].notna().to_numpy()).sum()) / span
        if coverage < min_stock_coverage:
            rep.excluded_stocks[symbol] = f"coverage {coverage:.0%} < {min_stock_coverage:.0%} of sessions"
            continue
        f = _features_for_stock(g, cal, horizon, max_daily_move)
        f.insert(0, "symbol", symbol)
        f.insert(0, "stock_id", sid)
        frames.append(f.reset_index(names="trading_date"))

    if not frames:
        raise RuntimeError("No stocks left after filters; check coverage thresholds and the export.")
    ds = pd.concat(frames, ignore_index=True)
    rep.n_rows_before_filters = len(ds)

    ds = ds[ds[TARGET_COL].notna()]
    rep.n_dropped_missing_label = rep.n_rows_before_filters - len(ds)
    n0 = len(ds)
    ds = ds.dropna(subset=FEATURES)
    ds = ds[np.isfinite(ds[FEATURES].to_numpy(dtype=float)).all(axis=1)]
    rep.n_dropped_missing_features = n0 - len(ds)
    rep.n_suspect_corp_action = int(ds["corp_action_suspect_5d"].sum())
    rep.n_zero_return_labels = int((ds[RETURN_COL] == 0).sum())

    ds[TARGET_COL] = ds[TARGET_COL].astype("int8")
    cols = NON_FEATURE_COLS + FEATURES
    return ds[cols].sort_values(["trading_date", "symbol"]).reset_index(drop=True), rep
