import numpy as np
import pandas as pd
import pytest

from marketsense.features.model_a import FEATURES, NON_FEATURE_COLS, RETURN_COL, TARGET_COL, build_calendar, build_dataset


def _stock(sid, sym, dates, seed, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    df = pd.DataFrame({"stock_id": sid, "symbol": sym, "trading_date": dates, "open": close * 0.998,
                       "high": close * 1.01, "low": close * 0.99, "close": close,
                       "volume": rng.integers(1_000, 90_000, len(dates))})
    # indicators: same causal formulas as the repo so tests exercise realistic values
    from marketsense.features.technical import compute_indicators
    ind = compute_indicators(df[["trading_date", "close"]]).set_index("trading_date")
    for c in ("rsi", "macd", "macd_signal", "macd_histogram"):
        df[c] = ind[c].to_numpy()
    return df


@pytest.fixture()
def panel():
    dates = pd.bdate_range("2023-01-02", periods=160)
    a = _stock(1, "AAA", dates, 1)
    b = _stock(2, "BBB", dates, 2)
    c = _stock(3, "CCC", dates, 3)
    return pd.concat([a, b, c], ignore_index=True), dates


def test_label_uses_market_sessions_not_row_offsets(panel):
    prices, dates = panel
    # Remove 3 rows from BBB so a naive shift(-5) on its own rows would look 8 sessions ahead.
    gap_days = dates[60:63]
    prices = prices[~((prices.symbol == "BBB") & prices.trading_date.isin(gap_days))]
    ds, _ = build_dataset(prices, min_stock_coverage=0.9)
    row = ds[(ds.symbol == "BBB") & (ds.trading_date == dates[58])].iloc[0]
    assert row.label_end_date == dates[63]                        # exactly 5 market sessions later
    close = prices[prices.symbol == "BBB"].set_index("trading_date")["close"]
    assert row[RETURN_COL] == pytest.approx(close[dates[63]] / close[dates[58]] - 1)
    # label whose target session is missing for that stock is dropped, never filled
    assert ((ds.symbol == "BBB") & (ds.trading_date == dates[57])).sum() == 0     # t+5 = dates[62] missing


def test_last_five_sessions_unlabelled(panel):
    prices, dates = panel
    ds, _ = build_dataset(prices)
    assert ds.trading_date.max() == dates[-6]
    assert ds.label_end_date.max() == dates[-1]
    assert (ds.label_end_date > ds.trading_date).all()


def test_target_definition(panel):
    prices, _ = panel
    ds, _ = build_dataset(prices)
    assert set(ds[TARGET_COL].unique()) <= {0, 1}
    assert ((ds[RETURN_COL] > 0).astype(int) == ds[TARGET_COL]).all()


def test_features_exclude_outcomes_and_ids():
    assert not set(FEATURES) & set(NON_FEATURE_COLS)
    assert not any("forward" in f or "target" in f or "label" in f for f in FEATURES)


def test_features_do_not_change_when_future_changes(panel):
    """Leakage test: rewriting every price AFTER time t must not change any feature at or before t."""
    prices, dates = panel
    t = dates[100]
    base, _ = build_dataset(prices)
    mutated = prices.copy()
    future = mutated.trading_date > t
    mutated.loc[future, ["open", "high", "low", "close"]] *= 3.7
    mutated.loc[future, "volume"] *= 5
    # indicators in the DB are causal, so recompute them on the mutated series the same way the repo does
    from marketsense.features.technical import compute_indicators
    for sid, g in mutated.groupby("stock_id"):
        ind = compute_indicators(g[["trading_date", "close"]].reset_index(drop=True))
        for c in ("rsi", "macd", "macd_signal", "macd_histogram"):
            mutated.loc[g.index, c] = ind[c].to_numpy()
    after, _ = build_dataset(mutated)
    a = base[base.trading_date <= t].set_index(["symbol", "trading_date"])[FEATURES]
    b = after[after.trading_date <= t].set_index(["symbol", "trading_date"])[FEATURES]
    common = a.index.intersection(b.index)
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common], check_exact=False, rtol=1e-9)
    assert len(common) > 150   # 3 stocks x ~68 rows after the 33-row indicator warm-up


def test_corp_action_flag_catches_large_discontinuity(panel):
    prices, dates = panel
    prices = prices.copy()
    m = (prices.symbol == "AAA") & (prices.trading_date >= dates[90])
    prices.loc[m, ["open", "high", "low", "close"]] *= 0.6              # 40% raw drop (e.g. bonus issue)
    ds, rep = build_dataset(prices, max_daily_move=0.10)
    flagged = ds[(ds.symbol == "AAA") & ds.corp_action_suspect_5d].trading_date
    assert set(flagged) == set(dates[85:90])                             # the 5 rows whose window contains the drop
    assert not ds[(ds.symbol == "BBB")].corp_action_suspect_5d.any()


def test_sparse_stock_excluded_and_reported(panel):
    prices, dates = panel
    keep = prices[(prices.symbol != "CCC") | (prices.trading_date.isin(dates[::3]))]
    ds, rep = build_dataset(keep, min_stock_coverage=0.8)
    assert "CCC" in rep.excluded_stocks and "CCC" not in set(ds.symbol)


def test_stray_date_not_a_session(panel):
    prices, dates = panel
    stray = prices[(prices.symbol == "AAA") & (prices.trading_date == dates[10])].copy()
    stray["trading_date"] = pd.Timestamp("2023-02-04")                   # a Saturday that only one stock has
    cal, dropped = build_calendar(pd.concat([prices, stray]))
    assert pd.Timestamp("2023-02-04") not in cal and "2023-02-04" in dropped
