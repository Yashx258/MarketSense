"""Publication-time sentiment features on the Model A market calendar."""
import numpy as np
import pandas as pd

from marketsense.sentiment.features import cutoff_for_trading_date

VARIANTS = {"headline": "hl", "headline_lead": "hl_lead"}
WINDOWS = (1, 3, 5)
SCORES = ["pos", "neg", "neu", "net_sentiment"]


def _rolling_features(articles, calendar, prefix, scope):
    totals = articles.groupby("trading_date")[SCORES].sum().reindex(calendar, fill_value=0.0)
    counts = articles.groupby("trading_date").size().reindex(calendar, fill_value=0)
    result = {}
    flag = "has_news" if scope == "market" else "has_company_news"
    for width in WINDOWS:
        suffix = f"{width}d"
        count = counts.rolling(width, min_periods=1).sum().astype("int32")
        sums = totals.rolling(width, min_periods=1).sum()
        means = sums.div(count.where(count.gt(0)), axis=0).fillna(0.0)
        for score in SCORES:
            result[f"{prefix}_{scope}_{score}_{suffix}"] = means[score]
        result[f"{prefix}_{scope}_news_count_{suffix}"] = count
        result[f"{prefix}_{flag}_{suffix}"] = count.gt(0).astype("int8")
    return pd.DataFrame(result, index=calendar)


def add_sentiment_features(base: pd.DataFrame, articles: pd.DataFrame, calendar: pd.DatetimeIndex):
    """Attach article-weighted market and company aggregates without altering Model A columns.

    An exact timestamp is allocated to the first cutoff at or after publication.
    Articles before or at the first available cutoff and after the last are
    excluded. The calendar includes the warm-up sessions preceding Model A rows.
    """
    if (not calendar.is_unique or not calendar.is_monotonic_increasing
            or not base["trading_date"].isin(calendar[5:]).all()):
        raise ValueError("Expected a sorted market calendar with five warm-up sessions.")
    if articles.duplicated(["news_id", "text_variant"]).any():
        raise ValueError("Expected one score per article and variant.")

    cutoffs = pd.DatetimeIndex([cutoff_for_trading_date(d.date(), "17:00") for d in calendar])
    publication = pd.to_datetime(articles["published_at"], utc=True)
    bins = cutoffs.searchsorted(publication, side="left")
    in_range = publication.notna() & (bins > 0) & (bins < len(calendar))
    dated = articles.loc[in_range].copy()
    dated["trading_date"] = calendar.take(bins[in_range])
    dated["net_sentiment"] = dated["pos"] - dated["neg"]
    date_index = pd.Index(base["trading_date"])
    key_index = pd.MultiIndex.from_frame(base[["symbol", "trading_date"]])
    symbols = pd.Index(base["symbol"].unique())

    blocks = [base]
    groups = {}
    for variant, prefix in VARIANTS.items():
        scored = dated[dated["text_variant"] == variant]
        market = _rolling_features(scored, calendar, prefix, "market").reindex(date_index)
        tagged = scored.explode("matched_tickers").rename(columns={"matched_tickers": "symbol"})
        tagged = tagged.drop_duplicates(["news_id", "symbol"])
        by_symbol = dict(tuple(tagged.groupby("symbol")))
        empty = tagged.iloc[:0]
        company = pd.concat(
            {symbol: _rolling_features(by_symbol.get(symbol, empty), calendar, prefix, "company")
             for symbol in symbols},
            names=["symbol", "trading_date"],
        ).reindex(key_index)
        groups[prefix] = list(market.columns) + list(company.columns)
        blocks.extend([market.set_axis(base.index), company.set_axis(base.index)])

    result = pd.concat(blocks, axis=1)
    columns = [column for group in groups.values() for column in group]
    if not np.isfinite(result[columns].to_numpy()).all():
        raise RuntimeError("Non-finite sentiment features.")
    return result, groups
