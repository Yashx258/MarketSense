import pandas as pd
import pytest

from marketsense.features.model_b import VARIANTS, add_sentiment_features


def test_cutoff_and_rolling_market_and_company_features():
    calendar = pd.bdate_range("2026-08-03", periods=12)
    friday, monday, tuesday = calendar[9:]
    base = pd.MultiIndex.from_product(
        [["AAA", "BBB", "CCC"], calendar[5:]], names=["symbol", "trading_date"],
    ).to_frame(index=False)
    base["target_direction_5d"] = 1
    examples = [
        (1, f"{friday.date()} 17:00:00+05:00", ["AAA", "BBB"], 0.1),
        (2, f"{friday.date()} 17:01:00+05:00", ["AAA", "AAA"], 0.9),
        (3, f"{monday.date()} 10:00:00+05:00", [], 0.8),
        (4, f"{tuesday.date()} 17:01:00+05:00", ["CCC"], 1.0),
        (5, None, ["AAA"], 1.0),
        (6, f"{calendar[6].date()} 11:00:00+05:00", ["AAA"], 0.6),
    ]
    articles = pd.DataFrame(
        [(news_id, variant, timestamp, symbols, pos if variant == "headline" else 1 - pos,
          1 - pos if variant == "headline" else pos, 0.0)
         for variant in VARIANTS for news_id, timestamp, symbols, pos in examples],
        columns=["news_id", "text_variant", "published_at", "matched_tickers", "pos", "neg", "neu"],
    )
    result, groups = add_sentiment_features(base, articles, calendar)
    pd.testing.assert_frame_equal(result[base.columns], base)
    assert [len(group) for group in groups.values()] == [36, 36]
    assert result.notna().all().all()
    rows = result.set_index(["symbol", "trading_date"])
    friday_a = rows.loc[("AAA", friday)]
    monday_a = rows.loc[("AAA", monday)]
    assert friday_a.hl_market_news_count_1d == friday_a.hl_company_news_count_1d == 1
    assert friday_a.hl_market_news_count_3d == 1
    assert friday_a.hl_market_news_count_5d == 2
    assert friday_a.hl_company_pos_1d == pytest.approx(0.1)
    assert friday_a.hl_lead_company_pos_1d == pytest.approx(0.9)
    assert monday_a.hl_company_news_count_1d == 1
    assert monday_a.hl_company_news_count_3d == 2
    assert monday_a.hl_market_news_count_1d == 2
    assert monday_a.hl_market_news_count_3d == 3
    assert monday_a.hl_market_pos_3d == pytest.approx(0.6)
    assert monday_a.hl_company_net_sentiment_1d == pytest.approx(0.8)
    assert rows.loc[("BBB", friday), "hl_company_news_count_1d"] == 1
    assert rows.loc[("BBB", monday), "hl_has_company_news_1d"] == 0
    assert rows.loc[("CCC", monday), "hl_market_pos_3d"] == monday_a.hl_market_pos_3d
    assert rows.loc[("CCC", tuesday), "hl_has_company_news_5d"] == 0
    assert rows.loc[("AAA", tuesday), "hl_has_news_1d"] == 0
