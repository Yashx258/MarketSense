from marketsense.collectors.news_relevance import classify_news


def test_direct_psx_headline_is_included():
    d = classify_news("PSX: selling at bourse, KSE-100 down over 800 points")
    assert d.status == "include"


def test_sbp_monetary_policy_is_included():
    d = classify_news("Hike or hold: SBP to announce policy rate today")
    assert d.status == "include"


def test_listed_company_material_event_is_included():
    d = classify_news("Textile giant Gul Ahmed approves acquisition of 43.75 acres")
    assert d.status == "include"


def test_obituary_is_rejected():
    d = classify_news("Former army chief general (retd) Abdul Waheed Kakar dies aged 89")
    assert d.status == "reject"


def test_retail_gold_price_is_rejected():
    d = classify_news("Gold prices in Pakistan rise by Rs1,400 per tola")
    assert d.status == "reject"


def test_foreign_bond_story_is_rejected():
    d = classify_news("India bonds to come under selling pressure at start of new week")
    assert d.status == "reject"


def test_generic_bilateral_partnership_is_rejected():
    d = classify_news("Pakistan, Qatar explore new avenues for trade, investment cooperation")
    assert d.status == "reject"


def test_travel_cost_is_rejected():
    d = classify_news("Travel cost falls 7.9pc since May in Pakistan")
    assert d.status == "reject"


def test_auto_policy_opinion_is_review_not_auto_included():
    d = classify_news("Pakistan's auto policy needs predictable taxes, not just low ones")
    assert d.status == "review"


def test_fbr_reforms_is_review_not_auto_included():
    d = classify_news("FBR reforms, again?")
    assert d.status == "review"


def test_privatisation_policy_is_review_not_auto_included():
    d = classify_news("Govt introduces bill for umbrella privatisation entity")
    assert d.status == "review"


def test_partner_specific_trade_gap_is_reviewed_without_a_figure():
    d = classify_news("Pakistan's trade gap with Gulf partners narrows")
    assert d.status == "review"


def test_quantified_trade_metric_is_included():
    d = classify_news("Pakistan exports rise 12 percent in September")
    assert d.status == "include"


def test_general_farm_size_story_is_not_auto_included():
    d = classify_news("Brewing crisis of shrinking farm size in Pakistan")
    assert d.status == "review"


def test_editorial_tag_routes_to_review():
    d = classify_news("Pakistan inflation rises in September", categories=["Opinion"])
    assert d.status == "review"
