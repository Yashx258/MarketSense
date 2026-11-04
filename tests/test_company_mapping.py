"""Company mapping: evidence-based links only. Regression cases come from the user's real 4,385-article file."""
from sqlalchemy import select

from marketsense.collectors.base import NewsItem
from marketsense.collectors.news_store import (
    AMBIGUOUS_TICKER_TOKENS,
    GENERIC_ACRONYMS,
    build_stock_index,
    load_aliases,
    match_stocks,
    store_news,
)
from marketsense.db.models import NewsArticle, NewsArticleStock, Stock
from tests.conftest import utc


def idx_with(s, extra=(), aliases=None):
    s.add_all([Stock(symbol=sym, company_name=name) for sym, name in extra]); s.commit()
    return build_stock_index(s, aliases=aliases or {})


def test_pol_in_pakistani_news_means_petroleum_not_pakistan_oilfields(news_session):
    idx = idx_with(news_session, [("POL", "Pakistan Oilfields Limited"), ("NPL", "Nishat Power Limited"), ("PTC", "Pakistan Telecommunication Company Limited")])
    # real headlines from the audited file: all 13 were false positives of the first matcher
    for h in ["New POL taxes may burden masses with Rs25bn", "FBR asked to crack down on smuggled Iranian POL products",
              "OGRA takes notice of 'artificial' shortage of POL products", "Banks' NPL ratio rises as NPLs climb"]:
        assert match_stocks(h, idx) == {}, h
    assert {"POL", "NPL"} <= AMBIGUOUS_TICKER_TOKENS
    # ...but evidence that IS specific still links
    sid = idx.by_symbol["POL"]
    assert match_stocks("Pakistan Oilfields posts record profit", idx) == {sid: "name"}              # by company name
    assert match_stocks("PTC promotes Syed Ali Akbar to CEO", idx) == {idx.by_symbol["PTC"]: "symbol"}   # unambiguous token stays valid


def test_ambiguous_ticker_still_links_when_the_source_tags_it_explicitly(news_session):
    s = news_session
    idx = idx_with(s, [("POL", "Pakistan Oilfields Limited")])
    store_news(s, [NewsItem(headline="New POL taxes may burden masses", source="x", url="https://x.com/1",
                            published_at=utc(2026, 1, 1), symbols=["POL"])], idx)
    how = s.execute(select(NewsArticleStock.matched_by)).scalars().all()
    assert how == ["source"]                                                                       # the dataset's own tag is honoured


def test_alias_file_loading_and_unknown_symbol_aliases_are_ignored(news_session, tmp_path):
    f = tmp_path / "aliases.csv"
    f.write_text("alias,symbol,note\nOGDCL,OGDC,x\nGHOSTCO,NOPE,points at a symbol we do not have\n, ,blank\n", encoding="utf-8")
    assert load_aliases(f) == {"OGDCL": "OGDC", "GHOSTCO": "NOPE"} and load_aliases(tmp_path / "missing.csv") == {}
    idx = build_stock_index(news_session, aliases=load_aliases(f))
    assert idx.alias_symbols == {"OGDCL": idx.by_symbol["OGDC"]}                                   # NOPE not in stocks -> dropped, never guessed
    assert match_stocks("GHOSTCO expands", idx) == {} and match_stocks("OGDCL strikes oil", idx) == {idx.by_symbol["OGDC"]: "alias"}


def test_source_evidence_beats_text_evidence_and_nothing_is_assigned_without_evidence(news_session):
    s = news_session
    idx = build_stock_index(s, aliases={})
    store_news(s, [NewsItem(headline="Lucky Cement posts profit", source="x", url="https://x.com/a", published_at=utc(2026, 1, 1), symbols=["LUCK"]),
                   NewsItem(headline="Budget talks continue in Islamabad", source="x", url="https://x.com/b", published_at=utc(2026, 1, 2))], idx)
    got = dict(s.execute(select(NewsArticle.headline, NewsArticle.stock_id)).all())
    assert got["Lucky Cement posts profit"] is not None and got["Budget talks continue in Islamabad"] is None
    assert s.execute(select(NewsArticleStock.matched_by)).scalars().all() == ["source"]            # explicit tag recorded first; text match did not overwrite it


def test_dawn_and_business_recorder_slug_and_id_urls_are_one_article(news_session):
    """Prevents double counting between an imported archive (slug URLs) and the RSS collectors (id URLs)."""
    s = news_session
    idx = build_stock_index(s, aliases={})
    imported = NewsItem(headline="Pak Suzuki to shut down production for four days", source="dawn_business",
                        url="https://www.dawn.com/news/1526991/pak-suzuki-to-shut-down-production-for-four-days", published_at=utc(2020, 1, 3, 5, 0))
    rss = NewsItem(headline="Pak Suzuki to shut down production for four days", source="dawn_business",
                   url="https://www.dawn.com/news/1526991", published_at=utc(2020, 1, 3, 5, 0))
    assert store_news(s, [imported], idx).inserted == 1
    again = store_news(s, [rss], idx)
    assert (again.inserted, again.duplicates) == (0, 1)
    br_a = NewsItem(headline="Rupee to rise", source="business_recorder", url="https://www.brecorder.com/news/40445246/rupee-to-rise", published_at=utc(2026, 10, 21, 3, 16))
    br_b = NewsItem(headline="Indian rupee to rise", source="business_recorder", url="https://www.brecorder.com/news/40445246/indian-rupee-to-rise", published_at=utc(2026, 10, 21, 3, 16))
    rep = store_news(s, [br_a, br_b], idx)
    assert (rep.inserted, rep.duplicates_in_batch) == (1, 1)                                       # the exact pair found in the real file
