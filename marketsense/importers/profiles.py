"""Import profiles: how a dataset's columns map onto news_articles. Only HF's schema is verified (from its viewer);
everything else goes through the `generic` profile after `--inspect`, never through a guessed mapping."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImportProfile:
    name: str
    description: str
    source: str | None = None                  # value stored in news_articles.source (None => --source-name required)
    doc_type: str = "news"
    columns: dict[str, tuple[str, ...]] = field(default_factory=dict)   # canonical field -> candidate column names
    tz: str | None = None                      # timezone of NAIVE times in the file (None => time-of-day is dropped)
    dayfirst: bool | None = None
    language: str = "en"
    ignore: tuple[str, ...] = ()               # columns deliberately NOT imported (reported)


# canonical fields: headline, published, time, url, text, symbol, record_id, language
GENERIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "headline": ("headline", "title", "story heading", "news headline", "heading", "news", "story title"),
    "published": ("published at", "published", "publish date", "published date", "date", "datetime", "timestamp",
                  "pub date", "created at", "date time", "publishing date"),
    "time": ("time",),
    "url": ("url", "link", "article url", "source url", "pdf url", "href"),
    "text": ("article text", "text", "content", "body", "story excerpt", "excerpt", "summary", "description",
             "extracted text"),
    "symbol": ("symbol", "ticker", "stock", "stock symbol", "code", "scrip"),
    "record_id": ("id", "news id", "article id", "doc id", "record id"),
    "language": ("language", "lang"),
    "category": ("category", "section", "relevance bucket", "topic"),   # dataset-supplied label, stored as-is
    "source": (),                                                     # per-row publisher; only via --map source=<column>
}

PROFILES: dict[str, ImportProfile] = {
    "psx_news_primary_candidates": ImportProfile(
        name="psx_news_primary_candidates",
        description="psx_news_primary_sentiment_candidates.csv (schema verified from the file: article_id,source,url,published_at,"
                    "published_date_pkt,title,clean_text,word_count,companies,is_market_wide,content_hash,...,relevance_bucket,...). "
                    "Timestamps carry explicit +05:00 offsets; `source` holds the publisher per row; `companies` is pipe-separated tickers.",
        source=None, doc_type="news", tz=None,
        columns={"headline": ("title",), "published": ("published at",), "url": ("url",), "text": ("clean text",),
                 "symbol": ("companies",), "record_id": ("article id",), "source": ("source",), "category": ("relevance bucket",)},
        # derived by whoever prepared the file; we neither trust nor reuse them (our own checks run instead)
        ignore=("published date pkt", "word count", "is market wide", "content hash", "company count", "cleaned word count",
                "text quality", "short or ambiguous title", "classification reason", "classification confidence",
                "include primary sentiment", "needs manual review", "title date duplicate", "published at utc normalized",
                "published date pkt normalized", "sentiment score", "sentiment impact", "sentiment signals",
                "company", "period ended")),
    "generic": ImportProfile(
        name="generic", description="Auto-detects common column names; run --inspect first and override with --map.",
        columns=GENERIC_COLUMNS),
}
