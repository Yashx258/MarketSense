"""Persisting collected news: URL normalisation, de-duplication, ticker matching.

Design (follows the existing schema; see sql/schema.sql):
  * news_articles.url has a PARTIAL unique index (WHERE url IS NOT NULL). PostgreSQL requires the predicate in
    ON CONFLICT, otherwise the insert fails. That was the bug in the first store_news().
  * Items WITHOUT a url are de-duplicated best-effort on (source, headline, published_at). Items with neither a
    url nor a timestamp are rejected: they cannot be de-duplicated and could never be used leakage-safely.
  * One row per article. news_articles.stock_id is set only when exactly ONE stock is mentioned; every mentioned
    stock (one or many) is recorded in news_article_stocks. No match => stock_id NULL (market-wide article).
  * All timestamps are converted to UTC before binding. Naive datetimes are NOT guessed at (PKT vs UTC is a
    5-hour leakage risk): they are stored as NULL and the article is excluded from features.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import insert, select, text
from sqlalchemy.orm import Session

from marketsense.collectors.base import NewsItem
from marketsense.db.connection import dialect_insert
from marketsense.db.models import NewsArticle, NewsArticleStock, Stock

log = logging.getLogger(__name__)

_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src", "_ga", "yclid", "msclkid"}
_DEFAULT_PORTS = {"http": 80, "https": 443}

# All-caps tokens that are NOT stock symbols even if a symbol with the same letters ever exists.
GENERIC_ACRONYMS = frozenset({
    "PSX", "KSE", "SBP", "FBR", "IMF", "SECP", "NEPRA", "OGRA", "GDP", "CPI", "USD", "PKR", "IPO", "CEO", "CFO",
    "EPS", "ECC", "PIB", "NAB", "PTI", "PPP", "PML", "ADB", "UAE", "USA", "NATO", "OPEC", "EFF", "RSF", "SLA",
    "AGM", "EOGM", "FY", "YOY", "MOM", "QOQ", "SPI", "CPEC", "LNG", "RLNG", "CNG", "LPG", "HSD", "MS", "PBS",
})
_NAME_SUFFIXES = {"limited", "ltd", "pvt", "private", "corporation", "corp", "company", "co", "plc", "inc", "the"}
_SYMBOL_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{2,9})(?![A-Za-z0-9])")


# ------------------------------------------------------------------------------------------ URL normalisation
def normalize_url(url: str | None) -> str | None:
    """Canonical form used as the dedup key. Returns None for empty / non-http(s) / host-less input.

    lower-case scheme+host, drop fragment and default port, remove tracking params (utm_*, fbclid, ...),
    sort the remaining query, strip a trailing slash (except the root path).
    """
    if not url or not url.strip():
        return None
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in ("http", "https") or not host:
        return None
    netloc = host if parts.port in (None, _DEFAULT_PORTS[scheme]) else f"{host}:{parts.port}"
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                             if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS))
    return urlunsplit((scheme, netloc, path, query, ""))


# ------------------------------------------------------------------------------------------ ticker matching
def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower().replace("&", " and ")).split())


def _name_stem(company_name: str | None) -> str | None:
    """'Lucky Cement Limited' -> 'lucky cement'. Single short words ('Engro') are too ambiguous -> None."""
    if not company_name:
        return None
    tokens = _norm(company_name).split()
    while tokens and tokens[-1] in _NAME_SUFFIXES:
        tokens.pop()
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    stem = " ".join(tokens)
    return stem if (len(tokens) >= 2 or len(stem) >= 8) else None


@dataclass
class StockIndex:
    by_symbol: dict[str, int] = field(default_factory=dict)      # 'LUCK' -> stock_id
    names: list[tuple[str, int]] = field(default_factory=list)   # [('lucky cement', stock_id), ...] longest first


def build_stock_index(session: Session) -> StockIndex:
    rows = session.execute(select(Stock.stock_id, Stock.symbol, Stock.company_name)).all()
    idx = StockIndex(by_symbol={sym.upper(): sid for sid, sym, _ in rows})
    stems: dict[str, set[int]] = {}
    for sid, _, name in rows:
        stem = _name_stem(name)
        if stem:
            stems.setdefault(stem, set()).add(sid)
    # a stem shared by several stocks (e.g. two share classes) is ambiguous: never guess
    idx.names = sorted(((st, next(iter(ids))) for st, ids in stems.items() if len(ids) == 1),
                       key=lambda t: -len(t[0]))
    return idx


def match_stocks(text_: str, index: StockIndex) -> dict[int, str]:
    """{stock_id: 'symbol'|'name'} for stocks the text mentions. Conservative: precision over recall.

    * symbol: an UPPER-CASE token (>=3 chars) equal to a known symbol and not a generic acronym ('PSX', 'IMF', ...)
    * name:   normalised company-name stem (>=2 words or >=8 chars) appearing on word boundaries
    """
    found: dict[int, str] = {}
    for tok in _SYMBOL_TOKEN.findall(text_ or ""):
        sid = index.by_symbol.get(tok)
        if sid is not None and tok not in GENERIC_ACRONYMS:
            found[sid] = "symbol"
    padded = f" {_norm(text_ or '')} "
    for stem, sid in index.names:
        if sid not in found and f" {stem} " in padded:
            found[sid] = "name"
    return found


# ------------------------------------------------------------------------------------------ storing
@dataclass
class StoreReport:
    received: int = 0
    inserted: int = 0
    duplicates: int = 0              # already stored (same normalised url, or same source+headline+time when no url)
    duplicates_in_batch: int = 0     # repeated inside this batch
    rejected: int = 0                # empty headline, or neither a usable url nor a timestamp
    stored_without_timestamp: int = 0   # stored, but excluded from features (no published_at)
    articles_with_stocks: int = 0
    stock_links: int = 0
    unknown_symbols: dict[str, int] = field(default_factory=dict)   # explicit source tickers not in `stocks`

    def summary(self) -> str:
        lines = [f"received={self.received} inserted={self.inserted} duplicates={self.duplicates} "
                 f"duplicates_in_batch={self.duplicates_in_batch} rejected={self.rejected}",
                 f"articles_with_stocks={self.articles_with_stocks} stock_links={self.stock_links} "
                 f"stored_without_timestamp={self.stored_without_timestamp}"]
        if self.unknown_symbols:
            lines.append("unknown source tickers (not in stocks): " + ", ".join(f"{k}x{v}" for k, v in sorted(self.unknown_symbols.items())))
        return "\n".join(lines)


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None or dt.utcoffset() is None:
        log.warning("Naive datetime %s refused (timezone unknown); storing NULL published_at", dt)
        return None
    return dt.astimezone(timezone.utc)


def _insert_article(session: Session, row: dict) -> int | None:
    """Insert one article; return news_id, or None if it already exists."""
    if row["url"]:
        stmt = (dialect_insert(session)(NewsArticle).values(**row)
                .on_conflict_do_nothing(index_elements=["url"], index_where=text("url IS NOT NULL"))
                .returning(NewsArticle.news_id))
        return session.execute(stmt).scalar_one_or_none()
    existing = session.execute(select(NewsArticle.news_id).where(
        NewsArticle.url.is_(None), NewsArticle.source == row["source"],
        NewsArticle.headline == row["headline"], NewsArticle.published_at == row["published_at"])).first()
    if existing:
        return None
    return session.execute(insert(NewsArticle).values(**row).returning(NewsArticle.news_id)).scalar_one()


def store_news(session: Session, items: list[NewsItem], index: StockIndex | None = None) -> StoreReport:
    """Persist items idempotently. Caller owns the transaction (use `session_scope`). Safe to re-run."""
    rep = StoreReport(received=len(items))
    index = index or build_stock_index(session)
    seen: set[tuple] = set()

    # oldest first: if the same article repeats in a batch (e.g. an edited headline), the EARLIEST published_at is the
    # time it first became available - the only leakage-safe choice. Items without a timestamp go last.
    ordered = sorted(items, key=lambda i: (i.published_at is None or i.published_at.tzinfo is None,
                                           _to_utc(i.published_at).timestamp() if _to_utc(i.published_at) else 0.0))

    for it in ordered:
        headline = " ".join((it.headline or "").split())
        url = normalize_url(it.url)
        published = _to_utc(it.published_at)
        if not headline or (url is None and published is None):
            rep.rejected += 1
            continue
        key = ("url", url) if url else ("nourl", it.source, headline, published)
        if key in seen:
            rep.duplicates_in_batch += 1
            continue
        seen.add(key)

        # --- which stocks does it mention? (explicit source tags + conservative text matching)
        stocks: dict[int, str] = {}
        for sym in it.symbols:
            sid = index.by_symbol.get((sym or "").upper())
            if sid is None:
                rep.unknown_symbols[(sym or "").upper()] = rep.unknown_symbols.get((sym or "").upper(), 0) + 1
            else:
                stocks[sid] = "source"
        for sid, how in match_stocks(f"{headline}. {it.article_text or ''}", index).items():
            stocks.setdefault(sid, how)

        row = dict(stock_id=next(iter(stocks)) if len(stocks) == 1 else None, headline=headline,
                   article_text=it.article_text or None, source=it.source, url=url,
                   published_at=published, language=(it.language or "en")[:10])
        news_id = _insert_article(session, row)
        if news_id is None:
            rep.duplicates += 1
            continue
        rep.inserted += 1
        rep.stored_without_timestamp += published is None
        if stocks:
            session.execute(dialect_insert(session)(NewsArticleStock).values(
                [dict(news_id=news_id, stock_id=sid, matched_by=how) for sid, how in stocks.items()]
            ).on_conflict_do_nothing())
            rep.articles_with_stocks += 1
            rep.stock_links += len(stocks)

    session.flush()
    return rep
