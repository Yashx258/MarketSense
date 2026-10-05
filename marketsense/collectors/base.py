"""Shared plumbing for pluggable, terms-respecting collectors."""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from urllib import robotparser
from urllib.parse import urlparse

log = logging.getLogger(__name__)
USER_AGENT = "MarketSenseAcademicBot/0.1 (student research project)"


def robots_allows(url: str, user_agent: str = USER_AGENT) -> bool:
    """Fail CLOSED: if robots.txt can't be read, treat the URL as disallowed."""
    p = urlparse(url)
    rp = robotparser.RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        log.warning("Could not read robots.txt for %s; skipping", p.netloc)
        return False
    return rp.can_fetch(user_agent, url)


@dataclass
class NewsItem:
    headline: str
    source: str
    url: str | None = None
    article_text: str | None = None
    published_at: datetime | None = None
    language: str = "en"
    symbol: str | None = None       # optional stock tag; None = market-wide


class NewsCollector(ABC):
    """Implement `fetch()` for one source. Prefer official RSS/APIs; check ToS + robots.txt first."""
    name: str
    enabled: bool = False           # flip only after reviewing the site's terms

    @abstractmethod
    def fetch(self, since: datetime | None = None) -> list[NewsItem]: ...


class FundamentalsCollector(ABC):
    name: str

    @abstractmethod
    def fetch(self, symbols: list[str]) -> list[dict]:
        """Return dicts: symbol, as_of_date, pe_ratio, eps, market_cap."""
