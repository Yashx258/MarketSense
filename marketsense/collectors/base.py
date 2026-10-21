"""Shared plumbing for pluggable, terms-respecting collectors."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from urllib import robotparser
from urllib.parse import urlparse

log = logging.getLogger(__name__)
USER_AGENT = "MarketSenseAcademicBot/0.1 (student research project)"


class NewsFetchError(RuntimeError):
    """A feed could not be fetched or parsed (network, 5xx, malformed XML, not a feed)."""


class NewsFetchBlocked(NewsFetchError):
    """The source refused or disallowed access (robots.txt, HTTP 401/403/429). Do NOT retry or work around."""


def robots_allows(url: str, user_agent: str = USER_AGENT, http=None, timeout: float = 10.0) -> bool:
    """Is `user_agent` allowed to fetch `url` according to the site's robots.txt?

    Fetched with `requests` (explicit UA + timeout; urllib's default UA is often answered with 403) and parsed
    with the stdlib parser. Mirrors the stdlib's status handling:
      401/403 -> disallowed; other 4xx (e.g. 404, no robots.txt) -> allowed; 5xx / network error -> disallowed
    (fail CLOSED when we cannot tell).
    """
    import requests
    http = http or requests
    p = urlparse(url)
    try:
        r = http.get(f"{p.scheme}://{p.netloc}/robots.txt", headers={"User-Agent": user_agent}, timeout=timeout)
    except Exception as exc:                                    # noqa: BLE001 - any failure => fail closed
        log.warning("Could not read robots.txt for %s (%s); treating as disallowed", p.netloc, exc)
        return False
    if r.status_code in (401, 403):
        return False
    if 400 <= r.status_code < 500:
        return True
    if r.status_code != 200:
        return False
    rp = robotparser.RobotFileParser()
    rp.parse(r.text.splitlines())
    return rp.can_fetch(user_agent, url)


@dataclass
class NewsItem:
    headline: str
    source: str
    url: str | None = None
    article_text: str | None = None
    published_at: datetime | None = None      # timezone-aware, UTC. None = the source gave no usable timestamp
    language: str = "en"
    symbols: list[str] = field(default_factory=list)   # tickers the SOURCE tagged explicitly (rare); text matching is separate


class NewsCollector(ABC):
    """Implement `fetch()` for one source. Prefer official RSS/APIs; check ToS + robots.txt first."""
    name: str
    enabled: bool = False           # True only for sources whose feed + terms have been reviewed AND implemented

    @abstractmethod
    def fetch(self, since: datetime | None = None) -> list[NewsItem]: ...


class FundamentalsCollector(ABC):
    name: str

    @abstractmethod
    def fetch(self, symbols: list[str]) -> list[dict]:
        """Return dicts: symbol, as_of_date, pe_ratio, eps, market_cap."""
