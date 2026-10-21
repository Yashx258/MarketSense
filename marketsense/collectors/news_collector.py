"""News collectors.

IMPLEMENTED (feed + terms reviewed, see README "News & sentiment"):
    dawn_business   Dawn Business RSS  https://www.dawn.com/feeds/business

NOT IMPLEMENTED (feed endpoint and terms NOT verified, so deliberately left as placeholders):
    business_recorder, profit_pakistan_today

Storage lives in news_store.py (`store_news` is re-exported here for backward compatibility).
"""
from __future__ import annotations

import calendar
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit

import feedparser
import requests

from marketsense.collectors.base import NewsCollector, NewsFetchBlocked, NewsFetchError, NewsItem, robots_allows
from marketsense.collectors.news_store import normalize_url, store_news  # noqa: F401  (store_news re-exported)
from marketsense.config import Settings

log = logging.getLogger(__name__)

_LINK_ONLY = re.compile(r"https?://\S+")
_FUTURE_TOLERANCE = timedelta(minutes=10)


# ------------------------------------------------------------------------------------------ HTML -> text
class _TextExtractor(HTMLParser):
    _BLOCK = {"p", "br", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "tr", "hr", "figure"}
    _SKIP = {"script", "style", "iframe", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str | None) -> str:
    """Plain text from feed HTML: scripts/iframes dropped, block tags become line breaks, whitespace collapsed."""
    p = _TextExtractor()
    p.feed(html or "")
    p.close()
    lines = (" ".join(line.split()) for line in "".join(p.parts).splitlines())
    return "\n".join(line for line in lines if line)


# ------------------------------------------------------------------------------------------ generic RSS collector
class RssNewsCollector(NewsCollector):
    """Fetch + parse one RSS 2.0 feed politely: robots.txt check, explicit User-Agent, timeout, bounded retries.

    HTTP 401/403/429 are NEVER retried or worked around (NewsFetchBlocked). `http` and `sleep` are injectable so
    unit tests never touch the network.
    """
    name: str = ""
    feed_url: str = ""
    language: str = "en"
    max_attempts: int = 3

    def __init__(self, settings: Settings | None = None, http=None, sleep=time.sleep, now=None,
                 feed_url: str | None = None, store_article_text: bool | None = None):
        self.settings = settings or Settings()
        self.http = http or requests.Session()
        self._sleep = sleep
        self._now = now or (lambda: datetime.now(timezone.utc))
        if feed_url:
            self.feed_url = feed_url
        # Dawn's terms allow copying content only for personal non-commercial use -> article bodies are OPT-IN.
        self.store_article_text = self.settings.news_store_article_text if store_article_text is None else store_article_text
        self.stats: dict[str, int] = {}

    # ---- network
    def fetch_bytes(self) -> bytes:
        ua, timeout = self.settings.news_user_agent, self.settings.news_http_timeout
        if not robots_allows(self.feed_url, ua, http=self.http, timeout=timeout):
            raise NewsFetchBlocked(f"robots.txt disallows (or could not be read for) {self.feed_url}")
        headers = {"User-Agent": ua, "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8"}
        last = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            try:
                r = self.http.get(self.feed_url, headers=headers, timeout=timeout)
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if r.status_code in (401, 403, 429):
                    raise NewsFetchBlocked(f"HTTP {r.status_code} from {self.feed_url} (Retry-After={r.headers.get('Retry-After')}). "
                                           "Stop and review the source's terms; do not retry or work around.")
                if r.status_code == 200:
                    return r.content
                if r.status_code < 500:
                    raise NewsFetchError(f"HTTP {r.status_code} from {self.feed_url}")
                last = f"HTTP {r.status_code}"
            if attempt < self.max_attempts:
                log.warning("fetch attempt %d/%d failed (%s); retrying", attempt, self.max_attempts, last)
                self._sleep(2 ** (attempt - 1))
        raise NewsFetchError(f"giving up after {self.max_attempts} attempts: {last}")

    def fetch(self, since: datetime | None = None) -> list[NewsItem]:
        return self.parse(self.fetch_bytes(), since)

    # ---- hooks a source may override
    def canonical_url(self, entry) -> str | None:
        return entry.get("link") or None

    # ---- parsing (pure: no network, used directly by tests)
    def _published(self, entry) -> datetime | None:
        """Aware UTC datetime, or None. Never guesses a zone for naive strings; rejects future-dated items."""
        dt = None
        raw = entry.get("published") or entry.get("updated")
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                dt = None
            if dt is not None and dt.tzinfo is None:
                dt = None
        if dt is None and entry.get("published_parsed"):          # non-RFC822 formats (feedparser normalises to UTC)
            dt = datetime.fromtimestamp(calendar.timegm(entry["published_parsed"]), tz=timezone.utc)
        if dt is None:
            return None
        dt = dt.astimezone(timezone.utc)
        if dt > self._now() + _FUTURE_TOLERANCE:
            self.stats["future_dated"] += 1
            log.warning("future-dated item %s ignored (published_at=NULL)", dt.isoformat())
            return None
        return dt

    def parse(self, content: bytes, since: datetime | None = None) -> list[NewsItem]:
        feed = feedparser.parse(content)
        # An HTML block/captcha page served with HTTP 200 parses as bozo=False, version='' and zero entries: without the
        # version check it would be reported as a successful (empty) run. A genuine empty feed has version 'rss20' etc.
        if not feed.entries and (feed.get("bozo") or not feed.get("version")):
            raise NewsFetchError(f"could not parse feed ({feed.get('bozo_exception') or 'response is not an RSS/Atom feed'})")
        self.stats = dict(entries=0, items=0, no_title=0, no_timestamp=0, future_dated=0, older_than_since=0,
                          with_body=0, link_only_body=0)
        lang = (feed.feed.get("language") or self.language).replace("_", "-").split("-")[0].lower()[:10] or self.language
        items: list[NewsItem] = []
        for e in feed.entries:                      # NB: feeds are NOT sorted by date; never stop at the first old item
            self.stats["entries"] += 1
            headline = " ".join((e.get("title") or "").split())
            if not headline:
                self.stats["no_title"] += 1
                continue
            published = self._published(e)
            if published is None:
                self.stats["no_timestamp"] += 1
            elif since is not None and published < since:
                self.stats["older_than_since"] += 1
                continue
            body_html = (e.get("content") or [{}])[0].get("value") or e.get("summary") or ""
            body = html_to_text(body_html)
            if body and _LINK_ONLY.fullmatch(body):
                self.stats["link_only_body"] += 1          # stub item: body is only a pointer to another article
                body = ""
            self.stats["with_body"] += bool(body)
            items.append(NewsItem(headline=headline, source=self.name, url=self.canonical_url(e),
                                  article_text=body if (body and self.store_article_text) else None,
                                  published_at=published, language=lang))
        self.stats["items"] = len(items)
        items.sort(key=lambda i: (i.published_at is None, -(i.published_at.timestamp() if i.published_at else 0)))
        return items


# ------------------------------------------------------------------------------------------ sources
class DawnBusinessCollector(RssNewsCollector):
    """Dawn Business RSS.

    VERIFIED (live fetch): RSS 2.0, <ttl>60</ttl>, pubDate with +0500 offset, full-article HTML in content:encoded,
    <guid> is the stable id URL (https://www.dawn.com/news/<id>) while <link> adds a headline slug, the 'business'
    feed also carries World/Pakistan items, a few items are link-only stubs, and items are not date-sorted.
    Terms (https://www.dawn.com/terms/): copying/reuse of content only for personal, non-commercial use;
    anything else needs Dawn's prior written permission. => bodies are opt-in (NEWS_STORE_ARTICLE_TEXT).
    """
    name = "dawn_business"
    feed_url = "https://www.dawn.com/feeds/business"
    enabled = True

    def canonical_url(self, entry) -> str | None:
        # the id URL survives headline edits (the slug in <link> may change), so it is the better dedup key
        guid = normalize_url(entry.get("id") or "")
        if guid and urlsplit(guid).hostname in ("www.dawn.com", "dawn.com") and "/news/" in guid:
            return guid
        return entry.get("link") or None


class _UnverifiedSource(NewsCollector):
    base_url: str = ""

    def fetch(self, since: datetime | None = None) -> list[NewsItem]:
        raise NotImplementedError(
            f"{self.name}: not implemented. The feed endpoint and terms of use have NOT been verified. "
            f"Review {self.base_url}/robots.txt and the site's terms, confirm an RSS feed/licensed API exists, "
            "then subclass RssNewsCollector (see DawnBusinessCollector) and set enabled=True.")


class BusinessRecorderCollector(_UnverifiedSource):
    name, base_url = "business_recorder", "https://www.brecorder.com"


class ProfitPakistanTodayCollector(_UnverifiedSource):
    name, base_url = "profit_pakistan_today", "https://profit.pakistantoday.com.pk"


REGISTRY: dict[str, type[NewsCollector]] = {c.name: c for c in
    (DawnBusinessCollector, BusinessRecorderCollector, ProfitPakistanTodayCollector)}
