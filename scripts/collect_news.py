"""Collect news from ONE source and store it in PostgreSQL. Collection only - no sentiment scoring here.

  python -m scripts.collect_news --source dawn_business --check      # robots.txt + feed health, NO database
  python -m scripts.collect_news --source dawn_business --dry-run    # fetch + parse, print items, NO database
  python -m scripts.collect_news --source dawn_business              # fetch + store (needs NEWS_COLLECTION_ENABLED=true)

Options: --hours N (default 72) | --since 2026-10-01T00:00:00+05:00 | --limit N | --feed-url URL (mirror/testing)
Exit codes: 0 ok | 1 error (network, parse, database) | 2 refused (not enabled, robots/HTTP 403/429, unimplemented source)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from marketsense.collectors.base import NewsFetchBlocked, NewsFetchError
from marketsense.collectors.news_collector import REGISTRY, RssNewsCollector, store_news
from marketsense.config import Settings, setup_logging
from marketsense.db.connection import session_scope


def _since(args) -> datetime | None:
    if args.since:
        dt = datetime.fromisoformat(args.since)
        if dt.tzinfo is None:
            raise SystemExit("--since needs a timezone offset, e.g. 2026-10-01T00:00:00+05:00 (naive times are ambiguous)")
        return dt.astimezone(timezone.utc)
    return datetime.now(timezone.utc) - timedelta(hours=args.hours)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect news from one source.")
    ap.add_argument("--source", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--check", action="store_true", help="verify robots.txt + feed health only; no database")
    ap.add_argument("--dry-run", action="store_true", help="fetch and parse, print items; no database")
    ap.add_argument("--hours", type=float, default=72.0, help="keep items published in the last N hours (default 72)")
    ap.add_argument("--since", help="ISO-8601 with offset; overrides --hours")
    ap.add_argument("--limit", type=int, help="store at most N (newest) items")
    ap.add_argument("--feed-url", help="override the feed URL (mirror / local test server)")
    args = ap.parse_args(argv)
    setup_logging()
    settings = Settings()

    cls = REGISTRY[args.source]
    if not issubclass(cls, RssNewsCollector):
        try:
            cls().fetch()
        except NotImplementedError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
    if not (args.check or args.dry_run) and not settings.news_enabled:
        print("REFUSED: NEWS_COLLECTION_ENABLED is not 'true'. Review the source's terms (README), then set it in .env.\n"
              "         (--check and --dry-run work without it.)", file=sys.stderr)
        return 2

    collector = cls(settings=settings, feed_url=args.feed_url)
    try:
        content = collector.fetch_bytes()
        items = collector.parse(content, None if args.check else _since(args))
    except NewsFetchBlocked as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except NewsFetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    st = collector.stats
    stamps = [i.published_at for i in items if i.published_at]
    if args.check:
        print(f"source: {args.source}   feed: {collector.feed_url}\nrobots.txt: allowed   fetched: {len(content)} bytes")
        print(f"items parsed: {st['items']} (entries={st['entries']} no_title={st['no_title']} no_timestamp={st['no_timestamp']} "
              f"future_dated={st['future_dated']} link_only_bodies={st['link_only_body']})")
        if stamps:
            print(f"newest: {max(stamps).isoformat()}   oldest: {min(stamps).isoformat()}")
        print(f"article bodies saved: {settings.news_store_article_text} (NEWS_STORE_ARTICLE_TEXT; see terms in README)")
        return 0

    if args.limit:
        items = items[:args.limit]
    if args.dry_run:
        print(f"{len(items)} item(s) would be stored (nothing written):")
        for i in items[:15]:
            print(f"  {i.published_at.isoformat() if i.published_at else 'NO-TIMESTAMP':<25} {i.headline[:70]}  {i.url}")
        return 0

    try:
        with session_scope() as s:
            report = store_news(s, items)
    except Exception as exc:                                       # noqa: BLE001 - report DB problems clearly
        print(f"ERROR storing news: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"source={args.source} fetched={len(items)} window={'since ' + (_since(args).isoformat())}")
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
