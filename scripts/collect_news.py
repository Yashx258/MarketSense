"""Collect news from ONE source and store only high-confidence PSX-relevant items.

Examples:
  python -m scripts.collect_news --source dawn_business --check
  python -m scripts.collect_news --source dawn_business --dry-run
  python -m scripts.collect_news --source business_recorder --dry-run --show-rejected

Only relevance decisions marked `include` are eligible for PostgreSQL storage. Items marked
`review` are shown in dry-run output but are NOT stored. Relevance is not sentiment scoring.
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


def _print_examples(label: str, examples: list[tuple[str, str, str]], limit: int = 10) -> None:
    if not examples:
        return
    print(f"\n{label} (showing {min(len(examples), limit)} of {len(examples)}):")
    for headline, reason, topic in examples[:limit]:
        print(f"  [{topic}] {headline[:110]}\n      reason: {reason}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Collect only high-confidence PSX-relevant news from one source.")
    ap.add_argument("--source", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--check", action="store_true", help="verify robots.txt + feed health and relevance counts; no database")
    ap.add_argument("--dry-run", action="store_true", help="fetch and preview included/reviewed/rejected items; no database")
    ap.add_argument("--hours", type=float, default=72.0, help="keep items published in the last N hours (default 72)")
    ap.add_argument("--since", help="ISO-8601 with offset; overrides --hours")
    ap.add_argument("--limit", type=int, help="store/preview at most N newest included items")
    ap.add_argument("--feed-url", help="override the feed URL (testing/verified alternate endpoint)")
    ap.add_argument("--show-rejected", action="store_true", help="also show rejected headlines in dry-run output")
    ap.add_argument("--examples", type=int, default=10, help="maximum review/rejected examples to display (default 10)")
    args = ap.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    if args.examples < 0:
        ap.error("--examples must be >= 0")
    if args.hours <= 0 and not args.since:
        ap.error("--hours must be > 0")

    setup_logging()
    settings = Settings()
    cls = REGISTRY[args.source]

    # `enabled` does not override robots.txt, publisher terms, or NEWS_COLLECTION_ENABLED.
    if not getattr(cls, "enabled", False):
        print(f"REFUSED: {args.source} is disabled in code; verify its feed and usage terms first.", file=sys.stderr)
        return 2
    if not issubclass(cls, RssNewsCollector):
        print(f"REFUSED: {args.source} is not an RSS collector.", file=sys.stderr)
        return 2
    if not (args.check or args.dry_run) and not settings.news_enabled:
        print("REFUSED: NEWS_COLLECTION_ENABLED is not 'true'. Review publisher terms before enabling persistent collection.\n"
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
    stamps = [item.published_at for item in items if item.published_at]
    print(f"source={args.source} entries={st['entries']} parsed={st['parsed']} "
          f"included={st['relevant']} review={st['review']} rejected={st['rejected_irrelevant']} "
          f"no_timestamp={st['no_timestamp']} future_dated={st['future_dated']}")

    if args.check:
        print(f"feed={collector.feed_url}\nrobots.txt: allowed   fetched={len(content)} bytes")
        if stamps:
            print(f"newest={max(stamps).isoformat()} oldest={min(stamps).isoformat()}")
        print(f"article bodies configured for storage: {settings.news_store_article_text} "
              "(confirm publisher terms before enabling)")
        _print_examples("Manual-review queue (not stored)", st["review_examples"], args.examples)
        return 0

    if args.limit is not None:
        items = items[:args.limit]

    if args.dry_run:
        print(f"{len(items)} high-confidence relevant item(s) would be stored (nothing written):")
        for item in items[:20]:
            print(f"  {item.published_at.isoformat() if item.published_at else 'NO-TIMESTAMP':<25} "
                  f"{item.headline[:100]}  {item.url}")
        _print_examples("Manual-review queue (NOT stored)", st["review_examples"], args.examples)
        if args.show_rejected:
            _print_examples("Rejected headlines", st["rejected_examples"], args.examples)
        return 0

    try:
        with session_scope() as session:
            report = store_news(session, items)
    except Exception as exc:  # noqa: BLE001 - report DB problems clearly
        print(f"ERROR storing news: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    since = _since(args)
    print(f"stored_source={args.source} stored_candidates={len(items)} "
          f"window={'since ' + since.isoformat() if since else 'all available'}")
    print(report.summary())
    print(f"Not stored: {st['review']} review item(s), {st['rejected_irrelevant']} rejected item(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
