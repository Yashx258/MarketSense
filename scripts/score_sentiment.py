"""Score pending news articles with FinBERT and store them in sentiment_features. Scoring only - no collection.

  python -m scripts.score_sentiment --smoke              # load the model, score 3 sentences, NO database
  python -m scripts.score_sentiment --limit 200          # score up to 200 articles that have no score yet
  python -m scripts.score_sentiment --batch-size 8 --model ProsusAI/finbert

Needs: pip install -r requirements-sentiment.txt   (first run downloads ~440 MB of weights from Hugging Face)
Safe to re-run: already-scored articles are skipped and duplicates cannot be created.
Exit codes: 0 ok | 1 some articles failed / smoke check failed / model or DB error
"""
from __future__ import annotations

import argparse
import sys

from marketsense.config import Settings, setup_logging
from marketsense.db.connection import session_scope
from marketsense.sentiment.processor import SentimentProcessor

SMOKE = [("Lucky Cement reports record quarterly profit and raises its dividend", "positive"),
         ("Bank posts sharp loss as non-performing loans surge", "negative"),
         ("The company will hold its annual general meeting on Tuesday", None)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score pending news with FinBERT.")
    ap.add_argument("--smoke", action="store_true", help="score 3 sample sentences; no database")
    ap.add_argument("--limit", type=int, help="examine at most N pending articles")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model", help="override SENTIMENT_MODEL (default ProsusAI/finbert)")
    args = ap.parse_args(argv)
    setup_logging()
    proc = SentimentProcessor(model_name=args.model or Settings().sentiment_model, batch_size=args.batch_size)

    if args.smoke:
        ok = True
        try:
            for text, expected in SMOKE:
                sc = proc.score(text)
                mark = "" if expected is None else ("  OK" if sc.label == expected else f"  UNEXPECTED (wanted {expected})")
                ok &= expected is None or sc.label == expected
                print(f"{sc.label:<8} pos={sc.positive:.3f} neu={sc.neutral:.3f} neg={sc.negative:.3f}  {text}{mark}")
        except Exception as exc:                                   # noqa: BLE001 - e.g. torch missing, no network for weights
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0 if ok else 1

    try:
        with session_scope() as s:
            report = proc.process_pending(s, limit=args.limit)
    except Exception as exc:                                       # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(report.summary())
    if report.pending_seen == 0:
        print("nothing pending.")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
