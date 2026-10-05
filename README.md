# MarketSense - data foundation 

Scope: PostgreSQL schema, KSE-100 price collection via `psxdata`, RSI/MACD features.

## Setup
1. python -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. createdb marketsense   (and a DB user)
4. cp .env.example .env   -> set DB_USER / DB_PASSWORD / DB_NAME 
5. python -m scripts.init_db                      # applies sql/schema.sql (safe to re-run)
6. python -m scripts.collect_prices --symbols ENGRO   # try one stock first
7. python -m scripts.collect_prices               # full KSE-100 universe
8. python -m scripts.compute_technical            # RSI/MACD -> technical_features
9. pytest -q                                      # SQLite by default; TEST_DATABASE_URL=... for PostgreSQL

## Design rules
- Raw tables (stocks, price_history, news_articles, fundamentals) are separate from derived ones
  (technical_features, sentiment_features). Derived tables can be dropped and recomputed.
- Universe is configurable: STOCK_UNIVERSE_SOURCE=psxdata (live KSE100 constituents) or file.
  Stocks that leave the index get is_kse100=false; history is kept.
- No fabricated data: bad/anomalous rows are rejected and counted; indicator warm-up rows are NULL.
- collect_prices prints stocks with no data / failures and exits non-zero if any exist.
- Re-runs are idempotent (UNIQUE (stock_id, trading_date) + ON CONFLICT DO NOTHING).

## Placeholders (not implemented)
- News: Dawn Business, Business Recorder, Profit (collectors/news_collector.py). Check each site's
  terms + robots.txt (robots_allows() helper fails closed); prefer RSS/official APIs.
- Yahoo Finance fundamentals (collectors/fundamentals_collector.py). Verify PSX coverage (.KA suffix) first.
- FinBERT sentiment (sentiment/processor.py).

## Known limits
- psxdata scrapes dps.psx.com.pk and always downloads full history per symbol (cached on disk by psxdata).
- Fundamentals table stays empty this phase.
- Tested in this environment against a fake psxdata module (no network access to PSX from here) and
  against real PostgreSQL 16 for schema/constraint checks. Treat the first live run as the real
  integration test with the actual psxdata network calls.
