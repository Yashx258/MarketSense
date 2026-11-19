# MarketSense

MarketSense is an analysis and decision support platform for the Pakistan Stock Exchange (PSX) KSE-100 index. It combines historical daily market data, technical indicators, financial news sentiment, and explainable machine learning to evaluate 5-session directional stock movements under strict point-in-time constraints.

> **Note:** MarketSense is built strictly for analytical research and decision support only.

## Overview
- **Database & Schema Architecture:** Dockerized PostgreSQL database storing stock universe metadata, daily OHLCV price series (`2020–2026`), financial news articles, and FinBERT sentiment probabilities, managed via sequential SQL migrations (`sql/migrations/002_news_imports.sql`, `003_news_relevance.sql`).
- **News Collection & Relevance Pipeline:** Ingests and deduplicates financial articles across Dawn Business, Business Recorder, and Profit Pakistan Today (`8,149` included articles and `5,039` manual-review articles). Uses versioned relevance annotations (`include` / `review` / `reject`), stable-ID URL canonicalization, and non-destructive text hygiene (`clean_article_text()`) at scoring time.
- **Model A (Technical Baseline):** Predicts 5-trading-session price direction (`target_direction_5d`) across 94 PSX stocks (`132,899` stock-session rows from `2020-02-18` to `2026-09-25`) using 18 OHLCV, return, volatility, volume, RSI, and MACD features.
- **Model B (Market + FinBERT Sentiment):** Scores all `8,149` included articles on GPU with `ProsusAI/finbert` across two text variants (`headline` only and `headline + cleaned lead`, `16,298` score rows) and aggregates 72 point-in-time market-wide and company-specific rolling sentiment features (`85.13%` pre-cutoff market-wide session coverage).
- **Leakage-Free Validation:** Enforces an inclusive `17:00 PKT` post-market-close cutoff (rolling after-hours news to the next session), chronological date splits (`85,853` train / `21,055` val / `22,166` test rows with `879` rows purged across 5-session label boundaries), and 10-day moving-block bootstrap confidence intervals (`300` replicates), backed by `224` standard tests and `7` PostgreSQL migration tests.

## Results Summary
Evaluated on identical chronological purged splits using XGBoost (selected by validation ROC-AUC) and Logistic Regression:

- **Model A (OHLCV + Technicals Baseline):** Validation AUC `0.5166` (`0.5011` LR) | Test AUC `0.5022` (`0.5081` LR) — confirms near-chance pooled directional predictability from historical price and technical features alone.
- **Model B — Headline Sentiment:** Validation AUC `0.5309` (`ΔAUC = +0.0142`, `p = 0.233`) | Test AUC `0.4964` (`ΔAUC = -0.0058`, `p = 0.721`).
- **Model B — Headline + Lead Sentiment:** Validation AUC `0.5283` (`ΔAUC = +0.0117`, `p = 0.538`) | Test AUC `0.5030` (`ΔAUC = +0.0008`, `p = 0.983`).
- **Cross-Sectional Reversal:** Detected a weak short-term reversal signal (daily Rank-IC ~0.05) across liquid stocks.

## Repository Structure
```text
MarketSense/
├── config/              # Configuration files
├── data/                # Local data storage, parquet datasets, and metadata
├── db_setup/            # PostgreSQL Docker setup
├── marketsense/         # Core python package (preprocessing, collectors, importers & features)
├── notebooks/           # Jupyter/Colab notebooks for FinBERT scoring & model training
├── scripts/             # DB init, data collection, FinBERT export/import & dataset build scripts
├── sql/                 # Base SQL schema and sequential database migrations
├── tests/               # Unit, integration, and PostgreSQL migration tests
├── project.md           # Detailed system specification and architectural roadmap
└── README.md