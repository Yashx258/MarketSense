# MarketSense

MarketSense is an analysis and decision support platform for the Pakistan Stock Exchange (PSX) KSE-100 index.

## Overview
- **Infrastructure:** Dockerized PostgreSQL housing stock universe metadata and OHLCV price series (2020–present).
- **Model A Baseline:** Evaluates 5-session price direction using OHLCV data, RSI, and MACD across KSE-100 stocks.
- **Validation:** Chronological purged splits and block-bootstrap evaluation to prevent data leakage.

## Results Summary
- **Pooled Direction:** Test AUC of ~0.51, indicating no absolute directional prediction capability from price and technical indicators alone.
- **Cross-Sectional Reversal:** Detected a weak short-term reversal signal (daily Rank-IC ~0.05) across liquid stocks.

## Repository Structure
```text
MarketSense/
├── config/              # Configuration files
├── data/                # Local data storage and metadata
├── db_setup/            # PostgreSQL Docker setup
├── marketsense/         # Core python package (preprocessing & features)
├── notebooks/           # Jupyter notebooks for model training
├── scripts/             # Data collection and dataset build scripts
├── sql/                 # SQL schemas and queries
├── tests/               # Unit tests
└── README.md