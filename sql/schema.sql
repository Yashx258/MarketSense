-- MarketSense: PostgreSQL schema (raw data + derived features kept in separate tables)
-- Run:  psql "$DATABASE_URL" -f sql/schema.sql   (or: python -m scripts.init_db)

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------- RAW / REFERENCE
CREATE TABLE IF NOT EXISTS stocks (
    stock_id      BIGSERIAL PRIMARY KEY,
    symbol        VARCHAR(20)  NOT NULL UNIQUE,
    company_name  VARCHAR(255),
    sector        VARCHAR(100),
    is_kse100     BOOLEAN      NOT NULL DEFAULT FALSE,
    status        VARCHAR(20)  NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active', 'inactive', 'delisted')),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CHECK (symbol = upper(symbol))
);
CREATE INDEX IF NOT EXISTS ix_stocks_kse100 ON stocks (is_kse100) WHERE is_kse100;
DROP TRIGGER IF EXISTS trg_stocks_updated_at ON stocks;
CREATE TRIGGER trg_stocks_updated_at BEFORE UPDATE ON stocks
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS price_history (
    price_id      BIGSERIAL PRIMARY KEY,
    stock_id      BIGINT       NOT NULL REFERENCES stocks(stock_id) ON DELETE CASCADE,
    trading_date  DATE         NOT NULL,
    open          NUMERIC(14,4),
    high          NUMERIC(14,4),
    low           NUMERIC(14,4),
    close         NUMERIC(14,4) NOT NULL,
    volume        BIGINT,
    source        VARCHAR(50)  NOT NULL DEFAULT 'psxdata',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_price_stock_date UNIQUE (stock_id, trading_date),
    CONSTRAINT ck_price_positive   CHECK (close > 0 AND (open IS NULL OR open > 0)
                                          AND (high IS NULL OR high > 0) AND (low IS NULL OR low > 0)),
    CONSTRAINT ck_price_hl         CHECK (high IS NULL OR low IS NULL OR high >= low),
    CONSTRAINT ck_price_volume     CHECK (volume IS NULL OR volume >= 0)
);
CREATE INDEX IF NOT EXISTS ix_price_date ON price_history (trading_date);

CREATE TABLE IF NOT EXISTS news_articles (
    news_id       BIGSERIAL PRIMARY KEY,
    stock_id      BIGINT       REFERENCES stocks(stock_id) ON DELETE SET NULL,  -- NULL = market-wide article
    headline      TEXT         NOT NULL,
    article_text  TEXT,
    source        VARCHAR(100) NOT NULL,
    url           TEXT,
    published_at  TIMESTAMPTZ,
    retrieved_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    language      VARCHAR(10)  NOT NULL DEFAULT 'en',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
-- URL unique only when present (some sources may lack a stable URL)
CREATE UNIQUE INDEX IF NOT EXISTS uq_news_url ON news_articles (url) WHERE url IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_news_stock_pub ON news_articles (stock_id, published_at DESC);
CREATE INDEX IF NOT EXISTS ix_news_published ON news_articles (published_at DESC);

CREATE TABLE IF NOT EXISTS fundamentals (
    fundamental_id BIGSERIAL PRIMARY KEY,
    stock_id       BIGINT      NOT NULL REFERENCES stocks(stock_id) ON DELETE CASCADE,
    as_of_date     DATE        NOT NULL,
    pe_ratio       NUMERIC(14,4),
    eps            NUMERIC(14,4),
    market_cap     NUMERIC(24,2),
    source         VARCHAR(50) NOT NULL,
    retrieved_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_fund_stock_date_source UNIQUE (stock_id, as_of_date, source)
);
CREATE INDEX IF NOT EXISTS ix_fund_stock_date ON fundamentals (stock_id, as_of_date DESC);

-- ---------------------------------------------------------------- DERIVED
CREATE TABLE IF NOT EXISTS technical_features (
    technical_id   BIGSERIAL PRIMARY KEY,
    stock_id       BIGINT      NOT NULL REFERENCES stocks(stock_id) ON DELETE CASCADE,
    trading_date   DATE        NOT NULL,
    rsi            NUMERIC(8,4),
    macd           NUMERIC(14,6),
    macd_signal    NUMERIC(14,6),
    macd_histogram NUMERIC(14,6),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_tech_stock_date UNIQUE (stock_id, trading_date),
    CONSTRAINT ck_rsi_range CHECK (rsi IS NULL OR (rsi >= 0 AND rsi <= 100))
);
CREATE INDEX IF NOT EXISTS ix_tech_date ON technical_features (trading_date);

CREATE TABLE IF NOT EXISTS sentiment_features (
    sentiment_id   BIGSERIAL PRIMARY KEY,
    news_id        BIGINT      NOT NULL REFERENCES news_articles(news_id) ON DELETE CASCADE,
    stock_id       BIGINT      NOT NULL REFERENCES stocks(stock_id) ON DELETE CASCADE,
    positive_score NUMERIC(6,5) NOT NULL CHECK (positive_score BETWEEN 0 AND 1),
    neutral_score  NUMERIC(6,5) NOT NULL CHECK (neutral_score  BETWEEN 0 AND 1),
    negative_score NUMERIC(6,5) NOT NULL CHECK (negative_score BETWEEN 0 AND 1),
    sentiment_label VARCHAR(10) NOT NULL CHECK (sentiment_label IN ('positive','neutral','negative')),
    model_name     VARCHAR(100) NOT NULL,
    processed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_sent_news_model UNIQUE (news_id, model_name)
);
CREATE INDEX IF NOT EXISTS ix_sent_stock_time ON sentiment_features (stock_id, processed_at DESC);
