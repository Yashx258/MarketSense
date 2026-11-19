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

-- Which stocks an article mentions (an article can mention several; news_articles.stock_id is only set when exactly one).
CREATE TABLE IF NOT EXISTS news_article_stocks (
    news_id     BIGINT      NOT NULL REFERENCES news_articles(news_id) ON DELETE CASCADE,
    stock_id    BIGINT      NOT NULL REFERENCES stocks(stock_id) ON DELETE CASCADE,
    matched_by  VARCHAR(20) NOT NULL DEFAULT 'symbol' CHECK (matched_by IN ('symbol', 'name', 'source')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (news_id, stock_id)
);
CREATE INDEX IF NOT EXISTS ix_nas_stock ON news_article_stocks (stock_id);

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
    stock_id       BIGINT      REFERENCES stocks(stock_id) ON DELETE CASCADE,   -- NULL = market-wide / multi-stock article
    positive_score NUMERIC(6,5) NOT NULL CHECK (positive_score BETWEEN 0 AND 1),
    neutral_score  NUMERIC(6,5) NOT NULL CHECK (neutral_score  BETWEEN 0 AND 1),
    negative_score NUMERIC(6,5) NOT NULL CHECK (negative_score BETWEEN 0 AND 1),
    sentiment_label VARCHAR(10) NOT NULL CHECK (sentiment_label IN ('positive','neutral','negative')),
    model_name     VARCHAR(100) NOT NULL,
    processed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    text_variant   VARCHAR(20) NOT NULL DEFAULT 'legacy_full_text',
    CONSTRAINT uq_sent_news_model_variant UNIQUE (news_id, model_name, text_variant),
    CONSTRAINT ck_sent_text_variant CHECK (text_variant IN ('legacy_full_text', 'headline', 'headline_lead'))
);
CREATE INDEX IF NOT EXISTS ix_sent_stock_time ON sentiment_features (stock_id, processed_at DESC);

-- Migration for databases created before news support (idempotent; also applied by `python -m scripts.init_db`):
-- market-wide and multi-stock articles have no single stock, so sentiment rows must allow NULL stock_id.
ALTER TABLE sentiment_features ALTER COLUMN stock_id DROP NOT NULL;

-- ======================================================================================================
-- 002: News import provenance and publication-time precision.
-- ADDITIVE and IDEMPOTENT: nothing is dropped or recreated, no existing row is lost. Applied by
-- `python -m scripts.init_db` (fresh or existing database). The same SQL is included in sql/schema.sql.
-- ======================================================================================================
CREATE TABLE IF NOT EXISTS import_batches (
    batch_id       BIGSERIAL PRIMARY KEY,
    source_name    VARCHAR(100) NOT NULL,
    file_name      TEXT,
    file_sha256    CHAR(64),
    status         VARCHAR(20)  NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'superseded')),
    started_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    rows_read      INTEGER,
    rows_inserted  INTEGER,
    rows_duplicate INTEGER,
    rows_rejected  INTEGER,
    report         TEXT                                  -- full import report as JSON text
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_import_source_file
    ON import_batches (source_name, file_sha256) WHERE file_sha256 IS NOT NULL AND status = 'completed';

ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS doc_type         VARCHAR(20) NOT NULL DEFAULT 'news';
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS published_date   DATE;           -- PKT calendar date, when known
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS time_precision   VARCHAR(10) NOT NULL DEFAULT 'exact';
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS source_record_id VARCHAR(100);   -- id in the original source/dataset
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS import_batch_id  BIGINT REFERENCES import_batches(batch_id) ON DELETE SET NULL;
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS category VARCHAR(60);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_news_doc_type' AND conrelid = 'news_articles'::regclass) THEN
        ALTER TABLE news_articles ADD CONSTRAINT ck_news_doc_type CHECK (doc_type IN ('news', 'other'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_news_time_precision' AND conrelid = 'news_articles'::regclass) THEN
        ALTER TABLE news_articles ADD CONSTRAINT ck_news_time_precision CHECK (time_precision IN ('exact', 'date', 'none'));
    END IF;
    -- allow evidence 'alias' (company alias list) in the stock link table
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_nas_matched_by' AND conrelid = 'news_article_stocks'::regclass) THEN
        ALTER TABLE news_article_stocks DROP CONSTRAINT IF EXISTS news_article_stocks_matched_by_check;
        ALTER TABLE news_article_stocks ADD CONSTRAINT ck_nas_matched_by CHECK (matched_by IN ('symbol', 'name', 'source', 'alias'));
    END IF;
END $$;

-- Legacy rows (written before this migration): derive, never invent.
UPDATE news_articles SET time_precision = 'none'
 WHERE published_at IS NULL AND published_date IS NULL AND time_precision = 'exact';
UPDATE news_articles SET published_date = (published_at AT TIME ZONE 'Asia/Karachi')::date
 WHERE published_at IS NOT NULL AND published_date IS NULL;

CREATE INDEX IF NOT EXISTS ix_news_doctype_pub ON news_articles (doc_type, published_at DESC);
CREATE INDEX IF NOT EXISTS ix_news_pubdate     ON news_articles (published_date);
CREATE INDEX IF NOT EXISTS ix_news_batch       ON news_articles (import_batch_id);

-- 003: Persist relevance decisions for all stored articles.
-- Legacy rows remain under review until the application annotates them.
ALTER TABLE news_articles
    ADD COLUMN IF NOT EXISTS relevance_status VARCHAR(10) NOT NULL DEFAULT 'review',
    ADD COLUMN IF NOT EXISTS relevance_topic VARCHAR(60) NOT NULL DEFAULT 'unclassified',
    ADD COLUMN IF NOT EXISTS relevance_reason TEXT NOT NULL DEFAULT 'Awaiting relevance annotation',
    ADD COLUMN IF NOT EXISTS relevance_version VARCHAR(30) NOT NULL DEFAULT 'unclassified';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_news_relevance_status' AND conrelid = 'news_articles'::regclass
    ) THEN
        ALTER TABLE news_articles ADD CONSTRAINT ck_news_relevance_status
            CHECK (relevance_status IN ('include', 'review', 'reject'));
    END IF;
END $$;
