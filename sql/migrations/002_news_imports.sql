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
