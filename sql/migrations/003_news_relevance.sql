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
