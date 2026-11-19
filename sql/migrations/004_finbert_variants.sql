-- Keep existing full-text scores separate from the Colab scoring variants.
ALTER TABLE sentiment_features
    ADD COLUMN IF NOT EXISTS text_variant VARCHAR(20) NOT NULL DEFAULT 'legacy_full_text';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'sentiment_features'::regclass AND conname = 'uq_sent_news_model_variant'
    ) THEN
        ALTER TABLE sentiment_features
            DROP CONSTRAINT IF EXISTS uq_sent_news_model,
            ADD CONSTRAINT uq_sent_news_model_variant UNIQUE (news_id, model_name, text_variant),
            ADD CONSTRAINT ck_sent_text_variant
                CHECK (text_variant IN ('legacy_full_text', 'headline', 'headline_lead'));
    END IF;
END
$$;
