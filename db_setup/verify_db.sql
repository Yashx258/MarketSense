SELECT current_database() AS database_name,
       current_user AS database_user,
       version() AS postgres_version;

\dt

SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

SELECT
    (SELECT count(*) FROM stocks) AS stocks,
    (SELECT count(*) FROM price_history) AS price_history,
    (SELECT count(*) FROM news_articles) AS news_articles,
    (SELECT count(*) FROM fundamentals) AS fundamentals,
    (SELECT count(*) FROM technical_features) AS technical_features,
    (SELECT count(*) FROM sentiment_features) AS sentiment_features;


-- Validation queries for news ingestion + sentiment. Run:
--   macOS/Linux/Git Bash:  docker exec -i marketsense-postgres sh -c 'psql -U $POSTGRES_USER -d $POSTGRES_DB' < db_setup/verify_news.sql
--   PowerShell:            Get-Content db_setup\verify_news.sql | docker exec -i marketsense-postgres sh -c 'psql -U $POSTGRES_USER -d $POSTGRES_DB'
\echo '== N1 counts (no_timestamp rows are stored but NEVER usable as features)'
SELECT count(*) AS articles, count(published_at) AS with_timestamp,
       count(*) FILTER (WHERE published_at IS NULL) AS no_timestamp FROM news_articles;
\echo '== N2 newest 10 (Pakistan time)'
SELECT news_id, to_char(published_at AT TIME ZONE 'Asia/Karachi','YYYY-MM-DD HH24:MI') AS published_pkt,
       left(headline,50) AS headline, stock_id, source
FROM news_articles ORDER BY published_at DESC NULLS LAST LIMIT 10;
\echo '== N3 duplicate URLs (expect 0 rows)'
SELECT url, count(*) FROM news_articles WHERE url IS NOT NULL GROUP BY url HAVING count(*) > 1;
\echo '== N4 articles per mentioned stock'
SELECT s.symbol, count(*) AS articles FROM news_article_stocks l JOIN stocks s USING (stock_id)
GROUP BY s.symbol ORDER BY 2 DESC, 1 LIMIT 15;
\echo '== N5 stock_id set <=> exactly one stock link (expect 0 rows)'
SELECT n.news_id FROM news_articles n
LEFT JOIN (SELECT news_id, count(*) c FROM news_article_stocks GROUP BY news_id) l USING (news_id)
WHERE (n.stock_id IS NOT NULL) <> (COALESCE(l.c,0) = 1);
\echo '== N6 future-dated rows (expect 0)'
SELECT count(*) AS future_dated FROM news_articles WHERE published_at > now() + interval '10 minutes';
\echo '== S1 sentiment by label'
SELECT sentiment_label, count(*) FROM sentiment_features GROUP BY 1 ORDER BY 1;
\echo '== S2 articles still pending for FinBERT (0 after a full scoring run)'
SELECT count(*) AS pending FROM news_articles n
WHERE NOT EXISTS (SELECT 1 FROM sentiment_features s WHERE s.news_id = n.news_id AND s.model_name = 'ProsusAI/finbert');
\echo '== S3 duplicate (news_id, model) rows (expect 0 rows)'
SELECT news_id, model_name, count(*) FROM sentiment_features GROUP BY 1,2 HAVING count(*) > 1;
\echo '== S4 scores that do not sum to 1 (expect 0 rows)'
SELECT sentiment_id FROM sentiment_features WHERE abs(positive_score + neutral_score + negative_score - 1) > 0.0001;
