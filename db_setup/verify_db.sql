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
