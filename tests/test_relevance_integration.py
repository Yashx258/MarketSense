from sqlalchemy import select

from marketsense.db.models import NewsArticle, SentimentFeature
from marketsense.importers.pipeline import import_file
from marketsense.importers.profiles import PROFILES
from marketsense.sentiment.features import articles_available_by, sentiment_available_by
from marketsense.sentiment.processor import MODEL_NAME, SentimentProcessor
from tests.conftest import utc
from tests.test_sentiment import FakeBackend


def test_csv_retains_all_statuses_and_only_include_reaches_features(news_session, tmp_path):
    path = tmp_path / "judgments.csv"
    path.write_text(
        "title,published_at\n"
        "PSX gains as HBL rallies,2026-03-02T10:00:00+05:00\n"
        "HBL considers FBR reforms,2026-03-02T11:00:00+05:00\n"
        "HBL obituary,2026-03-02T12:00:00+05:00\n",
        encoding="utf-8",
    )
    report = import_file(news_session, path, PROFILES["generic"], source_name="archive", now=utc(2026, 10, 12))
    assert report.store["inserted"] == 3
    assert report.store["relevance_counts"] == {"include": 1, "review": 1, "reject": 1}
    articles = {a.relevance_status: a for a in news_session.scalars(select(NewsArticle))}
    eligible = articles["include"].news_id
    scored = SentimentProcessor(FakeBackend()).process_pending(news_session)
    assert scored.pending_seen == scored.scored == 1
    assert list(news_session.scalars(select(SentimentFeature.news_id))) == [eligible]

    # Older scoring runs may already have produced scores for excluded articles.
    for status in ("review", "reject"):
        article = articles[status]
        news_session.add(SentimentFeature(news_id=article.news_id, stock_id=article.stock_id,
                                         positive_score=.6, neutral_score=.3, negative_score=.1,
                                         sentiment_label="positive", model_name=MODEL_NAME))
    news_session.flush()
    cutoff = utc(2026, 3, 2, 8)
    assert list(articles_available_by(news_session, cutoff).news_id) == [eligible]
    assert list(sentiment_available_by(news_session, cutoff).news_id) == [eligible]
