"""FinBERT sentiment scoring + the batch job that fills `sentiment_features`.

* `FinBertBackend` loads ProsusAI/finbert LAZILY (importing this module never imports torch/transformers).
* `SentimentProcessor.process_pending()` scores articles that have no row for the model yet, in batches, and is
  repeatable: unique (news_id, model_name) + NOT EXISTS selection + ON CONFLICT DO NOTHING mean re-runs and
  concurrent runs never create duplicates. A failing batch falls back to one-by-one so a single bad text cannot
  block the rest; failures are reported and simply stay 'pending' for the next run.
* Scores are a pure function of the text, so a score never "leaks" information; WHEN an article was available
  is governed by news_articles.published_at (see sentiment/features.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session

from marketsense.db.connection import dialect_insert
from marketsense.db.models import NewsArticle, SentimentFeature

log = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
LABELS = ("positive", "neutral", "negative")


@dataclass
class SentimentScores:
    positive: float
    neutral: float
    negative: float

    @property
    def label(self) -> str:
        return max(("positive", self.positive), ("neutral", self.neutral), ("negative", self.negative),
                   key=lambda t: t[1])[0]


class SentimentBackend(Protocol):
    def predict(self, texts: Sequence[str]) -> list[SentimentScores]: ...


class FinBertBackend:
    """HuggingFace FinBERT. Label order is read from the model config (never hard-coded)."""

    def __init__(self, model_name: str = MODEL_NAME, max_length: int = 512, device: str | None = None):
        self.model_name, self.max_length, self._device_pref = model_name, max_length, device
        self._tok = self._model = self._torch = None
        self._id2label: dict[int, str] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("FinBERT needs torch + transformers: pip install -r requirements-sentiment.txt") from exc
        self._torch = torch
        self._device = self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading %s on %s (first run downloads the weights from Hugging Face)", self.model_name, self._device)
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self._device).eval()
        id2label = {int(i): str(label).lower() for i, label in model.config.id2label.items()}
        if set(id2label.values()) != set(LABELS):
            raise RuntimeError(f"{self.model_name} has labels {sorted(id2label.values())}; expected {sorted(LABELS)}")
        self._id2label, self._model = id2label, model

    def predict(self, texts: Sequence[str]) -> list[SentimentScores]:
        self._load()
        enc = self._tok(list(texts), padding=True, truncation=True, max_length=self.max_length,
                        return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            probs = self._torch.softmax(self._model(**enc).logits, dim=-1).cpu().tolist()
        out = []
        for row in probs:
            by = {self._id2label[i]: float(p) for i, p in enumerate(row)}
            out.append(SentimentScores(by["positive"], by["neutral"], by["negative"]))
        return out


def build_text(headline: str | None, article_text: str | None) -> str:
    """Headline alone unless a body was stored (bodies are opt-in). Long text is truncated by the tokenizer."""
    head = " ".join((headline or "").split())
    body = " ".join((article_text or "").split())
    return f"{head}. {body}" if head and body else (head or body)


@dataclass
class ProcessReport:
    model_name: str
    pending_seen: int = 0
    scored: int = 0
    skipped_blank: int = 0
    failed: dict[int, str] = field(default_factory=dict)       # news_id -> error (stays pending; retried next run)

    def summary(self) -> str:
        lines = [f"model={self.model_name} pending_seen={self.pending_seen} scored={self.scored} "
                 f"skipped_blank={self.skipped_blank} failed={len(self.failed)}"]
        lines += [f"  FAILED news_id={k}: {v}" for k, v in sorted(self.failed.items())[:10]]
        return "\n".join(lines)


class SentimentProcessor:
    def __init__(self, backend: SentimentBackend | None = None, model_name: str = MODEL_NAME, batch_size: int = 16):
        self.model_name, self.batch_size = model_name, max(1, int(batch_size))
        self.backend = backend or FinBertBackend(model_name)

    # ---- scoring
    def score_batch(self, texts: Sequence[str]) -> list[SentimentScores]:
        out = self.backend.predict(list(texts))
        if len(out) != len(texts):
            raise RuntimeError(f"backend returned {len(out)} results for {len(texts)} texts")
        return out

    def score(self, text: str) -> SentimentScores:
        if not (text or "").strip():
            raise ValueError("cannot score empty text")
        return self.score_batch([text])[0]

    # ---- batch job
    def process_pending(self, session: Session, limit: int | None = None) -> ProcessReport:
        """Score articles lacking a row for this model. Commits after every batch. `limit` caps articles examined."""
        rep = ProcessReport(self.model_name)
        last_id, remaining = 0, limit
        while remaining is None or remaining > 0:
            n = self.batch_size if remaining is None else min(self.batch_size, remaining)
            rows = session.execute(
                select(NewsArticle.news_id, NewsArticle.stock_id, NewsArticle.headline, NewsArticle.article_text)
                .where(NewsArticle.news_id > last_id,
                       ~exists().where(and_(SentimentFeature.news_id == NewsArticle.news_id,
                                            SentimentFeature.model_name == self.model_name)))
                .order_by(NewsArticle.news_id).limit(n)).all()
            if not rows:
                break
            last_id = rows[-1].news_id
            rep.pending_seen += len(rows)
            remaining = None if remaining is None else remaining - len(rows)

            work = [(r, build_text(r.headline, r.article_text)) for r in rows]
            rep.skipped_blank += sum(1 for _, t in work if not t)
            work = [(r, t) for r, t in work if t]
            if not work:
                continue
            scored = self._score_with_fallback(work, rep)
            if scored:
                res = session.execute(dialect_insert(session)(SentimentFeature).values([
                    dict(news_id=r.news_id, stock_id=r.stock_id,
                         positive_score=_p(sc.positive), neutral_score=_p(sc.neutral), negative_score=_p(sc.negative),
                         sentiment_label=sc.label, model_name=self.model_name)
                    for r, sc in scored]).on_conflict_do_nothing(index_elements=["news_id", "model_name"]))
                rep.scored += max(res.rowcount or 0, 0)
            session.commit()
        return rep

    def _score_with_fallback(self, work, rep: ProcessReport):
        try:
            return list(zip((r for r, _ in work), self.score_batch([t for _, t in work])))
        except Exception as exc:                                       # noqa: BLE001 - isolate the bad item(s)
            log.warning("batch of %d failed (%s); retrying one by one", len(work), exc)
        out = []
        for r, t in work:
            try:
                out.append((r, self.score_batch([t])[0]))
            except Exception as exc:                                   # noqa: BLE001
                rep.failed[r.news_id] = f"{type(exc).__name__}: {exc}"[:300]
        return out


def _p(x: float) -> float:
    return round(min(max(float(x), 0.0), 1.0), 5)
