import math
import subprocess
import sys
import types
from contextlib import contextmanager

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

import scripts.score_sentiment as cli
from marketsense.db.models import NewsArticle, SentimentFeature
from marketsense.sentiment.processor import (FinBertBackend, SentimentProcessor, SentimentScores, _p, build_text)


class FakeBackend:
    """Deterministic: 'profit' -> positive, 'loss' -> negative, else neutral. Records every batch it is asked to score."""
    def __init__(self):
        self.batches = []

    def predict(self, texts):
        self.batches.append(list(texts))
        out = []
        for t in texts:
            if "POISON" in t:
                raise ValueError("cannot score this text")
            out.append(SentimentScores(0.8, 0.15, 0.05) if "profit" in t else
                       SentimentScores(0.05, 0.15, 0.8) if "loss" in t else SentimentScores(0.1, 0.8, 0.1))
        return out


def add(session, headline, text=None, stock_id=None):
    a = NewsArticle(headline=headline, article_text=text, source="t", stock_id=stock_id)
    session.add(a); session.flush()
    return a.news_id


def rows(session):
    return session.execute(select(SentimentFeature.news_id, SentimentFeature.sentiment_label, SentimentFeature.model_name)
                           .order_by(SentimentFeature.news_id)).all()


def n_sent(session):
    return session.scalar(select(func.count()).select_from(SentimentFeature))


# ------------------------------------------------------------------------------------------ job behaviour
def test_scores_pending_and_stores_probabilities(news_session):
    from marketsense.db.models import Stock
    luck = news_session.scalar(select(Stock.stock_id).where(Stock.symbol == "LUCK"))
    a = add(news_session, "Cement profit rises", stock_id=luck)
    b = add(news_session, "Bank reports loss")
    c = add(news_session, "AGM on Tuesday")
    rep = SentimentProcessor(FakeBackend(), batch_size=2).process_pending(news_session)
    assert (rep.pending_seen, rep.scored, rep.skipped_blank, rep.failed) == (3, 3, 0, {})
    assert rows(news_session) == [(a, "positive", "ProsusAI/finbert"), (b, "negative", "ProsusAI/finbert"), (c, "neutral", "ProsusAI/finbert")]
    sf = news_session.scalars(select(SentimentFeature).order_by(SentimentFeature.news_id)).all()
    assert [s.stock_id for s in sf] == [luck, None, None]            # single-stock article keeps its stock; others NULL
    assert all(abs(float(s.positive_score) + float(s.neutral_score) + float(s.negative_score) - 1) < 1e-4 for s in sf)


def test_text_is_headline_plus_body_only_when_body_exists(news_session):
    add(news_session, "Cement profit", text="Despatches rose.")
    add(news_session, "Bank loss")
    be = FakeBackend()
    SentimentProcessor(be, batch_size=10).process_pending(news_session)
    assert be.batches == [["Cement profit. Despatches rose.", "Bank loss"]]
    assert build_text("  A  b ", None) == "A b" and build_text(None, "body") == "body" and build_text("", "") == ""


def test_rerun_is_a_no_op_and_new_articles_are_picked_up(news_session):
    add(news_session, "Cement profit")
    be = FakeBackend(); p = SentimentProcessor(be)
    assert p.process_pending(news_session).scored == 1
    again = p.process_pending(news_session)
    assert (again.pending_seen, again.scored, n_sent(news_session)) == (0, 0, 1)
    add(news_session, "Bank loss")
    third = p.process_pending(news_session)
    assert (third.pending_seen, third.scored, n_sent(news_session)) == (1, 1, 2)
    assert be.batches[-1] == ["Bank loss"]                              # old article was not re-scored


def test_batch_size_and_limit(news_session):
    for i in range(7):
        add(news_session, f"item {i}")
    be = FakeBackend()
    rep = SentimentProcessor(be, batch_size=3).process_pending(news_session)
    assert [len(b) for b in be.batches] == [3, 3, 1] and rep.scored == 7
    news_session.execute(SentimentFeature.__table__.delete()); news_session.commit()
    be2 = FakeBackend()
    rep = SentimentProcessor(be2, batch_size=3).process_pending(news_session, limit=4)
    assert rep.pending_seen == 4 and rep.scored == 4 and n_sent(news_session) == 4


def test_one_poison_text_does_not_block_the_batch_and_is_retried_later(news_session):
    ok1, bad, ok2 = add(news_session, "profit a"), add(news_session, "POISON text"), add(news_session, "loss b")
    be = FakeBackend(); p = SentimentProcessor(be, batch_size=10)
    rep = p.process_pending(news_session)
    assert rep.scored == 2 and list(rep.failed) == [bad] and "cannot score" in rep.failed[bad]
    assert [r[0] for r in rows(news_session)] == [ok1, ok2]
    assert be.batches[0] == ["profit a", "POISON text", "loss b"] and len(be.batches) == 1 + 3   # batch, then one by one
    # the failure stays pending: next run retries ONLY it, still fails, and creates no duplicates
    rep2 = p.process_pending(news_session)
    assert (rep2.pending_seen, rep2.scored, list(rep2.failed)) == (1, 0, [bad]) and n_sent(news_session) == 2
    news_session.execute(NewsArticle.__table__.update().where(NewsArticle.news_id == bad).values(headline="profit fixed")); news_session.commit()
    assert p.process_pending(news_session).scored == 1 and n_sent(news_session) == 3


def test_failure_is_not_retried_in_a_loop_within_one_run(news_session):
    add(news_session, "POISON only")
    be = FakeBackend()
    SentimentProcessor(be, batch_size=1).process_pending(news_session)
    assert len(be.batches) == 2                                         # batch attempt + single retry, then moves on


def test_blank_text_is_skipped_not_scored(news_session):
    add(news_session, "   ")
    ok = add(news_session, "profit")
    rep = SentimentProcessor(FakeBackend()).process_pending(news_session)
    assert (rep.skipped_blank, rep.scored) == (1, 1) and [r[0] for r in rows(news_session)] == [ok]


def test_wrong_length_backend_output_is_a_failure_not_silent_misalignment(news_session):
    add(news_session, "a"); add(news_session, "b")
    class Short:
        def predict(self, texts): return [SentimentScores(1, 0, 0)]
    rep = SentimentProcessor(Short(), batch_size=2).process_pending(news_session)
    # the 2-text batch comes back with 1 result -> rejected (never zipped/misaligned) -> one-by-one fallback, where each
    # single-text call IS length-correct, so both articles end up scored with the right row each
    assert (rep.scored, rep.failed) == (2, {}) and n_sent(news_session) == 2


def test_second_model_gets_its_own_rows_and_duplicates_are_impossible(news_session):
    a = add(news_session, "profit")
    SentimentProcessor(FakeBackend(), model_name="m1").process_pending(news_session)
    SentimentProcessor(FakeBackend(), model_name="m2").process_pending(news_session)
    assert [(r[0], r[2]) for r in rows(news_session)] == [(a, "m1"), (a, "m2")]
    news_session.add(SentimentFeature(news_id=a, positive_score=.5, neutral_score=.3, negative_score=.2,
                                      sentiment_label="positive", model_name="m1"))
    with pytest.raises(IntegrityError):
        news_session.flush()
    news_session.rollback()


def test_concurrent_worker_inserting_the_same_row_causes_no_error_or_duplicate(news_session):
    a = add(news_session, "profit")
    class Racing(FakeBackend):
        def predict(self, texts):                                       # another worker finishes first
            news_session.add(SentimentFeature(news_id=a, positive_score=.9, neutral_score=.05, negative_score=.05,
                                              sentiment_label="positive", model_name="ProsusAI/finbert"))
            news_session.flush()
            return super().predict(texts)
    rep = SentimentProcessor(Racing()).process_pending(news_session)
    assert rep.scored == 0 and n_sent(news_session) == 1 and not rep.failed


def test_scores_and_labels():
    assert SentimentScores(.2, .5, .3).label == "neutral" and SentimentScores(.6, .1, .3).label == "positive"
    assert _p(1.0000004) == 1.0 and _p(-1e-9) == 0.0 and _p(0.123456789) == 0.12346
    with pytest.raises(ValueError):
        SentimentProcessor(FakeBackend()).score("   ")
    assert SentimentProcessor(FakeBackend()).score("big profit").label == "positive"


# ------------------------------------------------------------------------------------------ lazy loading + real backend wiring
def test_importing_the_processor_does_not_import_torch_or_transformers():
    code = "import sys, marketsense.sentiment.processor; assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode == 0


def test_missing_torch_gives_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)                     # makes `import torch` raise ImportError
    with pytest.raises(RuntimeError, match="requirements-sentiment.txt"):
        FinBertBackend().predict(["x"])


def _install_fake_transformers(monkeypatch, id2label, seen):
    """Fake torch + transformers using FinBERT's REAL label order {0: positive, 1: negative, 2: neutral}."""
    class T:
        def __init__(self, d): self.d = d
        def cpu(self): return self
        def tolist(self): return self.d
    class Enc(dict):
        def to(self, device): seen["device"] = device; return self
    class Tok:
        def __call__(self, texts, **kw): seen["tok_kwargs"] = kw; return Enc(texts=texts)
    class Model:
        config = types.SimpleNamespace(id2label=id2label)
        def to(self, d): return self
        def eval(self): return self
        def __call__(self, texts):
            rows = []
            for t in texts:
                v = [0.0, 0.0, 0.0]
                v[0 if "profit" in t else 1 if "loss" in t else 2] = 5.0
                rows.append(v)
            return types.SimpleNamespace(logits=T(rows))
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.no_grad = contextmanager(lambda: (yield))
    torch.softmax = lambda t, dim: T([[math.exp(x) / sum(math.exp(y) for y in r) for x in r] for r in t.d])
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: Tok())
    tf.AutoModelForSequenceClassification = types.SimpleNamespace(from_pretrained=lambda name: Model())
    monkeypatch.setitem(sys.modules, "torch", torch); monkeypatch.setitem(sys.modules, "transformers", tf)


FINBERT_ORDER = {0: "positive", 1: "negative", 2: "neutral"}


def test_finbert_backend_maps_labels_from_model_config_and_keeps_batch_order(monkeypatch):
    seen = {}
    _install_fake_transformers(monkeypatch, FINBERT_ORDER, seen)
    out = FinBertBackend().predict(["loss shock", "AGM notice", "record profit"])
    assert [s.label for s in out] == ["negative", "neutral", "positive"]     # NOT index order -> mapping is read from config
    assert all(abs(s.positive + s.neutral + s.negative - 1) < 1e-9 for s in out)
    assert seen["tok_kwargs"] == dict(padding=True, truncation=True, max_length=512, return_tensors="pt")
    assert seen["device"] == "cpu"


def test_finbert_backend_accepts_capitalised_labels_and_rejects_unexpected_ones(monkeypatch):
    _install_fake_transformers(monkeypatch, {0: "Positive", 1: "Negative", 2: "Neutral"}, {})
    assert FinBertBackend().predict(["profit"])[0].label == "positive"
    _install_fake_transformers(monkeypatch, {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, {})
    with pytest.raises(RuntimeError, match="expected"):
        FinBertBackend().predict(["x"])


def test_processor_end_to_end_through_the_real_backend_wiring(news_session, monkeypatch):
    _install_fake_transformers(monkeypatch, FINBERT_ORDER, {})
    a, b = add(news_session, "record profit"), add(news_session, "heavy loss")
    SentimentProcessor(batch_size=8).process_pending(news_session)
    assert rows(news_session) == [(a, "positive", "ProsusAI/finbert"), (b, "negative", "ProsusAI/finbert")]


# ------------------------------------------------------------------------------------------ CLI
def _patch_cli(monkeypatch, news_session, backend):
    @contextmanager
    def scope(*a, **k):
        yield news_session
        news_session.commit()
    monkeypatch.setattr(cli, "session_scope", scope)
    monkeypatch.setattr(cli, "SentimentProcessor", lambda model_name, batch_size: SentimentProcessor(backend, model_name, batch_size))


def test_cli_scores_pending_and_reports(news_session, monkeypatch, capsys):
    add(news_session, "profit"); add(news_session, "loss")
    _patch_cli(monkeypatch, news_session, FakeBackend())
    assert cli.main(["--limit", "10"]) == 0
    assert "scored=2" in capsys.readouterr().out
    assert cli.main([]) == 0 and "nothing pending" in capsys.readouterr().out


def test_cli_exit_code_1_when_items_fail(news_session, monkeypatch, capsys):
    add(news_session, "POISON")
    _patch_cli(monkeypatch, news_session, FakeBackend())
    assert cli.main([]) == 1 and "FAILED news_id" in capsys.readouterr().out


def test_cli_smoke_checks_polarity(news_session, monkeypatch, capsys):
    _patch_cli(monkeypatch, news_session, FakeBackend())
    # the CLI's sample sentences contain neither 'profit' nor 'loss' in a way the fake knows, except these:
    monkeypatch.setattr(cli, "SMOKE", [("big profit", "positive"), ("big loss", "negative"), ("agm", None)])
    assert cli.main(["--smoke"]) == 0 and "OK" in capsys.readouterr().out
    monkeypatch.setattr(cli, "SMOKE", [("big profit", "negative")])
    assert cli.main(["--smoke"]) == 1 and "UNEXPECTED" in capsys.readouterr().out


def test_cli_smoke_reports_missing_dependencies_cleanly(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)                     # torch not installed
    assert cli.main(["--smoke"]) == 1
    err = capsys.readouterr().err
    assert "ERROR: RuntimeError" in err and "requirements-sentiment.txt" in err and "Traceback" not in err
