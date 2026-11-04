import pytest
from marketsense.db.models import NewsArticle
import csv, pandas as pd
from contextlib import contextmanager

@pytest.fixture()
def hf_csv(tmp_path):
    p = tmp_path / "hf.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(HEADER); w.writerows(ROWS)
    return p


@pytest.fixture()
def hf_parquet(tmp_path):
    p = tmp_path / "hf.parquet"
    pd.DataFrame(ROWS, columns=HEADER).to_parquet(p, index=False)
    return p


def run(s, path, profile="psx_news_primary_candidates", **kw):
    return import_file(s, path, PROFILES[profile], now=NOW, index=build_stock_index(s, aliases={}), **kw)


def n(s, model=NewsArticle):
    return s.scalar(select(func.count()).select_from(model))


# ------------------------------------------------------------------------------------------ the main import
@pytest.mark.parametrize("fixture_name", ["hf_csv", "hf_parquet"])
def test_third_party_sentiment_columns_are_ignored_and_reported(
    news_session, request, fixture_name
):
    s = news_session
    input_file = request.getfixturevalue(fixture_name)

    rep = run(s, input_file)
    s.commit()

    assert set(rep.columns_ignored) == {
        "sentiment_score",
        "sentiment_impact",
        "sentiment_signals",
        "company",
        "period_ended",
    }
    assert n(s, SentimentFeature) == 0                                               # never mixed with our FinBERT scores


def test_article_text_is_opt_in(news_session, hf_csv):
    s = news_session
    rep = run(s, hf_csv); s.commit()
    assert rep.rows_with_text == 1 and rep.text_stored is False
    assert s.scalars(select(NewsArticle.article_text).where(NewsArticle.source_record_id == "269200")).one() is None
    s.execute(NewsArticle.__table__.delete()); s.execute(ImportBatch.__table__.delete()); s.commit()
    run(s, hf_csv, store_text=True); s.commit()
    assert s.scalars(select(NewsArticle.article_text).where(NewsArticle.source_record_id == "269200")).one() == "Some body text"


# ------------------------------------------------------------------------------------------ provenance, idempotence, dry-run
def test_same_file_twice_is_a_noop_and_force_creates_no_duplicates(news_session, hf_csv):
    s = news_session
    run(s, hf_csv); s.commit()
    again = run(s, hf_csv)
    assert again.already_imported and "already imported" in again.summary() and n(s) == 7 and n(s, ImportBatch) == 1
    forced = run(s, hf_csv, force=True); s.commit()
    assert forced.store["inserted"] == 0 and forced.store["duplicates"] == 7 and n(s) == 7


def test_edited_file_with_overlapping_rows_dedups_by_document_id(news_session, hf_csv, tmp_path):
    s = news_session
    run(s, hf_csv); s.commit()
    p2 = tmp_path / "hf2.csv"
    with open(p2, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(HEADER); w.writerows(ROWS[:3]); w.writerow(["OGDC", "New one", "Feb 9, 2026 10:00 AM", D + "document/270000.pdf"] + [""] * 6)
    rep = run(s, p2); s.commit()
    assert rep.store["inserted"] == 1 and rep.store["duplicates"] == 2 and n(s) == 8 and n(s, ImportBatch) == 2


def test_dry_run_is_exact_and_writes_nothing(news_session, hf_csv):
    s = news_session
    dry = run(s, hf_csv, dry_run=True)
    assert n(s) == 0 and n(s, ImportBatch) == 0 and n(s, NewsArticleStock) == 0
    assert dry.store["inserted"] == 7 and dry.batch_id is None
    real = run(s, hf_csv); s.commit()
    assert real.store["inserted"] == 7 and n(s) == 7                                     # rollback really left no trace


def test_failure_marks_batch_failed(news_session, hf_csv, monkeypatch):
    s = news_session
    monkeypatch.setattr(pipeline, "store_news", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(RuntimeError):
        run(s, hf_csv)
    b = s.scalars(select(ImportBatch)).one()
    assert b.status == "failed" and "disk full" in b.report and n(s) == 0
    ok = run(s, hf_csv) if False else None                                               # a failed batch does not block a retry
    monkeypatch.undo()
    assert run(s, hf_csv).store["inserted"] == 7


# ------------------------------------------------------------------------------------------ company mapping on import
def kaggle_like(tmp_path):
    p = tmp_path / "news.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Story Heading", "Story Excerpt", "Timestamp", "Section"])
        w.writerow(["Lucky Cement posts profit", "excerpt one", "2026-03-02 09:15:00", "Business"])
        w.writerow(["HBL raises dividend", "excerpt two", "2026-03-03 10:00:00", "Business"])
        w.writerow(["Ambiguous date row", "x", "03/04/2026", "Business"])
        w.writerow(["Same headline", "x", "2026-03-04", "Business"])
        w.writerow(["Same headline", "x", "2026-03-04", "Business"])
    return p


def test_inspect_reports_mapping_and_warns_about_missing_timezone(kaggle_like):
    info = inspect_file(kaggle_like, PROFILES["generic"])
    assert info["mapping"]["headline"] == "Story Heading" and info["mapping"]["published"] == "Timestamp"
    assert info["mapping"]["text"] == "Story Excerpt" and info["rows"] == 5 and info["encoding"]
    assert info["date_parse_sample"] == {"ok": 4, "ambiguous_date": 1} and "DROPPED to date-only" in info["warning"]
    assert inspect_file(kaggle_like, PROFILES["generic"])["warning"].startswith("2 of the sampled rows")
    assert inspect_file(kaggle_like, PROFILES["generic"], tz="Asia/Karachi")["warning"] is None


def test_generic_without_timezone_keeps_dates_only_and_rejects_ambiguous(news_session, kaggle_like):
    s = news_session
    rep = run(s, kaggle_like, profile="generic", source_name="kaggle_test", dataset="kaggle_test_v1"); s.commit()
    assert rep.notes == {"time_dropped_no_timezone": 2} and rep.rejected_by_reason == {"ambiguous_date": 1}
    assert rep.precision == {"date": 4} and rep.store["inserted"] == 3 and rep.store["duplicates_in_batch"] == 1   # no URL: dedup on source+headline+date
    assert set(s.scalars(select(NewsArticle.published_at))) == {None} and set(s.scalars(select(NewsArticle.source))) == {"kaggle_test"}


def test_manual_mapping_overrides_and_errors(news_session, tmp_path):
    p = tmp_path / "odd.csv"
    p.write_text("foo,bar,baz\nLucky Cement up,2026-03-02,https://x.com/a\n", encoding="utf-8")
    with pytest.raises(ImportSchemaError, match="Detected columns"):
        run(news_session, p, profile="generic", source_name="x")
    rep = run(news_session, p, profile="generic", source_name="x", overrides={"headline": "foo", "published": "bar", "url": "baz"})
    assert rep.store["inserted"] == 1
    with pytest.raises(ImportSchemaError, match="no such column"):
        resolve_columns(["a", "b"], PROFILES["generic"], {"headline": "nope"})
    with pytest.raises(ImportSchemaError, match="unknown field"):
        resolve_columns(["a"], PROFILES["generic"], {"bogus": "a"})


def test_generic_requires_a_source_label(news_session, kaggle_like):
    with pytest.raises(ImportSchemaError, match="--source-name"):
        run(news_session, kaggle_like, profile="generic")


# ------------------------------------------------------------------------------------------ file robustness
def test_reader_handles_encodings_delimiters_and_bad_files(tmp_path):
    latin = tmp_path / "l.csv"; latin.write_bytes("title;date\nCaf\xe9 profit;2026-01-02\n".encode("latin-1"))
    t = read_table(latin)
    assert t.delimiter == ";" and t.df.iloc[0, 0] == "Café profit" and t.encoding in ("cp1252", "latin-1")
    bom = tmp_path / "b.csv"; bom.write_bytes("\ufefftitle,date\nx,2026-01-02\n".encode("utf-8"))
    assert list(read_table(bom).df.columns) == ["title", "date"]
    ragged = tmp_path / "r.csv"; ragged.write_text("title,date\na,2026-01-01\nb,2026-01-02,EXTRA,COLS\nc,2026-01-03\n", encoding="utf-8")
    r = read_table(ragged)
    assert len(r.df) == 2 and r.malformed_count == 1 and "EXTRA" in r.malformed[0]       # skipped AND reported
    for name, content, msg in [("e.csv", "", "empty"), ("x.xlsx", "zz", "unsupported"), ("missing.csv", None, "not found")]:
        p = tmp_path / name
        if content is not None:
            p.write_text(content)
        with pytest.raises(ImportSchemaError, match=msg):
            read_table(p)


def test_malformed_lines_are_surfaced_in_the_import_report(news_session, tmp_path):
    p = tmp_path / "r.csv"; p.write_text("title,date,url\na,2026-01-01,https://x.com/1\nb,2026-01-02,https://x.com/2,EXTRA\n", encoding="utf-8")
    rep = run(news_session, p, profile="generic", source_name="x")
    assert rep.malformed_lines_skipped == 1 and rep.store["inserted"] == 1 and "malformed_lines_skipped=1" in rep.summary()


# ------------------------------------------------------------------------------------------ CLI
@pytest.fixture()
def scoped(monkeypatch, news_session):
    @contextmanager
    def scope(*a, **k):
        yield news_session
        news_session.commit()
    monkeypatch.setattr(cli, "session_scope", scope)
    monkeypatch.setattr(pipeline, "build_stock_index", lambda s: build_stock_index(s, aliases={}))


def test_cli_schema_problem_exits_2(tmp_path, scoped, capsys):
    p = tmp_path / "bad.csv"; p.write_text("a,b\n1,2\n")
    assert cli.main([str(p), "--profile", "generic", "--source-name", "x"]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert cli.main([str(p), "--profile", "generic"]) == 2                                # missing --source-name


def test_inspect_does_not_cry_wolf_when_timestamps_carry_offsets(tmp_path):
    p = tmp_path / "off.csv"
    p.write_text("title,published_at,url\nA,2026-03-02T09:15:00+05:00,https://x.com/1\nB,2026-03-03T10:00:00.123456+05:00,https://x.com/2\n", encoding="utf-8")
    info = inspect_file(p, PROFILES["generic"])
    assert info["warning"] is None and info["precision_sample"] == {"exact": 2}
