"""Timestamp rules: exact only with a known timezone; never guess zone or day/month order; reject nonsense."""
from datetime import date, datetime, time, timezone

import pandas as pd
import pytest

from marketsense.timeparse import is_missing, parse_datetime
from tests.conftest import utc

NOW = utc(2026, 10, 12, 0, 0)
PKT = "Asia/Karachi"


def p(value, tm=None, **kw):
    return parse_datetime(value, tm, now=NOW, **kw)


def test_named_format_with_declared_timezone_is_exact_utc():
    r = p("Feb 6, 2026 4:10 PM", tz=PKT)
    assert (r.precision, r.published_at, str(r.published_date), r.error) == ("exact", utc(2026, 2, 6, 11, 10), "2026-02-06", None)


def test_same_text_without_timezone_drops_the_time_instead_of_guessing():
    r = p("Feb 6, 2026 4:10 PM")
    assert (r.precision, r.published_at, str(r.published_date), r.note) == ("date", None, "2026-02-06", "time_dropped_no_timezone")


def test_pkt_date_can_differ_from_utc_date():
    r = p("Feb 6, 2026 2:00 AM", tz=PKT)                    # 2026-02-05 21:00 UTC, but the PKT calendar day is Feb 6
    assert r.published_at == utc(2026, 2, 5, 21, 0) and str(r.published_date) == "2026-02-06"


@pytest.mark.parametrize("text", ["2026-02-06T16:10:00+05:00", "2026-02-06 16:10:00+0500", "2026-02-06T11:10:00Z"])
def test_explicit_offset_wins_over_declared_timezone(text):
    r = p(text, tz="America/New_York")
    assert r.precision == "exact" and r.published_at == utc(2026, 2, 6, 11, 10)


@pytest.mark.parametrize("text", ["2026-02-06", "06-Feb-2026", "6 February 2026", "Feb 6, 2026", "2026/02/06"])
def test_date_only_formats(text):
    r = p(text, tz=PKT)
    assert (r.precision, r.published_at, str(r.published_date)) == ("date", None, "2026-02-06")


def test_separate_time_column_combines_when_timezone_is_known():
    assert p("2026-02-06", "3:28 PM", tz=PKT).published_at == utc(2026, 2, 6, 10, 28)
    assert p("2026-02-06", "15:28", tz=PKT).published_at == utc(2026, 2, 6, 10, 28)
    assert p("2026-02-06", "3:28 PM").precision == "date"                    # no timezone -> stays date-only
    assert p("2026-02-06", "not a time", tz=PKT).precision == "date"


def test_numeric_day_month_is_never_guessed():
    assert p("03/04/2026").error == "ambiguous_date"
    assert str(p("03/04/2026", dayfirst=True).published_date) == "2026-04-03"
    assert str(p("03/04/2026", dayfirst=False).published_date) == "2026-03-04"
    assert str(p("13/04/2026").published_date) == "2026-04-13"               # day > 12: unambiguous
    assert str(p("05/05/2026").published_date) == "2026-05-05"               # same either way
    assert p("32/01/2026").error == "invalid_date"
    r = p("03/04/2026 4:10 PM", tz=PKT, dayfirst=True)
    assert r.published_at == utc(2026, 4, 3, 11, 10)


@pytest.mark.parametrize("v", ["", "  ", None, float("nan"), "NaN", "N/A", "null", "-", pd.NA, pd.NaT])
def test_missing_values(v):
    assert is_missing(v)
    assert p(v).error == "missing_date"


def test_garbage_and_out_of_range():
    assert p("sometime last week").error == "unparseable_date"
    assert p("1985-01-01").error == "out_of_range_date"
    assert p("2099-01-01").error == "out_of_range_date"
    assert p("Dec 31, 2099 1:00 PM", tz=PKT).error == "out_of_range_date"
    assert p("2026-10-13T00:00:00+05:00").error == "future_timestamp"         # 19:00Z on the 12th is > now + 10 min
    assert p("2026-10-11T23:55:00Z").error is None


def test_python_and_pandas_objects():
    assert p(datetime(2026, 2, 6, 16, 10, tzinfo=timezone.utc)).published_at == utc(2026, 2, 6, 16, 10)
    assert p(datetime(2026, 2, 6, 16, 10), tz=PKT).published_at == utc(2026, 2, 6, 11, 10)
    assert p(datetime(2026, 2, 6, 16, 10)).precision == "date"                # naive, no tz
    assert p(datetime(2026, 2, 6)).precision == "date"                        # midnight naive = just a date
    assert p(date(2026, 2, 6)).precision == "date"
    assert p(pd.Timestamp("2026-02-06 16:10:00"), tz=PKT).published_at == utc(2026, 2, 6, 11, 10)
    assert p(pd.Timestamp("2026-02-06 16:10:00", tz="UTC")).published_at == utc(2026, 2, 6, 16, 10)
