"""Timestamp normalisation for collectors and importers. Rules (leakage safety first):

* A timestamp is stored EXACT only if its timezone is known (explicit offset, or a timezone the caller declares).
* A naive date-time with no declared timezone is NEVER guessed: it degrades to a DATE-ONLY value (precision 'date').
* Numeric day/month dates (03/04/2026) are NEVER guessed: they need an explicit `dayfirst`, else the row is rejected
  as 'ambiguous_date' unless the day is > 12 (unambiguous).
* Dates outside 1990..tomorrow and timestamps more than 10 minutes in the future are rejected.
* Date-only values become available to features only on the day AFTER (see sentiment/features.py).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

PKT = ZoneInfo("Asia/Karachi")
MISSING = {"", "nan", "nat", "none", "null", "n/a", "na", "-", "--"}
MIN_YEAR = 1990

_DATETIME_FORMATS = ("%b %d, %Y %I:%M %p", "%B %d, %Y %I:%M %p", "%b %d, %Y %I:%M:%S %p", "%Y-%m-%d %H:%M:%S",
                     "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y/%m/%d %H:%M:%S",
                     "%d-%b-%Y %H:%M:%S", "%d %b %Y %H:%M", "%d %B %Y %H:%M")
_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y", "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%b %d %Y",
                 "%d-%b-%y")
_TIME_FORMATS = ("%I:%M %p", "%I:%M:%S %p", "%H:%M:%S", "%H:%M", "%I %p")
_NUMERIC = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?:[ T](\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?))?$")


@dataclass
class ParsedTime:
    published_at: datetime | None = None      # aware UTC, only when precision == 'exact'
    published_date: date | None = None        # PKT calendar date (exact) or the source's date (date-only)
    precision: str = "none"                   # 'exact' | 'date' | 'none'
    error: str | None = None                  # set => reject the row
    note: str | None = None                   # e.g. 'time_dropped_no_timezone'


def is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if type(v).__name__ in ("NAType", "NaTType"):          # pandas pd.NA / pd.NaT (nullable columns, parquet)
        return True
    return str(v).strip().lower() in MISSING


def _try(formats, text: str):
    for f in formats:
        try:
            return datetime.strptime(text, f)
        except ValueError:
            continue
    return None


def _parse_time_of_day(v) -> time | None:
    if is_missing(v):
        return None
    if isinstance(v, time):
        return v
    t = " ".join(str(v).split()).upper().replace(".", "")
    d = _try(_TIME_FORMATS, t)
    return d.time() if d else None


def _validate(pt: ParsedTime, now: datetime) -> ParsedTime:
    d = pt.published_date
    if d is not None and (d.year < MIN_YEAR or d > (now.astimezone(PKT).date() + timedelta(days=1))):
        return ParsedTime(error="out_of_range_date")
    if pt.published_at is not None and pt.published_at > now + timedelta(minutes=10):
        return ParsedTime(error="future_timestamp")
    return pt


def _finish(naive: datetime, has_time: bool, tz: str | None, now: datetime) -> ParsedTime:
    if has_time and tz:
        aware = naive.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)
        return _validate(ParsedTime(aware, aware.astimezone(PKT).date(), "exact"), now)
    note = "time_dropped_no_timezone" if has_time else None
    return _validate(ParsedTime(None, naive.date(), "date", note=note), now)


def parse_datetime(value, time_value=None, *, tz: str | None = None, dayfirst: bool | None = None,
                   now: datetime | None = None) -> ParsedTime:
    """Normalise one source timestamp. `tz` = IANA zone the SOURCE uses for naive values (e.g. 'Asia/Karachi')."""
    now = now or datetime.now(timezone.utc)
    if is_missing(value):
        return ParsedTime(error="missing_date")

    # ---- already a python/pandas datetime or date
    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            u = value.astimezone(timezone.utc)
            return _validate(ParsedTime(u, u.astimezone(PKT).date(), "exact"), now)
        has_time = value.time() != time(0, 0)
        return _finish(value, has_time, tz, now)
    if isinstance(value, date):
        return _validate(ParsedTime(None, value, "date"), now)

    text = " ".join(str(value).split())

    # ---- ISO-8601 with an explicit offset / Z  -> exact
    if re.search(r"(Z|[+-]\d{2}:?\d{2})$", text) and re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text):
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                u = dt.astimezone(timezone.utc)
                return _validate(ParsedTime(u, u.astimezone(PKT).date(), "exact"), now)
        except ValueError:
            pass

    # ---- numeric day/month/year: never guess the order
    m = _NUMERIC.match(text)
    if m:
        a, b, y, tpart = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        if dayfirst is None and a <= 12 and b <= 12 and a != b:
            return ParsedTime(error="ambiguous_date")
        day_first = dayfirst if dayfirst is not None else (a > 12 or (a == b))
        d, mo = (a, b) if day_first else (b, a)
        try:
            naive = datetime(y, mo, d)
        except ValueError:
            return ParsedTime(error="invalid_date")
        has_time = False
        if tpart:
            tod = _parse_time_of_day(tpart)
            if tod is None:
                return ParsedTime(error="unparseable_date")
            naive, has_time = datetime.combine(naive.date(), tod), True
        elif not is_missing(time_value):
            tod = _parse_time_of_day(time_value)
            if tod is not None:
                naive, has_time = datetime.combine(naive.date(), tod), True
        return _finish(naive, has_time, tz, now)

    # ---- named / ISO formats without offset
    dt = _try(_DATETIME_FORMATS, text)
    if dt is not None:
        return _finish(dt, True, tz, now)
    d = _try(_DATE_FORMATS, text)
    if d is not None:
        has_time = False
        if not is_missing(time_value):
            tod = _parse_time_of_day(time_value)
            if tod is not None:
                d, has_time = datetime.combine(d.date(), tod), True
        return _finish(d, has_time, tz, now)
    return ParsedTime(error="unparseable_date")
