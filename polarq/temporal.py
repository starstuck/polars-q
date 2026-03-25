from polarq.types import QAtom, QVector
import polars as pl
from datetime import date, time, datetime, timedelta

# q epoch: 2000-01-01  (unlike Unix 1970-01-01)
Q_EPOCH_NS = 946684800_000_000_000   # nanoseconds from Unix to q epoch

def from_q_timestamp(ns: int) -> QAtom:
    """Convert q nanosecond timestamp (since 2000) → QAtom[datetime]."""
    unix_ns = ns + Q_EPOCH_NS
    dt = datetime.utcfromtimestamp(unix_ns / 1e9)
    return QAtom(dt, "p")

def to_q_timestamp(dt: datetime) -> int:
    unix_ns = int(dt.timestamp() * 1e9)
    return unix_ns - Q_EPOCH_NS

def timestamp_series(s: pl.Series) -> QVector:
    """Wrap a Polars ns-datetime Series as a QVector[p]."""
    if s.dtype != pl.Datetime("ns"):
        s = s.cast(pl.Datetime("ns"))
    return QVector(s, "p")

# q date arithmetic: 2024.01.15 + 1  = 2024.01.16
def date_add(d: QAtom, n: QAtom) -> QAtom:
    from datetime import timedelta
    return QAtom(d.value + timedelta(days=n.value), "d")

# Temporal extractors (maps to Polars dt namespace)
TEMPORAL_ATTRS = {
    "date":   lambda s: s.dt.date(),
    "time":   lambda s: s.dt.time(),
    "year":   lambda s: s.dt.year(),
    "month":  lambda s: s.dt.month(),
    "dd":     lambda s: s.dt.day(),
    "hh":     lambda s: s.dt.hour(),
    "mm":     lambda s: s.dt.minute(),
    "ss":     lambda s: s.dt.second(),
    "ns":     lambda s: s.dt.nanosecond(),
}

def parse_date_lit(s: str) -> QAtom:
    """Parse a q date literal string '2024.01.15' into a QAtom[d]."""
    y, m, d_ = int(s[:4]), int(s[5:7]), int(s[8:10])
    return QAtom(date(y, m, d_), "d")

def parse_time_lit(s: str) -> QAtom:
    """Parse a q time literal string 'HH:MM:SS.mmm' into a QAtom[t]."""
    h, mn, sc = int(s[:2]), int(s[3:5]), int(s[6:8])
    frac = s[9:] if len(s) > 8 else "0"
    ms = int(frac.ljust(3, "0")[:3])
    return QAtom(time(h, mn, sc, ms * 1000), "t")

def parse_timestamp_lit(s: str) -> QAtom:
    """Parse a q timestamp literal 'YYYY.MM.DDDhh:mm:ss.nnnnnnnnn' into QAtom[p]."""
    date_str, time_str = s.split("D", 1)
    y, mo, d_ = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    h, mn, sc = int(time_str[:2]), int(time_str[3:5]), int(time_str[6:8])
    frac = time_str[9:] if len(time_str) > 8 else "0"
    ns = int(frac.ljust(9, "0")[:9])
    us = ns // 1000
    return QAtom(datetime(y, mo, d_, h, mn, sc, us), "p")

def parse_month_lit(s: str) -> QAtom:
    """Parse a q month literal '2024.01m' into QAtom[m]."""
    y, m = int(s[:4]), int(s[5:7])
    return QAtom(date(y, m, 1), "m")

def extract(attr: str, x: QVector) -> QVector:
    fn = TEMPORAL_ATTRS.get(attr)
    if fn is None: raise QDomainError(f"unknown temporal attr: {attr}")
    return QVector(fn(x.series), "j")
