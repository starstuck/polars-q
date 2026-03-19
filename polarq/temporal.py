from polarq.types import QAtom, QVector
import polars as pl
from datetime import date, time, datetime

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

def extract(attr: str, x: QVector) -> QVector:
    fn = TEMPORAL_ATTRS.get(attr)
    if fn is None: raise QDomainError(f"unknown temporal attr: {attr}")
    return QVector(fn(x.series), "j")
