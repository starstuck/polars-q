from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
import polars as pl
import numpy as np

from polarq.errors import QError, QTypeError, QLengthError, QRankError, QDomainError  # noqa: F401

# ── Null sentinel ────────────────────────────────────────────────────────────

class QNull:
    """Generic null ::  — also used as gap sentinel in projections."""
    _instance = None
    def __new__(cls): 
        if not cls._instance: cls._instance = super().__new__(cls)
        return cls._instance
    def __repr__(self): return "::"

qnull = QNull()

# Typed nulls — 0N 0n 0Nd etc.
TYPED_NULLS = {
    "j": None,   "i": None,   "h": None,
    "f": float("nan"),        "e": float("nan"),
    "s": "",     "c": " ",    "p": None,
    "d": None,   "t": None,
}

# ── Atoms ────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class QAtom:
    value: Any
    kind:  str   # single char: b i j h f e c s p d t u v n

    def __repr__(self):
        if   self.kind == "s": return f"`{self.value}"
        elif self.kind == "b": return "1b" if self.value else "0b"
        elif self.kind == "c": return f'"{self.value}"'
        elif self.kind == "f":
            v = self.value
            # Match q console: 5.0 → "5f", 3.14 → "3.14f", nan → "0nf"
            if isinstance(v, float) and v % 1 == 0 and v == v:
                return f"{int(v)}f"
            return f"{v}f"
        else:                  return repr(self.value)

    def __eq__(self, other) -> bool:
        if isinstance(other, QAtom): return self.value == other.value
        return self.value == other

    def __hash__(self) -> int:
        return hash(self.value)

    def __sub__(self, other):
        """Support Python arithmetic for test assertions (e.g. abs(atom - 3.14))."""
        if isinstance(other, QAtom): return self.value - other.value
        return self.value - other

    def is_null(self) -> bool:
        if self.kind in ("f","e"): return self.value != self.value  # nan
        return self.value is None

# ── Vectors ───────────────────────────────────────────────────────────────────

KIND_TO_POLARS = {
    "b": pl.Boolean, "i": pl.Int32,   "j": pl.Int64,
    "h": pl.Int16,   "e": pl.Float32, "f": pl.Float64,
    "c": pl.Utf8,    "s": pl.Categorical,
    "p": pl.Datetime("ns"), "d": pl.Date, "t": pl.Time,
}

@dataclass(slots=True)
class QVector:
    """
    Homogeneous typed vector. Polars Series is the source of truth.
    All arithmetic routes through Series — zero Python loops.
    """
    series: pl.Series
    kind:   str

    @classmethod
    def from_items(cls, items: list, kind: str) -> QVector:
        dtype = KIND_TO_POLARS.get(kind)
        return cls(pl.Series(values=items, dtype=dtype), kind)

    @classmethod
    def from_series(cls, s: pl.Series) -> QVector:
        # infer kind from polars dtype
        kind = POLARS_TO_KIND.get(s.dtype, "j")
        return cls(s, kind)

    def __len__(self):     return len(self.series)
    def __repr__(self):
        if self.kind == "c":
            return " ".join(f'"{v}"' for v in self.series)
        return repr(list(self.series))

    # q-style indexing: v[2], v[0 1 2], v[::] (elision)
    def __getitem__(self, idx):
        if isinstance(idx, QAtom):
            return QAtom(self.series[idx.value], self.kind)
        if isinstance(idx, QVector):
            return QVector(self.series[idx.series.to_list()], self.kind)
        return QAtom(self.series[idx], self.kind)

# ── Mixed / Nested list ───────────────────────────────────────────────────────

@dataclass(slots=True)
class QList:
    """General (mixed / jagged) list — Python list of QValues."""
    items: list

    def __len__(self):  return len(self.items)
    def __repr__(self): return f"({';'.join(repr(i) for i in self.items)})"

# ── Dict & Table ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class QDict:
    keys:   QValue   # usually QVector[sym]
    values: QValue

    def to_polars(self) -> dict:
        """Convert to {col: Series} for DataFrame construction."""
        ks = self.keys.series.to_list() if isinstance(self.keys, QVector) else self.keys.items
        vs = self.values.items if isinstance(self.values, QList) else [self.values]
        return dict(zip(ks, vs))

@dataclass
class QTable:
    """
    Unkeyed table.  LazyFrame is authoritative — .collect() only when
    user explicitly asks for results or IPC serialization happens.
    """
    frame: pl.LazyFrame
    attrs: dict = field(default_factory=dict)  # `s# `u# `p#

    @classmethod
    def from_dict(cls, d: dict[str, list]) -> QTable:
        return cls(pl.DataFrame(d).lazy())

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> QTable:
        return cls(df.lazy())

    def meta(self) -> dict:
        schema = self.frame.schema
        return {"cols": list(schema.keys()), "types": list(schema.values())}

    def __repr__(self):
        return self.frame.collect().__repr__()

@dataclass
class QKeyedTable:
    key_table: QTable
    val_table: QTable

    def to_polars(self) -> pl.LazyFrame:
        """Merge key and value columns for join operations."""
        return pl.concat([self.key_table.frame, self.val_table.frame], how="horizontal")

# ── Functions ─────────────────────────────────────────────────────────────────

@dataclass
class QLambda:
    params:  list[str]     # empty → implicit x y z
    body:    Callable      # already-compiled Python callable from bytecode
    closure: QEnv

    def __call__(self, *args):
        from polarq.dispatch import apply_lambda
        return apply_lambda(self, list(args))

@dataclass
class QPartial:
    """Projection: f[;1] leaves a gap, produces a new partial."""
    func: QValue
    args: list   # None = unfilled gap

@dataclass
class QBuiltin:
    name:  str
    monad: Callable   # f(x)
    dyad:  Callable   # f(x, y) — None if monad-only

    def __call__(self, *args):
        if len(args) == 1: return self.monad(args[0])
        if len(args) == 2: return self.dyad(args[0], args[1])
        raise QError(f"{self.name}: wrong number of args")

@dataclass
class QAdverb:
    adverb: str    # "/" "\\" "'" "/:" "\\:"
    verb:   QValue

QValue = (
    QNull | QAtom | QVector | QList | QDict |
    QTable | QKeyedTable | QLambda | QPartial | QBuiltin | QAdverb
)
