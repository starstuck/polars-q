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
        elif self.kind == "h": return f"{self.value}h"
        elif self.kind == "i": return f"{self.value}i"
        elif self.kind == "f":
            v = self.value
            if v != v: return "0nf"          # nan
            return f"{v:.4g}f"               # 4 sig figs, matches q console
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

    def __bool__(self) -> bool:
        if self.kind == "b": return bool(self.value)
        if self.kind in ("f", "e"):
            return self.value == self.value and self.value != 0  # nan → False
        if self.kind in ("j", "i", "h"): return self.value != 0
        return self.value is not None

    def __index__(self) -> int:
        if self.kind in ("j", "i", "h"): return int(self.value)
        raise TypeError(f"QAtom of kind '{self.kind}' cannot be used as an integer index")

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

POLARS_TO_KIND = {v: k for k, v in KIND_TO_POLARS.items()}

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

    def __str__(self):
        """q-style display: space-separated elements."""
        if self.kind == "b":
            return "".join("1" if v else "0" for v in self.series) + "b"
        elif self.kind == "s":
            return "".join(f"`{v}" for v in self.series)
        elif self.kind == "c":
            return " ".join(f'"{v}"' for v in self.series)
        elif self.kind in ("f", "e"):
            return " ".join(f"{v:.4g}" for v in self.series.to_list())
        else:
            return " ".join(str(v) for v in self.series.to_list())

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
    def __repr__(self): return repr(self.items)
    def __str__(self):
        """q-style display: () empty, ,x single, (a;b;c) multi."""
        if len(self.items) == 0:
            return "()"
        if len(self.items) == 1:
            return "," + str(self.items[0])
        return "(" + ";".join(str(item) for item in self.items) + ")"

# ── Dict & Table ──────────────────────────────────────────────────────────────

@dataclass(slots=True)
class QDict:
    keys:   QValue   # usually QVector[sym]
    values: QValue

    def _key_list(self):
        if isinstance(self.keys, QVector):
            return self.keys.series.to_list()
        if isinstance(self.keys, QList):
            return [i.value if isinstance(i, QAtom) else i for i in self.keys.items]
        return [self.keys]

    def _val_list(self):
        if isinstance(self.values, QVector):
            return [QAtom(v, self.values.kind) for v in self.values.series.to_list()]
        if isinstance(self.values, QList):
            return self.values.items
        return [self.values]

    def __str__(self) -> str:
        ks = self._key_list()
        vs = self._val_list()
        return "\n".join(f"{k}|{v}" for k, v in zip(ks, vs))

    def __eq__(self, other) -> bool:
        if not isinstance(other, QDict):
            return False
        # Compare keys and values using _match-style logic
        def _vec_eq(a, b):
            if isinstance(a, QVector) and isinstance(b, QVector):
                if len(a.series) != len(b.series):
                    return False
                return bool((a.series == b.series).all())
            return a == b
        return _vec_eq(self.keys, other.keys) and _vec_eq(self.values, other.values)

    def __call__(self, key):
        """Dict indexing: d[`a] → value for key `a."""
        k = key.value if isinstance(key, QAtom) else key
        ks = self._key_list()
        vs = self._val_list()
        for ki, vi in zip(ks, vs):
            if ki == k:
                return vi
        raise KeyError(f"key not found: {k!r}")

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

    def __str__(self) -> str:
        df = self.frame.collect()
        if df.is_empty():
            return "(empty table)"
        cols = df.columns
        # Format each cell value in q style
        def _fmt(val) -> str:
            if val is None:
                return ""
            if isinstance(val, float):
                import math
                if math.isnan(val):
                    return "0n"
                # Use 4 sig figs without trailing 'f' for table display
                return f"{val:.6g}"
            return str(val)

        str_data = {c: [_fmt(v) for v in df[c]] for c in cols}
        widths   = {c: max(len(c), max((len(s) for s in str_data[c]), default=0))
                    for c in cols}
        header = " ".join(c.ljust(widths[c]) for c in cols)
        sep    = "-" * sum(widths[c] + 1 for c in cols[:-1]) + "-" * widths[cols[-1]]
        rows   = [" ".join(str_data[c][i].ljust(widths[c]) for c in cols)
                  for i in range(len(df))]
        return "\n".join([header, sep] + rows)

    def __repr__(self):
        return self.frame.collect().__repr__()

@dataclass
class QKeyedTable:
    key_table: QTable
    val_table: QTable

    def to_polars(self) -> pl.LazyFrame:
        """Merge key and value columns for join operations."""
        return pl.concat([self.key_table.frame, self.val_table.frame], how="horizontal")

    def __str__(self) -> str:
        key_df = self.key_table.frame.collect()
        val_df = self.val_table.frame.collect()

        def _fmt(val) -> str:
            if val is None:
                return ""
            if isinstance(val, float):
                import math
                if math.isnan(val):
                    return "0n"
                return f"{val:.6g}"
            return str(val)

        key_cols = key_df.columns
        val_cols = val_df.columns
        key_str  = {c: [_fmt(v) for v in key_df[c]] for c in key_cols}
        val_str  = {c: [_fmt(v) for v in val_df[c]] for c in val_cols}
        kw = {c: max(len(c), max((len(s) for s in key_str[c]), default=0)) for c in key_cols}
        vw = {c: max(len(c), max((len(s) for s in val_str[c]), default=0)) for c in val_cols}

        key_part_w = sum(kw[c] for c in key_cols) + len(key_cols) - 1
        val_part_w = sum(vw[c] for c in val_cols) + len(val_cols) - 1

        def _row(kdata, vdata):
            kp = " ".join(kdata[c].ljust(kw[c]) for c in key_cols)
            vp = " ".join(vdata[c].ljust(vw[c]) for c in val_cols)
            return kp + "| " + vp

        header = _row({c: c for c in key_cols}, {c: c for c in val_cols})
        sep    = "-" * key_part_w + "| " + "-" * val_part_w
        rows   = [_row({c: key_str[c][i] for c in key_cols},
                       {c: val_str[c][i] for c in val_cols})
                  for i in range(len(key_df))]
        return "\n".join([header, sep] + rows)

    def __repr__(self) -> str:
        return self.__str__()

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
