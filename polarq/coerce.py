from polarq.types import *
import polars as pl

def promote(x: QValue, y: QValue) -> tuple[QValue, QValue]:
    """Lift atoms to same rank as their partner before an operation."""
    if type(x) == type(y):
        return x, y
    # atom op vector → broadcast atom
    if isinstance(x, QAtom) and isinstance(y, QVector):
        return QVector(pl.Series([x.value] * len(y), dtype=y.series.dtype), x.kind), y
    if isinstance(x, QVector) and isinstance(y, QAtom):
        return x, QVector(pl.Series([y.value] * len(x), dtype=x.series.dtype), y.kind)
    # scalar op table column → handled in verb dispatch
    return x, y

def unify_kind(k1: str, k2: str) -> str:
    """q type promotion lattice — mirrors q's implicit casting rules."""
    RANK = {"b":0, "i":1, "j":2, "h":3, "e":4, "f":5, "s":6, "c":7}
    if k1 == k2:   return k1
    r1, r2 = RANK.get(k1, 99), RANK.get(k2, 99)
    return k1 if r1 > r2 else k2

def to_polars_expr(val: QValue, colname: str = None) -> pl.Expr:
    """
    Convert a QValue to a Polars expression for use inside
    .select() / .with_columns() / .filter() chains.
    """
    if isinstance(val, QAtom):
        return pl.lit(val.value)
    if isinstance(val, QVector):
        return pl.lit(val.series)
    if isinstance(val, pl.Expr):
        return val
    if isinstance(val, str):
        return pl.col(val)   # bare string treated as column name
    raise QTypeError(f"Cannot convert {type(val)} to Polars expression")
