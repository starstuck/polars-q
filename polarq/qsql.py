"""
polarq.qsql — runtime helpers called by transpiled qSQL code.

The transpiler converts QSelect/QUpdate/QExec/QDelete AST nodes into calls
to these functions, passing Polars Expr objects for columns/conditions.
"""
from __future__ import annotations
import polars as pl
from polarq.types import QTable, QKeyedTable, QVector, QAtom, KIND_TO_POLARS
from polarq.errors import QTypeError


_POLARS_KIND = {
    pl.Boolean:     "b",
    pl.Int16:       "h",
    pl.Int32:       "i",
    pl.Int64:       "j",
    pl.Float32:     "e",
    pl.Float64:     "f",
    pl.Utf8:        "c",
    pl.String:      "c",
    pl.Categorical: "s",
}


def q_tbl_col(v):
    """Convert a q value to a Polars Series for use in pl.DataFrame construction."""
    if isinstance(v, QVector):
        return v.series
    if isinstance(v, QAtom):
        return pl.Series([v.value])
    return v  # assume list or Series already


def q_select_rt(table: QTable,
                cols:  list,   # list of pl.Expr (may include .alias())
                where: list,   # list of bool pl.Expr
                by:    list,   # list of pl.Expr for group-by keys
                ):
    lf = table.frame
    for w in where:
        lf = lf.filter(w)
    if by:
        by_names = [e.meta.output_name() for e in by]
        if cols:
            result_lf = lf.group_by(by).agg(cols).sort(by_names)
        else:
            result_lf = lf.group_by(by).agg(pl.all()).sort(by_names)
        all_cols = list(result_lf.collect_schema().names())
        val_names = [c for c in all_cols if c not in by_names]
        key_lf = result_lf.select(by_names)
        val_lf = result_lf.select(val_names)
        return QKeyedTable(QTable(key_lf), QTable(val_lf))
    elif cols:
        lf = lf.select(cols)
    return QTable(lf)


def q_exec_rt(table: QTable,
              cols:  list,
              where: list,
              by:    list,
              ):
    """exec — like select but returns a vector (single col) or table."""
    result = q_select_rt(table, cols, where, by)
    if isinstance(result, QKeyedTable):
        result = result.val_table
    df = result.frame.collect()
    if len(df.columns) == 1:
        s = df.to_series(0)
        kind = _POLARS_KIND.get(s.dtype, "j")
        return QVector(s, kind)
    return result


def q_update_rt(table: QTable,
                cols:  list,   # list of pl.Expr with .alias()
                where: list,
                ) -> QTable:
    lf = table.frame
    if where:
        # Partial update: use when() ... otherwise(col)
        filters = where[0]
        for w in where[1:]:
            filters = filters & w
        updated = [pl.when(filters).then(e).otherwise(pl.col(e.meta.output_name()))
                   .alias(e.meta.output_name())
                   for e in cols]
        lf = lf.with_columns(updated)
    elif cols:
        lf = lf.with_columns(cols)
    return QTable(lf)


def q_delete_rt(table: QTable,
                cols:  list,   # column names to drop (usually empty for row delete)
                where: list,
                ) -> QTable:
    lf = table.frame
    if where:
        # Delete rows: invert the filter
        neg = ~where[0]
        for w in where[1:]:
            neg = neg & ~w
        lf = lf.filter(neg)
    if cols:
        lf = lf.drop([e.meta.output_name() for e in cols])
    return QTable(lf)


_META_TYPE_CHAR = {
    "Boolean": "b", "Int8": "x",  "Int16": "h", "Int32": "i", "Int64": "j",
    "UInt8": "x",   "UInt16": "h","UInt32": "i", "UInt64": "j",
    "Float32": "e", "Float64": "f",
    "Utf8": "c",    "String": "c",
    "Categorical": "s",
    "Datetime": "p", "Date": "d", "Time": "t",
}


def q_meta(table: QTable) -> str:
    """meta t — display table schema in q style."""
    schema = table.frame.collect_schema()
    col_w = max((len(c) for c in schema), default=1)
    lines = ["c".ljust(col_w) + "| t f a",
             "-" * col_w + "| " + "-" * 5]
    for col, dtype in schema.items():
        # Use base type name to handle parameterised types (e.g. Categorical(ordering=…))
        base = str(dtype).split("(")[0]
        tc = _META_TYPE_CHAR.get(base, "?")
        lines.append(f"{col:<{col_w}}| {tc}  ")
    return "\n".join(lines)
