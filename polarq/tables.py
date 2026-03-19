from polarq.types import *
import polars as pl

# Maps q agg function names to Polars Series/Expr methods
AGG_MAP = {
    "sum":   lambda e: e.sum(),
    "avg":   lambda e: e.mean(),
    "min":   lambda e: e.min(),
    "max":   lambda e: e.max(),
    "count": lambda e: e.count(),
    "dev":   lambda e: e.std(),
    "var":   lambda e: e.var(),
    "med":   lambda e: e.median(),
    "last":  lambda e: e.last(),
    "first": lambda e: e.first(),
}

def compile_select(table: QTable, columns, where, by) -> QTable:
    lf = table.frame

    # WHERE
    if where:
        for clause in where:
            lf = lf.filter(_to_expr(clause))

    # BY + aggregation
    if by:
        group_keys = [_to_expr(c) for c in by]
        if columns:
            aggs = [_col_to_agg_expr(c) for c in columns]
            lf   = lf.group_by(group_keys).agg(aggs)
        else:
            lf   = lf.group_by(group_keys).agg(pl.all())
    elif columns:
        lf = lf.select([_col_to_expr(c) for c in columns])

    return QTable(lf)


def compile_update(table: QTable, columns, where) -> QTable:
    lf = table.frame
    if where:
        for clause in where:
            lf = lf.filter(_to_expr(clause))
    if columns:
        lf = lf.with_columns([_col_to_expr(c) for c in columns])
    return QTable(lf)


def compile_aj(by_cols: list[str], t1: QTable, t2: QTable) -> QTable:
    """
    aj[`time`sym; trade; quote]
    Direct mapping to Polars join_asof with by= parameter.
    """
    time_col = by_cols[0]
    by       = by_cols[1:] if len(by_cols) > 1 else None
    result   = t1.frame.join_asof(
        t2.frame,
        on=time_col,
        by=by,
        strategy="backward"
    )
    return QTable(result)


def compile_wj(windows: tuple, by_cols: list[str],
               t1: QTable, t2: QTable, aggs) -> QTable:
    """
    wj[windows; `time`sym; trade; (quote; (avg;`bid); (avg;`ask))]
    Maps to Polars rolling join + group_by_dynamic pattern.
    """
    time_col = by_cols[0]
    by       = by_cols[1:] if len(by_cols) > 1 else None
    w_lo, w_hi = windows

    result = (
        t1.frame
        .join_asof(t2.frame, on=time_col, by=by, strategy="backward")
        .group_by_dynamic(
            time_col,
            every=f"{abs(w_lo)}ns",
            period=f"{w_hi - w_lo}ns",
            by=by
        )
        .agg([_compile_wj_agg(a) for a in aggs])
    )
    return QTable(result)


def _to_expr(node) -> pl.Expr:
    """Recursively compile a q expression AST node to a Polars Expr."""
    from polarq.verbs import VERB_TABLE
    match node:
        case str():              return pl.col(node)
        case QAtom(v, _):        return pl.lit(v)
        case QVector() as qv:    return pl.lit(qv.series)
        case ("binop", op, l, r):
            le, re = _to_expr(l), _to_expr(r)
            return {
                "+": le + re,  "-": le - re,
                "*": le * re,  "%": le / re,
                "<": le < re,  ">": le > re,
                "=": le == re, "~": le != re,
                "&": le & re,  "|": le | re,
            }[op]
        case ("call", name, arg):
            e = _to_expr(arg)
            agg = AGG_MAP.get(name)
            if agg: return agg(e)
            raise QTypeError(f"unknown column function: {name}")
        case _:
            raise QTypeError(f"cannot compile to Polars expr: {node!r}")


def _col_to_expr(col_def) -> pl.Expr:
    """Handle  newname:expr  aliasing."""
    if isinstance(col_def, tuple) and col_def[0] == "alias":
        _, name, inner = col_def
        return _to_expr(inner).alias(name)
    return _to_expr(col_def)


def _col_to_agg_expr(col_def) -> pl.Expr:
    return _col_to_expr(col_def)
