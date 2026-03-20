from polarq.types import *
import polars as pl
from functools import reduce

def over(verb: QBuiltin, x: QValue, init: QValue = None) -> QValue:
    """
    +/ — fold. Routes to Polars native agg when verb is a known builtin.
    Falls back to Python reduce otherwise.
    """
    POLARS_FOLDS = {
        "sum": lambda v: QAtom(v.series.sum(), v.kind),
        "mul": lambda v: QAtom(v.series.product(), v.kind),
        "min": lambda v: QAtom(v.series.min(), v.kind),
        "max": lambda v: QAtom(v.series.max(), v.kind),
    }
    if isinstance(verb, QBuiltin) and verb.name in POLARS_FOLDS:
        if isinstance(x, QVector):
            return POLARS_FOLDS[verb.name](x)

    # Convergence form: f/ x  (apply until stable)
    if init is None and isinstance(x, QAtom):
        prev, curr = None, x
        while prev is None or prev.value != curr.value:
            prev = curr
            curr = verb.monad(curr)
        return curr

    # Standard fold
    # QVector.series.to_list() produces plain Python scalars; wrap in QAtom so
    # that verb.dyad always receives proper QValue arguments.
    if isinstance(x, QVector):
        items = [QAtom(v, x.kind) for v in x.series.to_list()]
    else:
        items = x.items
    acc = init if init is not None else items[0]
    for item in (items if init is not None else items[1:]):
        acc = verb.dyad(acc, item)
    return acc


def scan(verb: QBuiltin, x: QValue, init: QValue = None) -> QVector:
    """
    +\\ — running fold. Routes to Polars cum_* for known verbs.
    """
    POLARS_SCANS = {
        "sum": lambda s: s.cum_sum(),
        "min": lambda s: s.cum_min(),
        "max": lambda s: s.cum_max(),
    }
    if isinstance(verb, QBuiltin) and verb.name in POLARS_SCANS:
        if isinstance(x, QVector):
            return QVector(POLARS_SCANS[verb.name](x.series), x.kind)

    items = x.series.to_list() if isinstance(x, QVector) else x.items
    acc   = init if init is not None else QAtom(items[0], x.kind)
    results = [acc]
    for item in (items if init is not None else items[1:]):
        item_val = QAtom(item, x.kind) if not isinstance(item, QValue) else item
        acc = verb.dyad(acc, item_val)
        results.append(acc)
    vals = [r.value if isinstance(r, QAtom) else r for r in results]
    return QVector(pl.Series(vals), x.kind)


def each(verb: QBuiltin, x: QValue) -> QValue:
    """
    f' — apply verb to each element. Vectorises over Series if possible.
    """
    if isinstance(x, QVector):
        # Try to stay in Polars: map known monads to Series methods
        POLARS_EACH = {
            "neg":     lambda s: -s,
            "reverse": lambda s: s.reverse(),  # no-op on atoms but safe
            "count":   lambda s: pl.Series([1]*len(s)),
        }
        if isinstance(verb, QBuiltin) and verb.name in POLARS_EACH:
            return QVector(POLARS_EACH[verb.name](x.series), x.kind)
        # Fallback: map_elements
        return QVector(
            x.series.map_elements(lambda v: verb.monad(QAtom(v, x.kind)).value),
            x.kind
        )
    if isinstance(x, QList):
        return QList([verb.monad(item) for item in x.items])
    if isinstance(x, QTable):
        # each over rows (expensive — warn)
        rows = x.frame.collect().to_dicts()
        return QList([verb.monad(_row_to_qtable(r)) for r in rows])
    return verb.monad(x)


def each_left(verb: QBuiltin, x: QValue, y: QValue) -> QList:
    """x f\\: y — apply (xi f y) for each xi in x."""
    items = _iter(x)
    return QList([verb.dyad(xi, y) for xi in items])


def each_right(verb: QBuiltin, x: QValue, y: QValue) -> QList:
    """x f/: y — apply (x f yi) for each yi in y."""
    items = _iter(y)
    return QList([verb.dyad(x, yi) for yi in items])


def each_both(verb: QBuiltin, x: QValue, y: QValue) -> QList:
    """x f': y — paired application."""
    xs, ys = _iter(x), _iter(y)
    if len(xs) != len(ys):
        raise QLengthError("each_both: length mismatch")
    return QList([verb.dyad(xi, yi) for xi, yi in zip(xs, ys)])


def _iter(x: QValue) -> list:
    if isinstance(x, QVector): return [QAtom(v, x.kind) for v in x.series.to_list()]
    if isinstance(x, QList):   return x.items
    return [x]
