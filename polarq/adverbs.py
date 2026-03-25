from polarq.types import *
import polars as pl


# ── Internal helpers ──────────────────────────────────────────────────────────

def _call(fn, *args):
    """Invoke fn, supporting both QBuiltin and plain Python callables."""
    if isinstance(fn, QBuiltin):
        if len(args) == 1:
            return fn.monad(args[0])
        return fn.dyad(args[0], args[1])
    return fn(*args)


def _iter(x: QValue) -> list:
    """Expand x into a list of QValues for element-wise iteration."""
    if isinstance(x, QVector):
        return [QAtom(v, x.kind) for v in x.series.to_list()]
    if isinstance(x, QList):
        return x.items
    return [x]


def _collect(results: list) -> QValue:
    """
    Combine a list of results into the most appropriate QValue:
      - single result  → return it directly (unwrap)
      - all same-kind QAtoms → QVector
      - mixed          → QList
    """
    if len(results) == 0:
        return QList([])
    if len(results) == 1:
        return results[0]
    if all(isinstance(r, QAtom) and r.kind == results[0].kind for r in results):
        kind = results[0].kind
        vals = [r.value for r in results]
        dtype = KIND_TO_POLARS.get(kind)
        return QVector(pl.Series(vals, dtype=dtype), kind)
    return QList(results)


# ── Public adverb functions ───────────────────────────────────────────────────

def over(fn, x, init=None):
    """
    f/ — fold (reduce).  Routes to Polars native agg for known QBuiltins.
    """
    POLARS_FOLDS = {
        "sum": lambda v: QAtom(v.series.sum(), v.kind),
        "mul": lambda v: QAtom(v.series.product(), v.kind),
        "min": lambda v: QAtom(v.series.min(), v.kind),
        "max": lambda v: QAtom(v.series.max(), v.kind),
    }
    if isinstance(fn, QBuiltin) and fn.name in POLARS_FOLDS and isinstance(x, QVector):
        return POLARS_FOLDS[fn.name](x)

    # Convergence form: f/ x (atom) — apply until stable
    if init is None and isinstance(x, QAtom):
        prev, curr = None, x
        while prev is None or prev.value != curr.value:
            prev = curr
            curr = _call(fn, curr)
        return curr

    if isinstance(x, QVector):
        items = [QAtom(v, x.kind) for v in x.series.to_list()]
    elif isinstance(x, QList):
        items = x.items
    else:
        items = [x]

    acc = init if init is not None else items[0]
    for item in (items if init is not None else items[1:]):
        acc = _call(fn, acc, item)
    return acc


def scan(fn, x, init=None):
    """
    f\\ — running fold.  Routes to Polars cum_* for known QBuiltins.
    """
    POLARS_SCANS = {
        "sum": lambda s: s.cum_sum(),
        "min": lambda s: s.cum_min(),
        "max": lambda s: s.cum_max(),
    }
    if isinstance(fn, QBuiltin) and fn.name in POLARS_SCANS and isinstance(x, QVector):
        return QVector(POLARS_SCANS[fn.name](x.series), x.kind)

    if isinstance(x, QVector):
        items = [QAtom(v, x.kind) for v in x.series.to_list()]
    elif isinstance(x, QList):
        items = x.items
    else:
        items = [x]

    acc = init if init is not None else items[0]
    results = [acc]
    for item in (items if init is not None else items[1:]):
        acc = _call(fn, acc, item)
        results.append(acc)
    # scan always returns a vector (never unwraps a single-element result)
    if all(isinstance(r, QAtom) and r.kind == results[0].kind for r in results):
        kind = results[0].kind
        dtype = KIND_TO_POLARS.get(kind)
        return QVector(pl.Series([r.value for r in results], dtype=dtype), kind)
    return _collect(results)


def each(fn, x):
    """f' / f each — apply fn to each element of x."""
    items = _iter(x)
    return _collect([_call(fn, xi) for xi in items])


def each_right(fn, y):
    """f/: y — apply fn to each element of y (right argument varies)."""
    items = _iter(y)
    return _collect([_call(fn, yi) for yi in items])


def each_left(fn, x):
    """f\\: x — apply fn to each element of x (left argument varies)."""
    items = _iter(x)
    return _collect([_call(fn, xi) for xi in items])


def each_prior(fn, x):
    """
    f': x — each-prior.
    Result[0] = x[0] (identity).
    Result[i] = fn(x[i-1], x[i]) for i > 0.
    """
    items = _iter(x)
    if not items:
        return QList([])
    results = [items[0]]
    for i in range(1, len(items)):
        results.append(_call(fn, items[i - 1], items[i]))
    return _collect(results)


def each_both(fn, x, y):
    """x f': y — paired application (zip)."""
    xs, ys = _iter(x), _iter(y)
    if len(xs) != len(ys):
        raise QLengthError("each_both: length mismatch")
    return _collect([_call(fn, xi, yi) for xi, yi in zip(xs, ys)])
