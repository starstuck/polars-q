from polarq.types import *
from polarq.coerce import promote, unify_kind
import polars as pl
import math
import fnmatch
import functools
import operator
import bisect

def _arith(polars_op, py_op):
    """
    Factory for arithmetic verbs.
    Routes: atom×atom → Python scalar
            vector×* → Polars Series op
            table col → Polars Expr (caller injects column context)
    """
    def dyad(x: QValue, y: QValue) -> QValue:
        x, y = promote(x, y)
        if isinstance(x, QAtom) and isinstance(y, QAtom):
            return QAtom(py_op(x.value, y.value), unify_kind(x.kind, y.kind))
        if isinstance(x, QVector) and isinstance(y, QVector):
            return QVector(polars_op(x.series, y.series), unify_kind(x.kind, y.kind))
        raise QTypeError(f"type mismatch")
    return dyad


def _date_add(x: QValue, y: QValue) -> QValue:
    """date + int  →  date shifted by n days."""
    from datetime import timedelta, date
    if isinstance(x, QAtom) and x.kind == "d" and isinstance(y, QAtom):
        return QAtom(x.value + timedelta(days=int(y.value)), "d")
    return _arith(lambda a, b: a + b, lambda a, b: a + b)(x, y)


def _date_sub(x: QValue, y: QValue) -> QValue:
    """date - date  →  int (days);  date - int  →  date shifted back."""
    from datetime import timedelta, date
    if isinstance(x, QAtom) and x.kind == "d" and isinstance(y, QAtom):
        if y.kind == "d":
            return QAtom((x.value - y.value).days, "j")
        return QAtom(x.value - timedelta(days=int(y.value)), "d")
    return _arith(lambda a, b: a - b, lambda a, b: a - b)(x, y)


# ── Core arithmetic ───────────────────────────────────────────────────────────

q_add = QBuiltin("add",
    monad = lambda x: x,                              # monadic + = flip for tables
    dyad  = _date_add,
)
q_sub = QBuiltin("sub",
    monad = lambda x: QAtom(-x.value, x.kind) if isinstance(x, QAtom)
                      else QVector(-x.series, x.kind),
    dyad  = _date_sub,
)
q_mul = QBuiltin("mul",
    monad = lambda x: x,                              # monadic * = first of list
    dyad  = _arith(lambda a,b: a*b, lambda a,b: a*b)
)
def _div_dyad(x: QValue, y: QValue) -> QValue:
    """q % always returns float."""
    x, y = promote(x, y)
    if isinstance(x, QAtom) and isinstance(y, QAtom):
        return QAtom(float(x.value) / float(y.value), "f")
    if isinstance(x, QVector) and isinstance(y, QVector):
        return QVector(x.series.cast(pl.Float64) / y.series.cast(pl.Float64), "f")
    raise QTypeError("type mismatch")

q_div = QBuiltin("div",
    monad = lambda x: x,                              # monadic % = matrix inverse (not impl)
    dyad  = _div_dyad,                                # NB: q % is divide not modulo
)

# ── Comparison ────────────────────────────────────────────────────────────────

def _cmp(op_name):
    OPS = {"lt": "__lt__", "gt": "__gt__", "eq": "__eq__", "ne": "__ne__",
           "le": "__le__", "ge": "__ge__"}
    def dyad(x, y):
        x, y = promote(x, y)
        if isinstance(x, QAtom) and isinstance(y, QAtom):
            return QAtom(getattr(x.value, OPS[op_name])(y.value), "b")
        if isinstance(x, QVector) and isinstance(y, QVector):
            s = getattr(x.series, OPS[op_name])(y.series)
            return QVector(s, "b")
    return dyad

def _match(x, y):
    """Structural equality — always returns a scalar boolean atom."""
    if isinstance(x, QAtom) and isinstance(y, QAtom):
        return QAtom(x.value == y.value, "b")
    if isinstance(x, QVector) and isinstance(y, QVector):
        if len(x.series) != len(y.series):
            return QAtom(False, "b")
        return QAtom(bool((x.series == y.series).all()), "b")
    if isinstance(x, QDict) and isinstance(y, QDict):
        return QAtom(x == y, "b")
    return QAtom(False, "b")

q_lt  = QBuiltin("lt",  monad=lambda x: x, dyad=_cmp("lt"))
q_gt  = QBuiltin("gt",  monad=lambda x: x, dyad=_cmp("gt"))
q_eq  = QBuiltin("eq",  monad=lambda x: x, dyad=_cmp("eq"))  # = in q
q_le  = QBuiltin("le",  monad=lambda x: x, dyad=_cmp("le"))  # <= in q
q_ge  = QBuiltin("ge",  monad=lambda x: x, dyad=_cmp("ge"))  # >= in q
q_not = QBuiltin("not", monad=lambda x: QAtom(not x.value, "b")
                               if isinstance(x, QAtom)
                               else QVector(~x.series, "b"),
                 dyad=_match)                                  # ~ dyadic = structural match

# ── Logic / bitwise (& = min/and, | = max/or) ─────────────────────────────────

def _and_dyad(x, y):
    x, y = promote(x, y)
    if isinstance(x, QAtom) and isinstance(y, QAtom):
        return QAtom(min(x.value, y.value), unify_kind(x.kind, y.kind))
    if isinstance(x, QVector) and isinstance(y, QVector):
        return QVector(pl.min_horizontal(x.series, y.series), unify_kind(x.kind, y.kind))
    raise QTypeError("type mismatch")

def _or_dyad(x, y):
    x, y = promote(x, y)
    if isinstance(x, QAtom) and isinstance(y, QAtom):
        return QAtom(max(x.value, y.value), unify_kind(x.kind, y.kind))
    if isinstance(x, QVector) and isinstance(y, QVector):
        return QVector(pl.max_horizontal(x.series, y.series), unify_kind(x.kind, y.kind))
    raise QTypeError("type mismatch")

q_and = QBuiltin("and", monad=lambda x: x, dyad=_and_dyad)  # & = min/and
q_or  = QBuiltin("or",  monad=lambda x: x, dyad=_or_dyad)   # | = max/or

def _all_m(x):
    if isinstance(x, QVector):
        return QAtom(bool(x.series.cast(pl.Boolean).all()), "b")
    if isinstance(x, QAtom):
        return QAtom(bool(x.value), "b")
    raise QTypeError("all expects vector")

def _any_m(x):
    if isinstance(x, QVector):
        return QAtom(bool(x.series.cast(pl.Boolean).any()), "b")
    if isinstance(x, QAtom):
        return QAtom(bool(x.value), "b")
    raise QTypeError("any expects vector")

q_all = QBuiltin("all", monad=_all_m, dyad=None)
q_any = QBuiltin("any", monad=_any_m, dyad=None)

# ── Math keyword verbs ────────────────────────────────────────────────────────

def _signum(v):
    if v > 0: return 1
    if v < 0: return -1
    return 0

def _math_monad(py_fn):
    def monad(x):
        if isinstance(x, QAtom):
            return QAtom(py_fn(float(x.value)), "f")
        if isinstance(x, QVector):
            return QVector(x.series.map_elements(lambda v: py_fn(float(v)), return_dtype=pl.Float64), "f")
        raise QTypeError("type mismatch")
    return monad

q_neg = QBuiltin("neg",
    monad=lambda x: QAtom(-x.value, x.kind) if isinstance(x, QAtom) else QVector(-x.series, x.kind),
    dyad=None)

q_abs = QBuiltin("abs",
    monad=lambda x: QAtom(abs(x.value), x.kind) if isinstance(x, QAtom) else QVector(x.series.abs(), x.kind),
    dyad=None)

q_signum = QBuiltin("signum",
    monad=lambda x: QAtom(_signum(x.value), x.kind) if isinstance(x, QAtom)
                    else QVector(x.series.map_elements(_signum, return_dtype=pl.Int64), "j"),
    dyad=None)

q_ceiling = QBuiltin("ceiling",
    monad=lambda x: QAtom(math.ceil(x.value), "j") if isinstance(x, QAtom)
                    else QVector(x.series.ceil().cast(pl.Int64), "j"),
    dyad=None)

q_floor = QBuiltin("floor",
    monad=lambda x: QAtom(math.floor(x.value), "j") if isinstance(x, QAtom)
                    else QVector(x.series.floor().cast(pl.Int64), "j"),
    dyad=None)

q_sqrt       = QBuiltin("sqrt",       monad=_math_monad(math.sqrt), dyad=None)
q_exp        = QBuiltin("exp",        monad=_math_monad(math.exp),  dyad=None)
q_log        = QBuiltin("log",        monad=_math_monad(math.log),  dyad=None)
q_reciprocal = QBuiltin("reciprocal", monad=_math_monad(lambda v: 1.0 / v), dyad=None)

q_xexp = QBuiltin("xexp",
    monad=lambda x: x,
    dyad=lambda x, y: QAtom(float(x.value) ** float(y.value), "f")
         if isinstance(x, QAtom) and isinstance(y, QAtom) else None)

q_xlog = QBuiltin("xlog",
    monad=lambda x: x,
    dyad=lambda x, y: QAtom(math.log(float(y.value)) / math.log(float(x.value)), "f")
         if isinstance(x, QAtom) and isinstance(y, QAtom) else None)

q_idiv = QBuiltin("idiv",
    monad=lambda x: x,
    dyad=lambda x, y: QAtom(int(x.value) // int(y.value), "j")
         if isinstance(x, QAtom) and isinstance(y, QAtom) else None)

q_mod = QBuiltin("mod",
    monad=lambda x: x,
    dyad=lambda x, y: QAtom(int(x.value) % int(y.value), "j")
         if isinstance(x, QAtom) and isinstance(y, QAtom) else None)

# ── List primitives ───────────────────────────────────────────────────────────

def q_count_m(x):
    if isinstance(x, QAtom):   return QAtom(1, "j")
    if isinstance(x, QVector): return QAtom(len(x.series), "j")
    if isinstance(x, QList):   return QAtom(len(x.items), "j")
    if isinstance(x, QDict):   return q_count_m(x.keys)
    if isinstance(x, QTable):  return QAtom(x.frame.collect().height, "j")
    return QAtom(0, "j")

def q_first_m(x):
    if isinstance(x, QVector): return QAtom(x.series[0], x.kind)
    if isinstance(x, QList):   return x.items[0]
    if isinstance(x, QTable):
        row = x.frame.first().collect()
        return QTable(row.lazy())
    return x

def q_reverse_m(x):
    if isinstance(x, QVector): return QVector(x.series.reverse(), x.kind)
    if isinstance(x, QList):   return QList(x.items[::-1])
    if isinstance(x, QTable):  return QTable(x.frame.reverse())
    return x

def q_where_m(x):
    """where — returns indices where boolean vector is 1b."""
    if isinstance(x, QVector) and x.kind == "b":
        idx = x.series.arg_true()
        return QVector(idx.cast(pl.Int64), "j")
    raise QTypeError("where expects boolean vector")

def q_distinct_m(x):
    if isinstance(x, QVector):
        return QVector(x.series.unique(maintain_order=True), x.kind)
    if isinstance(x, QTable):
        return QTable(x.frame.unique(maintain_order=True))
    return x

def q_group_m(x):
    """group — returns dict of sym→indices (like q's group)."""
    if isinstance(x, QVector):
        df = x.series.to_frame("v").with_row_index("i")
        grp = df.group_by("v").agg(pl.col("i").sort())
        keys   = QVector(grp["v"], x.kind)
        values = QList([QVector(row, "j") for row in grp["i"].to_list()])
        return QDict(keys, values)
    raise QTypeError("group expects vector")

q_count   = QBuiltin("count",   monad=q_count_m,   dyad=None)
q_first   = QBuiltin("first",   monad=q_first_m,   dyad=None)
q_last    = QBuiltin("last",    monad=lambda x: q_first_m(q_reverse_m(x)), dyad=None)
q_reverse = QBuiltin("reverse", monad=q_reverse_m, dyad=None)
q_where   = QBuiltin("where",   monad=q_where_m,   dyad=None)
q_distinct= QBuiltin("distinct",monad=q_distinct_m, dyad=None)
q_group   = QBuiltin("group",   monad=q_group_m,   dyad=None)

def q_til_m(x):
    n = x.value if isinstance(x, QAtom) else int(x)
    return QVector(pl.Series(values=list(range(n)), dtype=pl.Int64), "j")

def q_raze_m(x):
    items = x.items if isinstance(x, QList) else (x if isinstance(x, list) else [x])
    if items and all(isinstance(i, QVector) for i in items):
        return QVector(pl.concat([i.series for i in items]), items[0].kind)
    return QList(items)

q_til    = QBuiltin("til",    monad=q_til_m,            dyad=None)
q_enlist = QBuiltin("enlist", monad=lambda x: QList([x]), dyad=None)
q_raze   = QBuiltin("raze",   monad=q_raze_m,           dyad=None)

# ── Aggregation builtins (also used in qSQL compiler) ─────────────────────────

def _agg(series_method, py_fn):
    def monad(x):
        if isinstance(x, QVector): return QAtom(getattr(x.series, series_method)(), x.kind)
        if isinstance(x, QAtom):   return x
        if isinstance(x, QList):   return QAtom(py_fn(
            [i.value for i in x.items if isinstance(i, QAtom)]), "f")
    return monad

q_sum = QBuiltin("sum", monad=_agg("sum", sum), dyad=None)
q_min = QBuiltin("min", monad=_agg("min", min), dyad=None)
q_max = QBuiltin("max", monad=_agg("max", max), dyad=None)
q_avg = QBuiltin("avg", monad=lambda x: QAtom(x.series.mean(), "f")
                               if isinstance(x, QVector) else x, dyad=None)
q_dev = QBuiltin("dev", monad=lambda x: QAtom(x.series.std(ddof=0), "f")
                               if isinstance(x, QVector) else x, dyad=None)
q_med = QBuiltin("med", monad=lambda x: QAtom(x.series.median(), "f")
                               if isinstance(x, QVector) else x, dyad=None)
q_prd = QBuiltin("prd", monad=lambda x: QAtom(
                               functools.reduce(operator.mul, x.series.to_list(), 1), x.kind)
                               if isinstance(x, QVector) else x, dyad=None)
q_var = QBuiltin("var", monad=lambda x: QAtom(x.series.var(ddof=0), "f")
                               if isinstance(x, QVector) else x, dyad=None)

# ── Running / cumulative aggregations ─────────────────────────────────────────

q_sums = QBuiltin("sums", monad=lambda x: QVector(x.series.cum_sum(), x.kind)
                                if isinstance(x, QVector) else x, dyad=None)
q_prds = QBuiltin("prds", monad=lambda x: QVector(x.series.cum_prod(), x.kind)
                                if isinstance(x, QVector) else x, dyad=None)
q_maxs = QBuiltin("maxs", monad=lambda x: QVector(x.series.cum_max(), x.kind)
                                if isinstance(x, QVector) else x, dyad=None)
q_mins = QBuiltin("mins", monad=lambda x: QVector(x.series.cum_min(), x.kind)
                                if isinstance(x, QVector) else x, dyad=None)

def _q_avgs_m(x):
    if isinstance(x, QVector):
        s = x.series.cast(pl.Float64)
        n = pl.Series(list(range(1, len(s) + 1)), dtype=pl.Float64)
        return QVector(s.cum_sum() / n, "f")
    return x

def _q_deltas_m(x):
    if isinstance(x, QVector):
        s = x.series
        return QVector(s.diff(n=1).fill_null(int(s[0])), x.kind)
    return x

def _q_ratios_m(x):
    if isinstance(x, QVector):
        s = x.series.cast(pl.Float64)
        return QVector((s / s.shift(1)).fill_null(1.0), "f")
    return x

def _q_differ_m(x):
    if isinstance(x, QVector):
        s = x.series
        return QVector((s != s.shift(1)).fill_null(True), "b")
    return x

q_avgs   = QBuiltin("avgs",   monad=_q_avgs_m,   dyad=None)
q_deltas = QBuiltin("deltas", monad=_q_deltas_m, dyad=None)
q_ratios = QBuiltin("ratios", monad=_q_ratios_m, dyad=None)
q_differ = QBuiltin("differ", monad=_q_differ_m, dyad=None)

# ── Moving-window aggregations ─────────────────────────────────────────────────

def _win(n):
    return n.value if isinstance(n, QAtom) else int(n)

def _q_msum_d(n, x):
    return QVector(x.series.rolling_sum(window_size=_win(n), min_samples=1), x.kind)

def _q_mavg_d(n, x):
    return QVector(x.series.cast(pl.Float64).rolling_mean(window_size=_win(n), min_samples=1), "f")

def _q_mmin_d(n, x):
    return QVector(x.series.rolling_min(window_size=_win(n), min_samples=1), x.kind)

def _q_mmax_d(n, x):
    return QVector(x.series.rolling_max(window_size=_win(n), min_samples=1), x.kind)

def _q_mdev_d(n, x):
    result = x.series.cast(pl.Float64).rolling_std(
        window_size=_win(n), min_samples=1, ddof=0).round(4)
    return QVector(result, "f")

def _q_ema_d(alpha, x):
    a = alpha.value if isinstance(alpha, QAtom) else float(alpha)
    return QVector(x.series.cast(pl.Float64).ewm_mean(alpha=a, adjust=False), "f")

q_msum = QBuiltin("msum", monad=lambda x: x, dyad=_q_msum_d)
q_mavg = QBuiltin("mavg", monad=lambda x: x, dyad=_q_mavg_d)
q_mmin = QBuiltin("mmin", monad=lambda x: x, dyad=_q_mmin_d)
q_mmax = QBuiltin("mmax", monad=lambda x: x, dyad=_q_mmax_d)
q_mdev = QBuiltin("mdev", monad=lambda x: x, dyad=_q_mdev_d)
q_ema  = QBuiltin("ema",  monad=lambda x: x, dyad=_q_ema_d)

# ── Bucketing ─────────────────────────────────────────────────────────────────

def _q_xbar_d(n, x):
    nv = n.value if isinstance(n, QAtom) else int(n)
    if isinstance(x, QAtom):
        return QAtom((x.value // nv) * nv, x.kind)
    if isinstance(x, QVector):
        return QVector((x.series // nv) * nv, x.kind)
    raise QTypeError("xbar: right arg must be atom or vector")

def _q_bin_d(x, y):
    if isinstance(x, QVector):
        lst = x.series.to_list()
        v = y.value if isinstance(y, QAtom) else float(y)
        return QAtom(bisect.bisect_right(lst, v) - 1, "j")
    raise QTypeError("bin: left arg must be vector")

def _q_wavg_d(x, y):
    if not isinstance(y, QVector):
        raise QTypeError("wavg: right arg must be vector")
    y_s = y.series.cast(pl.Float64)
    if isinstance(x, QAtom):
        x_val = float(x.value)
        return QAtom(float((y_s * x_val).sum()) / x_val, "f")
    if isinstance(x, QVector):
        x_s = x.series.cast(pl.Float64)
        return QAtom(float((x_s * y_s).sum()) / float(x_s.sum()), "f")
    raise QTypeError("wavg: left arg must be atom or vector")

def _q_wsum_d(x, y):
    if isinstance(x, QVector) and isinstance(y, QVector):
        x_s = x.series.cast(pl.Float64)
        y_s = y.series.cast(pl.Float64)
        result = float((x_s * y_s).sum())
        if result == int(result):
            return QAtom(int(result), "j")
        return QAtom(result, "f")
    raise QTypeError("wsum: both args must be vectors")

q_xbar = QBuiltin("xbar", monad=lambda x: x, dyad=_q_xbar_d)
q_bin  = QBuiltin("bin",  monad=lambda x: x, dyad=_q_bin_d)
q_wavg = QBuiltin("wavg", monad=lambda x: x, dyad=_q_wavg_d)
q_wsum = QBuiltin("wsum", monad=lambda x: x, dyad=_q_wsum_d)

# ── String verbs ──────────────────────────────────────────────────────────────

def _str_val(x) -> str:
    """Extract a Python str from QAtom('c') or raw Python str."""
    if isinstance(x, str): return x
    if isinstance(x, QAtom): return str(x.value)
    raise QTypeError("type")

def _q_string_m(x) -> QAtom:
    if isinstance(x, QAtom):
        if x.kind == "s": v = x.value
        elif x.kind == "b": v = "1" if x.value else "0"
        elif x.kind == "f":
            v = str(int(x.value)) if x.value % 1 == 0 else str(x.value)
        else: v = str(x.value)
        return QAtom(v, "c")
    if isinstance(x, str): return QAtom(x, "c")
    raise QTypeError("string: unsupported type")

def _str_monad(fn):
    def monad(x):
        return QAtom(fn(_str_val(x)), "c")
    return monad

q_string = QBuiltin("string", monad=_q_string_m, dyad=None)
q_lower  = QBuiltin("lower",  monad=_str_monad(str.lower),  dyad=None)
q_upper  = QBuiltin("upper",  monad=_str_monad(str.upper),  dyad=None)
q_trim   = QBuiltin("trim",   monad=_str_monad(str.strip),  dyad=None)
q_ltrim  = QBuiltin("ltrim",  monad=_str_monad(str.lstrip), dyad=None)
q_rtrim  = QBuiltin("rtrim",  monad=_str_monad(str.rstrip), dyad=None)

def _like_dyad(x, y) -> QAtom:
    s, pat = _str_val(x), _str_val(y)
    return QAtom(fnmatch.fnmatch(s, pat), "b")

def _ss_dyad(x, y) -> QVector:
    """Find all start positions of y in x (0-indexed)."""
    s, sub = _str_val(x), _str_val(y)
    if not sub:
        return QVector(pl.Series(values=[], dtype=pl.Int64), "j")
    positions, pos = [], 0
    while True:
        idx = s.find(sub, pos)
        if idx == -1:
            break
        positions.append(idx)
        pos = idx + 1
    return QVector(pl.Series(values=positions, dtype=pl.Int64), "j")

def _sv_dyad(x, y) -> QAtom:
    """Delimiter sv string: join chars/strings with delimiter."""
    delim = _str_val(x)
    if isinstance(y, str):
        return QAtom(delim.join(y), "c")           # join chars of string
    if isinstance(y, QAtom) and y.kind == "c":
        return QAtom(delim.join(y.value), "c")
    if isinstance(y, QVector) and y.kind == "c":
        return QAtom(delim.join(y.series.to_list()), "c")
    if isinstance(y, QList):
        return QAtom(delim.join(_str_val(i) for i in y.items), "c")
    raise QTypeError("sv: right arg must be string or list of strings")

def _vs_dyad(x, y) -> QVector:
    """Delimiter vs string: split string by delimiter."""
    delim, s = _str_val(x), _str_val(y)
    parts = s.split(delim)
    return QVector(pl.Series(values=parts, dtype=pl.Utf8), "c")

def _join_dyad(x, y):
    if isinstance(x, str) and isinstance(y, str):
        return x + y
    if isinstance(x, QAtom) and isinstance(y, QAtom) and x.kind == "c" == y.kind:
        return QAtom(x.value + y.value, "c")
    # Numeric atoms of the same kind → 2-element vector
    if (isinstance(x, QAtom) and isinstance(y, QAtom)
            and x.kind == y.kind and x.kind not in ("c", "s")):
        from polarq.types import KIND_TO_POLARS
        dtype = KIND_TO_POLARS.get(x.kind)
        return QVector(pl.Series([x.value, y.value], dtype=dtype), x.kind)
    # Atom + vector or vector + atom → extend
    if isinstance(x, QAtom) and isinstance(y, QVector) and x.kind == y.kind:
        return QVector(
            pl.concat([pl.Series([x.value], dtype=y.series.dtype), y.series]), x.kind
        )
    if isinstance(x, QVector) and isinstance(y, QAtom) and x.kind == y.kind:
        return QVector(
            pl.concat([x.series, pl.Series([y.value], dtype=x.series.dtype)]), x.kind
        )
    if isinstance(x, (QAtom, str)) and isinstance(y, (QAtom, str)):
        return QAtom(_str_val(x) + _str_val(y), "c")
    if isinstance(x, QVector) and isinstance(y, QVector) and x.kind == y.kind:
        return QVector(pl.concat([x.series, y.series]), x.kind)
    lhs = x.items if isinstance(x, QList) else [x]
    rhs = y.items if isinstance(y, QList) else [y]
    return QList(lhs + rhs)

q_join = QBuiltin("join", monad=lambda x: QList([x]), dyad=_join_dyad)

q_like = QBuiltin("like", monad=lambda x: x, dyad=_like_dyad)
q_ss   = QBuiltin("ss",   monad=lambda x: x, dyad=_ss_dyad)
q_sv   = QBuiltin("sv",   monad=lambda x: x, dyad=_sv_dyad)
q_vs   = QBuiltin("vs",   monad=lambda x: x, dyad=_vs_dyad)

# ── Dictionary verbs ─────────────────────────────────────────────────────────

def q_dict_create(keys, values) -> QDict:
    """x!y — create a dict from keys vector and values vector/list."""
    return QDict(keys, values)

def q_key(x) -> QValue:
    """key d — return the keys of a dict."""
    if isinstance(x, QDict):
        return x.keys
    raise QTypeError("key: expected dict")

def q_value(x) -> QValue:
    """value d — return the values of a dict."""
    if isinstance(x, QDict):
        return x.values
    raise QTypeError("value: expected dict")


# ── Type introspection ────────────────────────────────────────────────────────

_KIND_TYPE_NUM = {
    "b": 1, "h": 5, "i": 6, "j": 7, "e": 8, "f": 9, "c": 10, "s": 11,
    "p": 12, "m": 13, "d": 14, "t": 19,
}

def q_type(x) -> QAtom:
    """type x — return q type number as a short atom."""
    if isinstance(x, QAtom):
        return QAtom(-_KIND_TYPE_NUM.get(x.kind, 0), "h")
    if isinstance(x, QVector):
        return QAtom(_KIND_TYPE_NUM.get(x.kind, 0), "h")
    if isinstance(x, str):       # Python str = q char vector
        return QAtom(10, "h")
    if isinstance(x, (QList, list)):
        return QAtom(0, "h")
    return QAtom(0, "h")


# ── Type casting ──────────────────────────────────────────────────────────────

_CAST_KIND = {
    "b": "b", "h": "h", "i": "i", "j": "j",
    "e": "e", "f": "f", "c": "c", "s": "s",
}
_CAST_DTYPE = {
    "b": pl.Boolean, "h": pl.Int16, "i": pl.Int32, "j": pl.Int64,
    "e": pl.Float32, "f": pl.Float64, "c": pl.Utf8, "s": pl.Categorical,
}
_CAST_PY = {
    "b": bool, "h": int, "i": int, "j": int,
    "e": float, "f": float, "c": str,
}

def q_cast(type_char: str, x) -> QValue:
    """
    \"x\"$y — cast y to the type indicated by type_char.
    Lowercase: numeric/type cast.  Uppercase: parse from string.
    """
    tc = type_char.lower()
    parse_mode = type_char.isupper()

    # Resolve the value to cast
    raw = x.value if isinstance(x, QAtom) else x

    if tc == "s":
        # cast to symbol: string → symbol atom
        val = str(raw)
        return QAtom(val, "s")

    if tc == "d":
        # cast to date: date → identity, int → days-from-2000-01-01
        from datetime import date as _date, timedelta
        if isinstance(raw, _date):
            return QAtom(raw, "d")
        _epoch = _date(2000, 1, 1)
        return QAtom(_epoch + timedelta(days=int(raw)), "d")

    if tc not in _CAST_PY:
        raise QTypeError(f"cast: unknown type char {type_char!r}")

    if parse_mode:
        # uppercase: parse the string representation
        src = str(raw)
        if tc in ("j", "h", "i"):
            val = int(src)
        elif tc in ("f", "e"):
            val = float(src)
        elif tc == "b":
            val = bool(int(src))
        else:
            val = src
    else:
        if tc in ("j", "h", "i"):
            val = int(raw)
        elif tc in ("f", "e"):
            val = float(raw)
        elif tc == "b":
            val = bool(raw)
        else:
            val = str(raw)

    kind = _CAST_KIND[tc]
    return QAtom(val, kind)


# ── Null checking ─────────────────────────────────────────────────────────────

def q_null(x) -> QAtom:
    """null x — return 1b if x is null, 0b otherwise."""
    if x is None:
        return QAtom(True, "b")
    if isinstance(x, QAtom):
        return QAtom(x.is_null(), "b")
    return QAtom(False, "b")


# ── Table operations ──────────────────────────────────────────────────────────

def q_flip_m(x):
    """flip x — dict→table or table→dict."""
    if isinstance(x, QDict):
        ks = x._key_list()
        vs = x._val_list()
        data = {}
        for k, v in zip(ks, vs):
            if isinstance(v, QVector):
                data[k] = v.series
            elif isinstance(v, QAtom):
                data[k] = pl.Series([v.value])
            elif isinstance(v, QList):
                raw = [i.value if isinstance(i, QAtom) else i for i in v.items]
                data[k] = raw
            else:
                data[k] = [v]
        return QTable(pl.DataFrame(data).lazy())
    if isinstance(x, QTable):
        df = x.frame.collect()
        keys = QVector.from_items(df.columns, "s")
        vals = QList([QVector.from_series(df[c]) for c in df.columns])
        return QDict(keys, vals)
    return x  # vectors: identity for 1D


def q_asc_m(x):
    """asc x — sort ascending."""
    if isinstance(x, QTable):
        cols = list(x.frame.collect_schema().names())
        return QTable(x.frame.sort(cols))
    if isinstance(x, QVector):
        return QVector(x.series.sort(), x.kind)
    return x


def q_xasc_d(cols, table):
    """cols xasc table — sort table by named column(s) ascending."""
    if not isinstance(table, QTable):
        raise QTypeError("xasc: right arg must be a table")
    if isinstance(cols, QAtom) and cols.kind == "s":
        sort_cols = [cols.value]
    elif isinstance(cols, QVector) and cols.kind == "s":
        sort_cols = cols.series.to_list()
    else:
        sort_cols = [str(cols.value if isinstance(cols, QAtom) else cols)]
    return QTable(table.frame.sort(sort_cols))


def q_lj_d(t1, t2):
    """t1 lj t2 — left join keyed table t2 onto t1."""
    if not isinstance(t1, QTable):
        raise QTypeError("lj: left arg must be a table")
    if not isinstance(t2, QKeyedTable):
        raise QTypeError("lj: right arg must be a keyed table")
    key_cols = list(t2.key_table.frame.collect_schema().names())
    t2_full = t2.to_polars()
    return QTable(t1.frame.join(t2_full, on=key_cols, how="left"))


q_flip = QBuiltin("flip", monad=q_flip_m, dyad=None)
q_asc  = QBuiltin("asc",  monad=q_asc_m,  dyad=None)
q_xasc = QBuiltin("xasc", monad=None,     dyad=q_xasc_d)
q_lj   = QBuiltin("lj",   monad=None,     dyad=q_lj_d)


# ── The global verb table (maps q token → QBuiltin) ───────────────────────────

VERB_TABLE: dict[str, QBuiltin] = {
    "+": q_add,  "-": q_sub,  "*": q_mul,  "%": q_div,
    "<": q_lt,   ">": q_gt,   "=": q_eq,   "~": q_not,
    "<=": q_le,  ">=": q_ge,
    "&": q_and,  "|": q_or,
    "all": q_all, "any": q_any,
    "#": QBuiltin("take",  monad=q_count_m,   dyad=lambda x,y: ...),
    "_": QBuiltin("drop",  monad=q_first_m,   dyad=lambda x,y: ...),
    "!": QBuiltin("key",   monad=q_group_m,   dyad=q_dict_create),
    "?": QBuiltin("find",  monad=q_distinct_m,dyad=lambda x,y: ...),
    "@": QBuiltin("index", monad=q_first_m,   dyad=lambda x,y: ...),
    ",": q_join,
    # string verbs
    "string": q_string, "lower": q_lower, "upper": q_upper,
    "trim": q_trim, "ltrim": q_ltrim, "rtrim": q_rtrim,
    "like": q_like, "ss": q_ss, "sv": q_sv, "vs": q_vs,
    # named verbs — core list
    "til": q_til, "enlist": q_enlist, "raze": q_raze,
    "count": q_count, "first": q_first, "last": q_last,
    "reverse": q_reverse, "where": q_where, "distinct": q_distinct, "group": q_group,
    # named verbs — aggregations
    "sum": q_sum, "min": q_min, "max": q_max, "avg": q_avg,
    "dev": q_dev, "med": q_med, "prd": q_prd, "var": q_var,
    # running aggs
    "sums": q_sums, "prds": q_prds, "maxs": q_maxs, "mins": q_mins,
    "avgs": q_avgs, "deltas": q_deltas, "ratios": q_ratios, "differ": q_differ,
    # moving-window aggs (dyadic)
    "msum": q_msum, "mavg": q_mavg, "mmin": q_mmin, "mmax": q_mmax,
    "mdev": q_mdev, "ema": q_ema,
    # bucketing (dyadic)
    "xbar": q_xbar, "bin": q_bin, "wavg": q_wavg, "wsum": q_wsum,
    # math keywords
    "neg": q_neg, "abs": q_abs, "signum": q_signum,
    "ceiling": q_ceiling, "floor": q_floor,
    "sqrt": q_sqrt, "exp": q_exp, "log": q_log, "reciprocal": q_reciprocal,
    "xexp": q_xexp, "xlog": q_xlog,
    "div": q_idiv, "mod": q_mod,
}
