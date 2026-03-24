from polarq.types import *
from polarq.coerce import promote, unify_kind
import polars as pl
import math
import fnmatch

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

# ── Core arithmetic ───────────────────────────────────────────────────────────

q_add = QBuiltin("add",
    monad = lambda x: x,                              # monadic + = flip for tables
    dyad  = _arith(lambda a,b: a+b, lambda a,b: a+b)
)
q_sub = QBuiltin("sub",
    monad = lambda x: QAtom(-x.value, x.kind) if isinstance(x, QAtom)
                      else QVector(-x.series, x.kind),
    dyad  = _arith(lambda a,b: a-b, lambda a,b: a-b)
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
q_dev = QBuiltin("dev", monad=lambda x: QAtom(x.series.std(), "f")
                               if isinstance(x, QVector) else x, dyad=None)
q_med = QBuiltin("med", monad=lambda x: QAtom(x.series.median(), "f")
                               if isinstance(x, QVector) else x, dyad=None)

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

# ── The global verb table (maps q token → QBuiltin) ───────────────────────────

VERB_TABLE: dict[str, QBuiltin] = {
    "+": q_add,  "-": q_sub,  "*": q_mul,  "%": q_div,
    "<": q_lt,   ">": q_gt,   "=": q_eq,   "~": q_not,
    "<=": q_le,  ">=": q_ge,
    "&": q_and,  "|": q_or,
    "all": q_all, "any": q_any,
    "#": QBuiltin("take",  monad=q_count_m,   dyad=lambda x,y: ...),
    "_": QBuiltin("drop",  monad=q_first_m,   dyad=lambda x,y: ...),
    "!": QBuiltin("key",   monad=q_group_m,   dyad=lambda x,y: ...),
    "?": QBuiltin("find",  monad=q_distinct_m,dyad=lambda x,y: ...),
    "@": QBuiltin("index", monad=q_first_m,   dyad=lambda x,y: ...),
    ",": q_join,
    # string verbs
    "string": q_string, "lower": q_lower, "upper": q_upper,
    "trim": q_trim, "ltrim": q_ltrim, "rtrim": q_rtrim,
    "like": q_like, "ss": q_ss, "sv": q_sv, "vs": q_vs,
    # named verbs — aggregations and list
    "sum": q_sum, "min": q_min, "max": q_max, "avg": q_avg,
    "dev": q_dev, "med": q_med, "count": q_count,
    "first": q_first, "last": q_last, "reverse": q_reverse,
    "where": q_where, "distinct": q_distinct, "group": q_group,
    # math keywords
    "neg": q_neg, "abs": q_abs, "signum": q_signum,
    "ceiling": q_ceiling, "floor": q_floor,
    "sqrt": q_sqrt, "exp": q_exp, "log": q_log, "reciprocal": q_reciprocal,
    "xexp": q_xexp, "xlog": q_xlog,
    "div": q_idiv, "mod": q_mod,
}
