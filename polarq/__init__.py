"""
polarq — q/kdb+ semantics on Polars + Arrow Flight.

Typical usage in pure Python:
    from polarq import Q, QTable, QVector, qvec
    import polars as pl

    q = Q()
    trade = q.table(pl.DataFrame({
        "sym":   ["AAPL","GOOG","AAPL"],
        "price": [150.0, 280.0, 155.0],
        "qty":   [100, 200, 150],
    }))

    # qSQL via method chain:
    result = q.select(trade,
        cols=["sym", ("sum_qty", q.sum("qty")), ("avg_px", q.avg("price"))],
        by=["sym"]
    )

    # Or transpile a .q file:
    from polarq.transpiler import loadq
    loadq("strategy.q", q.env)
"""

from polarq.types     import (QAtom, QVector, QList, QDict,
                               QTable, QKeyedTable, QLambda, QFn,
                               QBuiltin, QAdverb, QNull, qnull)
from polarq.env       import QEnv
from polarq.verbs     import (
    VERB_TABLE,
    q_add, q_sub, q_mul, q_div,
    q_lt, q_gt, q_eq, q_le, q_ge, q_not,
    q_and, q_or, q_all, q_any,
    q_string, q_lower, q_upper, q_trim, q_ltrim, q_rtrim,
    q_like, q_ss, q_sv, q_vs, q_join,
    q_sum, q_min, q_max, q_avg, q_dev, q_med, q_prd, q_var,
    q_count, q_first, q_last, q_reverse, q_where, q_distinct, q_group,
    q_til, q_enlist, q_raze,
    q_type, q_cast, q_null,
    q_at_apply, q_dot_apply, q_amend_at, q_trap,
    q_dict_create, q_key, q_value, q_show, q_parse,
    q_flip, q_asc, q_xasc, q_lj,
    q_neg, q_abs, q_signum, q_ceiling, q_floor,
    q_sqrt, q_exp, q_log, q_reciprocal, q_xexp, q_xlog,
    q_idiv, q_mod,
    q_sums, q_prds, q_maxs, q_mins, q_avgs, q_deltas, q_ratios, q_differ,
    q_msum, q_mavg, q_mmin, q_mmax, q_mdev, q_ema,
    q_xbar, q_bin, q_wavg, q_wsum,
    q_zero_colon, q_one_colon, q_read0, q_read1, q_get, q_set,
    q_take, q_drop, q_rotate, q_sublist,
    q_find, q_in, q_within,
)

# Named keyword aliases — transpiled q code uses bare names like `neg x`, `abs x`
type       = q_type   # shadows Python builtin in q exec context
null       = q_null
key        = q_key
value      = q_value
show       = q_show
parse      = q_parse
flip       = q_flip
asc        = q_asc
xasc       = q_xasc
lj         = q_lj
neg        = q_neg
abs        = q_abs  # shadows Python builtin intentionally in q exec context
signum     = q_signum
ceiling    = q_ceiling
floor      = q_floor
sqrt       = q_sqrt
exp        = q_exp
log        = q_log
reciprocal = q_reciprocal
xexp       = q_xexp
xlog       = q_xlog
div        = q_idiv
mod        = q_mod
count      = q_count
first      = q_first
last       = q_last
reverse    = q_reverse
where      = q_where
distinct   = q_distinct
group      = q_group
til        = q_til
enlist     = q_enlist
raze       = q_raze
string     = q_string
lower      = q_lower
upper      = q_upper
trim       = q_trim
ltrim      = q_ltrim
rtrim      = q_rtrim
sum        = q_sum   # shadows Python builtin intentionally in q exec context
avg        = q_avg
min        = q_min   # shadows Python builtin
max        = q_max   # shadows Python builtin
dev        = q_dev
med        = q_med
prd        = q_prd
var        = q_var
sums       = q_sums
prds       = q_prds
maxs       = q_maxs
mins       = q_mins
avgs       = q_avgs
deltas     = q_deltas
ratios     = q_ratios
differ     = q_differ
msum       = q_msum
mavg       = q_mavg
mmin       = q_mmin
mmax       = q_mmax
mdev       = q_mdev
ema        = q_ema
xbar       = q_xbar
bin        = q_bin   # shadows Python builtin
wavg       = q_wavg
wsum       = q_wsum
# I/O
csv        = ","          # q's csv separator character
read0      = q_read0
read1      = q_read1
get        = q_get
set        = q_set        # shadows Python builtin in q exec context
from polarq.adverbs   import over, scan, each, each_left, each_right, each_prior
from polarq.qsql      import (q_select_rt, q_exec_rt, q_update_rt, q_delete_rt,
                               q_meta, q_tbl_col)
meta = q_meta
from polarq.tables    import compile_select, compile_update, compile_aj, compile_wj
from polarq.temporal  import from_q_timestamp, to_q_timestamp, timestamp_series
from polarq.ipc       import PolarQServer, PolarQClient
from polarq.errors    import QError, QTypeError, QLengthError, QRankError

def qvec(*args, kind=None) -> QVector:
    """Convenience: qvec(1,2,3) or qvec('a','b','c', kind='s')."""
    items = list(args[0]) if len(args)==1 and hasattr(args[0],'__iter__') else list(args)
    if kind is None:
        kind = "f" if isinstance(items[0], float) else \
               "s" if isinstance(items[0], str)   else "j"
    return QVector.from_items(items, kind)

class Q:
    """Stateful q session — holds env, wraps verbs as methods."""
    def __init__(self): self.env = QEnv()

    def table(self, df): return QTable(df.lazy())
    def select(self, t, cols=None, where=None, by=None):
        return compile_select(t, cols, where, by)
    def aj(self, cols, t1, t2): return compile_aj(cols, t1, t2)
    def sum(self, col):  return ("call", "sum",  col)
    def avg(self, col):  return ("call", "avg",  col)
    def max(self, col):  return ("call", "max",  col)
    def min(self, col):  return ("call", "min",  col)
    def count(self, col):return ("call", "count",col)

__all__ = [
    "Q", "qvec", "QAtom", "QVector", "QList", "QDict",
    "QTable", "QKeyedTable", "QLambda", "QFn", "QBuiltin", "QAdverb",
    "QNull", "qnull", "QEnv", "PolarQServer", "PolarQClient",
    "over", "scan", "each", "each_left", "each_right", "each_prior",
    "compile_select", "compile_update", "compile_aj", "compile_wj",
    "QError", "QTypeError", "QLengthError", "QRankError",
    # qSQL runtime functions
    "q_select_rt", "q_exec_rt", "q_update_rt", "q_delete_rt",
    "q_meta", "q_tbl_col",
    # verb functions (for transpiled code: `from polarq import *`)
    "q_type", "q_cast", "q_null",
    "q_at_apply", "q_dot_apply", "q_amend_at", "q_trap",
    "q_dict_create", "q_key", "q_value", "q_show", "q_parse",
    "q_flip", "q_asc", "q_xasc", "q_lj",
    "q_add", "q_sub", "q_mul", "q_div",
    "q_lt", "q_gt", "q_eq", "q_le", "q_ge", "q_not",
    "q_and", "q_or", "q_all", "q_any",
    "q_string", "q_lower", "q_upper", "q_trim", "q_ltrim", "q_rtrim",
    "q_like", "q_ss", "q_sv", "q_vs", "q_join",
    "q_sum", "q_min", "q_max", "q_avg", "q_dev", "q_med",
    "q_count", "q_first", "q_last", "q_reverse", "q_where", "q_distinct", "q_group",
    "q_til", "q_enlist", "q_raze",
    "q_neg", "q_abs", "q_signum", "q_ceiling", "q_floor",
    "q_sqrt", "q_exp", "q_log", "q_reciprocal", "q_xexp", "q_xlog",
    "q_idiv", "q_mod",
    "q_prd", "q_var",
    "q_sums", "q_prds", "q_maxs", "q_mins", "q_avgs",
    "q_deltas", "q_ratios", "q_differ",
    "q_msum", "q_mavg", "q_mmin", "q_mmax", "q_mdev", "q_ema",
    "q_xbar", "q_bin", "q_wavg", "q_wsum",
    # keyword aliases
    "type", "null", "meta",
    "key", "value", "show", "parse",
    "flip", "asc", "xasc", "lj",
    "neg", "abs", "signum", "ceiling", "floor",
    "sqrt", "exp", "log", "reciprocal", "xexp", "xlog",
    "div", "mod",
    "string", "lower", "upper", "trim", "ltrim", "rtrim",
    "count", "first", "last", "reverse", "where", "distinct", "group",
    "til", "enlist", "raze",
    "sum", "avg", "min", "max", "dev", "med", "prd", "var",
    "sums", "prds", "maxs", "mins", "avgs", "deltas", "ratios", "differ",
    "msum", "mavg", "mmin", "mmax", "mdev", "ema",
    "xbar", "bin", "wavg", "wsum",
    # find / membership
    "q_find", "q_in", "q_within",
    # slicing
    "q_take", "q_drop", "q_rotate", "q_sublist",
    # I/O
    "q_zero_colon", "q_one_colon", "q_read0", "q_read1", "q_get", "q_set",
    "csv", "read0", "read1", "get", "set",
]
