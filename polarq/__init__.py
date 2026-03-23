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
                               QTable, QKeyedTable, QLambda,
                               QBuiltin, QAdverb, QNull, qnull)
from polarq.env       import QEnv
from polarq.verbs     import (
    VERB_TABLE,
    q_add, q_sub, q_mul, q_div,
    q_lt, q_gt, q_eq, q_not,
    q_sum, q_min, q_max, q_avg, q_dev, q_med,
    q_count, q_first, q_last, q_reverse, q_where, q_distinct, q_group,
    q_neg, q_abs, q_signum, q_ceiling, q_floor,
    q_sqrt, q_exp, q_log, q_reciprocal, q_xexp, q_xlog,
    q_idiv, q_mod,
)

# Named keyword aliases — transpiled q code uses bare names like `neg x`, `abs x`
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
from polarq.adverbs   import over, scan, each, each_left, each_right
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
    "QTable", "QKeyedTable", "QLambda", "QBuiltin", "QAdverb",
    "QNull", "qnull", "QEnv", "PolarQServer", "PolarQClient",
    "over", "scan", "each", "each_left", "each_right",
    "compile_select", "compile_update", "compile_aj", "compile_wj",
    "QError", "QTypeError", "QLengthError", "QRankError",
    # verb functions (for transpiled code: `from polarq import *`)
    "q_add", "q_sub", "q_mul", "q_div",
    "q_lt", "q_gt", "q_eq", "q_not",
    "q_sum", "q_min", "q_max", "q_avg", "q_dev", "q_med",
    "q_count", "q_first", "q_last", "q_reverse", "q_where", "q_distinct", "q_group",
    "q_neg", "q_abs", "q_signum", "q_ceiling", "q_floor",
    "q_sqrt", "q_exp", "q_log", "q_reciprocal", "q_xexp", "q_xlog",
    "q_idiv", "q_mod",
    # keyword aliases
    "neg", "abs", "signum", "ceiling", "floor",
    "sqrt", "exp", "log", "reciprocal", "xexp", "xlog",
    "div", "mod",
]
