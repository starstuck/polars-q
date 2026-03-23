"""
Maps q verb/function token strings to the corresponding polarq.verbs symbols.

Used by the transpiler to emit calls to the runtime rather than raw Python ops.
"""

# q token → (module, name) reference in the polarq runtime
VERB_MAP: dict[str, tuple[str, str]] = {
    "+":        ("polarq.verbs", "q_add"),
    "-":        ("polarq.verbs", "q_sub"),
    "*":        ("polarq.verbs", "q_mul"),
    "%":        ("polarq.verbs", "q_div"),
    "<":        ("polarq.verbs", "q_lt"),
    ">":        ("polarq.verbs", "q_gt"),
    "=":        ("polarq.verbs", "q_eq"),
    "~":        ("polarq.verbs", "q_not"),
    "sum":      ("polarq.verbs", "q_sum"),
    "min":      ("polarq.verbs", "q_min"),
    "max":      ("polarq.verbs", "q_max"),
    "avg":      ("polarq.verbs", "q_avg"),
    "dev":      ("polarq.verbs", "q_dev"),
    "med":      ("polarq.verbs", "q_med"),
    "count":    ("polarq.verbs", "q_count"),
    "first":    ("polarq.verbs", "q_first"),
    "last":     ("polarq.verbs", "q_last"),
    "reverse":  ("polarq.verbs", "q_reverse"),
    "where":    ("polarq.verbs", "q_where"),
    "distinct": ("polarq.verbs", "q_distinct"),
    "group":    ("polarq.verbs", "q_group"),
    # dyadic named keywords — used via BinOp after parser fix
    "div":        ("polarq.verbs", "q_idiv"),
    "mod":        ("polarq.verbs", "q_mod"),
    "xexp":       ("polarq.verbs", "q_xexp"),
    "xlog":       ("polarq.verbs", "q_xlog"),
}

ADVERB_MAP: dict[str, tuple[str, str]] = {
    "/":   ("polarq.adverbs", "over"),
    "\\":  ("polarq.adverbs", "scan"),
    "'":   ("polarq.adverbs", "each"),
    "/:":  ("polarq.adverbs", "each_right"),
    "\\:": ("polarq.adverbs", "each_left"),
}
