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
    "<=":       ("polarq.verbs", "q_le"),
    ">=":       ("polarq.verbs", "q_ge"),
    "not":      ("polarq.verbs", "q_not"),
    "&":        ("polarq.verbs", "q_and"),
    "|":        ("polarq.verbs", "q_or"),
    "all":      ("polarq.verbs", "q_all"),
    "any":      ("polarq.verbs", "q_any"),
    "string":   ("polarq.verbs", "q_string"),
    "lower":    ("polarq.verbs", "q_lower"),
    "upper":    ("polarq.verbs", "q_upper"),
    "trim":     ("polarq.verbs", "q_trim"),
    "ltrim":    ("polarq.verbs", "q_ltrim"),
    "rtrim":    ("polarq.verbs", "q_rtrim"),
    "like":     ("polarq.verbs", "q_like"),
    "ss":       ("polarq.verbs", "q_ss"),
    "sv":       ("polarq.verbs", "q_sv"),
    "vs":       ("polarq.verbs", "q_vs"),
    "til":      ("polarq.verbs", "q_til"),
    "enlist":   ("polarq.verbs", "q_enlist"),
    "raze":     ("polarq.verbs", "q_raze"),
    ",":        ("polarq.verbs", "q_join"),
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
    # aggregation keywords
    "prd":        ("polarq.verbs", "q_prd"),
    "var":        ("polarq.verbs", "q_var"),
    # running aggs
    "sums":       ("polarq.verbs", "q_sums"),
    "prds":       ("polarq.verbs", "q_prds"),
    "maxs":       ("polarq.verbs", "q_maxs"),
    "mins":       ("polarq.verbs", "q_mins"),
    "avgs":       ("polarq.verbs", "q_avgs"),
    "deltas":     ("polarq.verbs", "q_deltas"),
    "ratios":     ("polarq.verbs", "q_ratios"),
    "differ":     ("polarq.verbs", "q_differ"),
    # moving-window aggs
    "msum":       ("polarq.verbs", "q_msum"),
    "mavg":       ("polarq.verbs", "q_mavg"),
    "mmin":       ("polarq.verbs", "q_mmin"),
    "mmax":       ("polarq.verbs", "q_mmax"),
    "mdev":       ("polarq.verbs", "q_mdev"),
    "ema":        ("polarq.verbs", "q_ema"),
    # bucketing
    "xbar":       ("polarq.verbs", "q_xbar"),
    "bin":        ("polarq.verbs", "q_bin"),
    "wavg":       ("polarq.verbs", "q_wavg"),
    "wsum":       ("polarq.verbs", "q_wsum"),
}

ADVERB_MAP: dict[str, tuple[str, str]] = {
    "/":   ("polarq.adverbs", "over"),
    "\\":  ("polarq.adverbs", "scan"),
    "'":   ("polarq.adverbs", "each"),
    "/:":  ("polarq.adverbs", "each_right"),
    "\\:": ("polarq.adverbs", "each_left"),
}
