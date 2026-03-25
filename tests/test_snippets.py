"""
tests/test_snippets.py — Living Documentation
══════════════════════════════════════════════
Organised to mirror the q language reference card:
  https://code.kx.com/q/ref/

Each pytest.param is simultaneously a runnable test and a documented q console
snippet showing expected REPL output.  pytest -v produces a feature inventory:

    PASSED  tests/test_snippets.py::TestAtoms::test_snippet[int-42]
    XFAIL   tests/test_snippets.py::TestArithmetic::test_snippet[add]

──────────────────────────────────────────────────────────────────────────────
Adding a new snippet
──────────────────────────────────────────────────────────────────────────────
  1. Find the relevant *_SNIPPETS list (or add a new section/class).
  2. Append  pytest.param("q source", "expected output", id="label").
  3. If not yet working: place it in the xfail class for that section.
  4. When a feature is implemented: move it to the passing class; if the
     whole section flips, remove pytestmark=xfail from the class.

Expected-output conventions
──────────────────────────────────────────────────────────────────────────────
  str          single output line must equal this string exactly
  list[str]    multi-line session; one element per output line
  callable     predicate f(lines: list[str]) → bool for non-deterministic output
"""

import pytest
from conftest import output_lines


# ── Core test helper ──────────────────────────────────────────────────────────

def check_snippet(
    q_code: "str | list[str]",
    expected: "str | list[str] | callable",
) -> None:
    lines = [q_code] if isinstance(q_code, str) else q_code
    result = output_lines(lines)
    if callable(expected):
        assert expected(result), f"q: {q_code!r}\npredicate failed, got: {result!r}"
    else:
        expected_list = [expected] if isinstance(expected, str) else expected
        assert result == expected_list, (
            f"q: {q_code!r}\n"
            f"expected: {expected_list!r}\n"
            f"got:      {result!r}"
        )


# ── xfail reason constants ────────────────────────────────────────────────────

# Transpiler emits q_add/q_mul/… calls but the runtime functions don't exist yet.
# Fix: implement q_add, q_sub, q_mul, q_div … in polarq/verbs.py and re-export
# them through polarq/__init__.py; also wrap IntLit/FloatLit as QAtom in the
# transpiler (polarq/transpiler/transpiler.py QToPythonTranspiler._expr).
_ARITH_XFAIL = (
    "Transpiler emits bare Python literals (not QAtom) and q_add/q_mul/… are "
    "not yet defined in the runtime. Fix: (1) wrap IntLit/FloatLit as "
    "QAtom(v,kind) in QToPythonTranspiler._expr(); (2) expose q_add, q_sub, "
    "q_mul, q_div, q_eq, q_lt, q_gt, q_le, q_ge, q_not, q_match in "
    "polarq/transpiler/builtins.py and polarq/__init__.py."
)

# Keywords parsed correctly but not exposed in the execution namespace.
# Fix: implement the function in polarq/verbs.py (or polarq/lists.py etc.)
# and add it to polarq/__init__.py __all__ + the REPL env seed.
_NYI_KEYWORDS = (
    "Function is recognised by the parser but not yet implemented in the "
    "polarq runtime. Fix: implement in the appropriate polarq module and "
    "export via polarq/__init__.py."
)

# Feature requires transpiler support that has not been added yet.
# Fix: add a handler in QToPythonTranspiler (polarq/transpiler/transpiler.py).
_NYI_TRANSPILER = (
    "Parser may handle this construct but the transpiler does not yet emit "
    "code for it. Fix: add a handler in QToPythonTranspiler "
    "(polarq/transpiler/transpiler.py)."
)

# Feature needs both a new parser rule and a transpiler/runtime implementation.
_NYI_FULL = (
    "Feature not yet implemented end-to-end (parser + transpiler + runtime). "
    "Check polarq/parser/pratt.py, polarq/transpiler/transpiler.py, and the "
    "relevant polarq runtime module."
)


# ╔══════════════════════════════════════════════════════════════════════════════
# § 1  ATOMS   ref: https://code.kx.com/q/ref/datatypes/
# ══════════════════════════════════════════════════════════════════════════════╗
# Transpiler wraps literals in QAtom, so output matches q console format
# (e.g. 1b → "1b", `AAPL → "`AAPL", 3.14f → "3.14f").

ATOM_SNIPPETS = [
    pytest.param("42",       "42",     id="int-42"),
    pytest.param("0",        "0",      id="int-zero"),
    pytest.param("-1",       "-1",     id="int-neg"),
    pytest.param("42j",      "42",     id="int-j-suffix"),
    pytest.param("42h",      "42",     id="int-h-suffix"),
    pytest.param("3.14",     "3.14f",  id="float"),
    pytest.param("1b",       "1b",     id="bool-true"),
    pytest.param("0b",       "0b",     id="bool-false"),
    pytest.param("`AAPL",    "`AAPL",  id="symbol"),
    pytest.param('"hello"',  "hello",  id="string"),
]


class TestAtoms:
    """Atom literals — bare values typed at the prompt."""

    @pytest.mark.parametrize("q_code,expected", ATOM_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# § 2  VECTORS   ref: https://code.kx.com/q/ref/datatypes/
# ══════════════════════════════════════════════════════════════════════════════╗

VECTOR_SNIPPETS = [
    pytest.param("1 2 3",         "1 2 3",       id="int-vector"),
    pytest.param("1.5 2.5 3.5",   "1.5 2.5 3.5", id="float-vector"),
    pytest.param("1b 0b 1b",      "101b",        id="bool-vector"),
    pytest.param("`a`b`c",        "`a`b`c",      id="sym-vector"),
]


class TestVectors:
    """Homogeneous typed vectors formed by juxtaposing atoms."""

    @pytest.mark.parametrize("q_code,expected", VECTOR_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# § 3  ASSIGNMENT   ref: https://code.kx.com/q/ref/assign/
# ══════════════════════════════════════════════════════════════════════════════╗

ASSIGNMENT_SNIPPETS = [
    pytest.param("x:42",        "42",               id="local-int"),
    pytest.param("x:3.14",      "3.14f",            id="local-float"),
    pytest.param("x:1b",        "1b",               id="local-bool"),
    pytest.param("x:`AAPL",     "`AAPL",            id="local-symbol"),
    pytest.param('x:"hello"',   "hello",            id="local-string"),
    pytest.param("v:1 2 3",     "1 2 3",    id="local-vector"),
    pytest.param("v:`a`b`c",    "`a`b`c",   id="local-sym-vec"),
]

NAME_LOOKUP_SNIPPETS = [
    pytest.param(["x:42",       "x"],           ["42",  "42"],              id="recall-int"),
    pytest.param(["v:1 2 3",    "v"],           ["1 2 3", "1 2 3"],         id="recall-vector"),
    pytest.param(["x:42","y:99","x"],           ["42","99","42"],            id="multi-binding"),
]


class TestAssignment:
    """Variable assignment displays the assigned value (q console behaviour)."""

    @pytest.mark.parametrize("q_code,expected", ASSIGNMENT_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestNameLookup:
    """Binding a name then recalling it across REPL lines."""

    @pytest.mark.parametrize("q_code,expected", NAME_LOOKUP_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# § 4  ARITHMETIC   ref: https://code.kx.com/q/ref/add/  … /divide/
# ══════════════════════════════════════════════════════════════════════════════╗

ARITHMETIC_SNIPPETS = [
    pytest.param("1+2",         "3",            id="add"),
    pytest.param("10-3",        "7",            id="sub"),
    pytest.param("3*4",         "12",           id="mul"),
    pytest.param("10%2",        "5f",           id="div-float"),        # % always returns float
    pytest.param("2+3*4",       "14",           id="rtl-no-prec"),      # right-to-left: 2+(3*4)
    pytest.param("7 div 2",     "3",            id="integer-div"),
    pytest.param("7 mod 3",     "1",            id="modulo"),
    pytest.param("1 2 3 + 10",  "11 12 13",     id="vector-add"),
    pytest.param("2 * 1 2 3",   "2 4 6",        id="vector-mul"),
]

MATH_KEYWORD_SNIPPETS = [
    pytest.param("neg 3",         "-3",    id="neg"),
    pytest.param("neg[-3]",       "3",     id="neg-neg"),
    pytest.param("abs[-5]",       "5",     id="abs"),
    pytest.param("signum[-3]",    "-1",    id="signum-neg"),
    pytest.param("signum 0",      "0",     id="signum-zero"),
    pytest.param("signum 3",      "1",     id="signum-pos"),
    pytest.param("ceiling 2.3",   "3",     id="ceiling"),
    pytest.param("floor 2.7",     "2",     id="floor"),
    pytest.param("sqrt 9",        "3f",    id="sqrt"),
    pytest.param("exp 0",         "1f",    id="exp-zero"),
    pytest.param("log 1",         "0f",    id="log-one"),
    pytest.param("reciprocal 4",  "0.25f", id="reciprocal"),
    pytest.param("2 xexp 10",     "1024f", id="xexp"),
    pytest.param("2 xlog 8",      "3f",    id="xlog"),
]


class TestArithmetic:
    """Arithmetic operators (+  -  *  %  div  mod) — vector and scalar."""

    @pytest.mark.parametrize("q_code,expected", ARITHMETIC_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestMathKeywords:
    """Named math functions (neg, abs, signum, ceiling, floor, sqrt, exp, log …)."""

    @pytest.mark.parametrize("q_code,expected", MATH_KEYWORD_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# § 5  COMPARISON   ref: https://code.kx.com/q/ref/equal/  … /match/
# ══════════════════════════════════════════════════════════════════════════════╗

COMPARISON_SNIPPETS = [
    pytest.param("1=1",             "1b",   id="eq-true"),
    pytest.param("1=2",             "0b",   id="eq-false"),
    pytest.param("1<>2",            "1b",   id="ne-true"),
    pytest.param("1<2",             "1b",   id="lt-true"),
    pytest.param("2>1",             "1b",   id="gt-true"),
    pytest.param("1<=1",            "1b",   id="le-eq"),
    pytest.param("1<=2",            "1b",   id="le-lt"),
    pytest.param("2<=1",            "0b",   id="le-gt"),
    pytest.param("2>=2",            "1b",   id="ge-eq"),
    pytest.param("2>=1",            "1b",   id="ge-gt"),
    pytest.param("1>=2",            "0b",   id="ge-lt"),
    pytest.param("(1 2 3)~(1 2 3)", "1b",   id="match-vec-eq"),
    pytest.param("(1 2 3)~(1 2 4)", "0b",   id="match-vec-ne"),
    pytest.param("not 1b",          "0b",   id="not-true"),
    pytest.param("not 0b",          "1b",   id="not-false"),
]

COMPARISON_NYI_SNIPPETS: list = []


class TestComparison:
    """Equality and ordering operators (=  <>  <  >  <=  >=  ~  not) — working."""

    @pytest.mark.parametrize("q_code,expected", COMPARISON_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# § 6  LOGIC / BITWISE   ref: https://code.kx.com/q/ref/lesser/ etc.
# ══════════════════════════════════════════════════════════════════════════════╗

LOGIC_SNIPPETS = [
    pytest.param("1b & 0b",    "0b",  id="and-false"),
    pytest.param("1b & 1b",    "1b",  id="and-true"),
    pytest.param("0b | 1b",    "1b",  id="or-true"),
    pytest.param("0b | 0b",    "0b",  id="or-false"),
    pytest.param("all 1 1 1b", "1b",  id="all-true"),
    pytest.param("all 1 0 1b", "0b",  id="all-false"),
    pytest.param("any 0 0 1b", "1b",  id="any-true"),
    pytest.param("any 0 0 0b", "0b",  id="any-false"),
    pytest.param("3 & 5",      "3",   id="min-int"),
    pytest.param("3 | 5",      "5",   id="max-int"),
]


class TestLogic:
    """Logical and bitwise operators (& | all any) — working."""

    @pytest.mark.parametrize("q_code,expected", LOGIC_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# § 7  STRING FUNCTIONS   ref: https://code.kx.com/q/ref/lower/ etc.
# ══════════════════════════════════════════════════════════════════════════════╗

STRING_SNIPPETS = [
    pytest.param('string 42',          '"42"',         id="string-int"),
    pytest.param('string `AAPL',       '"AAPL"',       id="string-symbol"),
    pytest.param('lower "Hello"',      '"hello"',      id="lower"),
    pytest.param('upper "hello"',      '"HELLO"',      id="upper"),
    pytest.param('trim "  hi  "',      '"hi"',         id="trim"),
    pytest.param('ltrim "  hi"',       '"hi"',         id="ltrim"),
    pytest.param('rtrim "hi  "',       '"hi"',         id="rtrim"),
    pytest.param('"hello" like "h*"',  "1b",           id="like-glob"),
    pytest.param('"hello" ss "l"',     "2 3",          id="ss-positions"),
    pytest.param('"," sv "a","b","c"', '"a,b,c"',      id="sv-join"),
    pytest.param('"," vs "a,b,c"',     '"a" "b" "c"',  id="vs-split"),
]


class TestStrings:
    """String functions — working."""

    @pytest.mark.parametrize("q_code,expected", STRING_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# § 8  LIST FUNCTIONS — CORE   ref: https://code.kx.com/q/ref/til/ etc.
# ══════════════════════════════════════════════════════════════════════════════╗

LIST_CORE_SNIPPETS = [
    pytest.param("til 5",             "0 1 2 3 4",       id="til"),
    pytest.param("count 1 2 3",       "3",               id="count-vector"),
    pytest.param("count `a`b`c",      "3",               id="count-syms"),
    pytest.param("v:1 2 3\ncount v",  "3",               id="count-named"),
    pytest.param("enlist 42",         ",42",             id="enlist-atom"),
    pytest.param("enlist 1 2 3",      ",1 2 3",          id="enlist-vector"),
    pytest.param("first 1 2 3",       "1",               id="first"),
    pytest.param("last 1 2 3",        "3",               id="last"),
    pytest.param("reverse 1 2 3",     "3 2 1",           id="reverse"),
    pytest.param("distinct 1 2 1 3",  "1 2 3",           id="distinct"),
    pytest.param("raze (1 2;3 4)",    "1 2 3 4",         id="raze"),
]

LIST_SLICE_SNIPPETS = [
    pytest.param("3 # 1 2 3 4 5",    "1 2 3",        id="take"),
    pytest.param("-2 # 1 2 3 4 5",   "4 5",          id="take-last"),
    pytest.param("2 _ 1 2 3 4 5",    "3 4 5",        id="drop"),
    pytest.param("-2 _ 1 2 3 4 5",   "1 2 3",        id="drop-last"),
    pytest.param("2 rotate 1 2 3 4", "3 4 1 2",      id="rotate"),
    pytest.param("2 sublist 1 2 3 4","1 2",           id="sublist"),
]

LIST_SEARCH_SNIPPETS = [
    pytest.param("(1 2 3)?2",        "1",            id="find"),
    pytest.param("(1 2 3)?9",        "3",            id="find-miss"),    # count x if not found
    pytest.param("2 in 1 2 3",       "1b",           id="in-true"),
    pytest.param("9 in 1 2 3",       "0b",           id="in-false"),
    pytest.param("2 within 1 3",     "1b",           id="within-true"),
    pytest.param("where 1b 0b 1b",   "0 2",          id="where"),
    pytest.param("group 1 2 1 3 2",  "1|0 2\n2|1 4\n3|,3",  id="group"),
]

LIST_SET_SNIPPETS = [
    pytest.param("(1 2 3) union (2 3 4)", "1 2 3 4",  id="union"),
    pytest.param("(1 2 3) inter (2 3 4)", "2 3",       id="inter"),
    pytest.param("(1 2 3) except (2 3)",  "1",         id="except"),
]

LIST_NAV_SNIPPETS = [
    pytest.param("next 1 2 3",        "2 3 0N",         id="next"),
    pytest.param("prev 1 2 3",        "0N 1 2",         id="prev"),
    pytest.param("2 xprev 1 2 3 4 5", "0N 0N 1 2 3",   id="xprev"),
]

LIST_NULL_SNIPPETS = [
    pytest.param("null 0N",            "1b",              id="null-int"),
    pytest.param("null 42",            "0b",              id="null-false"),
    pytest.param("0N ^ 1 0N 3",        "1 1 3",           id="fill"),
    pytest.param("fills 1 0N 0N 4",    "1 1 1 4",         id="fills"),
]


class TestListCore:
    """Core list functions (til, count, enlist, first, last, reverse …) — working."""

    @pytest.mark.parametrize("q_code,expected", LIST_CORE_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestListSlicing:
    """List slicing: take (#), drop (_), rotate, sublist — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_TRANSPILER, strict=True)

    @pytest.mark.parametrize("q_code,expected", LIST_SLICE_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestListSearch:
    """List search: find (?), in, within, where, group — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_KEYWORDS, strict=True)

    @pytest.mark.parametrize("q_code,expected", LIST_SEARCH_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestListSets:
    """Set operations: union, inter, except — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_KEYWORDS, strict=True)

    @pytest.mark.parametrize("q_code,expected", LIST_SET_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestListNav:
    """Navigation: next, prev, xprev — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_KEYWORDS, strict=True)

    @pytest.mark.parametrize("q_code,expected", LIST_NAV_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestListNull:
    """Null handling: null, fill (^), fills — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_TRANSPILER, strict=True)

    @pytest.mark.parametrize("q_code,expected", LIST_NULL_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# § 9  AGGREGATIONS   ref: https://code.kx.com/q/ref/sum/ etc.
# ══════════════════════════════════════════════════════════════════════════════╗

AGG_SNIPPETS = [
    pytest.param("sum 1 2 3 4",       "10",    id="sum"),
    pytest.param("prd 1 2 3 4",       "24",    id="prd"),
    pytest.param("avg 1 2 3",         "2f",    id="avg"),
    pytest.param("min 3 1 4 1 5",     "1",     id="min"),
    pytest.param("max 3 1 4 1 5",     "5",     id="max"),
    pytest.param("med 1 2 3 4 5",     "3f",    id="med"),
    pytest.param("dev 1 2 3",         "0.8165f", id="dev"),
    pytest.param("var 1 2 3",         "0.6667f", id="var"),
]

RUNNING_SNIPPETS = [
    pytest.param("sums 1 2 3 4",      "1 3 6 10",       id="sums"),
    pytest.param("prds 1 2 3 4",      "1 2 6 24",       id="prds"),
    pytest.param("maxs 3 1 4 1 5",    "3 3 4 4 5",      id="maxs"),
    pytest.param("mins 3 1 4 1 5",    "3 1 1 1 1",      id="mins"),
    pytest.param("avgs 2 4 6",        "2 3 4",           id="avgs"),
    pytest.param("deltas 1 3 6 10",   "1 2 3 4",         id="deltas"),
    pytest.param("ratios 1 2 4 8",    "1 2 2 2",         id="ratios"),
    pytest.param("differ 1 1 2 2 3",  "10101b",          id="differ"),
]

MOVING_SNIPPETS = [
    pytest.param("3 msum 1 2 3 4 5",  "1 3 6 9 12",              id="msum"),
    pytest.param("3 mavg 1 2 3 4 5",  "1 1.5 2 3 4",             id="mavg"),
    pytest.param("3 mmin 5 3 4 1 2",  "5 3 3 1 1",               id="mmin"),
    pytest.param("3 mmax 1 5 2 4 3",  "1 5 5 5 4",               id="mmax"),
    pytest.param("3 mdev 1 2 3 4 5",  "0 0.5 0.8165 0.8165 0.8165", id="mdev"),
    pytest.param("0.1 ema 1 2 3",     "1 1.1 1.29",              id="ema"),
]

BUCKET_SNIPPETS = [
    pytest.param("10 xbar 15",          "10",   id="xbar"),
    pytest.param("10 xbar 25",          "20",   id="xbar-25"),
    pytest.param("(0 1 2 3) bin 1.5",   "1",    id="bin"),
    pytest.param("2.1 wavg 1 2 3",      "6f",   id="wavg"),
    pytest.param("1 2 3 wsum 4 5 6",    "32",   id="wsum"),
]


class TestAggregations:
    """Aggregate functions (sum, avg, min, max …)."""

    @pytest.mark.parametrize("q_code,expected", AGG_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestRunningAgg:
    """Running/cumulative aggregations (sums, maxs, deltas …)."""

    @pytest.mark.parametrize("q_code,expected", RUNNING_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestMovingAgg:
    """Moving-window aggregations (msum, mavg, ema …)."""

    @pytest.mark.parametrize("q_code,expected", MOVING_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestBucketing:
    """Bucketing: xbar, bin, wavg, wsum."""

    @pytest.mark.parametrize("q_code,expected", BUCKET_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# §10  ITERATORS (ADVERBS)   ref: https://code.kx.com/q/ref/iterators/
# ══════════════════════════════════════════════════════════════════════════════╗

ITER_OVER_SNIPPETS = [
    pytest.param("{[a;b] a+b}/[1 2 3 4]",  "10",        id="over-add-sum"),
    pytest.param("{[a;b] a*b}/[1 2 3 4]",  "24",        id="over-mul-prd"),
    pytest.param("{[a;b] a+b}\\ [1 2 3 4]", "1 3 6 10", id="scan-add"),
]

ITER_EACH_SNIPPETS = [
    pytest.param("{[x] x*x} each 1 2 3",       "1 4 9",        id="each-square"),
    pytest.param("{[x;y] x+y}[1 2 3]/: 10",    "11 12 13",     id="each-right"),
    pytest.param("{[x;y] x+y}[10]\\: 1 2 3",   "11 12 13",     id="each-left"),
    pytest.param("{[a;b] a,b}': 1 2 3",         "(1;1 2;2 3)",  id="each-prior"),
]


class TestIterators:
    """Iterators: over (/), scan (\\), each ('), each-right (/:), each-left (\\:)."""

    @pytest.mark.parametrize("q_code,expected", ITER_OVER_SNIPPETS + ITER_EACH_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# §11  LAMBDAS   ref: https://code.kx.com/q/ref/apply/
# ══════════════════════════════════════════════════════════════════════════════╗

_is_lambda = lambda lines: len(lines) == 1 and "<function" in lines[0]  # noqa: E731

LAMBDA_SNIPPETS = [
    # Definition
    pytest.param("{x*x}",       _is_lambda,  id="implicit-def"),
    pytest.param("{[a;b] a}",   _is_lambda,  id="explicit-def"),
    # Application — identity lambdas avoid the arithmetic gap
    pytest.param(["{[x;y] x}[3;4]"],        ["3"],  id="identity-first"),
    pytest.param(["{[x;y] y}[3;4]"],        ["4"],  id="identity-second"),
    pytest.param(["f:{[x;y] x}", "f[10;20]"],
                 [lambda l: "<function" in l[0], "10"],
                 id="named-apply"),
]


class TestLambdas:
    """Lambda definition and application (non-arithmetic examples)."""

    @pytest.mark.parametrize("q_code,expected", LAMBDA_SNIPPETS)
    def test_snippet(self, q_code, expected):
        # Support mixed list of strings and predicates in expected
        lines = [q_code] if isinstance(q_code, str) else q_code
        result = output_lines(lines)
        if callable(expected):
            assert expected(result), f"predicate failed, got: {result!r}"
        elif isinstance(expected, list) and any(callable(e) for e in expected):
            assert len(result) == len(expected), (
                f"expected {len(expected)} lines, got {len(result)}: {result!r}"
            )
            for r, e in zip(result, expected):
                if callable(e):
                    assert e([r]), f"predicate failed for {r!r}"
                else:
                    assert r == e, f"expected {e!r}, got {r!r}"
        else:
            expected_list = [expected] if isinstance(expected, str) else expected
            assert result == expected_list, (
                f"expected: {expected_list!r}\ngot: {result!r}"
            )


# ╔══════════════════════════════════════════════════════════════════════════════
# §12  CONTROL FLOW   ref: https://code.kx.com/q/ref/if/ etc.
# ══════════════════════════════════════════════════════════════════════════════╗

CONTROL_SNIPPETS = [
    pytest.param("$[1b;42;99]",                      "42",   id="cond-true"),
    pytest.param("$[0b;42;99]",                      "99",   id="cond-false"),
    pytest.param("$[1b;`yes;0b;`maybe;`no]",         "yes",  id="cond-chain"),
    pytest.param("x:0\nif[1b;x:42]\nx",              "42",   id="if-true"),
    pytest.param("x:0\nif[0b;x:99]\nx",              "0",    id="if-false"),
    pytest.param("x:0\ndo[3;x:x+1]\nx",              "3",    id="do-loop"),
    pytest.param("x:0\nwhile[x<5;x:x+1]\nx",         "5",    id="while-loop"),
]


class TestControlFlow:
    """Control flow: $[cond;t;f], if, do, while — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_TRANSPILER, strict=True)

    @pytest.mark.parametrize("q_code,expected", CONTROL_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# §13  TYPES AND CASTING   ref: https://code.kx.com/q/ref/cast/
# ══════════════════════════════════════════════════════════════════════════════╗

TYPE_SNIPPETS = [
    pytest.param("type 42",          "-7h",  id="type-long"),
    pytest.param("type 42h",         "-5h",  id="type-short"),
    pytest.param("type 1b",          "-1h",  id="type-bool"),
    pytest.param("type 3.14",        "-9h",  id="type-float"),
    pytest.param("type `sym",        "-11h", id="type-symbol"),
    pytest.param('type "hi"',        "10h",  id="type-string"),
    pytest.param("type 1 2 3",       "7h",   id="type-long-vec"),
    pytest.param("type (1;`a;2)",    "0h",   id="type-general-list"),
]

CAST_SNIPPETS = [
    pytest.param('"i"$42',           "42i",   id="cast-to-int"),
    pytest.param('"f"$42',           "42f",   id="cast-to-float"),
    pytest.param('"b"$1',            "1b",    id="cast-to-bool"),
    pytest.param('"j"$3.7',          "3",     id="cast-float-to-long"),
    pytest.param('"s"$"AAPL"',       "`AAPL", id="cast-to-sym"),
    pytest.param('"I"$"42"',         "42i",   id="tok-string-to-int"),
    pytest.param('"F"$"3.14"',       "3.14f", id="tok-string-to-float"),
    pytest.param("null 0N",          "1b",    id="null-int"),
    pytest.param("null 42",          "0b",    id="null-not"),
]


class TestTypes:
    """Type introspection (type) — xfail until q type system exposed."""

    pytestmark = pytest.mark.xfail(reason=_NYI_KEYWORDS, strict=True)

    @pytest.mark.parametrize("q_code,expected", TYPE_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestCasting:
    """Type casting ($) and null checking — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_TRANSPILER, strict=True)

    @pytest.mark.parametrize("q_code,expected", CAST_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# §14  DICTIONARIES   ref: https://code.kx.com/q/ref/dict/
# ══════════════════════════════════════════════════════════════════════════════╗

DICT_SNIPPETS = [
    pytest.param("`a`b`c!1 2 3",              "a|1\nb|2\nc|3",  id="create"),
    pytest.param("d:`a`b!1 2\nd`a",           "1",              id="lookup"),
    pytest.param("key `a`b!1 2",              "`a`b",           id="keys"),
    pytest.param("value `a`b!1 2",            "1 2",            id="values"),
    pytest.param("`a`b!1 2 ~ `a`b!1 2",       "1b",             id="match-dicts"),
    pytest.param("count `a`b`c!1 2 3",        "3",              id="count-dict"),
]

ENUM_SNIPPETS = [
    pytest.param("sym:`AAPL`GOOG\nx:`sym$`AAPL\nx",  "`AAPL", id="enumerate"),
    pytest.param("sym:`AAPL`GOOG\nsym?`MSFT",
                 "`sym$`AAPL`GOOG`MSFT",              id="enum-extend"),
]


class TestDictionaries:
    """Dictionary construction and access — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_TRANSPILER, strict=True)

    @pytest.mark.parametrize("q_code,expected", DICT_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestEnumerations:
    """Enumerations ($, ?) — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_TRANSPILER, strict=True)

    @pytest.mark.parametrize("q_code,expected", ENUM_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# §15  TABLES AND qSQL   ref: https://code.kx.com/q/ref/select/
# ══════════════════════════════════════════════════════════════════════════════╗

TABLE_SETUP = "trade:([] sym:`AAPL`GOOG`AAPL; price:100 200 110; vol:1000 500 800)"

TABLE_SNIPPETS = [
    pytest.param(
        [TABLE_SETUP, "count trade"],
        ["...", "3"],
        id="table-count",
    ),
    pytest.param(
        [TABLE_SETUP, "meta trade"],
        ["...", "c  | t f a", "-----...", "sym  | s  ", "price| j  ", "vol  | j  "],
        id="table-meta",
    ),
    pytest.param(
        [TABLE_SETUP, "select from trade where sym=`AAPL"],
        ["...", "sym  price vol", "------...", "AAPL 100   1000", "AAPL 110   800"],
        id="select-where"),
    pytest.param(
        [TABLE_SETUP, "select avg price by sym from trade"],
        ["...", "sym | price", "----...", "AAPL| 105f", "GOOG| 200f"],
        id="select-by-avg",
    ),
    pytest.param(
        [TABLE_SETUP, "exec price from trade where sym=`AAPL"],
        ["...", "100 110"],
        id="exec-col",
    ),
    pytest.param(
        [TABLE_SETUP, "update price:price*1.1 from trade where sym=`AAPL"],
        ["...", "sym  price vol", "------...", "AAPL 110   1000", "GOOG 200   500", "AAPL 121   800"],
        id="update-where",
    ),
    pytest.param(
        [TABLE_SETUP, "delete from trade where sym=`GOOG"],
        ["...", "sym  price vol", "------...", "AAPL 100   1000", "AAPL 110   800"],
        id="delete-where",
    ),
]

TABLE_OP_SNIPPETS = [
    pytest.param(
        [TABLE_SETUP, "asc trade"],
        ["...", "..."],
        id="asc-table",
    ),
    pytest.param(
        [TABLE_SETUP, "`price xasc trade"],
        ["...", "..."],
        id="xasc",
    ),
    pytest.param(
        [TABLE_SETUP, "flip `sym`price`vol!((`AAPL;100;1000))"],
        ["..."],
        id="flip-dict-to-table",
    ),
    pytest.param(
        [TABLE_SETUP, "10 xbar 15"],
        ["...", "10"],
        id="xbar-bucket",
    ),
    pytest.param(
        ["t1:([] k:`a`b; v:1 2)",
         "t2:([k:`a`b] x:10 20)",
         "t1 lj t2"],
        ["...", "...", "k  v  x", "-------", "a  1  10", "b  2  20"],
        id="lj",
    ),
]


class TestQSQL:
    """qSQL: select, exec, update, delete with where/by — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_FULL, strict=True)

    @pytest.mark.parametrize("q_code,expected", TABLE_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestTableOps:
    """Table operations: sort, flip, xbar, joins — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_FULL, strict=True)

    @pytest.mark.parametrize("q_code,expected", TABLE_OP_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# §16  TEMPORAL TYPES   ref: https://code.kx.com/q/ref/datatypes/#temporal
# ══════════════════════════════════════════════════════════════════════════════╗

# .z namespace variables are already wired into the runtime.
TEMPORAL_ENV_SNIPPETS = [
    pytest.param(".z.d", lambda l: len(l) == 1, id="z-d-today"),
    pytest.param(".z.t", lambda l: len(l) == 1, id="z-t-now"),
]

TEMPORAL_SNIPPETS = [
    pytest.param("2024.01.15",                    "2024.01.15",             id="date-literal"),
    pytest.param("12:30:00.000",                  "12:30:00.000",           id="time-literal"),
    pytest.param("2024.01.15D12:30:00.000000000", "2024.01.15D12:30:00.000000000", id="timestamp"),
    pytest.param("2024.01m",                      "2024.01m",               id="month-literal"),
    pytest.param("type 2024.01.15",               "-14h",                   id="type-date"),
    pytest.param("type 12:30:00.000",             "-19h",                   id="type-time"),
    pytest.param('"d"$2024.01.15',                "2024.01.15",             id="cast-date"),
    pytest.param("2024.01.15 + 7",                "2024.01.22",             id="date-add"),
    pytest.param("2024.01.20 - 2024.01.15",       "5",                      id="date-diff"),
]


class TestTemporalEnv:
    """.z namespace: current date/time already wired into the runtime."""

    @pytest.mark.parametrize("q_code,expected", TEMPORAL_ENV_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


class TestTemporal:
    """Temporal literals, type codes, arithmetic — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_FULL, strict=True)

    @pytest.mark.parametrize("q_code,expected", TEMPORAL_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# §17  META / INTROSPECTION   ref: https://code.kx.com/q/ref/meta/
# ══════════════════════════════════════════════════════════════════════════════╗

META_SNIPPETS = [
    pytest.param("x:42\ntype x",            "-7h",             id="type-long"),
    pytest.param("v:1 2 3\nkey v",          "7h",              id="key-vector-type"),
    pytest.param("x:42\nshow x",            "42",              id="show"),
    pytest.param("value {x+y}",             "({x+y};+;x;y)",   id="value-lambda"),
    pytest.param("parse \"1+2\"",           "(+;1;2)",         id="parse"),
]


class TestMeta:
    """Meta functions: type, count, key, show, value, parse — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_KEYWORDS, strict=True)

    @pytest.mark.parametrize("q_code,expected", META_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# §18  APPLY / INDEX / AMEND   ref: https://code.kx.com/q/ref/apply/
# ══════════════════════════════════════════════════════════════════════════════╗

APPLY_SNIPPETS = [
    pytest.param("{[x;y] x+y} . (3;4)",         "7",   id="apply-dot"),
    pytest.param("{x*x} @ 5",                   "25",  id="apply-at"),
    pytest.param("(1 2 3 4 5) @ 2",             "3",   id="index-at"),
    pytest.param("(1 2 3;4 5 6) . (1;0)",       "4",   id="index-dot-nested"),
    pytest.param("@[(1 2 3);1;{x*10}]",         "1 20 3", id="amend-at"),
    pytest.param(".[add;(3;4);{\"err:\",x}]",   "7",   id="trap-success"),
    pytest.param(".[{x%0};(1f);{\"err\"}]",     "err", id="trap-error"),
]


class TestApply:
    """Apply (. @) and index — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_TRANSPILER, strict=True)

    @pytest.mark.parametrize("q_code,expected", APPLY_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)


# ╔══════════════════════════════════════════════════════════════════════════════
# §19  I/O   ref: https://code.kx.com/q/ref/read0/ etc.
# (File I/O — integration tests; these require filesystem access)
# ══════════════════════════════════════════════════════════════════════════════╗

IO_SNIPPETS = [
    pytest.param(
        ['`:tmp_test.csv 0: csv 0: ([] x:1 2 3)',
         'read0 `:tmp_test.csv'],
        ['...', '"x"', '"1"', '"2"', '"3"'],
        id="csv-roundtrip",
    ),
    pytest.param(
        ['`:tmp_test.dat set 1 2 3',
         'get `:tmp_test.dat'],
        ['...', '1 2 3'],
        id="binary-roundtrip",
    ),
]


class TestIO:
    """File I/O: 0:, 1:, read0, read1, get, set — xfail."""

    pytestmark = pytest.mark.xfail(reason=_NYI_FULL, strict=True)

    @pytest.mark.parametrize("q_code,expected", IO_SNIPPETS)
    def test_snippet(self, q_code, expected):
        check_snippet(q_code, expected)
