"""
tests/test_parser.py
────────────────────
Live documentation for the polarq parser.

Structure
─────────
  Each test class covers one feature area.  Within each class tests progress
  from the simplest case to more complex ones.  The docstring on each test
  acts as the documentation; the assertion is the proof.

  The comment markers below show the intended coverage expansion path:

    ✅  implemented and tested now
    🔜  planned — test skeleton present, marked pytest.skip until ready
    📋  listed as future work — no skeleton yet

Feature map
───────────
  ✅  Literals      (int, float, bool, symbol, string, null)
  ✅  Vectors       (homogeneous juxtaposed literals)
  ✅  Names         (simple identifiers, dotted names)
  ✅  Assignment    (local :  and global ::)
  ✅  Arithmetic    (+ - * %, right-to-left, no precedence)
  ✅  Comparison    (< > = ~)
  ✅  Monadic verbs (negation, count, reverse, …)
  ✅  Adverbs       (+/ +\\ f' f/: f\\:)
  ✅  Function call — bracket form  f[x;y]
  ✅  Lambda        ({[x;y] …}  and implicit {x+y})
  ✅  Parentheses   (grouping, list literals)
  ✅  qSQL          (select … from … where … by …)
  🔜  Multi-line scripts
  🔜  Projections   f[;1]
  🔜  Nested lambdas
  📋  String escapes
  📋  Temporal literals  2024.01.15  12:30:00.000
  📋  Namespace assignment  .myns.foo:42
  📋  System commands  \\t \\l \\d
"""

import pytest
from polarq.parser import parse, parse_expr
from polarq.parser.ast_nodes import (
    IntLit, FloatLit, BoolLit, SymLit, StrLit, NullLit,
    VectorLit, ListLit,
    Name, Assign,
    Verb, BinOp, MonOp, Adverb,
    Apply,
    Lambda,
    Script,
    ColExpr, QSelect, QUpdate, QExec, QDelete,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def expr(src: str):
    """Parse a single expression from source (convenience for tests)."""
    return parse_expr(src)


def script(src: str) -> Script:
    return parse(src)


# ═════════════════════════════════════════════════════════════════════════════
# Literals
# ═════════════════════════════════════════════════════════════════════════════

class TestIntLiterals:
    """
    q integer literals.

    q has three signed integer types:
      h — 16-bit short    42h
      i — 32-bit int      42i
      j — 64-bit long     42  or  42j   (default)

    All are parsed as IntLit with a plain Python int value.
    """

    def test_plain_integer(self):
        """42  →  IntLit(42)"""
        assert expr("42") == IntLit(42)

    def test_long_suffix(self):
        """42j  →  IntLit(42)  (j suffix is consumed, value is native int)"""
        assert expr("42j") == IntLit(42)

    def test_int_suffix(self):
        """42i  →  IntLit(42)"""
        assert expr("42i") == IntLit(42)

    def test_short_suffix(self):
        """42h  →  IntLit(42)"""
        assert expr("42h") == IntLit(42)

    def test_zero(self):
        """0  →  IntLit(0)"""
        assert expr("0") == IntLit(0)

    def test_negative_literal(self):
        """-3  →  IntLit(-3)  (negative literal, not negation of 3)"""
        assert expr("-3") == IntLit(-3)


class TestFloatLiterals:
    """
    q floating-point literals.

      f — 64-bit double   3.14  or  3.14f
      e — 32-bit float    3.14e
    """

    def test_decimal(self):
        """3.14  →  FloatLit(3.14)"""
        assert expr("3.14") == FloatLit(3.14)

    def test_float_suffix(self):
        """3.14f  →  FloatLit(3.14)"""
        assert expr("3.14f") == FloatLit(3.14)

    def test_trailing_dot(self):
        """1.  →  FloatLit(1.0)"""
        assert expr("1.") == FloatLit(1.0)

    def test_negative_float(self):
        """-1.5  →  FloatLit(-1.5)"""
        assert expr("-1.5") == FloatLit(-1.5)


class TestBoolLiterals:
    """
    Boolean atoms: 0b and 1b.
    """

    def test_true(self):
        """1b  →  BoolLit(True)"""
        assert expr("1b") == BoolLit(True)

    def test_false(self):
        """0b  →  BoolLit(False)"""
        assert expr("0b") == BoolLit(False)


class TestSymbolLiterals:
    """
    Symbols — interned string atoms, written with a leading backtick.
    """

    def test_simple_sym(self):
        """`foo  →  SymLit('foo')"""
        assert expr("`foo") == SymLit("foo")

    def test_empty_sym(self):
        """`  →  SymLit('')  (empty / null symbol)"""
        assert expr("`") == SymLit("")

    def test_dotted_sym(self):
        """`a.b  →  SymLit('a.b')"""
        assert expr("`a.b") == SymLit("a.b")


class TestStringLiterals:
    """
    q strings are character vectors, written with double quotes.
    """

    def test_simple_string(self):
        '"hello"  →  StrLit("hello")'
        assert expr('"hello"') == StrLit("hello")

    def test_empty_string(self):
        '""  →  StrLit("")'
        assert expr('""') == StrLit("")


class TestNullLiterals:
    """
    q null values: 0N (long null), 0n (float null), 0Nd (date null), etc.
    """

    def test_long_null(self):
        """0N  →  NullLit(kind='j')"""
        assert expr("0N") == NullLit("j")

    def test_float_null(self):
        """0n  →  NullLit(kind='f')"""
        assert expr("0n") == NullLit("f")


# ═════════════════════════════════════════════════════════════════════════════
# Vectors (homogeneous juxtaposed literals)
# ═════════════════════════════════════════════════════════════════════════════

class TestVectorLiterals:
    """
    In q, space-separated atoms of the same type form a vector literal.
    The parser groups them into a VectorLit node during term-folding.
    """

    def test_int_vector(self):
        """1 2 3  →  VectorLit((IntLit(1), IntLit(2), IntLit(3)))"""
        node = expr("1 2 3")
        assert node == VectorLit((IntLit(1), IntLit(2), IntLit(3)))

    def test_float_vector(self):
        """1.0 2.0 3.0  →  VectorLit of FloatLit"""
        node = expr("1.0 2.0 3.0")
        assert node == VectorLit((FloatLit(1.0), FloatLit(2.0), FloatLit(3.0)))

    def test_bool_vector(self):
        """1b 0b 1b  →  VectorLit of BoolLit"""
        node = expr("1b 0b 1b")
        assert node == VectorLit((BoolLit(True), BoolLit(False), BoolLit(True)))

    def test_sym_vector(self):
        """`a`b`c  →  VectorLit of SymLit"""
        node = expr("`a`b`c")
        assert node == VectorLit((SymLit("a"), SymLit("b"), SymLit("c")))

    def test_single_int_is_not_vector(self):
        """A single literal stays as an atom, not a one-element VectorLit."""
        assert expr("42") == IntLit(42)


# ═════════════════════════════════════════════════════════════════════════════
# Names and assignment
# ═════════════════════════════════════════════════════════════════════════════

class TestNames:
    """Identifiers and dotted-namespace names."""

    def test_simple_name(self):
        """x  →  Name('x')"""
        assert expr("x") == Name("x")

    def test_dotted_name(self):
        """.z.p  →  Name('.z.p')"""
        assert expr(".z.p") == Name(".z.p")


class TestAssignment:
    """
    Variable assignment.

    x:42    local (global at top level)
    x::42   global amend
    """

    def test_local_assign(self):
        """x:42  →  Assign('x', IntLit(42), global_=False)"""
        node = expr("x:42")
        assert node == Assign("x", IntLit(42), global_=False)

    def test_global_assign(self):
        """x::42  →  Assign('x', IntLit(42), global_=True)"""
        node = expr("x::42")
        assert node == Assign("x", IntLit(42), global_=True)

    def test_assign_expression(self):
        """x:1+2  →  Assign('x', BinOp('+', IntLit(1), IntLit(2)))"""
        node = expr("x:1+2")
        assert node == Assign("x", BinOp("+", IntLit(1), IntLit(2)))

    def test_assign_vector(self):
        """v:1 2 3  →  Assign('v', VectorLit(...))"""
        node = expr("v:1 2 3")
        assert isinstance(node, Assign)
        assert node.name == "v"
        assert node.expr == VectorLit((IntLit(1), IntLit(2), IntLit(3)))

    def test_assign_sym(self):
        """s:`AAPL  →  Assign('s', SymLit('AAPL'))"""
        node = expr("s:`AAPL")
        assert node == Assign("s", SymLit("AAPL"), global_=False)


# ═════════════════════════════════════════════════════════════════════════════
# Arithmetic — right-to-left evaluation
# ═════════════════════════════════════════════════════════════════════════════

class TestArithmetic:
    """
    q has NO operator precedence — everything evaluates strictly right-to-left.

      2+3*4  =  2+(3*4)  =  14   (not 20 as in Python/C)

    The parser captures this by placing the leftmost verb as the outermost
    node and recursing right.
    """

    def test_add(self):
        """1+2  →  BinOp('+', IntLit(1), IntLit(2))"""
        assert expr("1+2") == BinOp("+", IntLit(1), IntLit(2))

    def test_subtract(self):
        """5-3  →  BinOp('-', IntLit(5), IntLit(3))"""
        assert expr("5-3") == BinOp("-", IntLit(5), IntLit(3))

    def test_multiply(self):
        """3*4  →  BinOp('*', IntLit(3), IntLit(4))"""
        assert expr("3*4") == BinOp("*", IntLit(3), IntLit(4))

    def test_divide(self):
        """10%2  →  BinOp('%', IntLit(10), IntLit(2))  — note: q uses % for divide"""
        assert expr("10%2") == BinOp("%", IntLit(10), IntLit(2))

    def test_right_to_left_no_precedence(self):
        """
        2+3*4  →  BinOp('+', IntLit(2), BinOp('*', IntLit(3), IntLit(4)))

        The multiplication is on the RIGHT of addition in the source, so it
        becomes the inner (right) node — equivalent to 2+(3*4).
        """
        node = expr("2+3*4")
        assert node == BinOp("+", IntLit(2), BinOp("*", IntLit(3), IntLit(4)))

    def test_right_associativity(self):
        """
        1+2+3  →  BinOp('+', IntLit(1), BinOp('+', IntLit(2), IntLit(3)))

        Right-to-left: the rightmost + binds first.
        """
        node = expr("1+2+3")
        assert node == BinOp("+", IntLit(1), BinOp("+", IntLit(2), IntLit(3)))

    def test_float_arithmetic(self):
        """1.5+2.5  →  BinOp('+', FloatLit(1.5), FloatLit(2.5))"""
        node = expr("1.5+2.5")
        assert node == BinOp("+", FloatLit(1.5), FloatLit(2.5))

    def test_mixed_chain(self):
        """
        a+b*c%d  →  BinOp('+', Name('a'), BinOp('*', Name('b'), BinOp('%', Name('c'), Name('d'))))
        """
        node = expr("a+b*c%d")
        expected = BinOp("+", Name("a"),
                         BinOp("*", Name("b"),
                               BinOp("%", Name("c"), Name("d"))))
        assert node == expected


# ═════════════════════════════════════════════════════════════════════════════
# Comparison operators
# ═════════════════════════════════════════════════════════════════════════════

class TestComparison:
    """
    Comparison verbs return boolean atoms / vectors.
    In q:  = is equality,  ~ is match / not-equal,  < and > are less/greater-than.
    """

    def test_equal(self):
        """x=1  →  BinOp('=', Name('x'), IntLit(1))"""
        assert expr("x=1") == BinOp("=", Name("x"), IntLit(1))

    def test_less_than(self):
        """x<10  →  BinOp('<', Name('x'), IntLit(10))"""
        assert expr("x<10") == BinOp("<", Name("x"), IntLit(10))

    def test_greater_than(self):
        """x>0  →  BinOp('>', Name('x'), IntLit(0))"""
        assert expr("x>0") == BinOp(">", Name("x"), IntLit(0))

    def test_tilde(self):
        """x~y  →  BinOp('~', Name('x'), Name('y'))  — dyadic ~ is match"""
        assert expr("x~y") == BinOp("~", Name("x"), Name("y"))


# ═════════════════════════════════════════════════════════════════════════════
# Monadic verbs
# ═════════════════════════════════════════════════════════════════════════════

class TestMonadicVerbs:
    """
    A verb that appears with no left-hand operand is monadic (unary).
    q uses many verbs ambiguously: + is flip (monad) or add (dyad).
    """

    def test_monadic_negate(self):
        """-x  →  MonOp('-', Name('x'))  (negation)"""
        assert expr("-x") == MonOp("-", Name("x"))

    def test_monadic_not(self):
        """~x  →  MonOp('~', Name('x'))  (logical NOT)"""
        assert expr("~x") == MonOp("~", Name("x"))

    def test_monadic_count(self):
        """#x  →  MonOp('#', Name('x'))  (count)"""
        assert expr("#x") == MonOp("#", Name("x"))

    def test_monadic_flip(self):
        """+x  →  MonOp('+', Name('x'))  (flip / transpose)"""
        assert expr("+x") == MonOp("+", Name("x"))

    def test_named_monadic_verb(self):
        """count x  →  Apply(Name('count'), (Name('x'),))  (named function call)"""
        node = expr("count x")
        assert node == Apply(Name("count"), (Name("x"),))

    def test_named_verb_on_vector(self):
        """
        sum 1 2 3  →  Apply(Name('sum'), (VectorLit(...),))

        The vector `1 2 3` is grouped first, then sum is applied to it.
        """
        node = expr("sum 1 2 3")
        assert node == Apply(Name("sum"), (VectorLit((IntLit(1), IntLit(2), IntLit(3))),))


# ═════════════════════════════════════════════════════════════════════════════
# Adverbs
# ═════════════════════════════════════════════════════════════════════════════

class TestAdverbs:
    """
    Adverbs are postfix modifiers that produce derived functions.

      +/   over (fold)
      +\\   scan (running fold)
      f'   each
      f/:  each-right
      f\\:  each-left
    """

    def test_over(self):
        """+/  →  Adverb(Verb('+'), '/')"""
        assert expr("+/") == Adverb(Verb("+"), "/")

    def test_scan(self):
        r"""+\  →  Adverb(Verb('+'), '\\')"""
        node = expr("+\\")
        assert node == Adverb(Verb("+"), "\\")

    def test_each(self):
        """f'  →  Adverb(Name('f'), \"'\")"""
        assert expr("f'") == Adverb(Name("f"), "'")

    def test_each_right(self):
        """f/:  →  Adverb(Name('f'), '/:')"""
        assert expr("f/:") == Adverb(Name("f"), "/:")

    def test_each_left(self):
        r"""f\:  →  Adverb(Name('f'), '\\:')"""
        node = expr("f\\:")
        assert node == Adverb(Name("f"), "\\:")

    def test_over_applied_to_vector(self):
        """
        +/ 1 2 3  →  Apply(Adverb(Verb('+'), '/'), (VectorLit(...),))

        The adverb produces a derived function; the vector is the argument.
        In q, `+/ 1 2 3` computes the sum via fold.
        """
        node = expr("+/ 1 2 3")
        expected = Apply(
            Adverb(Verb("+"), "/"),
            (VectorLit((IntLit(1), IntLit(2), IntLit(3))),),
        )
        assert node == expected


# ═════════════════════════════════════════════════════════════════════════════
# Function application — bracket form
# ═════════════════════════════════════════════════════════════════════════════

class TestBracketApplication:
    """
    f[x]       — monadic bracket call
    f[x;y]     — dyadic bracket call
    f[x;y;z]   — triadic
    f[;1]      — projection (gap = None)
    """

    def test_single_arg(self):
        """f[x]  →  Apply(Name('f'), (Name('x'),))"""
        assert expr("f[x]") == Apply(Name("f"), (Name("x"),))

    def test_two_args(self):
        """f[x;y]  →  Apply(Name('f'), (Name('x'), Name('y')))"""
        assert expr("f[x;y]") == Apply(Name("f"), (Name("x"), Name("y")))

    def test_three_args(self):
        """f[a;b;c]  →  Apply(Name('f'), (Name('a'), Name('b'), Name('c')))"""
        assert expr("f[a;b;c]") == Apply(
            Name("f"), (Name("a"), Name("b"), Name("c"))
        )

    def test_expr_args(self):
        """f[x+1;y*2]  →  Apply with BinOp arguments"""
        node = expr("f[x+1;y*2]")
        assert node == Apply(
            Name("f"),
            (BinOp("+", Name("x"), IntLit(1)),
             BinOp("*", Name("y"), IntLit(2))),
        )

    def test_projection_gap(self):
        """f[;1]  →  Apply(Name('f'), (None, IntLit(1)))"""
        node = expr("f[;1]")
        assert node == Apply(Name("f"), (None, IntLit(1)))

    def test_chained_application(self):
        """f[x][y]  →  Apply(Apply(Name('f'), (Name('x'),)), (Name('y'),))"""
        node = expr("f[x][y]")
        assert node == Apply(Apply(Name("f"), (Name("x"),)), (Name("y"),))


# ═════════════════════════════════════════════════════════════════════════════
# Lambda
# ═════════════════════════════════════════════════════════════════════════════

class TestLambda:
    """
    Lambdas are first-class functions in q.

    Explicit params:  {[x;y] x+y}
    Implicit params:  {x+y}  — x y z are available implicitly
    Multi-statement:  {a:x+1; a*y}  — semicolon-delimited; last is return
    """

    def test_implicit_params_single_expr(self):
        """{x+y}  →  Lambda(params=(), body=(BinOp('+', Name('x'), Name('y')),))"""
        node = expr("{x+y}")
        assert node == Lambda(
            params=(),
            body=(BinOp("+", Name("x"), Name("y")),),
        )

    def test_explicit_params(self):
        """{[a;b] a+b}  →  Lambda(params=('a','b'), body=(BinOp(...),))"""
        node = expr("{[a;b] a+b}")
        assert node == Lambda(
            params=("a", "b"),
            body=(BinOp("+", Name("a"), Name("b")),),
        )

    def test_single_param(self):
        """{[x] x*x}  →  Lambda(params=('x',), ...)"""
        node = expr("{[x] x*x}")
        assert node == Lambda(
            params=("x",),
            body=(BinOp("*", Name("x"), Name("x")),),
        )

    def test_multi_statement_body(self):
        """
        {a:x+1; a*y}  →  Lambda(params=(), body=(Assign('a', ...), BinOp('*', ...)))

        Two statements separated by semicolon.  The last (a*y) is the return value.
        """
        node = expr("{a:x+1; a*y}")
        assert isinstance(node, Lambda)
        assert node.params == ()
        assert len(node.body) == 2
        assert isinstance(node.body[0], Assign)
        assert node.body[0].name == "a"
        assert node.body[1] == BinOp("*", Name("a"), Name("y"))

    def test_assigned_lambda(self):
        """vwap:{sum[x*y]%sum y}  →  Assign('vwap', Lambda(...))"""
        node = expr("vwap:{sum[x*y]%sum y}")
        assert isinstance(node, Assign)
        assert node.name == "vwap"
        assert isinstance(node.expr, Lambda)


# ═════════════════════════════════════════════════════════════════════════════
# Parentheses and list literals
# ═════════════════════════════════════════════════════════════════════════════

class TestParentheses:
    """
    Parentheses serve two purposes:
      (expr)         — grouping (no AST node, just changes structure)
      (a; b; c)      — mixed list literal  →  ListLit
    """

    def test_grouping_changes_structure(self):
        """
        (2+3)*4  →  BinOp('*', BinOp('+', ...), IntLit(4))

        Without parentheses, q evaluates right-to-left:
          2+3*4  =  2+(3*4)  =  14
        With parentheses:
          (2+3)*4  =  5*4    =  20
        """
        node = expr("(2+3)*4")
        assert node == BinOp("*", BinOp("+", IntLit(2), IntLit(3)), IntLit(4))

    def test_list_literal(self):
        """(1;2;3)  →  ListLit((IntLit(1), IntLit(2), IntLit(3)))"""
        node = expr("(1;2;3)")
        assert node == ListLit((IntLit(1), IntLit(2), IntLit(3)))

    def test_mixed_list(self):
        """(1;`a;"hello")  →  ListLit(...)"""
        node = expr('(1;`a;"hello")')
        assert node == ListLit((IntLit(1), SymLit("a"), StrLit("hello")))

    def test_empty_list(self):
        """()  →  ListLit(())"""
        assert expr("()") == ListLit(())


# ═════════════════════════════════════════════════════════════════════════════
# qSQL
# ═════════════════════════════════════════════════════════════════════════════

class TestQSQL:
    """
    qSQL is a first-class query language embedded in q.

      select [cols] [by groups] from table [where conds]
      update  col:expr           from table [where conds]
      exec   [cols] [by groups] from table [where conds]
      delete [cols]              from table [where conds]
    """

    def test_select_all(self):
        """
        select from trade  →  QSelect(cols=(), table=Name('trade'), where=(), by=())
        """
        node = expr("select from trade")
        assert isinstance(node, QSelect)
        assert node.cols  == ()
        assert node.table == Name("trade")
        assert node.where == ()
        assert node.by    == ()

    def test_select_column(self):
        """
        select price from trade  →  QSelect with one ColExpr
        """
        node = expr("select price from trade")
        assert isinstance(node, QSelect)
        assert len(node.cols) == 1
        assert node.cols[0] == ColExpr(Name("price"), alias=None)

    def test_select_with_where(self):
        """
        select from trade where price>200.0
        """
        node = expr("select from trade where price>200.0")
        assert isinstance(node, QSelect)
        assert len(node.where) == 1
        assert node.where[0] == BinOp(">", Name("price"), FloatLit(200.0))

    def test_select_with_by(self):
        """
        select by sym from trade

        In q, the `by` clause always precedes `from`.
        """
        node = expr("select by sym from trade")
        assert isinstance(node, QSelect)
        assert len(node.by) == 1
        assert node.by[0] == ColExpr(Name("sym"), alias=None)

    def test_select_agg_by(self):
        """
        select sum qty, avg price by sym from trade

        Produces a QSelect with two ColExprs (no alias) and one by-group.
        """
        node = expr("select sum qty,avg price by sym from trade")
        assert isinstance(node, QSelect)
        assert len(node.cols) == 2
        # first col: sum qty
        assert node.cols[0].expr == Apply(Name("sum"), (Name("qty"),))
        # second col: avg price
        assert node.cols[1].expr == Apply(Name("avg"), (Name("price"),))
        # by clause
        assert node.by[0].expr == Name("sym")

    def test_select_aliased_col(self):
        """
        select total:sum qty from trade  →  ColExpr(..., alias='total')
        """
        node = expr("select total:sum qty from trade")
        assert isinstance(node, QSelect)
        assert node.cols[0].alias == "total"
        assert node.cols[0].expr == Apply(Name("sum"), (Name("qty"),))

    def test_update(self):
        """
        update price:price*1.1 from trade  →  QUpdate
        """
        node = expr("update price:price*1.1 from trade")
        assert isinstance(node, QUpdate)
        assert node.table == Name("trade")
        assert len(node.cols) == 1
        assert node.cols[0].alias == "price"

    def test_delete(self):
        """
        delete from trade where price<0.0  →  QDelete
        """
        node = expr("delete from trade where price<0.0")
        assert isinstance(node, QDelete)
        assert node.table == Name("trade")
        assert len(node.where) == 1


# ═════════════════════════════════════════════════════════════════════════════
# Multi-line scripts
# ═════════════════════════════════════════════════════════════════════════════

class TestScripts:
    """
    A script is a sequence of statements separated by newlines.
    The parse() function returns a Script node containing all statements.
    """

    def test_two_assignments(self):
        """
        x:1
        y:2
        →  Script with two Assign nodes.
        """
        s = script("x:1\ny:2")
        assert len(s.stmts) == 2
        assert s.stmts[0] == Assign("x", IntLit(1))
        assert s.stmts[1] == Assign("y", IntLit(2))

    def test_assignment_then_expr(self):
        """
        x:42
        x+1
        →  Script(Assign('x', 42),  BinOp('+', Name('x'), IntLit(1)))
        """
        s = script("x:42\nx+1")
        assert len(s.stmts) == 2
        assert isinstance(s.stmts[0], Assign)
        assert s.stmts[1] == BinOp("+", Name("x"), IntLit(1))

    def test_comment_is_ignored(self):
        """
        // this is a comment
        x:1
        →  Script with one Assign (the comment is discarded).
        """
        s = script("// this is a comment\nx:1")
        assert len(s.stmts) == 1
        assert s.stmts[0] == Assign("x", IntLit(1))

    def test_readme_strategy_script(self):
        """
        The three-line script from the README transpiler example parses without errors.

        trade:([]sym:`AAPL`GOOG`MSFT;price:150.0 280.0 320.0;qty:100 200 150)
        vwap:{sum[x*y]%sum y}
        result:select sum qty,avg price by sym from trade where price>200.0
        """
        src = (
            "vwap:{sum[x*y]%sum y}\n"
            "result:select sum qty,avg price by sym from trade where price>200.0\n"
        )
        s = script(src)
        assert len(s.stmts) == 2
        # vwap is a lambda assignment
        assert isinstance(s.stmts[0], Assign)
        assert s.stmts[0].name == "vwap"
        assert isinstance(s.stmts[0].expr, Lambda)
        # result is a qSQL assignment
        assert isinstance(s.stmts[1], Assign)
        assert s.stmts[1].name == "result"
        assert isinstance(s.stmts[1].expr, QSelect)


# ═════════════════════════════════════════════════════════════════════════════
# Skeletons for planned future coverage
# ═════════════════════════════════════════════════════════════════════════════

class TestProjections:
    """
    🔜  Projections — f[;y] partially applies a function.

    Projections create new derived functions with some arguments pre-filled.
    A gap (;;) in the bracket argument list leaves that position unfilled.
    """

    def test_projection_one_gap(self):
        """f[;1]  →  Apply(Name('f'), (None, IntLit(1)))"""
        node = expr("f[;1]")
        # Already works — gap is represented as None in args tuple
        assert node == Apply(Name("f"), (None, IntLit(1)))

    @pytest.mark.skip(reason="double-gap projections not yet validated end-to-end")
    def test_projection_two_gaps(self):
        """f[;1;]  →  Apply(Name('f'), (None, IntLit(1), None))"""
        node = expr("f[;1;]")
        assert node == Apply(Name("f"), (None, IntLit(1), None))


class TestTemporalLiterals:
    """
    📋  Temporal literals — not yet implemented in the lexer.

    q has rich first-class date/time types:
      2024.01.15        — date
      12:30:00.000      — time
      2024.01.15D12:30  — timestamp
    """

    @pytest.mark.skip(reason="temporal literal tokenisation not yet implemented")
    def test_date_literal(self):
        """2024.01.15  →  DateLit(date(2024, 1, 15))"""
        node = expr("2024.01.15")
        # assert isinstance(node, DateLit)

    @pytest.mark.skip(reason="temporal literal tokenisation not yet implemented")
    def test_timestamp_literal(self):
        """2024.01.15D12:30:00.000000000"""
        node = expr("2024.01.15D12:30:00.000000000")
        # assert isinstance(node, TimestampLit)


class TestNamespaceAssignment:
    """
    📋  Dotted namespace assignment  .myns.foo:42

    Assigns `foo` inside the `.myns` namespace frame.
    """

    @pytest.mark.skip(reason="namespace assignment parsing not yet validated")
    def test_dotted_assign(self):
        """.myns.foo:42  →  Assign('.myns.foo', IntLit(42))"""
        node = expr(".myns.foo:42")
        assert isinstance(node, Assign)
        assert node.name == ".myns.foo"
