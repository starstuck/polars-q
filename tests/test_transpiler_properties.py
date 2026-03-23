"""
tests/test_transpiler_properties.py
────────────────────────────────────
Property-based (Hypothesis / QuickCheck-style) tests for the polarq transpiler
and q runtime.

Philosophy
──────────
Rather than testing specific inputs, we specify *invariants* that must hold for
all inputs in some domain.  Hypothesis explores that domain automatically and
shrinks any counterexample to its simplest form.

Three layers are tested:

  Layer 1 — Transpiler contract
    Properties about what the transpiler is *allowed* to do.
    Given any valid q AST node, the transpiler must either:
      (a) return a well-formed ``ast.Module``, or
      (b) raise ``NotImplementedError`` (for unimplemented constructs).
    Any other exception is a bug.

  Layer 2 — Transpiler output structure
    Properties about the *shape* of the Python AST the transpiler emits.
    e.g. IntLit always becomes ast.Constant; BinOp('+',…) always becomes
    a Call to q_add with two arguments; VectorLit always becomes a Call to
    QVector.from_items with the right kind string.

  Layer 3 — q runtime semantics
    Properties about the runtime verbs and adverbs, tested directly (bypassing
    the transpiler).  These are the semantic ground truth; the transpiler is
    correct when it emits code that calls these functions appropriately.
    Examples: addition commutativity, fold identity, vector broadcast.

Known gap (tracked, not hidden)
────────────────────────────────
The transpiler currently emits bare Python literals (e.g. ``q_add(1, 2)``)
rather than wrapped atoms (``q_add(QAtom(1,'j'), QAtom(2,'j'))``).  The q
runtime's ``_arith`` dispatch does not handle plain Python ints, so end-to-end
execution of transpiled arithmetic fails with QTypeError.  The property
``test_transpiled_arithmetic_executes`` documents this gap explicitly; it will
become a passing test once the transpiler wraps literals in QAtom calls.
"""

import ast as py_ast
import math

import pytest
from hypothesis import HealthCheck, assume, given, note, settings
from hypothesis import strategies as st

from polarq.errors import QTypeError
from polarq.parser.ast_nodes import (
    Adverb,
    Apply,
    Assign,
    BinOp,
    BoolLit,
    FloatLit,
    IntLit,
    Lambda,
    ListLit,
    MonOp,
    Name,
    NullLit,
    Script,
    StrLit,
    SymLit,
    VectorLit,
    Verb,
)
from polarq.transpiler.builtins import ADVERB_MAP, VERB_MAP
from polarq.transpiler.transpiler import QToPythonTranspiler
from polarq.types import QAtom, QVector
from polarq.verbs import q_add, q_sub, q_mul, q_div, q_avg, q_sum, q_min, q_max
from polarq.adverbs import over, scan


# ── Shared strategies ─────────────────────────────────────────────────────────

# Integer bounds chosen to stay well inside 64-bit range and avoid overflow
# in arithmetic expressions with several levels of nesting.
SAFE_INT = st.integers(min_value=-(10 ** 9), max_value=10 ** 9)
# Floats: no NaN / inf — those would propagate unexpectedly through arithmetic
SAFE_FLOAT = st.floats(
    min_value=-(10 ** 9),
    max_value=10 ** 9,
    allow_nan=False,
    allow_infinity=False,
)
# Identifiers that are valid Python names (required for transpiled output)
IDENT = st.from_regex(r"[a-z][a-z0-9]{0,7}", fullmatch=True)
# Symbols (q backtick names — same shape as identifiers here)
SYM_NAME = st.from_regex(r"[a-z][a-z0-9]{0,7}", fullmatch=True)

# Arithmetic operators supported by the transpiler (% excluded: division by zero)
ARITH_OP = st.sampled_from(["+", "-", "*"])
# All operators that appear in VERB_MAP
KNOWN_OP = st.sampled_from(list(VERB_MAP.keys()))
# All adverbs that appear in ADVERB_MAP
KNOWN_ADVERB = st.sampled_from(list(ADVERB_MAP.keys()))

# ── Leaf literal strategies ───────────────────────────────────────────────────

int_lit    = SAFE_INT.map(IntLit)
float_lit  = SAFE_FLOAT.map(FloatLit)
bool_lit   = st.booleans().map(BoolLit)
sym_lit    = SYM_NAME.map(SymLit)
str_lit    = st.text(
    alphabet=st.characters(whitelist_categories=["Ll", "Lu", "Nd"], max_codepoint=127),
    max_size=32,
).map(StrLit)
null_lit   = st.just(NullLit())
name_node  = IDENT.map(Name)

any_leaf = st.one_of(int_lit, float_lit, bool_lit, sym_lit, str_lit, null_lit)

# ── Recursive arithmetic expression strategy ─────────────────────────────────

def _arith_children(children):
    return st.builds(BinOp, op=ARITH_OP, left=children, right=children)

arith_expr = st.recursive(
    st.one_of(int_lit, float_lit),
    _arith_children,
    max_leaves=10,
)

# ── Vector literal strategies ─────────────────────────────────────────────────

int_vec_lit = st.lists(SAFE_INT, min_size=2, max_size=20).map(
    lambda vs: VectorLit(tuple(IntLit(v) for v in vs))
)
float_vec_lit = st.lists(SAFE_FLOAT, min_size=2, max_size=20).map(
    lambda vs: VectorLit(tuple(FloatLit(v) for v in vs))
)
bool_vec_lit = st.lists(st.booleans(), min_size=2, max_size=20).map(
    lambda vs: VectorLit(tuple(BoolLit(v) for v in vs))
)
sym_vec_lit = st.lists(SYM_NAME, min_size=2, max_size=10).map(
    lambda vs: VectorLit(tuple(SymLit(v) for v in vs))
)

# ── Script builder ────────────────────────────────────────────────────────────

def make_script(*nodes) -> Script:
    return Script(tuple(nodes))


def transpile_node(node) -> py_ast.Module:
    """Convenience: wrap a single node in a Script and transpile it."""
    t = QToPythonTranspiler()
    return t.transpile(make_script(node))


def transpile_expr_node(node) -> py_ast.expr:
    """Return just the expression AST node from a single-expression script."""
    mod = transpile_node(node)
    # mod.body = [import polars, from polarq import *, Expr(node)]
    stmt = mod.body[-1]
    assert isinstance(stmt, py_ast.Expr)
    return stmt.value


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1 — Transpiler contract
# ═════════════════════════════════════════════════════════════════════════════

class TestTranspilerContract:
    """
    The transpiler must obey a strict contract: for any valid q AST node it
    either succeeds (returns ast.Module) or raises NotImplementedError.
    No other exception is acceptable.
    """

    def _assert_contract(self, node):
        try:
            mod = transpile_node(node)
            # Success path: must return an ast.Module
            assert isinstance(mod, py_ast.Module), (
                f"transpile() returned {type(mod).__name__}, expected ast.Module"
            )
        except NotImplementedError:
            pass   # Explicitly allowed — unimplemented but acknowledged
        except Exception as exc:
            pytest.fail(
                f"transpiler raised unexpected {type(exc).__name__}: {exc}\n"
                f"  node = {node!r}"
            )

    @given(int_lit)
    def test_contract_int_lit(self, node):
        """IntLit never causes an unexpected exception."""
        self._assert_contract(node)

    @given(float_lit)
    def test_contract_float_lit(self, node):
        """FloatLit never causes an unexpected exception."""
        self._assert_contract(node)

    @given(bool_lit)
    def test_contract_bool_lit(self, node):
        """BoolLit never causes an unexpected exception."""
        self._assert_contract(node)

    @given(sym_lit)
    def test_contract_sym_lit(self, node):
        """SymLit never causes an unexpected exception."""
        self._assert_contract(node)

    @given(str_lit)
    def test_contract_str_lit(self, node):
        """StrLit never causes an unexpected exception."""
        self._assert_contract(node)

    @given(st.just(NullLit()))
    def test_contract_null_lit(self, node):
        """NullLit never causes an unexpected exception."""
        self._assert_contract(node)

    @given(int_vec_lit)
    def test_contract_int_vec(self, node):
        """Integer VectorLit never causes an unexpected exception."""
        self._assert_contract(node)

    @given(float_vec_lit)
    def test_contract_float_vec(self, node):
        """Float VectorLit never causes an unexpected exception."""
        self._assert_contract(node)

    @given(sym_vec_lit)
    def test_contract_sym_vec(self, node):
        """Symbol VectorLit never causes an unexpected exception."""
        self._assert_contract(node)

    @given(name_node)
    def test_contract_name(self, node):
        """Name node never causes an unexpected exception."""
        self._assert_contract(node)

    @given(st.builds(BinOp, op=KNOWN_OP, left=int_lit, right=int_lit))
    def test_contract_binop_known_ops(self, node):
        """BinOp with any known operator + literal operands never crashes unexpectedly."""
        self._assert_contract(node)

    @given(st.builds(MonOp, op=KNOWN_OP, right=int_lit))
    def test_contract_monop_known_ops(self, node):
        """MonOp with any known operator + literal operand never crashes unexpectedly."""
        self._assert_contract(node)

    @given(arith_expr)
    @settings(max_examples=200)
    def test_contract_deep_arith_tree(self, node):
        """Arbitrarily nested arithmetic expression never causes unexpected exception."""
        self._assert_contract(node)

    @given(
        st.builds(
            Apply,
            func=name_node,
            args=st.lists(int_lit, min_size=1, max_size=4).map(tuple),
        )
    )
    def test_contract_apply(self, node):
        """Apply node (function call) never causes unexpected exception."""
        self._assert_contract(node)

    @given(
        st.builds(
            Adverb,
            verb=st.builds(Verb, op=ARITH_OP),
            adverb=KNOWN_ADVERB,
        )
    )
    def test_contract_adverb(self, node):
        """Adverb node never causes unexpected exception."""
        self._assert_contract(node)

    @given(
        st.builds(
            Lambda,
            params=st.just(()),
            body=st.tuples(int_lit),
        )
    )
    def test_contract_simple_lambda(self, node):
        """Single-expression implicit-param lambda never causes unexpected exception."""
        self._assert_contract(node)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2 — Transpiler output structure
# ═════════════════════════════════════════════════════════════════════════════

class TestTranspilerOutputStructure:
    """
    Properties about *what* the transpiler emits, not just whether it succeeds.
    """

    # ── Literal → ast.Constant ───────────────────────────────────────────────

    @given(SAFE_INT)
    def test_int_lit_becomes_constant(self, v):
        """IntLit(v) transpiles to QAtom(v, 'j') call."""
        node = transpile_expr_node(IntLit(v))
        assert isinstance(node, py_ast.Call)
        assert isinstance(node.func, py_ast.Name) and node.func.id == "QAtom"
        assert node.args[0].value == v
        assert node.args[1].value == "j"

    @given(SAFE_FLOAT)
    def test_float_lit_becomes_constant(self, v):
        """FloatLit(v) transpiles to QAtom(v, 'f') call."""
        node = transpile_expr_node(FloatLit(v))
        assert isinstance(node, py_ast.Call)
        assert isinstance(node.func, py_ast.Name) and node.func.id == "QAtom"
        raw = node.args[0].value
        assert raw == v or (math.isnan(raw) and math.isnan(v))
        assert node.args[1].value == "f"

    @given(st.booleans())
    def test_bool_lit_becomes_constant(self, v):
        """BoolLit(v) transpiles to QAtom(v, 'b') call."""
        node = transpile_expr_node(BoolLit(v))
        assert isinstance(node, py_ast.Call)
        assert isinstance(node.func, py_ast.Name) and node.func.id == "QAtom"
        assert node.args[0].value == v
        assert node.args[1].value == "b"

    @given(SYM_NAME)
    def test_sym_lit_becomes_string_constant(self, v):
        """SymLit(v) transpiles to QAtom(v, 's') call."""
        node = transpile_expr_node(SymLit(v))
        assert isinstance(node, py_ast.Call)
        assert isinstance(node.func, py_ast.Name) and node.func.id == "QAtom"
        assert node.args[0].value == v
        assert node.args[1].value == "s"

    def test_null_lit_becomes_none_constant(self):
        """NullLit() transpiles to ast.Constant(value=None)."""
        node = transpile_expr_node(NullLit())
        assert isinstance(node, py_ast.Constant)
        assert node.value is None

    # ── Name → ast.Name ──────────────────────────────────────────────────────

    @given(IDENT)
    def test_name_becomes_ast_name(self, name):
        """Name(n) transpiles to ast.Name(id=n)."""
        node = transpile_expr_node(Name(name))
        assert isinstance(node, py_ast.Name)
        assert node.id == name

    # ── BinOp → ast.Call with correct function ────────────────────────────────

    @given(ARITH_OP, int_lit, int_lit)
    def test_binop_becomes_call(self, op, left, right):
        """BinOp(op, l, r) transpiles to ast.Call wrapping the runtime verb."""
        node = transpile_expr_node(BinOp(op, left, right))
        assert isinstance(node, py_ast.Call)
        assert len(node.args) == 2

    @given(ARITH_OP, int_lit, int_lit)
    def test_binop_calls_correct_verb(self, op, left, right):
        """The function name in the emitted Call matches VERB_MAP."""
        node = transpile_expr_node(BinOp(op, left, right))
        _, expected_name = VERB_MAP[op]
        assert isinstance(node.func, py_ast.Name)
        assert node.func.id == expected_name

    @given(ARITH_OP, int_lit, int_lit)
    def test_binop_args_preserve_order(self, op, left, right):
        """
        BinOp(op, left, right) → Call(verb, [left_expr, right_expr]).
        Argument order must be preserved — q is not commutative for subtraction/division.
        """
        node = transpile_expr_node(BinOp(op, left, right))
        left_node  = node.args[0]
        right_node = node.args[1]
        # left_node / right_node are now QAtom(v, kind) Call nodes
        assert isinstance(left_node,  py_ast.Call) and left_node.args[0].value  == left.value
        assert isinstance(right_node, py_ast.Call) and right_node.args[0].value == right.value

    # ── MonOp → ast.Call with one argument ───────────────────────────────────

    @given(ARITH_OP, int_lit)
    def test_monop_becomes_single_arg_call(self, op, operand):
        """MonOp(op, x) transpiles to ast.Call with exactly one argument."""
        node = transpile_expr_node(MonOp(op, operand))
        assert isinstance(node, py_ast.Call)
        assert len(node.args) == 1

    @given(ARITH_OP, int_lit)
    def test_monop_calls_correct_verb(self, op, operand):
        """The verb name in the monadic Call matches VERB_MAP."""
        node = transpile_expr_node(MonOp(op, operand))
        _, expected_name = VERB_MAP[op]
        assert isinstance(node.func, py_ast.Name)
        assert node.func.id == expected_name

    # ── VectorLit → QVector.from_items call ──────────────────────────────────

    @given(int_vec_lit)
    def test_int_vec_becomes_from_items_call(self, node):
        """Integer VectorLit transpiles to QVector.from_items([...], 'j')."""
        out = transpile_expr_node(node)
        assert isinstance(out, py_ast.Call)
        assert isinstance(out.func, py_ast.Attribute)
        assert out.func.attr == "from_items"
        # Kind argument is 'j' for integers
        kind_arg = out.args[1]
        assert isinstance(kind_arg, py_ast.Constant)
        assert kind_arg.value == "j"

    @given(float_vec_lit)
    def test_float_vec_becomes_from_items_call(self, node):
        """Float VectorLit transpiles to QVector.from_items([...], 'f')."""
        out = transpile_expr_node(node)
        kind_arg = out.args[1]
        assert isinstance(kind_arg, py_ast.Constant)
        assert kind_arg.value == "f"

    @given(bool_vec_lit)
    def test_bool_vec_becomes_from_items_call(self, node):
        """Boolean VectorLit transpiles to QVector.from_items([...], 'b')."""
        out = transpile_expr_node(node)
        kind_arg = out.args[1]
        assert isinstance(kind_arg, py_ast.Constant)
        assert kind_arg.value == "b"

    @given(sym_vec_lit)
    def test_sym_vec_becomes_from_items_call(self, node):
        """Symbol VectorLit transpiles to QVector.from_items([...], 's')."""
        out = transpile_expr_node(node)
        kind_arg = out.args[1]
        assert isinstance(kind_arg, py_ast.Constant)
        assert kind_arg.value == "s"

    @given(int_vec_lit)
    def test_vec_items_count_preserved(self, node):
        """The number of elements in the vector is preserved through transpilation."""
        out = transpile_expr_node(node)
        items_list = out.args[0]
        assert isinstance(items_list, py_ast.List)
        assert len(items_list.elts) == len(node.items)

    # ── Apply → ast.Call ─────────────────────────────────────────────────────

    @given(
        IDENT,
        st.lists(int_lit, min_size=1, max_size=5).map(tuple),
    )
    def test_apply_becomes_call(self, fname, args):
        """Apply(Name(f), (a, b, …)) transpiles to a Python function call."""
        node = Apply(Name(fname), args)
        out = transpile_expr_node(node)
        assert isinstance(out, py_ast.Call)
        assert len(out.args) == len(args)

    @given(
        IDENT,
        st.lists(int_lit, min_size=1, max_size=4).map(tuple),
    )
    def test_apply_arity_preserved(self, fname, args):
        """The number of arguments in Apply is preserved."""
        node = Apply(Name(fname), args)
        out = transpile_expr_node(node)
        assert len(out.args) == len(args)

    # ── Module structure ─────────────────────────────────────────────────────

    @given(any_leaf)
    def test_module_always_has_imports(self, node):
        """Every transpiled module starts with the standard polarq imports."""
        mod = transpile_node(node)
        assert len(mod.body) >= 3   # import polars, from polarq import *, + stmt
        first  = mod.body[0]
        second = mod.body[1]
        assert isinstance(first,  py_ast.Import)
        assert isinstance(second, py_ast.ImportFrom)
        assert second.module == "polarq"

    @given(any_leaf)
    def test_module_is_compilable(self, node):
        """Every module produced by the transpiler compiles without SyntaxError."""
        mod = transpile_node(node)
        code = compile(mod, "<property-test>", "exec")
        assert code is not None

    @given(arith_expr)
    @settings(max_examples=300)
    def test_arith_module_is_compilable(self, node):
        """Arbitrarily nested arithmetic expression always produces compilable Python."""
        mod = transpile_node(node)
        code = compile(mod, "<property-test>", "exec")
        assert code is not None

    # ── Source unparse is valid Python ───────────────────────────────────────

    @given(any_leaf)
    def test_to_source_is_valid_python(self, node):
        """to_source() output can always be re-parsed by Python's ast.parse()."""
        t = QToPythonTranspiler()
        src = t.to_source(make_script(node))
        reparsed = py_ast.parse(src)
        assert isinstance(reparsed, py_ast.Module)

    @given(arith_expr)
    @settings(max_examples=200)
    def test_arith_to_source_is_valid_python(self, node):
        """Arbitrary arithmetic tree always produces valid Python source."""
        t = QToPythonTranspiler()
        src = t.to_source(make_script(node))
        reparsed = py_ast.parse(src)
        assert isinstance(reparsed, py_ast.Module)

    # ── Assign ───────────────────────────────────────────────────────────────

    @given(IDENT, int_lit)
    def test_assign_becomes_ast_assign(self, name, rhs):
        """Assign(name, IntLit) transpiles to ast.Assign targeting that name."""
        node = Assign(name, rhs)
        mod  = transpile_node(node)
        last = mod.body[-1]
        assert isinstance(last, py_ast.Assign)
        assert isinstance(last.targets[0], py_ast.Name)
        assert last.targets[0].id == name

    @given(IDENT, arith_expr)
    def test_assign_rhs_is_call_for_binop(self, name, rhs):
        """Assign with BinOp RHS produces an ast.Assign whose value is ast.Call."""
        if not isinstance(rhs, BinOp):
            return  # only check when RHS is actually a BinOp
        node = Assign(name, rhs)
        mod  = transpile_node(node)
        last = mod.body[-1]
        assert isinstance(last, py_ast.Assign)
        assert isinstance(last.value, py_ast.Call)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3 — q runtime semantics (verbs and adverbs tested directly)
# ═════════════════════════════════════════════════════════════════════════════

class TestQRuntimeProperties:
    """
    These tests bypass the transpiler and check the q runtime (verbs, adverbs)
    directly.  They establish the semantic ground truth that transpiled code is
    expected to reproduce.
    """

    # ── Atom arithmetic ───────────────────────────────────────────────────────

    @given(SAFE_INT, SAFE_INT)
    def test_add_atoms_correct(self, a, b):
        """q_add on integer atoms gives the Python sum."""
        result = q_add(QAtom(a, "j"), QAtom(b, "j"))
        assert isinstance(result, QAtom)
        assert result.value == a + b

    @given(SAFE_INT, SAFE_INT)
    def test_add_atoms_commutative(self, a, b):
        """Integer addition is commutative: q_add(x,y) == q_add(y,x)."""
        lhs = q_add(QAtom(a, "j"), QAtom(b, "j"))
        rhs = q_add(QAtom(b, "j"), QAtom(a, "j"))
        assert lhs.value == rhs.value

    @given(SAFE_INT, SAFE_INT, SAFE_INT)
    def test_add_atoms_associative(self, a, b, c):
        """Integer addition is associative."""
        qa, qb, qc = QAtom(a, "j"), QAtom(b, "j"), QAtom(c, "j")
        lhs = q_add(q_add(qa, qb), qc)
        rhs = q_add(qa, q_add(qb, qc))
        assert lhs.value == rhs.value

    @given(SAFE_INT)
    def test_add_zero_identity(self, a):
        """Adding zero is the identity: q_add(x, 0) == x."""
        result = q_add(QAtom(a, "j"), QAtom(0, "j"))
        assert result.value == a

    @given(SAFE_INT, SAFE_INT)
    def test_sub_atoms_correct(self, a, b):
        """q_sub on integer atoms gives the Python difference."""
        result = q_sub(QAtom(a, "j"), QAtom(b, "j"))
        assert result.value == a - b

    @given(SAFE_INT, SAFE_INT)
    def test_sub_antisymmetric(self, a, b):
        """Subtraction is antisymmetric: q_sub(x,y) == -q_sub(y,x)."""
        ab = q_sub(QAtom(a, "j"), QAtom(b, "j"))
        ba = q_sub(QAtom(b, "j"), QAtom(a, "j"))
        assert ab.value == -ba.value

    @given(SAFE_INT, SAFE_INT)
    def test_mul_atoms_correct(self, a, b):
        """q_mul on integer atoms gives the Python product."""
        result = q_mul(QAtom(a, "j"), QAtom(b, "j"))
        assert result.value == a * b

    @given(SAFE_INT, SAFE_INT)
    def test_mul_atoms_commutative(self, a, b):
        """Multiplication is commutative."""
        lhs = q_mul(QAtom(a, "j"), QAtom(b, "j"))
        rhs = q_mul(QAtom(b, "j"), QAtom(a, "j"))
        assert lhs.value == rhs.value

    @given(SAFE_INT)
    def test_mul_one_identity(self, a):
        """Multiplying by one is the identity: q_mul(x, 1) == x."""
        result = q_mul(QAtom(a, "j"), QAtom(1, "j"))
        assert result.value == a

    @given(SAFE_INT)
    def test_mul_zero_absorbing(self, a):
        """Multiplying by zero always gives zero."""
        result = q_mul(QAtom(a, "j"), QAtom(0, "j"))
        assert result.value == 0

    @given(SAFE_FLOAT, SAFE_FLOAT)
    def test_div_atoms_correct(self, a, b):
        """q_div (%) on float atoms gives the Python quotient."""
        assume(abs(b) > 1e-9)   # avoid division by zero
        result = q_div(QAtom(a, "f"), QAtom(b, "f"))
        assert abs(result.value - a / b) < 1e-6

    # ── Atom–vector broadcast ─────────────────────────────────────────────────

    @given(SAFE_INT, st.lists(SAFE_INT, min_size=1, max_size=50))
    def test_add_atom_to_vector_broadcasts(self, scalar, items):
        """
        q_add(atom, vector) broadcasts the scalar: each element gets scalar added.
        """
        atom = QAtom(scalar, "j")
        vec  = QVector.from_items(items, "j")
        result = q_add(atom, vec)
        assert isinstance(result, QVector)
        assert len(result.series) == len(items)
        for i, item in enumerate(items):
            assert result.series[i] == scalar + item

    @given(st.lists(SAFE_INT, min_size=1, max_size=50), SAFE_INT)
    def test_add_vector_to_atom_broadcasts(self, items, scalar):
        """q_add(vector, atom) also broadcasts."""
        vec  = QVector.from_items(items, "j")
        atom = QAtom(scalar, "j")
        result = q_add(vec, atom)
        assert isinstance(result, QVector)
        assert len(result.series) == len(items)

    @given(st.lists(SAFE_INT, min_size=1, max_size=50))
    def test_add_zero_vector_identity(self, items):
        """Adding a zero vector to a vector is the identity."""
        v = QVector.from_items(items, "j")
        z = QVector.from_items([0] * len(items), "j")
        result = q_add(v, z)
        assert list(result.series) == items

    # ── Adverbs: over (fold) ──────────────────────────────────────────────────

    @given(st.lists(SAFE_INT, min_size=1, max_size=50))
    def test_over_add_equals_sum(self, items):
        """
        +/ vec  (over q_add) equals the plain sum of the elements.
        This is the definition of fold-add.
        """
        vec    = QVector.from_items(items, "j")
        result = over(q_add, vec)
        assert isinstance(result, QAtom)
        assert result.value == sum(items)

    @given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=5))
    def test_over_mul_equals_product(self, items):
        """
        */ vec  (over q_mul) equals the plain product of the elements.

        Bounded to small integers (max 100^5 = 10^10) to stay within Int64 range
        and avoid q's defined integer-overflow behaviour diverging from Python's
        arbitrary-precision arithmetic.
        """
        vec    = QVector.from_items(items, "j")
        result = over(q_mul, vec)
        assert isinstance(result, QAtom)
        expected = 1
        for v in items:
            expected *= v
        assert result.value == expected

    @given(st.lists(SAFE_INT, min_size=2, max_size=50))
    def test_over_add_equals_q_sum(self, items):
        """+/ vec  produces the same answer as q_sum applied to the same vector."""
        vec = QVector.from_items(items, "j")
        assert over(q_add, vec).value == q_sum(vec).value

    # ── Adverbs: scan (running fold) ──────────────────────────────────────────

    @given(st.lists(SAFE_INT, min_size=1, max_size=30))
    def test_scan_add_last_equals_over_add(self, items):
        """
        The last element of +\\ vec equals +/ vec.
        Running fold's final value is the same as the complete fold.
        """
        vec  = QVector.from_items(items, "j")
        running = scan(q_add, vec)
        total   = over(q_add, vec)
        assert isinstance(running, QVector)
        assert running.series[-1] == total.value

    @given(st.lists(SAFE_INT, min_size=1, max_size=30))
    def test_scan_add_length_preserved(self, items):
        """+\\ vec has the same length as vec."""
        vec     = QVector.from_items(items, "j")
        running = scan(q_add, vec)
        assert len(running.series) == len(items)

    @given(st.lists(SAFE_INT, min_size=1, max_size=30))
    def test_scan_add_monotone_for_positive(self, items):
        """For a vector of non-negative integers, the running sum is non-decreasing."""
        pos_items = [abs(v) for v in items]
        vec     = QVector.from_items(pos_items, "j")
        running = scan(q_add, vec)
        vals    = running.series.to_list()
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]

    # ── Aggregation verbs ─────────────────────────────────────────────────────

    @given(st.lists(SAFE_INT, min_size=1, max_size=100))
    def test_sum_equals_python_sum(self, items):
        """q_sum of an integer vector equals Python's built-in sum."""
        vec = QVector.from_items(items, "j")
        assert q_sum(vec).value == sum(items)

    @given(st.lists(SAFE_INT, min_size=1, max_size=100))
    def test_min_equals_python_min(self, items):
        """q_min of an integer vector equals Python's built-in min."""
        vec = QVector.from_items(items, "j")
        assert q_min(vec).value == min(items)

    @given(st.lists(SAFE_INT, min_size=1, max_size=100))
    def test_max_equals_python_max(self, items):
        """q_max of an integer vector equals Python's built-in max."""
        vec = QVector.from_items(items, "j")
        assert q_max(vec).value == max(items)

    @given(st.lists(SAFE_INT, min_size=1, max_size=100))
    def test_min_leq_max(self, items):
        """q_min <= q_max for any non-empty vector."""
        vec = QVector.from_items(items, "j")
        assert q_min(vec).value <= q_max(vec).value

    @given(st.lists(SAFE_INT, min_size=1, max_size=100))
    def test_avg_in_range(self, items):
        """q_avg is always between q_min and q_max."""
        vec = QVector.from_items(items, "j")
        lo  = q_min(vec).value
        hi  = q_max(vec).value
        avg = q_avg(vec).value
        assert lo <= avg <= hi

    @given(st.lists(SAFE_INT, min_size=1, max_size=100))
    def test_sum_over_add_consistency(self, items):
        """q_sum and over(q_add, …) give the same result."""
        vec = QVector.from_items(items, "j")
        assert q_sum(vec).value == over(q_add, vec).value


# ═════════════════════════════════════════════════════════════════════════════
# End-to-end transpiler execution tests
# ═════════════════════════════════════════════════════════════════════════════

class TestKnownGaps:
    """End-to-end transpiler execution tests (formerly xfail; QAtom gap is now fixed)."""

    @given(SAFE_INT, SAFE_INT)
    def test_transpiled_int_arithmetic_executes(self, a, b):
        """
        End-to-end: transpile `a+b` and exec the result.
        Currently fails because transpiled literals are bare Python ints.
        Once the transpiler wraps literals in QAtom(), this test should pass
        — at which point it should be promoted out of xfail.
        """
        import polarq  # noqa: F401 — needed in exec namespace
        from polarq.verbs import q_add as _q_add  # noqa: F401

        t   = QToPythonTranspiler()
        mod = t.transpile(make_script(Assign("result", BinOp("+", IntLit(a), IntLit(b)))))
        ns  = {name: getattr(polarq, name) for name in polarq.__all__}
        exec(compile(mod, "<test>", "exec"), ns)
        assert ns["result"].value == a + b
