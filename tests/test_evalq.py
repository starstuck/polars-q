"""
tests/test_evalq.py
────────────────────
Integration tests for ``evalq`` — the parse → transpile → exec pipeline.

How to add more test cases
──────────────────────────
Each feature area has a parametrised table of ``(q_source, checker)`` pairs.
To add a new case, append a ``pytest.param`` to the relevant table:

    pytest.param(
        "x:99",                          # q source string
        lambda env: env.get("x") == 99,  # predicate on the resulting QEnv
        id="assign-99",                  # short label shown in pytest output
    ),

All cases in ``WORKING_CASES`` are expected to pass today.
Cases in the ``xfail`` tables will turn green once the matching transpiler gap
is resolved (currently: literals need to be wrapped as QAtom before being
passed to runtime verbs).

Structure
─────────
  TestLiteralAssignment   — atoms and vectors bound to names
  TestMultiStatement      — scripts with more than one statement
  TestLambdaDefinition    — {…} lambdas stored in the env
  TestArithmeticXfail     — arithmetic expressions (blocked by QAtom gap)
"""

import pytest
from polarq.env import QEnv
from polarq.transpiler import evalq
from polarq.types import QVector


# ── Helpers ───────────────────────────────────────────────────────────────────

def fresh_env() -> QEnv:
    return QEnv()


def run(source: str) -> QEnv:
    """Eval q source in a fresh environment and return the env for inspection."""
    env = fresh_env()
    evalq(source, env)
    return env


# ═════════════════════════════════════════════════════════════════════════════
# Literal assignment
# ═════════════════════════════════════════════════════════════════════════════

#: Table of (q_source, checker) pairs.
#: Each checker receives the QEnv after evalq and must return True.
LITERAL_ASSIGNMENT_CASES = [
    pytest.param(
        "x:42",
        lambda env: env.get("x") == 42,
        id="int-literal",
    ),
    pytest.param(
        "x:0",
        lambda env: env.get("x") == 0,
        id="int-zero",
    ),
    pytest.param(
        "x:3.14",
        lambda env: abs(env.get("x") - 3.14) < 1e-9,
        id="float-literal",
    ),
    pytest.param(
        "x:1b",
        lambda env: env.get("x") is True,
        id="bool-true",
    ),
    pytest.param(
        "x:0b",
        lambda env: env.get("x") is False,
        id="bool-false",
    ),
    pytest.param(
        "s:`AAPL",
        lambda env: env.get("s") == "AAPL",
        id="symbol",
    ),
    pytest.param(
        's:"hello"',
        lambda env: env.get("s") == "hello",
        id="string",
    ),
    pytest.param(
        's:""',
        lambda env: env.get("s") == "",
        id="empty-string",
    ),
    # ── Vector literals ───────────────────────────────────────────────────────
    pytest.param(
        "v:1 2 3",
        lambda env: (
            isinstance(env.get("v"), QVector)
            and list(env.get("v").series) == [1, 2, 3]
        ),
        id="int-vector",
    ),
    pytest.param(
        "v:1.0 2.0 3.0",
        lambda env: (
            isinstance(env.get("v"), QVector)
            and len(env.get("v").series) == 3
        ),
        id="float-vector",
    ),
    pytest.param(
        "v:1b 0b 1b",
        lambda env: (
            isinstance(env.get("v"), QVector)
            and list(env.get("v").series) == [True, False, True]
        ),
        id="bool-vector",
    ),
    pytest.param(
        "v:`a`b`c",
        lambda env: (
            isinstance(env.get("v"), QVector)
            and len(env.get("v").series) == 3
        ),
        id="sym-vector",
    ),
    pytest.param(
        "v:1 2 3",
        lambda env: env.get("v").kind == "j",
        id="int-vector-kind",
    ),
    pytest.param(
        "v:1.0 2.0",
        lambda env: env.get("v").kind == "f",
        id="float-vector-kind",
    ),
]


class TestLiteralAssignment:
    """evalq correctly stores literal values and vectors in the environment."""

    @pytest.mark.parametrize("source,check", LITERAL_ASSIGNMENT_CASES)
    def test_literal(self, source, check):
        env = run(source)
        assert check(env), f"check failed for: {source!r}"


# ═════════════════════════════════════════════════════════════════════════════
# Multi-statement scripts
# ═════════════════════════════════════════════════════════════════════════════

MULTI_STATEMENT_CASES = [
    pytest.param(
        "x:1\ny:2",
        lambda env: env.get("x") == 1 and env.get("y") == 2,
        id="two-int-assignments",
    ),
    pytest.param(
        "a:10\nb:20\nc:30",
        lambda env: env.get("a") == 10 and env.get("b") == 20 and env.get("c") == 30,
        id="three-assignments",
    ),
    pytest.param(
        "sym:`GOOG\nval:99",
        lambda env: env.get("sym") == "GOOG" and env.get("val") == 99,
        id="mixed-types",
    ),
    pytest.param(
        "v:1 2 3\nn:3",
        lambda env: (
            isinstance(env.get("v"), QVector)
            and env.get("n") == 3
        ),
        id="vector-and-int",
    ),
    # Later assignment overwrites earlier one in the same script
    pytest.param(
        "x:1\nx:2",
        lambda env: env.get("x") == 2,
        id="overwrite-same-name",
    ),
]


class TestMultiStatement:
    """evalq handles newline-separated multi-statement scripts."""

    @pytest.mark.parametrize("source,check", MULTI_STATEMENT_CASES)
    def test_multi(self, source, check):
        env = run(source)
        assert check(env), f"check failed for: {source!r}"

    def test_second_evalq_sees_first_bindings(self):
        """
        Bindings from a first evalq call persist in the env and are
        visible to a second evalq call.
        """
        env = fresh_env()
        evalq("x:10", env)
        evalq("y:20", env)
        assert env.get("x") == 10
        assert env.get("y") == 20

    def test_comment_lines_ignored(self):
        """// comment lines are stripped by the lexer and do not affect the env."""
        env = run("// this is a comment\nx:7")
        assert env.get("x") == 7


# ═════════════════════════════════════════════════════════════════════════════
# Lambda definition
# ═════════════════════════════════════════════════════════════════════════════

LAMBDA_CASES = [
    pytest.param(
        "f:{x}",
        lambda env: callable(env.get("f")),
        id="identity-lambda-callable",
    ),
    pytest.param(
        "double:{x+x}",
        lambda env: callable(env.get("double")),
        id="double-lambda-callable",
    ),
    pytest.param(
        "sq:{x*x}",
        lambda env: callable(env.get("sq")),
        id="square-lambda-callable",
    ),
    pytest.param(
        "add:{[a;b] a+b}",
        lambda env: callable(env.get("add")),
        id="explicit-params-callable",
    ),
    pytest.param(
        "f:{x}\ng:{y}",
        lambda env: callable(env.get("f")) and callable(env.get("g")),
        id="two-lambdas",
    ),
]


class TestLambdaDefinition:
    """evalq stores lambdas as callables in the environment."""

    @pytest.mark.parametrize("source,check", LAMBDA_CASES)
    def test_lambda(self, source, check):
        env = run(source)
        assert check(env), f"check failed for: {source!r}"


# ═════════════════════════════════════════════════════════════════════════════
# Arithmetic — blocked by the QAtom-wrapping gap (xfail)
#
# These tests document expressions that *should* work end-to-end but currently
# fail because the transpiler emits bare Python literals (e.g. q_add(1, 2))
# instead of QAtom-wrapped ones (q_add(QAtom(1,'j'), QAtom(2,'j'))).
#
# To unlock: in QToPythonTranspiler._expr(), change IntLit/FloatLit to emit
#   QAtom(v, 'j') / QAtom(v, 'f') calls instead of plain ast.Constant nodes.
# Once fixed, move the case to the WORKING_CASES tables above.
# ═════════════════════════════════════════════════════════════════════════════

_ARITH_XFAIL_REASON = (
    "Transpiler emits bare Python literals; q runtime verbs expect QAtom. "
    "Fix: wrap IntLit/FloatLit as QAtom(v, kind) in the transpiler."
)

ARITHMETIC_XFAIL_CASES = [
    pytest.param(
        "x:1+2",
        lambda env: env.get("x").value == 3,
        id="int-add",
    ),
    pytest.param(
        "x:10-3",
        lambda env: env.get("x").value == 7,
        id="int-sub",
    ),
    pytest.param(
        "x:3*4",
        lambda env: env.get("x").value == 12,
        id="int-mul",
    ),
    pytest.param(
        "x:10%2",
        lambda env: abs(env.get("x").value - 5.0) < 1e-9,
        id="int-div",
    ),
    pytest.param(
        "x:2+3*4",
        lambda env: env.get("x").value == 14,   # right-to-left: 2+(3*4)
        id="rtl-no-precedence",
    ),
    pytest.param(
        "x:1+2+3+4",
        lambda env: env.get("x").value == 10,
        id="chain-add",
    ),
    pytest.param(
        "x:1.5+2.5",
        lambda env: abs(env.get("x").value - 4.0) < 1e-9,
        id="float-add",
    ),
]


class TestArithmeticXfail:
    """
    Arithmetic expressions that should work but are currently blocked by the
    QAtom-wrapping gap in the transpiler.  All are marked xfail(strict=True):
    they must fail today and will become normal passing tests once the
    transpiler wraps literals correctly.
    """

    @pytest.mark.parametrize("source,check", ARITHMETIC_XFAIL_CASES)
    @pytest.mark.xfail(reason=_ARITH_XFAIL_REASON, strict=True)
    def test_arithmetic(self, source, check):
        env = run(source)
        assert check(env), f"check failed for: {source!r}"
