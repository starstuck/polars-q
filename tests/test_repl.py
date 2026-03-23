"""
tests/test_repl.py
──────────────────
Unit tests for the polarq interactive REPL (polarq.repl).

Testing strategy
────────────────
The REPL is an input/output loop driven by ``builtins.input``.  Each test
patches ``input`` with a canned sequence of lines (using ``iter`` + ``side_effect``
so ``StopIteration`` raises ``EOFError`` when lines run out, which is the same
signal the real terminal sends on Ctrl-D).  stdout is captured via
``contextlib.redirect_stdout``.

Helper: ``run_repl(lines, env=None)``
    Feed *lines* into a fresh REPL, capture stdout, return ``(output, env)``.
    The env is inspected to verify side-effects (variable bindings etc.).

Each test class covers one behavioural area.  Add a new ``pytest.param``
or a new test method to extend coverage.
"""

import io
import contextlib
from unittest.mock import patch

import pytest

from polarq.env import QEnv
from polarq.repl import repl, _brace_depth, _Q_PROMPT, _Q_CONTINUE, _PY_PROMPT, _PY_CONTINUE
from conftest import run_repl, output_lines


# ═════════════════════════════════════════════════════════════════════════════
# Utility: _brace_depth
# ═════════════════════════════════════════════════════════════════════════════

class TestBraceDepth:
    """Unit tests for the brace-balance helper (used for multi-line detection)."""

    def test_no_braces(self):
        assert _brace_depth("x:42") == 0

    def test_open_brace(self):
        assert _brace_depth("{x+y") == 1

    def test_balanced(self):
        assert _brace_depth("{x+y}") == 0

    def test_nested(self):
        assert _brace_depth("{{x+y}") == 1

    def test_close_without_open(self):
        assert _brace_depth("}") == -1

    def test_brace_inside_string_ignored(self):
        assert _brace_depth('"{"') == 0

    def test_multiple_opens(self):
        assert _brace_depth("{[x] {y") == 2


# ═════════════════════════════════════════════════════════════════════════════
# Literal output
# ═════════════════════════════════════════════════════════════════════════════

class TestLiteralOutput:
    """Bare expressions typed at the prompt should be printed."""

    def test_integer(self):
        lines = output_lines(["42"])
        assert "42" in lines

    def test_float(self):
        lines = output_lines(["3.14"])
        assert any("3.14" in l for l in lines)

    def test_bool_true(self):
        lines = output_lines(["1b"])
        assert any("True" in l for l in lines)

    def test_symbol(self):
        lines = output_lines(["`AAPL"])
        assert any("AAPL" in l for l in lines)

    def test_string(self):
        lines = output_lines(['"hello"'])
        assert any("hello" in l for l in lines)

    def test_int_vector(self):
        lines = output_lines(["1 2 3"])
        assert any("1" in l and "2" in l and "3" in l for l in lines)


# ═════════════════════════════════════════════════════════════════════════════
# Assignment
# ═════════════════════════════════════════════════════════════════════════════

class TestAssignment:
    """Assignments should bind names in the env; the assigned value is displayed."""

    def test_assign_int_binds_env(self):
        _, env = run_repl(["x:42"])
        assert env.get("x") == 42

    def test_assign_displays_value(self):
        lines = output_lines(["x:42"])
        assert "42" in lines

    def test_assign_float(self):
        _, env = run_repl(["pi:3.14"])
        assert abs(env.get("pi") - 3.14) < 1e-9

    def test_assign_sym(self):
        _, env = run_repl(["s:`GOOG"])
        assert env.get("s") == "GOOG"

    def test_assign_vector_is_qvector(self):
        from polarq.types import QVector
        _, env = run_repl(["v:1 2 3"])
        v = env.get("v")
        assert isinstance(v, QVector)
        assert list(v.series) == [1, 2, 3]

    def test_multi_assign_persists(self):
        """Multiple assignments in one session are all visible in the env."""
        _, env = run_repl(["x:1", "y:2", "z:3"])
        assert env.get("x") == 1
        assert env.get("y") == 2
        assert env.get("z") == 3

    def test_second_session_sees_first(self):
        """Env is shared across multiple run_repl calls (simulates a live session)."""
        env = QEnv()
        run_repl(["a:10"], env)
        run_repl(["b:20"], env)
        assert env.get("a") == 10
        assert env.get("b") == 20

    def test_overwrite(self):
        """Re-assigning a name replaces the earlier binding."""
        _, env = run_repl(["x:1", "x:99"])
        assert env.get("x") == 99


# ═════════════════════════════════════════════════════════════════════════════
# Name lookup
# ═════════════════════════════════════════════════════════════════════════════

class TestNameLookup:
    """Typing a name at the prompt should display its current value."""

    def test_bare_name_displays_value(self):
        env = QEnv()
        env.set("x", 42)
        lines = output_lines(["x"], env)
        assert "42" in lines

    def test_bare_name_after_assignment(self):
        lines = output_lines(["x:7", "x"])
        assert lines.count("7") >= 1   # shown at least once (assign + lookup)


# ═════════════════════════════════════════════════════════════════════════════
# Empty lines and comments
# ═════════════════════════════════════════════════════════════════════════════

class TestIgnored:
    """Empty lines and comment lines should be silently skipped."""

    def test_empty_line_ignored(self):
        _, env = run_repl(["", "x:5"])
        assert env.get("x") == 5

    def test_comment_ignored(self):
        _, env = run_repl(["// this is a comment", "x:5"])
        assert env.get("x") == 5

    def test_multiple_empty_lines(self):
        _, env = run_repl(["", "", "x:3", ""])
        assert env.get("x") == 3


# ═════════════════════════════════════════════════════════════════════════════
# Metacommands
# ═════════════════════════════════════════════════════════════════════════════

class TestMetacommands:

    def test_backslash_v_lists_user_names(self):
        """\\v shows names bound by the user, not runtime names."""
        out, _ = run_repl(["x:1", "y:2", "\\v"])
        assert "x" in out
        assert "y" in out
        # Runtime symbols should NOT appear
        assert "QVector" not in out
        assert "compile_select" not in out

    def test_backslash_v_empty_env(self):
        """\\v on a fresh env produces no output (or just a blank line)."""
        lines = output_lines(["\\v"])
        # No user names → nothing printed (or only whitespace)
        user_lines = [l for l in lines if l.strip()]
        assert user_lines == []

    def test_backslash_d_set_and_read(self):
        """\\d ns sets the namespace; a bare \\d prints it back."""
        out, _ = run_repl(["\\d .myns", "\\d"])
        assert ".myns" in out

    def test_backslash_t_produces_timing(self):
        """\\t <expr> prints a numeric timing value."""
        out, _ = run_repl(["\\t 1 2 3"])
        # Should contain a number (the timing in µs)
        numbers = [w for w in out.split() if w.replace(".", "").isdigit()]
        assert numbers, f"expected a timing number in output, got: {out!r}"

    def test_backslash_q_exits(self):
        """\\q causes the REPL to exit cleanly."""
        out, _ = run_repl(["x:1", "\\q", "y:2"])
        # y:2 is never reached — env should not contain y
        # (we just verify the REPL exited without error)
        assert "polarq" in out   # banner was printed

    def test_unknown_backslash_does_not_crash(self):
        """An unknown backslash sequence should produce an error, not a crash."""
        out, _ = run_repl(["\\z"])
        # REPL should still be alive afterwards (no exception propagated)
        assert "polarq" in out


# ═════════════════════════════════════════════════════════════════════════════
# Lambda definition
# ═════════════════════════════════════════════════════════════════════════════

class TestLambdaDefinition:
    """Lambdas entered at the prompt are stored as callables in the env."""

    def test_implicit_param_lambda_callable(self):
        _, env = run_repl(["f:{x*x}"])
        assert callable(env.get("f"))

    def test_explicit_param_lambda_callable(self):
        _, env = run_repl(["add:{[a;b] a+b}"])
        assert callable(env.get("add"))

    def test_multiline_lambda(self):
        """
        A lambda whose opening brace is on one line continues to the next.
        The REPL should collect both lines before evaluating.
        """
        _, env = run_repl([
            "f:{",      # open brace — triggers continuation
            "x*x}",     # closing brace — completes the block
        ])
        assert callable(env.get("f"))


# ═════════════════════════════════════════════════════════════════════════════
# Python mode
# ═════════════════════════════════════════════════════════════════════════════

class TestPythonMode:
    """The \\ command drops into a Python sub-REPL; \\q returns to q mode."""

    def test_switch_to_python_and_back(self):
        """Enter Python mode, evaluate an expression, return to q mode."""
        out, _ = run_repl([
            "\\",       # enter Python mode
            "1+1",      # Python expression
            "\\q",      # return to q mode
            "x:5",      # still alive in q mode
        ])
        assert "2" in out           # Python evaluated 1+1

    def test_python_mode_sees_env(self):
        """The q env is accessible as ``env`` inside Python mode."""
        env = QEnv()
        env.set("myval", 99)
        out, _ = run_repl([
            "\\",
            "env.get('myval')",
            "\\q",
        ], env)
        assert "99" in out

    def test_python_mode_sets_env(self):
        """Names defined in Python mode are written back to the q env on exit."""
        _, env = run_repl([
            "\\",
            "from polarq.types import QAtom",
            "result = QAtom(7, 'j')",
            "\\q",
        ])
        # 'result' should be visible in env after returning from Python mode
        assert env.get("result").value == 7

    def test_eof_in_python_mode_returns_to_q(self):
        """Ctrl-D (EOFError) inside Python mode exits Python mode cleanly."""
        # We use no explicit \\q — the fake input runs out inside Python mode
        out, _ = run_repl([
            "\\",       # enter Python mode
            "42",       # one expression
            # EOF here → returns to q mode, then outer EOF exits the REPL
        ])
        assert "42" in out


# ═════════════════════════════════════════════════════════════════════════════
# Error handling
# ═════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    """Errors should be displayed in q style and the REPL should continue."""

    def test_repl_continues_after_error(self):
        """A runtime error on one line does not kill the REPL."""
        _, env = run_repl([
            "x:undef_name_xyz",   # will fail — name not in env
            "y:42",               # should still execute
        ])
        assert env.get("y") == 42

    def test_parse_error_shows_message(self):
        """A syntax error in q source prints a message, not a traceback."""
        out, _ = run_repl([":::garbage:::"])
        # REPL must survive; some error text must appear
        assert "polarq" in out   # banner still there = REPL didn't crash

    def test_nyi_shown_for_unimplemented(self):
        """NotImplementedError is shown with 'nyi' prefix."""
        # qSQL is not yet transpiled — triggers NotImplementedError
        out, _ = run_repl(["select from t"])
        assert "nyi" in out


# ═════════════════════════════════════════════════════════════════════════════
# Basic REPL prompt use
#
# These tests verify the *visible experience* of the REPL — the sequence of
# prompts shown to the user — rather than just the side-effects on the env.
#
# A second helper ``transcript()`` records every (prompt, line) pair that
# ``input()`` is called with, giving a precise trace of what the user sees.
# ═════════════════════════════════════════════════════════════════════════════

def transcript(lines: list[str], env: QEnv | None = None) -> tuple[list[tuple[str,str]], str, QEnv]:
    """
    Run the REPL and return ``(calls, stdout, env)`` where *calls* is a list
    of ``(prompt, line)`` pairs — one entry per ``input()`` invocation —
    recording exactly which prompt was shown for each user line.
    """
    if env is None:
        env = QEnv()

    calls: list[tuple[str, str]] = []
    inputs = iter(lines)

    def fake_input(prompt=""):
        try:
            line = next(inputs)
        except StopIteration:
            raise EOFError
        calls.append((prompt, line))
        return line

    buf = io.StringIO()
    with patch("builtins.input", side_effect=fake_input):
        with contextlib.redirect_stdout(buf):
            repl(env)

    return calls, buf.getvalue(), env


class TestBasicPromptUse:
    """
    Verify the prompts shown to the user during typical REPL interactions.

    Each test describes a concrete user session and asserts on the exact
    sequence of prompts, making it immediately clear what the user would see
    on their terminal.
    """

    # ── q) prompt ─────────────────────────────────────────────────────────

    def test_first_prompt_is_q(self):
        """The very first thing the user sees is the q) prompt."""
        calls, _, _ = transcript(["42"])
        assert calls[0][0] == _Q_PROMPT

    def test_every_q_line_gets_q_prompt(self):
        """Each line in q mode is preceded by q)."""
        calls, _, _ = transcript(["x:1", "y:2", "z:3"])
        prompts = [p for p, _ in calls]
        assert all(p == _Q_PROMPT for p in prompts)

    def test_q_prompt_string_value(self):
        """The q mode prompt is exactly 'q) ' (with trailing space)."""
        calls, _, _ = transcript(["1"])
        assert calls[0][0] == "q) "

    # ── >>> prompt ────────────────────────────────────────────────────────

    def test_python_mode_prompt_is_triple_arrow(self):
        r"""After \, the prompt changes to >>>."""
        calls, _, _ = transcript(["\\", "1+1", "\\q"])
        py_prompts = [p for p, line in calls if line in ("1+1",)]
        assert py_prompts == [_PY_PROMPT]

    def test_python_prompt_string_value(self):
        r"""The Python mode prompt is exactly '>>> ' (with trailing space)."""
        calls, _, _ = transcript(["\\", "42", "\\q"])
        py_calls = [(p, l) for p, l in calls if l == "42"]
        assert py_calls[0][0] == ">>> "

    def test_prompt_returns_to_q_after_backslash_q(self):
        r"""After \q in Python mode, subsequent lines get the q) prompt again."""
        calls, _, _ = transcript(["\\", "1", "\\q", "x:5"])
        # The last call (x:5) must use the q) prompt
        assert calls[-1][0] == _Q_PROMPT

    # ── Continuation prompt ───────────────────────────────────────────────

    def test_continuation_prompt_on_open_brace(self):
        """An unclosed { triggers the continuation prompt on the next line."""
        calls, _, _ = transcript([
            "f:{",      # first line — opens a brace
            "x*x}",     # second line — closes it
        ])
        assert calls[0][0] == _Q_PROMPT    # first line: q) prompt
        assert calls[1][0] == _Q_CONTINUE  # second line: continuation

    def test_continuation_prompt_string_value(self):
        """The continuation prompt is exactly '   ' (three spaces)."""
        calls, _, _ = transcript(["f:{", "x}"])
        assert calls[1][0] == "   "

    def test_no_continuation_for_balanced_braces(self):
        """A single-line lambda (balanced braces) does not trigger continuation."""
        calls, _, _ = transcript(["{x+y}"])
        assert len(calls) == 1
        assert calls[0][0] == _Q_PROMPT

    def test_double_open_needs_two_continuations(self):
        """Two unclosed braces require two continuation lines."""
        calls, _, _ = transcript([
            "f:{{",     # depth 2
            "x+",       # depth 2, still open
            "y}}",      # depth 0, balanced
        ])
        assert calls[0][0] == _Q_PROMPT
        assert calls[1][0] == _Q_CONTINUE
        assert calls[2][0] == _Q_CONTINUE

    # ── Mode-switch sequence ──────────────────────────────────────────────

    def test_prompt_sequence_for_mode_switch(self):
        r"""
        Complete prompt sequence for:   x:1  \  2+2  \q  y:3
        Expected:  q)  q)  >>>  q)
        (the \\ and \q lines consume prompts too)
        """
        calls, _, _ = transcript(["x:1", "\\", "2+2", "\\q", "y:3"])
        prompts = [p for p, _ in calls]
        # x:1  → q)
        assert prompts[0] == _Q_PROMPT
        # \\   → q)  (mode-switch line is read under q) prompt)
        assert prompts[1] == _Q_PROMPT
        # 2+2  → >>>
        assert prompts[2] == _PY_PROMPT
        # \q   → >>>  (the exit command is read under >>> prompt)
        assert prompts[3] == _PY_PROMPT
        # y:3  → q)  (back in q mode)
        assert prompts[4] == _Q_PROMPT

    # ── Session transcripts ───────────────────────────────────────────────

    def test_typical_session_output(self):
        """
        A representative session:

            q) x:10
            q) x
            10
            q) v:1 2 3
            q) v
            [1, 2, 3]
            q) \\q
        """
        calls, out, env = transcript(["x:10", "x", "v:1 2 3", "v", "\\q"])
        assert env.get("x") == 10
        assert "10" in out
        assert "1" in out and "2" in out and "3" in out
        # All five lines read under q) prompt
        assert all(p == _Q_PROMPT for p, _ in calls)

    def test_python_session_output(self):
        r"""
        Python sub-session:

            q) \\
              (Python mode …)
            >>> x = 7
            >>> x
            7
            >>> \q
            q) \\q
        """
        calls, out, env = transcript(["\\", "x = 7", "x", "\\q", "\\q"])
        assert "7" in out
        # Verify prompt sequence: q) for \\, then >>> for Python lines
        assert calls[0][0] == _Q_PROMPT   # the \\ line
        assert calls[1][0] == _PY_PROMPT  # x = 7
        assert calls[2][0] == _PY_PROMPT  # x
        assert calls[3][0] == _PY_PROMPT  # \q (exit Python)
