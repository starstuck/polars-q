"""
tests/conftest.py
─────────────────
Shared test helpers for the polarq test suite.

``run_repl`` and ``output_lines`` are used by both ``test_repl.py`` and
``test_snippets.py``.  pytest auto-discovers this file so no explicit import
path setup is needed — just ``from conftest import run_repl, output_lines``.
"""

import io
import contextlib
from unittest.mock import patch

from polarq.env import QEnv
from polarq.repl import repl


def run_repl(lines: list[str], env: QEnv | None = None) -> tuple[str, QEnv]:
    """
    Run the REPL with a scripted sequence of input lines.
    Returns (captured_stdout, env).

    The last line does NOT need to be ``\\q``; when the fake input is exhausted
    it raises EOFError, which the REPL treats as Ctrl-D (clean exit).
    """
    if env is None:
        env = QEnv()

    inputs = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    buf = io.StringIO()
    with patch("builtins.input", side_effect=fake_input):
        with contextlib.redirect_stdout(buf):
            repl(env)

    return buf.getvalue(), env


def output_lines(lines: list[str], env: QEnv | None = None) -> list[str]:
    """Run the REPL and return non-empty, non-banner output lines."""
    out, _ = run_repl(lines, env)
    result = []
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("polarq") or stripped.startswith("\\"):
            continue
        if stripped == "":
            continue
        result.append(stripped)
    return result
