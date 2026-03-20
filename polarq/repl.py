"""
polarq REPL — interactive q/Python shell.

Behaviour
─────────
  Starts in **q mode** with the prompt ``q) ``.
  Every line is parsed as q and evaluated via evalq().

  Switching modes
  ~~~~~~~~~~~~~~~
  ``\\``      — enter Python mode (prompt changes to ``>>> ``).
               In Python mode every line is evaluated by Python's built-in
               ``eval``/``exec``.  The q environment is available as ``env``.
  ``\\q``     — return to q mode from Python mode (or quit if already in q mode).

  Metacommands (q mode only)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ``\\t expr``  — time the q expression (microseconds).
  ``\\l path``  — load a .q file into the current environment.
  ``\\d ns``    — set / print the current default namespace.
  ``\\v``       — list all names visible in the current environment.

  Special inputs
  ~~~~~~~~~~~~~~
  Empty line    — ignored.
  Ctrl-D / EOF  — quit gracefully.
  Ctrl-C        — cancel the current line (stay in current mode).

  Multi-line input
  ~~~~~~~~~~~~~~~~
  A line ending with a ``{`` or containing an unclosed brace continues on the
  next prompt (indented ``   ``) until the braces balance.

Error display
~~~~~~~~~~~~~
  q errors are shown in q style:  ``'type``  ``'length``  etc.
  Python exceptions show the class name and message; no traceback by default
  (set POLARQ_TRACEBACK=1 in the environment to see full tracebacks).
"""

from __future__ import annotations

import code
import os
import sys
import time
import traceback
from typing import Optional

from polarq.env import QEnv
from polarq.errors import QError
from polarq.transpiler import evalq, loadq


# ── Prompts ────────────────────────────────────────────────────────────────

_Q_PROMPT      = "q) "
_Q_CONTINUE    = "   "   # continuation (open brace)
_PY_PROMPT     = ">>> "
_PY_CONTINUE   = "... "


# ── Result display ────────────────────────────────────────────────────────

def _display(value) -> None:
    """Print a q result value.  None (no-result statements) is silent."""
    if value is None:
        return
    print(value)


# ── Brace-balance helper for multi-line input ─────────────────────────────

def _brace_depth(line: str) -> int:
    """Return net open-brace count in line (ignoring string contents)."""
    depth = 0
    in_str = False
    for ch in line:
        if ch == '"' and not in_str:
            in_str = True
        elif ch == '"' and in_str:
            in_str = False
        elif not in_str:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
    return depth


def _read_block(first_line: str, prompt_cont: str) -> Optional[str]:
    """
    If *first_line* has unclosed braces, keep reading continuation lines
    until the braces balance.  Returns the complete block, or None if the
    user hit Ctrl-D mid-block.
    """
    lines = [first_line]
    depth = _brace_depth(first_line)
    while depth > 0:
        try:
            cont = input(prompt_cont)
        except EOFError:
            print()
            return None
        lines.append(cont)
        depth += _brace_depth(cont)
    return "\n".join(lines)


# ── q metacommand handling ─────────────────────────────────────────────────

def _handle_meta(cmd: str, env: QEnv) -> bool:
    """
    Handle a backslash metacommand.  Returns True if handled, False if the
    line should be passed to evalq unchanged (e.g. plain ``\\`` mode switch).
    """
    stripped = cmd.strip()

    # \t expr  — timing
    if stripped.startswith("\\t "):
        expr = stripped[3:].strip()
        if not expr:
            print("usage: \\t <expression>")
            return True
        start = time.perf_counter()
        try:
            result = evalq(expr, env)
        except Exception as exc:
            _show_q_error(exc)
            return True
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        print(f"{elapsed_us:.1f}")
        _display(result)
        return True

    # \l path  — load script
    if stripped.startswith("\\l "):
        path = stripped[3:].strip()
        try:
            loadq(path, env)
            print(f"loaded {path}")
        except FileNotFoundError:
            print(f"'os\n{path}: file not found")
        except Exception as exc:
            _show_q_error(exc)
        return True

    # \d ns  — set or show default namespace
    if stripped.startswith("\\d"):
        rest = stripped[2:].strip()
        if not rest:
            # show current namespace (stored as _namespace in env)
            try:
                ns = env.get("_namespace")
            except KeyError:
                ns = "."
            print(ns)
        else:
            env.set_global("_namespace", rest)
            print(rest)
        return True

    # \v  — list visible names (excluding private/runtime)
    if stripped == "\\v":
        names = [k for k in env.keys() if not k.startswith("_")]
        if names:
            print("  ".join(sorted(names)))
        return True

    return False


# ── Error display ──────────────────────────────────────────────────────────

def _show_q_error(exc: Exception) -> None:
    """Display an exception in q-style (short) or with traceback if POLARQ_TRACEBACK=1)."""
    if os.environ.get("POLARQ_TRACEBACK") == "1":
        traceback.print_exc()
        return
    if isinstance(exc, QError):
        # q style: 'type  'length  etc.
        kind = type(exc).__name__.replace("Q", "").replace("Error", "").lower()
        msg  = str(exc)
        # strip the "parse error at …" prefix for parse errors to keep it short
        if " — " in msg:
            msg = msg.split(" — ", 1)[1]
        print(f"'{kind}\n{msg}")
    else:
        print(f"{type(exc).__name__}: {exc}")


# ── Python sub-REPL ───────────────────────────────────────────────────────

def _python_repl(env: QEnv) -> None:
    """
    Drop into a Python interactive session.
    ``env`` is injected as a local so the user can inspect q bindings.
    Exit with ``\\q`` on a line by itself, or Ctrl-D.
    """
    # Build a namespace that mirrors env's bindings + the env itself
    ns: dict = {"env": env}
    for name in env.keys():
        try:
            ns[name] = env.get(name)
        except Exception:
            pass

    print("  (Python mode — type \\q or Ctrl-D to return to q)")

    while True:
        # Read one Python statement (handle continuations with code.compile_command)
        try:
            line = input(_PY_PROMPT)
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue

        if line.strip() in ("\\q", "\\"):
            break

        # Accumulate until we have a complete Python statement
        source = line
        while True:
            try:
                compiled = code.compile_command(source)
            except SyntaxError as exc:
                print(f"SyntaxError: {exc}")
                source = ""
                break
            if compiled is not None:
                break
            # Incomplete — read more
            try:
                more = input(_PY_CONTINUE)
            except EOFError:
                print()
                source = ""
                break
            source = source + "\n" + more

        if not source.strip():
            continue

        try:
            result = eval(compile(source, "<repl>", "eval"), ns)  # noqa: S307
            if result is not None:
                print(repr(result))
        except SyntaxError:
            try:
                exec(compile(source, "<repl>", "exec"), ns)  # noqa: S102
            except Exception as exc:
                if os.environ.get("POLARQ_TRACEBACK") == "1":
                    traceback.print_exc()
                else:
                    print(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            if os.environ.get("POLARQ_TRACEBACK") == "1":
                traceback.print_exc()
            else:
                print(f"{type(exc).__name__}: {exc}")

    # Write any new Python-side names back to the q env
    for key, val in ns.items():
        if not key.startswith("_") and key != "env":
            try:
                env.set_global(key, val)
            except Exception:
                pass


# ── Main REPL loop ────────────────────────────────────────────────────────

def repl(env: Optional[QEnv] = None) -> None:
    """
    Start the polarq interactive shell.

    Parameters
    ----------
    env:
        Optional pre-populated QEnv.  A fresh one is created if not supplied.
    """
    if env is None:
        env = QEnv()

    _print_banner()

    while True:
        # ── Read ──────────────────────────────────────────────────────────
        try:
            line = input(_Q_PROMPT)
        except EOFError:
            print("\n\\")   # q-style exit indicator
            break
        except KeyboardInterrupt:
            print()
            continue

        stripped = line.strip()

        # ── Empty line ────────────────────────────────────────────────────
        if not stripped:
            continue

        # ── Mode switch: bare \ → Python sub-REPL ────────────────────────
        if stripped == "\\":
            _python_repl(env)
            continue

        # ── Quit: \q ─────────────────────────────────────────────────────
        if stripped == "\\q":
            print("\\")
            break

        # ── Metacommands (\t \l \d \v) ───────────────────────────────────
        if stripped.startswith("\\") and _handle_meta(stripped, env):
            continue

        # ── Multi-line block (open brace) ─────────────────────────────────
        if _brace_depth(stripped) > 0:
            full = _read_block(stripped, _Q_CONTINUE)
            if full is None:
                continue
            stripped = full

        # ── Evaluate as q ─────────────────────────────────────────────────
        try:
            result = evalq(stripped, env)
            _display(result)
        except NotImplementedError as exc:
            print(f"nyi\n{exc}")
        except QError as exc:
            _show_q_error(exc)
        except Exception as exc:
            _show_q_error(exc)


# ── Banner ────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    print("polarq  —  q/kdb+ on Polars")
    print("  \\    enter Python mode     \\q   quit")
    print("  \\t   time expression       \\l   load script")
    print("  \\d   namespace             \\v   list names")
    print()


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """Console-script entry point: ``polarq`` command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="polarq",
        description="polarq — q/kdb+ semantics on Polars",
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Optional .q file to load before starting the REPL",
    )
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="Execute script and exit without starting the REPL",
    )
    args = parser.parse_args()

    env = QEnv()

    if args.script:
        try:
            loadq(args.script, env)
        except Exception as exc:
            print(f"error loading {args.script}: {exc}", file=sys.stderr)
            sys.exit(1)
        if args.no_repl:
            return

    repl(env)


if __name__ == "__main__":
    main()
