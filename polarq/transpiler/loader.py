"""
loadq(path, env)  — parse + transpile + exec a .q file into a QEnv.
evalq(source, env) — parse + transpile + exec a q expression string.
"""

from __future__ import annotations
import ast as py_ast

from polarq.env import QEnv
from polarq.parser import parse, parse_expr
from polarq.transpiler.transpiler import QToPythonTranspiler


def loadq(path: str, env: QEnv) -> None:
    """Load a .q file, transpile to Python, and exec into env."""
    with open(path) as fh:
        source = fh.read()
    script = parse(source)
    _exec_script(script, env)


def evalq(source: str, env: QEnv):
    """
    Parse and evaluate a q source string (one or more statements).
    Returns the value of the last assignment, or None for expression-only scripts.
    """
    script = parse(source)
    return _exec_script(script, env)


_RESULT_VAR = "__polarq_result__"


def _exec_script(script, env: QEnv):
    """Transpile script to Python and exec in a namespace derived from env."""
    t   = QToPythonTranspiler()
    mod = t.transpile(script)

    # If the last generated statement is a bare expression (py_ast.Expr),
    # rewrite it to  __polarq_result__ = <expr>  so we can read the value back.
    from polarq.parser.ast_nodes import Assign as QAssign
    last_q = script.stmts[-1] if script.stmts else None
    capture_result = last_q is not None and not isinstance(last_q, QAssign)
    if capture_result:
        last_py = mod.body[-1]
        if isinstance(last_py, py_ast.Expr):
            mod.body[-1] = py_ast.Assign(
                targets=[py_ast.Name(id=_RESULT_VAR, ctx=py_ast.Store())],
                value=last_py.value,
                lineno=0,
                col_offset=0,
            )
            py_ast.fix_missing_locations(mod)

    code = compile(mod, "<polarq>", "exec")

    # Build a namespace seeded with env bindings + polarq runtime
    ns = {k: env.get(k) for k in env.keys() if _safe_get(env, k)}
    import polarq
    _runtime_names = set(polarq.__all__) | {"pl"}
    for name in _runtime_names:
        if hasattr(polarq, name):
            ns[name] = getattr(polarq, name)
    # polars is imported as `pl` by the transpiler
    import polars as _pl
    ns["pl"] = _pl

    exec(code, ns)  # noqa: S102

    # Write new/updated names back to env.
    # Skip: dunder names, builtins, and names that were pre-seeded as runtime.
    _builtins = set(dir(__builtins__))
    _skip = _runtime_names | _builtins
    for key, val in ns.items():
        if not key.startswith("_") and key not in _skip:
            try:
                env.set_global(key, val)
            except Exception:
                pass

    # Return captured result
    if capture_result:
        return ns.get(_RESULT_VAR)
    if last_q is not None and isinstance(last_q, QAssign):
        try:
            return env.get(last_q.name)
        except KeyError:
            pass
    return None


def _safe_get(env: QEnv, key: str):
    try:
        env.get(key)
        return True
    except Exception:
        return False
