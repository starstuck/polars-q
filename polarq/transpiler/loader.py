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
    Parse and evaluate a single q expression string.
    Returns the result of the last statement.
    """
    from polarq.parser.ast_nodes import Script
    node = parse_expr(source)
    if not isinstance(node, Script):
        from polarq.parser.ast_nodes import Script as S
        script = S((node,))
    else:
        script = node
    return _exec_script(script, env)


def _exec_script(script, env: QEnv):
    """Transpile script to Python and exec in a namespace derived from env."""
    t   = QToPythonTranspiler()
    mod = t.transpile(script)
    code = compile(mod, "<polarq>", "exec")
    # Build a namespace that has access to the env's bindings
    ns = {k: env.get(k) for k in env.keys() if _safe_get(env, k)}
    # Import all polarq runtime symbols into the namespace
    import polarq
    for name in polarq.__all__:
        ns[name] = getattr(polarq, name)
    exec(code, ns)
    # Write results back to env
    for key, val in ns.items():
        if not key.startswith("_") and key not in dir(__builtins__):
            try:
                env.set_global(key, val)
            except Exception:
                pass
    # Return the value of the last assignment or expression
    stmts = script.stmts
    if stmts:
        from polarq.parser.ast_nodes import Assign
        last = stmts[-1]
        if isinstance(last, Assign):
            try:
                return env.get(last.name)
            except KeyError:
                pass
    return None


def _safe_get(env: QEnv, key: str):
    try:
        env.get(key)
        return True
    except Exception:
        return False
