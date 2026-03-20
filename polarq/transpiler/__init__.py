"""
polarq.transpiler — q AST → Python ast → CPython bytecode.

Phase 5 of the development roadmap.  The public surface is:

    from polarq.transpiler import load_q, to_source, quick_eval
"""

from polarq.transpiler.loader import load_q, quick_eval  # noqa: F401

__all__ = ["load_q", "quick_eval"]
