"""
qSQL sub-grammar helpers.

The actual qSQL parsing lives in pratt.Parser._parse_qsql() — keeping it
there avoids a circular import.  This module re-exports the qSQL AST nodes
and provides a standalone  parse_qsql()  convenience for tests and the REPL.
"""

from polarq.parser.ast_nodes import (   # noqa: F401  (re-export)
    ColExpr, QSelect, QUpdate, QExec, QDelete,
)
from polarq.parser.pratt import parse_expr


def parse_qsql(source: str):
    """Parse a single qSQL statement from source."""
    return parse_expr(source)
