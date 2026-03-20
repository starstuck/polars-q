"""
polarq.parser — q/kdb+ source parser.

Public API::

    from polarq.parser import parse, parse_expr

    ast = parse("x:1+2\\ny:x*3")          # Script node
    node = parse_expr("1+2*3")             # single expression node
"""

from polarq.parser.pratt import parse, parse_expr   # noqa: F401

__all__ = ["parse", "parse_expr"]
