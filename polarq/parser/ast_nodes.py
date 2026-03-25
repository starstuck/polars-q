"""
q AST node definitions.

All nodes are frozen dataclasses so they are hashable and
cannot be accidentally mutated after construction.

Naming convention mirrors q terminology:
  *Lit   — literal value nodes
  Name   — unresolved identifier
  Verb   — operator token (+, -, *, %, ...)
  BinOp  — dyadic verb application  (left verb right)
  MonOp  — monadic verb application (verb right)
  Adverb — verb with adverb suffix  (+/, +\\, f', f/:, f\\:)
  Apply  — function call, bracket or juxtaposition
  Lambda — anonymous function { ... }
  Assign — assignment stmt   name: expr  or  name:: expr
  Script — top-level sequence of statements
  QSelect / QUpdate / QExec / QDelete — qSQL statements
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ── Literals ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IntLit:
    value: int
    kind:  str = "j"   # "j" long (default), "i" int, "h" short

@dataclass(frozen=True)
class FloatLit:
    value: float

@dataclass(frozen=True)
class BoolLit:
    value: bool

@dataclass(frozen=True)
class SymLit:
    """Single symbol: `foo."""
    value: str   # without the backtick

@dataclass(frozen=True)
class StrLit:
    """Character-vector string: "hello"."""
    value: str

@dataclass(frozen=True)
class NullLit:
    """The generic null :: or typed nulls 0N 0n 0Nd etc."""
    kind: str = ""   # "" for generic, "j" for 0N, "f" for 0n, etc.


# ── Composite value nodes ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class VectorLit:
    """
    Homogeneous literal vector formed by juxtaposed atoms of the same type.
    e.g. 1 2 3  or  1.0 2.0 3.0  or  `a`b`c
    items is a tuple of IntLit/FloatLit/BoolLit/SymLit.
    """
    items: tuple

@dataclass(frozen=True)
class ListLit:
    """
    Mixed / general list formed by parenthesised semicolon-delimited exprs.
    e.g.  (1; `a; "hello")
    """
    items: tuple   # tuple of AST nodes


# ── Names and assignment ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Name:
    """Unresolved identifier — simple or dotted (.myns.foo)."""
    name: str

@dataclass(frozen=True)
class Assign:
    """
    name : expr    (local at lambda scope, global at script scope)
    name :: expr   (always global — amend)
    """
    name: str
    expr: Any
    global_: bool = False   # True for ::


# ── Verbs and operations ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Verb:
    """
    A raw verb token in the term-sequence before BinOp/MonOp resolution.
    op is the q operator string: "+", "-", "*", "%", "~", ...
    Named verbs like "sum", "count" are kept as Name nodes.
    """
    op: str

@dataclass(frozen=True)
class BinOp:
    """Dyadic verb application:  left verb right."""
    op: str
    left: Any
    right: Any

@dataclass(frozen=True)
class MonOp:
    """Monadic verb application:  verb right."""
    op: str
    right: Any

@dataclass(frozen=True)
class Adverb:
    """
    Verb + adverb modifier.
    adverb is one of: "/" "\\" "'" "/:" "\\:"
    verb is any AST node (typically a Verb, Name, or Lambda).
    """
    verb: Any
    adverb: str


# ── Function application ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Apply:
    """
    Function call — bracket form f[x;y] or juxtaposition f x.
    args is a tuple of AST nodes; a None entry means a projection gap: f[;1].
    """
    func: Any
    args: tuple   # tuple of AST nodes (or None for gaps)

@dataclass(frozen=True)
class Index:
    """
    Table / vector indexing: t[i]  t[i;j]
    Structurally identical to Apply but semantically distinct.
    The transpiler resolves the distinction based on runtime type.
    (We alias it to Apply at parse time; kept separate for clarity.)
    """
    obj: Any
    args: tuple


# ── Lambda ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Lambda:
    """
    Anonymous function.
    params: explicit parameter list from [x;y;z], or () for implicit {x+y}.
    body: tuple of statement AST nodes; last expression is the return value.
    """
    params: tuple   # tuple of str; empty = implicit x y z
    body: tuple     # tuple of AST nodes


# ── Statements and script ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Script:
    """Top-level sequence of statements (one per line or semicolon-delimited)."""
    stmts: tuple   # tuple of AST nodes


# ── Table literal ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TableLit:
    """
    ([] col:val; ...) — inline table constructor.
    cols is a tuple of (col_name: str, expr: Any) pairs.
    """
    cols: tuple   # ((name, expr), ...)

@dataclass(frozen=True)
class KeyedTableLit:
    """
    ([key_col:val; ...] val_col:val; ...) — keyed table constructor.
    key_cols: tuple of (name, expr) pairs for key columns.
    val_cols: tuple of (name, expr) pairs for value columns.
    """
    key_cols: tuple   # ((name, expr), ...)
    val_cols: tuple   # ((name, expr), ...)


# ── qSQL ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ColExpr:
    """
    A column expression inside qSQL, optionally with an alias.
    e.g.  sum qty           → ColExpr(expr=Apply(Name('sum'), ('qty',)), alias=None)
          total:sum qty     → ColExpr(expr=..., alias='total')
    """
    expr: Any
    alias: str | None = None

@dataclass(frozen=True)
class QSelect:
    """select [cols] [by cols] from table [where conds]"""
    cols:  tuple        # tuple of ColExpr (empty = select all)
    table: Any          # the from-expression
    where: tuple        # tuple of condition expressions
    by:    tuple        # tuple of group-by expressions

@dataclass(frozen=True)
class QUpdate:
    """update col:expr [by cols] from table [where conds]"""
    cols:  tuple
    table: Any
    where: tuple
    by:    tuple

@dataclass(frozen=True)
class QExec:
    """exec [cols] [by cols] from table [where conds]  → flat values, not table"""
    cols:  tuple
    table: Any
    where: tuple
    by:    tuple

@dataclass(frozen=True)
class QDelete:
    """delete [cols] from table [where conds]"""
    cols:  tuple
    table: Any
    where: tuple
