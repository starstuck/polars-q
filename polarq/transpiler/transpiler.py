"""
QToPythonTranspiler — walks a q AST and emits Python ``ast`` nodes.

Phase 5 of the roadmap; currently provides enough coverage for the core
constructs exercised by the tests and the README transpiler output example.

Unsupported constructs raise ``NotImplementedError`` with a helpful message
so that gaps are easy to identify and fill in.
"""

from __future__ import annotations
import ast as py_ast
from typing import Any

from polarq.parser.ast_nodes import (
    IntLit, FloatLit, BoolLit, SymLit, StrLit, NullLit,
    DateLit, TimeLit, TimestampLit, MonthLit,
    VectorLit, ListLit,
    Name, Assign,
    Verb, BinOp, MonOp, Adverb,
    Apply,
    Lambda,
    Script,
    TableLit, KeyedTableLit,
    ColExpr, QSelect, QUpdate, QExec, QDelete,
)

# Maps q aggregate function names to Polars Expr methods
_POLARS_AGG = {
    "avg":   "mean",
    "sum":   "sum",
    "min":   "min",
    "max":   "max",
    "count": "len",
    "first": "first",
    "last":  "last",
}

_Q_ARITH_OP = {
    "+": py_ast.Add,
    "-": py_ast.Sub,
    "*": py_ast.Mult,
    "%": py_ast.Div,
}

_Q_CMP_OP = {
    "=":  py_ast.Eq,
    "<":  py_ast.Lt,
    ">":  py_ast.Gt,
    "<=": py_ast.LtE,
    ">=": py_ast.GtE,
}
from polarq.transpiler.builtins import VERB_MAP, ADVERB_MAP


def _q_unparse(node) -> str:
    """Serialize a q AST node back to a q source string (best-effort)."""
    from polarq.parser.ast_nodes import (
        Lambda, BinOp, MonOp, Name, IntLit, FloatLit, SymLit, StrLit,
        Apply, VectorLit, Assign,
    )
    if isinstance(node, Lambda):
        body_str = ";".join(_q_unparse(s) for s in node.body)
        if node.params:
            return "{[" + ";".join(node.params) + "] " + body_str + "}"
        return "{" + body_str + "}"
    if isinstance(node, BinOp):
        return _q_unparse(node.left) + node.op + _q_unparse(node.right)
    if isinstance(node, MonOp):
        return node.op + _q_unparse(node.right)
    if isinstance(node, Assign):
        return node.name + ":" + _q_unparse(node.expr)
    if isinstance(node, Name):
        return node.name
    if isinstance(node, IntLit):
        suffix = {"j": "", "i": "i", "h": "h"}.get(node.kind, "")
        return str(node.value) + suffix
    if isinstance(node, FloatLit):
        return str(node.value)
    if isinstance(node, SymLit):
        return "`" + node.value
    if isinstance(node, StrLit):
        return '"' + node.value + '"'
    if isinstance(node, VectorLit):
        return " ".join(_q_unparse(i) for i in node.items)
    if isinstance(node, Apply):
        fn_str = _q_unparse(node.func)
        if len(node.args) == 1 and node.args[0] is not None:
            return fn_str + " " + _q_unparse(node.args[0])
        args_str = ";".join("" if a is None else _q_unparse(a) for a in node.args)
        return fn_str + "[" + args_str + "]"
    return "..."


class QToPythonTranspiler:
    """
    Walk a q AST and return a ``ast.Module`` ready for ``compile()``.
    """

    def __init__(self):
        self._lambda_counter = 0

    def transpile(self, script: Script) -> py_ast.Module:
        imports = [
            py_ast.Import(names=[py_ast.alias(name="polars", asname="pl")]),
            py_ast.ImportFrom(
                module="polarq",
                names=[py_ast.alias(name="*")],
                level=0,
            ),
            py_ast.ImportFrom(
                module="polarq.temporal",
                names=[
                    py_ast.alias(name="parse_date_lit"),
                    py_ast.alias(name="parse_time_lit"),
                    py_ast.alias(name="parse_timestamp_lit"),
                    py_ast.alias(name="parse_month_lit"),
                ],
                level=0,
            ),
            py_ast.ImportFrom(
                module="functools",
                names=[py_ast.alias(name="partial", asname="_q_partial")],
                level=0,
            ),
        ]
        body = imports + [self._stmt(s) for s in script.stmts]
        mod = py_ast.Module(body=body, type_ignores=[])
        py_ast.fix_missing_locations(mod)
        return mod

    def to_source(self, script: Script) -> str:
        mod = self.transpile(script)
        return py_ast.unparse(mod)

    # ── Statements ────────────────────────────────────────────────────────────

    def _stmt(self, node: Any) -> py_ast.stmt:
        if isinstance(node, Assign):
            target = py_ast.Name(id=node.name, ctx=py_ast.Store())
            value  = self._expr(node.expr)
            return py_ast.Assign(
                targets=[target], value=value, lineno=0, col_offset=0
            )
        # x:expr inside if/do/while args parses as BinOp(':', Name(x), expr)
        if isinstance(node, BinOp) and node.op == ":" and isinstance(node.left, Name):
            target = py_ast.Name(id=node.left.name, ctx=py_ast.Store())
            value  = self._expr(node.right)
            return py_ast.Assign(
                targets=[target], value=value, lineno=0, col_offset=0
            )
        # if[cond; stmts...]
        if isinstance(node, Apply) and isinstance(node.func, Name) and node.func.name == "if":
            cond = self._expr(node.args[0])
            body = [self._stmt(s) for s in node.args[1:]] or [py_ast.Pass()]
            return py_ast.If(test=cond, body=body, orelse=[], lineno=0, col_offset=0)
        # do[n; stmts...]
        if isinstance(node, Apply) and isinstance(node.func, Name) and node.func.name == "do":
            n_expr = self._expr(node.args[0])
            body   = [self._stmt(s) for s in node.args[1:]] or [py_ast.Pass()]
            return py_ast.For(
                target=py_ast.Name(id="_q_i", ctx=py_ast.Store()),
                iter=py_ast.Call(
                    func=py_ast.Name(id="range", ctx=py_ast.Load()),
                    args=[n_expr], keywords=[],
                ),
                body=body, orelse=[], lineno=0, col_offset=0,
            )
        # while[cond; stmts...]
        if isinstance(node, Apply) and isinstance(node.func, Name) and node.func.name == "while":
            cond = self._expr(node.args[0])
            body = [self._stmt(s) for s in node.args[1:]] or [py_ast.Pass()]
            return py_ast.While(test=cond, body=body, orelse=[], lineno=0, col_offset=0)
        # Any expression as a standalone statement
        return py_ast.Expr(value=self._expr(node), lineno=0, col_offset=0)

    # ── Expressions ──────────────────────────────────────────────────────────

    def _temporal_call(self, fn_name: str, raw: str) -> py_ast.expr:
        """Emit: parse_date_lit('2024.01.15') etc."""
        return py_ast.Call(
            func=py_ast.Name(id=fn_name, ctx=py_ast.Load()),
            args=[py_ast.Constant(value=raw)],
            keywords=[],
        )

    def _qatom_call(self, value: Any, kind: str) -> py_ast.expr:
        """Emit: QAtom(<value>, '<kind>')"""
        return py_ast.Call(
            func=py_ast.Name(id="QAtom", ctx=py_ast.Load()),
            args=[py_ast.Constant(value=value), py_ast.Constant(value=kind)],
            keywords=[],
        )

    def _expr(self, node: Any) -> py_ast.expr:
        match node:
            case IntLit():
                return self._qatom_call(node.value, node.kind)
            case FloatLit(v):
                return self._qatom_call(v, "f")
            case BoolLit(v):
                return self._qatom_call(v, "b")
            case SymLit(v):
                return self._qatom_call(v, "s")
            case StrLit(v):
                return py_ast.Constant(value=v)
            case NullLit():
                return py_ast.Constant(value=None)

            case DateLit(s):
                return self._temporal_call("parse_date_lit", s)
            case TimeLit(s):
                return self._temporal_call("parse_time_lit", s)
            case TimestampLit(s):
                return self._temporal_call("parse_timestamp_lit", s)
            case MonthLit(s):
                return self._temporal_call("parse_month_lit", s)

            case VectorLit(items):
                return self._vector_lit(items)

            case ListLit(items):
                elts = [self._expr(i) for i in items]
                return py_ast.Call(
                    func=py_ast.Name(id="QList", ctx=py_ast.Load()),
                    args=[py_ast.List(elts=elts, ctx=py_ast.Load())],
                    keywords=[],
                )

            case Name(n):
                return py_ast.Name(id=n, ctx=py_ast.Load())

            case BinOp(op, left, right):
                return self._bin_op(op, left, right)

            case MonOp(op, right):
                return self._mon_op(op, right)

            case Apply(func, args):
                # $[cond;t;f] conditional — handled as expression
                if isinstance(func, Verb) and func.op == "$":
                    return self._cond_expr(args)
                return self._apply(func, args)

            case Lambda(params, body):
                return self._lambda(params, body)

            case Adverb(verb, adv):
                return self._adverb(verb, adv)

            case TableLit(cols):
                return self._table_lit(cols)

            case KeyedTableLit(key_cols, val_cols):
                return self._keyed_table_lit(key_cols, val_cols)

            case QSelect() | QUpdate() | QExec() | QDelete():
                return self._qsql(node)

            case _:
                raise NotImplementedError(
                    f"transpiler: unsupported node type {type(node).__name__}: {node!r}"
                )

    def _cond_expr(self, args: tuple) -> py_ast.expr:
        """$[c;t;f] and chained $[c1;t1;c2;t2;f] → nested Python ternary."""
        # args must be odd-length ≥ 3: (c1,t1, c2,t2, ..., else)
        if len(args) < 3 or len(args) % 2 == 0:
            raise NotImplementedError(
                f"$[...] requires an odd number of args ≥ 3, got {len(args)}"
            )
        # Build right-to-left: start with the else branch
        result = self._expr(args[-1])
        # Walk pairs (cond, then) from right to left
        pairs = list(zip(args[::2], args[1::2]))  # [(c1,t1), (c2,t2), ...]
        for cond_node, then_node in reversed(pairs):
            result = py_ast.IfExp(
                test=self._expr(cond_node),
                body=self._expr(then_node),
                orelse=result,
            )
        return result

    def _vector_lit(self, items: tuple) -> py_ast.expr:
        # QVector.from_items([...], kind)
        # NullLit items emit None; all others emit their raw value
        elts = [
            py_ast.Constant(value=None) if isinstance(i, NullLit)
            else py_ast.Constant(value=i.value)
            for i in items
        ]
        first = next((i for i in items if not isinstance(i, NullLit)), None)
        if first is None:
            kind = "j"  # all-null vector defaults to long
        elif isinstance(first, IntLit):
            kind = first.kind
        elif isinstance(first, FloatLit):
            kind = "f"
        elif isinstance(first, BoolLit):
            kind = "b"
        elif isinstance(first, SymLit):
            kind = "s"
        else:
            kind = "j"
        return py_ast.Call(
            func=py_ast.Attribute(
                value=py_ast.Name(id="QVector", ctx=py_ast.Load()),
                attr="from_items",
                ctx=py_ast.Load(),
            ),
            args=[
                py_ast.List(elts=elts, ctx=py_ast.Load()),
                py_ast.Constant(value=kind),
            ],
            keywords=[],
        )

    def _bin_op(self, op: str, left: Any, right: Any) -> py_ast.expr:
        # `sym$y where sym is a multi-char symbol — enumeration cast, not type cast
        if op == "$" and isinstance(left, SymLit) and len(left.value) > 1:
            return py_ast.Call(
                func=py_ast.Name(id="q_enum_cast", ctx=py_ast.Load()),
                args=[
                    py_ast.Name(id=left.value, ctx=py_ast.Load()),
                    self._expr(right),
                ],
                keywords=[],
            )
        # sym?y where sym is a variable name — enum-extend (falls back to find for non-sym)
        if op == "?" and isinstance(left, Name):
            return py_ast.Call(
                func=py_ast.Name(id="q_enum_extend", ctx=py_ast.Load()),
                args=[
                    py_ast.Constant(value=left.name),
                    self._expr(left),
                    self._expr(right),
                ],
                keywords=[],
            )
        ref = VERB_MAP.get(op)
        if ref:
            module, name = ref
            fn = py_ast.Name(id=name, ctx=py_ast.Load())
            return py_ast.Call(
                func=fn,
                args=[self._expr(left), self._expr(right)],
                keywords=[],
            )
        raise NotImplementedError(f"transpiler: unknown dyadic verb {op!r}")

    def _mon_op(self, op: str, right: Any) -> py_ast.expr:
        ref = VERB_MAP.get(op)
        if ref:
            module, name = ref
            fn = py_ast.Name(id=name, ctx=py_ast.Load())
            return py_ast.Call(func=fn, args=[self._expr(right)], keywords=[])
        raise NotImplementedError(f"transpiler: unknown monadic verb {op!r}")

    def _apply(self, func: Any, args: tuple) -> py_ast.expr:
        # Apply(@) / Apply(.) — dispatch based on arity
        if isinstance(func, Verb) and func.op in ("@", "."):
            return self._apply_dot_at(func.op, args)

        # Apply(Adverb(verb, adv), args) → adverb_fn(verb_expr, *arg_exprs)
        if isinstance(func, Adverb):
            ref = ADVERB_MAP.get(func.adverb)
            if not ref:
                raise NotImplementedError(
                    f"transpiler: unknown adverb {func.adverb!r}"
                )
            _, name = ref
            verb_expr = self._expr(func.verb)
            arg_exprs = [
                self._expr(a) if a is not None else py_ast.Constant(value=None)
                for a in args
            ]
            return py_ast.Call(
                func=py_ast.Name(id=name, ctx=py_ast.Load()),
                args=[verb_expr] + arg_exprs,
                keywords=[],
            )

        fn    = self._expr(func)
        pargs = [
            self._expr(a) if a is not None else py_ast.Constant(value=None)
            for a in args
        ]

        # Partial application: Lambda with more params than args supplied
        if isinstance(func, Lambda):
            n_params = len(func.params) if func.params else 3  # implicit x y z
            if len(args) < n_params:
                return py_ast.Call(
                    func=py_ast.Name(id="_q_partial", ctx=py_ast.Load()),
                    args=[fn] + pargs,
                    keywords=[],
                )

        return py_ast.Call(func=fn, args=pargs, keywords=[])

    def _apply_dot_at(self, op: str, args: tuple) -> py_ast.expr:
        """
        Dispatch Apply(@/.) based on arity:
          @[x;i;fn]     → q_amend_at(x, i, fn)   (3-arg amend)
          @[x;i]        → q_at_apply(x, i)        (2-arg index)
          .[f;args;h]   → q_trap(f, args, h)      (3-arg trap)
          .[f;args]     → q_dot_apply(f, args)    (2-arg apply)
        """
        pargs = [self._expr(a) if a is not None else py_ast.Constant(value=None)
                 for a in args]
        if op == "@":
            fn_name = "q_amend_at" if len(args) >= 3 else "q_at_apply"
        else:  # "."
            fn_name = "q_trap" if len(args) >= 3 else "q_dot_apply"
        return py_ast.Call(
            func=py_ast.Name(id=fn_name, ctx=py_ast.Load()),
            args=pargs, keywords=[],
        )

    def _lambda(self, params: tuple, body: tuple) -> py_ast.expr:
        from polarq.parser.ast_nodes import Lambda as LambdaNode
        source = _q_unparse(LambdaNode(params=params, body=body))
        self._lambda_counter += 1
        param_names = params or ("x", "y", "z")
        # Implicit-param lambdas ({x+y}) use default None so they can be called
        # monadically / dyadically without raising TypeError for unused params.
        if params:
            defaults = []
        else:
            defaults = [py_ast.Constant(value=None)] * len(param_names)
        args    = py_ast.arguments(
            posonlyargs=[],
            args=[py_ast.arg(arg=p) for p in param_names],
            vararg=None, kwonlyargs=[], kw_defaults=[],
            kwarg=None, defaults=defaults,
        )
        if len(body) == 1:
            inner = py_ast.Lambda(args=args, body=self._expr(body[0]))
        else:
            # Multi-statement: cannot inline; caller should hoist to def
            raise NotImplementedError(
                "multi-statement lambdas require def-hoisting (not yet implemented)"
            )
        # Wrap in QFn so the function carries its q source for show/value.
        return py_ast.Call(
            func=py_ast.Name(id="QFn", ctx=py_ast.Load()),
            args=[inner, py_ast.Constant(value=source)],
            keywords=[],
        )

    def _adverb(self, verb: Any, adv: str) -> py_ast.expr:
        ref = ADVERB_MAP.get(adv)
        if not ref:
            raise NotImplementedError(f"transpiler: unknown adverb {adv!r}")
        module, name = ref
        return py_ast.Call(
            func=py_ast.Name(id=name, ctx=py_ast.Load()),
            args=[self._expr(verb)],
            keywords=[],
        )

    def _table_lit(self, cols: tuple) -> py_ast.expr:
        """([] col:val; ...) → QTable.from_dataframe(pl.DataFrame({col: q_tbl_col(val), ...}))"""
        keys = [py_ast.Constant(value=name) for name, _ in cols]
        vals = [
            py_ast.Call(
                func=py_ast.Name(id="q_tbl_col", ctx=py_ast.Load()),
                args=[self._expr(expr)], keywords=[],
            )
            for _, expr in cols
        ]
        dict_node = py_ast.Dict(keys=keys, values=vals)
        df_call = py_ast.Call(
            func=py_ast.Attribute(
                value=py_ast.Name(id="pl", ctx=py_ast.Load()),
                attr="DataFrame", ctx=py_ast.Load(),
            ),
            args=[dict_node], keywords=[],
        )
        return py_ast.Call(
            func=py_ast.Attribute(
                value=py_ast.Name(id="QTable", ctx=py_ast.Load()),
                attr="from_dataframe", ctx=py_ast.Load(),
            ),
            args=[df_call], keywords=[],
        )

    def _keyed_table_lit(self, key_cols: tuple, val_cols: tuple) -> py_ast.expr:
        """([key:val] val:val; ...) → QKeyedTable(key_QTable, val_QTable)"""
        def _make_tbl(cols):
            keys = [py_ast.Constant(value=name) for name, _ in cols]
            vals = [
                py_ast.Call(
                    func=py_ast.Name(id="q_tbl_col", ctx=py_ast.Load()),
                    args=[self._expr(expr)], keywords=[],
                )
                for _, expr in cols
            ]
            return py_ast.Call(
                func=py_ast.Attribute(
                    value=py_ast.Name(id="QTable", ctx=py_ast.Load()),
                    attr="from_dataframe", ctx=py_ast.Load(),
                ),
                args=[py_ast.Call(
                    func=py_ast.Attribute(
                        value=py_ast.Name(id="pl", ctx=py_ast.Load()),
                        attr="DataFrame", ctx=py_ast.Load(),
                    ),
                    args=[py_ast.Dict(keys=keys, values=vals)], keywords=[],
                )],
                keywords=[],
            )
        return py_ast.Call(
            func=py_ast.Name(id="QKeyedTable", ctx=py_ast.Load()),
            args=[_make_tbl(key_cols), _make_tbl(val_cols)],
            keywords=[],
        )

    # ── qSQL helpers ─────────────────────────────────────────────────────────

    def _polars_expr(self, node: Any) -> py_ast.expr:
        """Convert a q AST node to a Python AST representing a pl.Expr."""
        match node:
            case ColExpr(expr, alias):
                e = self._polars_expr(expr)
                if alias is not None:
                    e = py_ast.Call(
                        func=py_ast.Attribute(value=e, attr="alias", ctx=py_ast.Load()),
                        args=[py_ast.Constant(value=alias)], keywords=[],
                    )
                return e
            case Name(n):
                return py_ast.Call(
                    func=py_ast.Attribute(
                        value=py_ast.Name(id="pl", ctx=py_ast.Load()),
                        attr="col", ctx=py_ast.Load(),
                    ),
                    args=[py_ast.Constant(value=n)], keywords=[],
                )
            case IntLit(v, _):
                return self._pl_lit(v)
            case FloatLit(v):
                return self._pl_lit(v)
            case BoolLit(v):
                return self._pl_lit(v)
            case SymLit(v):
                return self._pl_lit(v)
            case StrLit(v):
                return self._pl_lit(v)
            case BinOp(op, left, right):
                return self._polars_binop(op, left, right)
            case Apply(Name(fname), args) if fname in _POLARS_AGG:
                method = _POLARS_AGG[fname]
                arg_expr = self._polars_expr(args[0]) if args else None
                return py_ast.Call(
                    func=py_ast.Attribute(value=arg_expr, attr=method, ctx=py_ast.Load()),
                    args=[], keywords=[],
                )
            case _:
                raise NotImplementedError(
                    f"polars_expr: unsupported {type(node).__name__}: {node!r}"
                )

    def _pl_lit(self, value) -> py_ast.expr:
        return py_ast.Call(
            func=py_ast.Attribute(
                value=py_ast.Name(id="pl", ctx=py_ast.Load()),
                attr="lit", ctx=py_ast.Load(),
            ),
            args=[py_ast.Constant(value=value)], keywords=[],
        )

    def _polars_binop(self, op: str, left: Any, right: Any) -> py_ast.expr:
        le = self._polars_expr(left)
        re = self._polars_expr(right)
        if op in _Q_ARITH_OP:
            return py_ast.BinOp(left=le, op=_Q_ARITH_OP[op](), right=re)
        if op in _Q_CMP_OP:
            return py_ast.Compare(left=le, ops=[_Q_CMP_OP[op]()], comparators=[re])
        if op == "&":
            return py_ast.BinOp(left=le, op=py_ast.BitAnd(), right=re)
        if op == "|":
            return py_ast.BinOp(left=le, op=py_ast.BitOr(), right=re)
        raise NotImplementedError(f"polars_binop: unknown op {op!r}")

    def _col_exprs(self, cols: tuple) -> py_ast.expr:
        return py_ast.List(
            elts=[self._polars_expr(c) for c in cols],
            ctx=py_ast.Load(),
        )

    def _where_exprs(self, where: tuple) -> py_ast.expr:
        return py_ast.List(
            elts=[self._polars_expr(w) for w in where],
            ctx=py_ast.Load(),
        )

    def _by_exprs(self, by: tuple) -> py_ast.expr:
        return py_ast.List(
            elts=[self._polars_expr(c.expr if isinstance(c, ColExpr) else c) for c in by],
            ctx=py_ast.Load(),
        )

    def _qsql(self, node: Any) -> py_ast.expr:
        if isinstance(node, (QSelect, QExec)):
            fn = "q_select_rt" if isinstance(node, QSelect) else "q_exec_rt"
            return py_ast.Call(
                func=py_ast.Name(id=fn, ctx=py_ast.Load()),
                args=[
                    self._expr(node.table),
                    self._col_exprs(node.cols),
                    self._where_exprs(node.where),
                    self._by_exprs(node.by),
                ],
                keywords=[],
            )
        if isinstance(node, QUpdate):
            return py_ast.Call(
                func=py_ast.Name(id="q_update_rt", ctx=py_ast.Load()),
                args=[
                    self._expr(node.table),
                    self._col_exprs(node.cols),
                    self._where_exprs(node.where),
                ],
                keywords=[],
            )
        if isinstance(node, QDelete):
            return py_ast.Call(
                func=py_ast.Name(id="q_delete_rt", ctx=py_ast.Load()),
                args=[
                    self._expr(node.table),
                    self._col_exprs(node.cols),
                    self._where_exprs(node.where),
                ],
                keywords=[],
            )
        raise NotImplementedError(f"_qsql: unknown node {type(node).__name__}")
