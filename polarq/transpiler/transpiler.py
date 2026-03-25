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
    VectorLit, ListLit,
    Name, Assign,
    Verb, BinOp, MonOp, Adverb,
    Apply,
    Lambda,
    Script,
    ColExpr, QSelect, QUpdate, QExec, QDelete,
)
from polarq.transpiler.builtins import VERB_MAP, ADVERB_MAP


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
        # Any expression as a standalone statement
        return py_ast.Expr(value=self._expr(node), lineno=0, col_offset=0)

    # ── Expressions ──────────────────────────────────────────────────────────

    def _qatom_call(self, value: Any, kind: str) -> py_ast.expr:
        """Emit: QAtom(<value>, '<kind>')"""
        return py_ast.Call(
            func=py_ast.Name(id="QAtom", ctx=py_ast.Load()),
            args=[py_ast.Constant(value=value), py_ast.Constant(value=kind)],
            keywords=[],
        )

    def _expr(self, node: Any) -> py_ast.expr:
        match node:
            case IntLit(v):
                return self._qatom_call(v, "j")
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

            case VectorLit(items):
                return self._vector_lit(items)

            case ListLit(items):
                elts = [self._expr(i) for i in items]
                return py_ast.List(elts=elts, ctx=py_ast.Load())

            case Name(n):
                return py_ast.Name(id=n, ctx=py_ast.Load())

            case BinOp(op, left, right):
                return self._bin_op(op, left, right)

            case MonOp(op, right):
                return self._mon_op(op, right)

            case Apply(func, args):
                return self._apply(func, args)

            case Lambda(params, body):
                return self._lambda(params, body)

            case Adverb(verb, adv):
                return self._adverb(verb, adv)

            case QSelect() | QUpdate() | QExec() | QDelete():
                return self._qsql(node)

            case _:
                raise NotImplementedError(
                    f"transpiler: unsupported node type {type(node).__name__}: {node!r}"
                )

    def _vector_lit(self, items: tuple) -> py_ast.expr:
        # QVector.from_items([...], kind)
        # Use raw constant values — QAtom wrapping is only for scalar atoms
        elts = [py_ast.Constant(value=i.value) for i in items]
        first = items[0]
        if isinstance(first, IntLit):
            kind = "j"
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

    def _lambda(self, params: tuple, body: tuple) -> py_ast.expr:
        self._lambda_counter += 1
        fn_name = f"_qfn_{self._lambda_counter}"
        args    = py_ast.arguments(
            posonlyargs=[],
            args=[py_ast.arg(arg=p) for p in (params or ("x", "y", "z"))],
            vararg=None, kwonlyargs=[], kw_defaults=[],
            kwarg=None, defaults=[],
        )
        fn_body = [self._stmt(s) for s in body[:-1]]
        fn_body.append(py_ast.Return(value=self._expr(body[-1])))
        # Emit a module-level def (referenced by name)
        # For now we produce a lambda-compatible expression for simple bodies
        if len(body) == 1:
            # Single-expression lambda — works regardless of whether params are
            # implicit or explicit, since Python lambda handles both.
            return py_ast.Lambda(args=args, body=self._expr(body[0]))
        # Multi-statement: cannot inline; caller should hoist to def
        raise NotImplementedError(
            "multi-statement lambdas require def-hoisting (not yet implemented)"
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

    def _qsql(self, node: Any) -> py_ast.expr:
        raise NotImplementedError(
            "qSQL transpilation to Polars chains not yet implemented"
        )
