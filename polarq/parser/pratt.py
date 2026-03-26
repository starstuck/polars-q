"""
Hand-written recursive-descent / Pratt-style parser for q expressions.

Key properties of q's grammar
───────────────────────────────
1. **Right-to-left evaluation, no operator precedence.**
   `2+3*4` = 2+(3*4) = 14.  Every verb has equal precedence; they are all
   right-associative.  This is implemented by the _parse_expr_seq() method,
   which collects a flat sequence of terms and then folds them right-to-left.

2. **Rank-ambiguous verbs.**
   `+` is monadic (flip/identity) or dyadic (add) depending on context.
   A verb at the start of a sequence is monadic; one with a left-hand term
   is dyadic.  The fold step resolves this after the sequence is collected.

3. **Adverbs are postfix.**
   `+/` is a derived function (over).  Adverbs attach to the immediately
   preceding verb or expression.

4. **Assignment at statement level.**
   `x:expr`   local (global at script top-level).
   `x::expr`  always global.
   `:expr`    return from lambda (bare colon).

5. **Lambda syntax.**
   `{[a;b] a+b}`  explicit params.
   `{x+y}`        implicit x y z.
   Lambda body is semicolon-delimited; last expression is the return value.

6. **Function application.**
   `f[x;y]`  bracket form — unambiguous multi-arg call.
   `f x`     juxtaposition — monadic application.
   Bracket form binds tighter than juxtaposition.

7. **Vector literals.**
   Consecutive same-kind atoms (int, float, bool, sym) form a VectorLit.
   `1 2 3`  →  VectorLit((IntLit(1), IntLit(2), IntLit(3))).

8. **qSQL.**
   `select … from … where … by …` is handled by _parse_qsql().
"""

from __future__ import annotations
from typing import Any

from polarq.errors import QParseError
from polarq.parser.lexer import Token, TT, tokenize as lex
from polarq.parser.ast_nodes import (
    IntLit, FloatLit, BoolLit, SymLit, StrLit, NullLit,
    DateLit, TimeLit, TimestampLit, MonthLit,
    VectorLit, ListLit,
    Name, Assign,
    Verb, BinOp, MonOp, Adverb,
    Apply,
    Lambda,
    Script,
    ColExpr, QSelect, QUpdate, QExec, QDelete,
    TableLit, KeyedTableLit,
)

# Verb token types → their q operator string
_VERB_TOKENS: dict[TT, str] = {
    TT.PLUS:    "+",
    TT.MINUS:   "-",
    TT.STAR:    "*",
    TT.PERCENT: "%",
    TT.BANG:    "!",
    TT.HASH:    "#",
    TT.UNDER:   "_",
    TT.TILDE:   "~",
    TT.AT:      "@",
    TT.QUESTION:"?",
    TT.DOT:     ".",
    TT.COMMA:   ",",
    TT.CARET:   "^",
    TT.DOLLAR:  "$",
    TT.AMP:     "&",
    TT.PIPE:    "|",
    TT.LT:      "<",
    TT.GT:      ">",
    TT.EQ:      "=",
    TT.LE:      "<=",
    TT.GE:      ">=",
    TT.KW_NOT:  "not",
    TT.KW_ALL:  "all",
    TT.KW_ANY:  "any",
    TT.COLON:   ":",
    TT.ZEROCOLON: "0:",
    TT.ONECOLON:  "1:",
}

# Adverb token types → their symbol string
_ADVERB_TOKENS: dict[TT, str] = {
    TT.ADVERB_OVER:       "/",
    TT.ADVERB_SCAN:       "\\",
    TT.ADVERB_EACH:       "'",
    TT.ADVERB_EACHRIGHT:  "/:",
    TT.ADVERB_EACHLEFT:   "\\:",
    TT.ADVERB_EACHPRIOR:  "':",
}

# Token types that can BEGIN a new expression term (after the previous term)
_EXPR_START = frozenset({
    TT.INT, TT.FLOAT, TT.BOOL, TT.SYM, TT.STRING, TT.NULL,
    TT.NAME,
    TT.KW_SELECT, TT.KW_UPDATE, TT.KW_EXEC, TT.KW_DELETE,
    TT.LPAREN, TT.LBRACE,
    # Verbs can start a monadic expression
    TT.PLUS, TT.MINUS, TT.STAR, TT.PERCENT, TT.BANG, TT.HASH,
    TT.UNDER, TT.TILDE, TT.AT, TT.COMMA, TT.CARET, TT.DOLLAR,
    TT.AMP, TT.PIPE, TT.LT, TT.GT, TT.EQ, TT.LE, TT.GE,
    TT.KW_NOT, TT.KW_ALL, TT.KW_ANY,
    TT.ZEROCOLON, TT.ONECOLON,
    TT.QUESTION,
    TT.KW_WHERE,   # `where` is also a monadic function outside qSQL
})

# Literal node types that can be grouped into a VectorLit
_SCALAR_LIT_TYPES = (IntLit, FloatLit, BoolLit, SymLit)


# ── Public entry points ───────────────────────────────────────────────────────

def parse(source: str) -> Script:
    """Parse a complete q script and return a Script AST node."""
    tokens = lex(source)
    p = Parser(tokens)
    return p.parse_script()


def parse_expr(source: str) -> Any:
    """Parse a single q expression (convenience for tests and REPL)."""
    tokens = lex(source)
    p = Parser(tokens)
    node = p._parse_stmt()
    return node


# ── Parser class ──────────────────────────────────────────────────────────────

class Parser:
    def __init__(self, tokens: list[Token]):
        self._tokens = tokens
        self._pos    = 0

    # ── Peek / consume helpers ────────────────────────────────────────────────

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return self._tokens[-1]   # EOF sentinel

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return tok

    def _expect(self, tt: TT) -> Token:
        tok = self._peek()
        if tok.type != tt:
            raise QParseError(
                f"expected {tt.name}, got {tok.type.name} ({tok.value!r})",
                tok.line, tok.col,
            )
        return self._advance()

    def _skip_newlines(self) -> None:
        while self._peek().type == TT.NEWLINE:
            self._advance()

    def _at_end(self) -> bool:
        return self._peek().type == TT.EOF

    # ── Script ────────────────────────────────────────────────────────────────

    def parse_script(self) -> Script:
        stmts = []
        self._skip_newlines()
        while not self._at_end():
            stmt = self._parse_stmt()
            if stmt is not None:
                stmts.append(stmt)
            # consume trailing newlines / semicolons between statements
            while self._peek().type in (TT.NEWLINE, TT.SEMI):
                self._advance()
        return Script(tuple(stmts))

    # ── Statement ────────────────────────────────────────────────────────────

    def _parse_stmt(self) -> Any:
        """
        A statement is one of:
          name : expr        local/global assignment
          name :: expr       global assignment (amend)
          : expr             bare colon — return from lambda
          expr               plain expression
        """
        self._skip_newlines()
        tok = self._peek()

        # Bare colon return: `:expr`
        if tok.type == TT.COLON:
            self._advance()
            expr = self._parse_expr_seq()
            return MonOp(":", expr)

        # Assignment? Look ahead: NAME COLON or NAME DCOLON
        if tok.type == TT.NAME:
            next_tok = self._peek(1)
            if next_tok.type == TT.COLON:
                name = self._advance().value
                self._advance()   # consume :
                # Check for :: (was split into : :)? No — DCOLON is its own token.
                expr = self._parse_expr_seq()
                return Assign(name, expr, global_=False)
            if next_tok.type == TT.DCOLON:
                name = self._advance().value
                self._advance()   # consume ::
                expr = self._parse_expr_seq()
                return Assign(name, expr, global_=True)

        return self._parse_expr_seq()

    # ── Expression sequence (right-to-left) ──────────────────────────────────

    def _parse_expr_seq(self) -> Any:
        """
        Collect a flat list of *terms* until a statement terminator is reached,
        then fold them right-to-left into the final AST.

        Statement terminators: NEWLINE, SEMI, RBRACE, EOF.
        (RPAREN and RBRACKET terminate sub-expressions but are handled by their
        respective callers.)
        """
        terms = []

        while True:
            tok = self._peek()
            if tok.type in (TT.NEWLINE, TT.SEMI, TT.RBRACE,
                            TT.RPAREN, TT.RBRACKET, TT.EOF):
                break

            # qSQL keywords start a qSQL expression
            if tok.type in (TT.KW_SELECT, TT.KW_UPDATE, TT.KW_EXEC, TT.KW_DELETE):
                if terms:
                    # e.g.  result: select …   — qSQL is the RHS
                    terms.append(self._parse_qsql())
                else:
                    return self._parse_qsql()
                break

            term = self._parse_term()
            terms.append(term)

        if not terms:
            raise QParseError(
                "empty expression",
                self._peek().line, self._peek().col,
            )

        return _fold_terms(terms)

    # ── Term (single unit in the expression sequence) ─────────────────────────

    def _parse_term(self) -> Any:
        """
        Parse one 'term' — an atom, sub-expression, lambda, or verb symbol —
        then check for postfix: bracket application and adverbs.
        """
        tok = self._peek()

        # ── Literals ──────────────────────────────────────────────────────────
        if tok.type == TT.INT:
            self._advance()
            int_val, int_kind = tok.value
            node: Any = IntLit(int_val, int_kind)
        elif tok.type == TT.FLOAT:
            self._advance()
            node = FloatLit(tok.value)
        elif tok.type == TT.BOOL:
            self._advance()
            node = BoolLit(tok.value)
        elif tok.type == TT.SYM:
            self._advance()
            node = SymLit(tok.value)
        elif tok.type == TT.STRING:
            self._advance()
            node = StrLit(tok.value)
        elif tok.type == TT.NULL:
            self._advance()
            node = NullLit(tok.value)
        elif tok.type == TT.DATE:
            self._advance()
            node = DateLit(tok.value)
        elif tok.type == TT.TIME:
            self._advance()
            node = TimeLit(tok.value)
        elif tok.type == TT.TIMESTAMP:
            self._advance()
            node = TimestampLit(tok.value)
        elif tok.type == TT.MONTH:
            self._advance()
            node = MonthLit(tok.value)

        # ── Name ──────────────────────────────────────────────────────────────
        elif tok.type == TT.NAME:
            self._advance()
            node = Name(tok.value)

        # ── Keywords that double as monadic functions outside qSQL ────────────
        elif tok.type == TT.KW_WHERE:
            self._advance()
            node = Name("where")

        # ── Parenthesised expression / list ──────────────────────────────────
        elif tok.type == TT.LPAREN:
            node = self._parse_paren()

        # ── Lambda ────────────────────────────────────────────────────────────
        elif tok.type == TT.LBRACE:
            node = self._parse_lambda()

        # ── Verb (will become Verb node; BinOp/MonOp resolved by _fold_terms) ─
        elif tok.type in _VERB_TOKENS:
            self._advance()
            node = Verb(_VERB_TOKENS[tok.type])

        else:
            raise QParseError(
                f"unexpected token {tok.type.name} ({tok.value!r})",
                tok.line, tok.col,
            )

        # ── Postfix: interleaved brackets and adverbs ─────────────────────────
        # e.g.  f[x]/:  (partial then adverb)  or  f/[x]  (adverb then apply)
        while True:
            if self._peek().type == TT.LBRACKET:
                self._advance()   # consume [
                args = self._parse_arg_list(TT.RBRACKET)
                self._expect(TT.RBRACKET)
                node = Apply(node, tuple(args))
            elif self._peek().type in _ADVERB_TOKENS:
                adv_tok = self._advance()
                node = Adverb(node, _ADVERB_TOKENS[adv_tok.type])
            else:
                break

        return node

    # ── Parenthesised expressions and general lists ───────────────────────────

    def _parse_paren(self) -> Any:
        """
        ( expr )            → the inner expression (grouping)
        ( expr ; expr ; … ) → ListLit
        ( )                 → ListLit(()) — empty list / null
        ([] col:val; …)     → TableLit
        """
        self._expect(TT.LPAREN)
        self._skip_newlines()

        # Table literal: ([] col:val; ...)  or keyed table ([key:val] val:val; ...)
        if self._peek().type == TT.LBRACKET:
            self._advance()   # consume [
            if self._peek().type == TT.RBRACKET:
                # Unkeyed: ([] col:val; ...)
                self._advance()   # consume ]
                self._skip_newlines()
                cols = []
                while self._peek().type != TT.RPAREN:
                    if self._peek().type == TT.SEMI:
                        self._advance(); self._skip_newlines(); continue
                    name_tok = self._expect(TT.NAME)
                    self._expect(TT.COLON)
                    cols.append((name_tok.value, self._parse_expr_seq()))
                    self._skip_newlines()
                self._expect(TT.RPAREN)
                return TableLit(tuple(cols))
            else:
                # Keyed: ([key_col:val; ...] val_col:val; ...)
                key_cols = []
                while self._peek().type != TT.RBRACKET:
                    if self._peek().type == TT.SEMI:
                        self._advance(); self._skip_newlines(); continue
                    name_tok = self._expect(TT.NAME)
                    self._expect(TT.COLON)
                    key_cols.append((name_tok.value, self._parse_qsql_col_expr()))
                    self._skip_newlines()
                self._expect(TT.RBRACKET)
                self._skip_newlines()
                val_cols = []
                while self._peek().type != TT.RPAREN:
                    if self._peek().type == TT.SEMI:
                        self._advance(); self._skip_newlines(); continue
                    name_tok = self._expect(TT.NAME)
                    self._expect(TT.COLON)
                    val_cols.append((name_tok.value, self._parse_expr_seq()))
                    self._skip_newlines()
                self._expect(TT.RPAREN)
                return KeyedTableLit(tuple(key_cols), tuple(val_cols))

        if self._peek().type == TT.RPAREN:
            self._advance()
            return ListLit(())

        items = []
        # First item
        if self._peek().type not in (TT.SEMI,):
            items.append(self._parse_expr_seq())
        else:
            items.append(NullLit())   # leading ; → null first element

        # Further items separated by ;
        while self._peek().type == TT.SEMI:
            self._advance()
            self._skip_newlines()
            if self._peek().type == TT.RPAREN:
                break
            items.append(self._parse_expr_seq())

        self._expect(TT.RPAREN)

        if len(items) == 1:
            return items[0]   # plain grouping
        return ListLit(tuple(items))

    # ── Lambda ────────────────────────────────────────────────────────────────

    def _parse_lambda(self) -> Lambda:
        """
        { [p;q] body }    explicit params
        { body }          implicit x y z
        """
        self._expect(TT.LBRACE)
        self._skip_newlines()

        # Optional explicit param list [p;q;r]
        params: tuple = ()
        if self._peek().type == TT.LBRACKET:
            self._advance()
            param_names = []
            while self._peek().type != TT.RBRACKET:
                tok = self._expect(TT.NAME)
                param_names.append(tok.value)
                if self._peek().type == TT.SEMI:
                    self._advance()
            self._expect(TT.RBRACKET)
            params = tuple(param_names)
            self._skip_newlines()

        # Body: semicolon-delimited statements
        body = []
        while self._peek().type not in (TT.RBRACE, TT.EOF):
            self._skip_newlines()
            if self._peek().type == TT.RBRACE:
                break
            stmt = self._parse_stmt()
            if stmt is not None:
                body.append(stmt)
            # Consume stmt separator
            while self._peek().type in (TT.SEMI, TT.NEWLINE):
                self._advance()

        self._expect(TT.RBRACE)
        return Lambda(params, tuple(body))

    # ── Argument list for f[…] calls ─────────────────────────────────────────

    def _parse_arg_list(self, close: TT) -> list:
        """
        Parse semicolon-separated arguments until `close` token.
        A bare semicolon (;;) means a gap/projection: None.
        """
        args = []
        if self._peek().type == close:
            return args

        # First arg
        if self._peek().type == TT.SEMI:
            args.append(None)   # gap
        else:
            args.append(self._parse_expr_seq())

        while self._peek().type == TT.SEMI:
            self._advance()
            if self._peek().type in (close, TT.SEMI):
                args.append(None)
            else:
                args.append(self._parse_expr_seq())

        return args

    # ── qSQL ─────────────────────────────────────────────────────────────────

    def _parse_qsql(self) -> Any:
        """
        select [cols] [by grp] from tbl [where cond]
        update  col:expr       from tbl [where cond]
        exec   [cols] [by grp] from tbl [where cond]
        delete [cols]          from tbl [where cond]
        """
        kw = self._advance()   # select / update / exec / delete

        # --- parse column list (optional, before 'by' or 'from') -------------
        cols = self._parse_qsql_cols()

        # --- optional  by  clause  -------------------------------------------
        by: tuple = ()
        if self._peek().type == TT.KW_BY:
            self._advance()
            by = tuple(self._parse_qsql_cols())

        # --- from  clause  ---------------------------------------------------
        self._expect(TT.KW_FROM)
        table = self._parse_term()   # one term — table name or expr

        # --- optional  where  clause  ----------------------------------------
        where: tuple = ()
        if self._peek().type == TT.KW_WHERE:
            self._advance()
            where = tuple(self._parse_where_list())

        cols_t = tuple(cols)
        if kw.type == TT.KW_SELECT:
            return QSelect(cols_t, table, where, by)
        if kw.type == TT.KW_UPDATE:
            return QUpdate(cols_t, table, where, by)
        if kw.type == TT.KW_EXEC:
            return QExec(cols_t, table, where, by)
        if kw.type == TT.KW_DELETE:
            return QDelete(cols_t, table, where)
        raise QParseError(f"unknown qSQL keyword {kw.value!r}", kw.line, kw.col)

    def _parse_qsql_cols(self) -> list[ColExpr]:
        """
        Parse a comma-separated column list before 'by' or 'from'.
        Each column is optionally aliased:  alias:expr  or just  expr.
        Returns [] if the next token is 'by' or 'from'.
        """
        cols = []
        _STOP = {TT.KW_BY, TT.KW_FROM, TT.KW_WHERE, TT.EOF, TT.NEWLINE}

        while self._peek().type not in _STOP:
            alias = None
            # Optional alias: NAME COLON
            if (self._peek().type == TT.NAME and
                    self._peek(1).type == TT.COLON):
                alias = self._advance().value
                self._advance()   # consume :

            expr = self._parse_qsql_col_expr()
            cols.append(ColExpr(expr, alias))

            if self._peek().type == TT.COMMA:
                self._advance()
            else:
                break

        return cols

    def _parse_qsql_col_expr(self) -> Any:
        """
        Parse one column expression in qSQL context.
        This is like _parse_expr_seq but stops at comma, 'by', 'from', 'where'.
        """
        terms = []
        _STOP = {TT.COMMA, TT.KW_BY, TT.KW_FROM, TT.KW_WHERE,
                 TT.SEMI, TT.NEWLINE, TT.EOF, TT.RBRACKET}

        while self._peek().type not in _STOP:
            terms.append(self._parse_term())

        if not terms:
            raise QParseError(
                "empty column expression in qSQL",
                self._peek().line, self._peek().col,
            )
        return _fold_terms(terms)

    def _parse_where_list(self) -> list:
        """
        Parse comma-separated where conditions.
        """
        conds = []
        _STOP = {TT.KW_BY, TT.KW_FROM, TT.SEMI, TT.NEWLINE, TT.EOF}

        while self._peek().type not in _STOP:
            conds.append(self._parse_qsql_col_expr())
            if self._peek().type == TT.COMMA:
                self._advance()
            else:
                break
        return conds


# ── Term-sequence folding (right-to-left) ─────────────────────────────────────

def _fold_terms(terms: list) -> Any:
    """
    Fold a flat list of term-AST-nodes into a structured AST, applying q's
    right-to-left evaluation rule.

    Steps:
    1. Group consecutive same-kind literal atoms into VectorLit nodes.
    2. Find the leftmost Verb node.  Everything left of it is the LHS;
       everything right is the RHS (evaluated recursively).
       If there is no LHS the verb is monadic; otherwise dyadic.
    3. If no verb: build a right-to-left application chain.
       Single-element → return as-is.
    """
    if len(terms) == 1:
        return terms[0]

    # Step 1: group adjacent same-kind scalar literals into vectors
    grouped = _group_literals(terms)

    if len(grouped) == 1:
        return grouped[0]

    # Step 2: find leftmost verb (BinOp / MonOp resolution)
    # Named dyadic keywords (e.g. `div`, `mod`, `xexp`) are treated as verbs
    # when they appear between two terms.
    _DYADIC_NAMED_VERBS = frozenset({
        "div", "mod", "xexp", "xlog", "like", "ss", "sv", "vs",
        "msum", "mavg", "mmin", "mmax", "mdev", "ema",
        "xbar", "bin", "wavg", "wsum",
        "xasc", "xdesc",
        "lj", "aj", "uj", "pj",
        "each",
        "set",
        "rotate", "sublist",
        "in", "within",
    })

    for i, node in enumerate(grouped):
        is_named_dyadic = isinstance(node, Name) and node.name in _DYADIC_NAMED_VERBS
        if isinstance(node, Verb) or is_named_dyadic:
            op = node.op if isinstance(node, Verb) else node.name
            left_terms  = grouped[:i]
            right_terms = grouped[i + 1:]

            if not right_terms:
                # Trailing verb — treat as monadic applied to LHS
                left = _fold_terms(left_terms) if left_terms else None
                return MonOp(op, left) if left else Verb(op)

            right = _fold_terms(right_terms) if len(right_terms) > 1 else right_terms[0]

            if left_terms:
                left = _fold_terms(left_terms) if len(left_terms) > 1 else left_terms[0]
                return BinOp(op, left, right)
            else:
                return MonOp(op, right)

    # Step 3: no verbs — right-to-left application chain
    # `f g x` = f[g[x]]  =  Apply(f, Apply(g, x))
    result = grouped[-1]
    for node in reversed(grouped[:-1]):
        result = Apply(node, (result,))
    return result


def _group_literals(terms: list) -> list:
    """
    Merge runs of same-kind scalar literals (IntLit, FloatLit, BoolLit, SymLit)
    into VectorLit nodes.  Mixed runs or single-element runs are left alone.

    Special case: a run of IntLit values immediately followed by a BoolLit (or
    a run starting with IntLit where any element is BoolLit) is treated as a
    boolean vector — in q, `0 0 1b` means the boolean vector 0b 0b 1b.
    """
    out = []
    i   = 0
    while i < len(terms):
        t = terms[i]
        if not isinstance(t, _SCALAR_LIT_TYPES):
            out.append(t)
            i += 1
            continue
        # Scan for a run of the same literal type
        j = i + 1
        while j < len(terms) and type(terms[j]) is type(t):
            j += 1
        run = terms[i:j]
        # If this is a run of IntLit and the very next item is a BoolLit,
        # absorb it: the trailing `b` suffix coerces the whole run to boolean.
        if (run and isinstance(run[0], IntLit)
                and j < len(terms) and isinstance(terms[j], BoolLit)):
            run = run + [terms[j]]
            j += 1
            run = [BoolLit(bool(n.value)) for n in run]
        if len(run) > 1:
            out.append(VectorLit(tuple(run)))
        else:
            out.append(run[0])
        i = j
    return out
