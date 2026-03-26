"""
Tokeniser for q/kdb+ source.

Design notes
────────────
• Uses Python's ``re`` module directly rather than Lark because q's lexical
  rules are context-sensitive in ways that confuse a pure context-free scanner
  (negative number literals vs. subtraction, adverb vs. division, etc.).

• Token patterns are tried in priority order (earlier = higher priority).

• Negative-number rule:
    -3    →  INT(-3)   when `-` immediately precedes a digit and the previous
                       significant token was an operator, open bracket, or BOF.
    a-3   →  NAME MINUS INT(3)
    1 -3  →  INT(1) MINUS INT(3)   (so `1 -3` parses as 1 minus 3 = 0,
                                    matching kdb+ behaviour)

• Symbol chaining:  `a`b`c  is tokenised as three SYM tokens (the parser
  groups them into a VectorLit).

• Keywords (select, update, from, …) are recognised from NAME tokens.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


# ── Token types ───────────────────────────────────────────────────────────────

class TT(Enum):
    # Literals
    BOOL      = auto()   # 0b  1b
    FLOAT     = auto()   # 3.14  1.5e  -2.0f
    INT       = auto()   # 42  42i  42j  42h   (may be negative)
    SYM       = auto()   # `foo  ` (empty sym)
    STRING    = auto()   # "hello world"
    NULL      = auto()   # 0N  0n  0Nd  0Nt  0Np  0Nv  0Nu  0Nz  ::
    # Temporal literals
    TIMESTAMP = auto()   # 2024.01.15D12:30:00.000000000
    MONTH     = auto()   # 2024.01m
    DATE      = auto()   # 2024.01.15
    TIME      = auto()   # 12:30:00.000
    # Identifiers / keywords
    NAME     = auto()
    KW_SELECT= auto()
    KW_UPDATE= auto()
    KW_EXEC  = auto()
    KW_DELETE= auto()
    KW_FROM  = auto()
    KW_WHERE = auto()
    KW_BY    = auto()
    KW_NOT   = auto()
    KW_ALL   = auto()
    KW_ANY   = auto()
    # Structural
    LPAREN   = auto()   # (
    RPAREN   = auto()   # )
    LBRACE   = auto()   # {
    RBRACE   = auto()   # }
    LBRACKET = auto()   # [
    RBRACKET = auto()   # ]
    SEMI     = auto()   # ;
    COLON    = auto()   # :
    DCOLON   = auto()   # ::
    # Verbs (single-character operators)
    PLUS    = auto()   # +
    MINUS   = auto()   # -
    STAR    = auto()   # *
    PERCENT = auto()   # %
    BANG    = auto()   # !
    HASH    = auto()   # #
    UNDER   = auto()   # _
    TILDE   = auto()   # ~
    AT      = auto()   # @
    DOT     = auto()   # .  (apply / index operator — standalone, not part of name)
    COMMA   = auto()   # ,
    CARET   = auto()   # ^
    DOLLAR  = auto()   # $
    AMP     = auto()   # &
    PIPE    = auto()   # |
    LT      = auto()   # <
    GT      = auto()   # >
    EQ      = auto()   # =
    LE      = auto()   # <=
    GE      = auto()   # >=
    # Adverbs
    ADVERB_OVER      = auto()   # /
    ADVERB_SCAN      = auto()   # \  (when not followed by :)
    ADVERB_EACH      = auto()   # '
    ADVERB_EACHRIGHT = auto()   # /:
    ADVERB_EACHLEFT  = auto()   # \:
    ADVERB_EACHPRIOR = auto()   # ':
    # Whitespace / structure
    NEWLINE = auto()
    EOF     = auto()


KEYWORDS: dict[str, TT] = {
    "select": TT.KW_SELECT,
    "update": TT.KW_UPDATE,
    "exec":   TT.KW_EXEC,
    "delete": TT.KW_DELETE,
    "from":   TT.KW_FROM,
    "where":  TT.KW_WHERE,
    "by":     TT.KW_BY,
    "not":    TT.KW_NOT,
    "all":    TT.KW_ALL,
    "any":    TT.KW_ANY,
}

# Token types that represent "value-like" tokens after which a `-` that is
# immediately followed by a digit should be treated as subtraction, not
# the start of a negative literal.
_VALUE_CLOSE = frozenset({
    TT.INT, TT.FLOAT, TT.BOOL, TT.SYM, TT.STRING, TT.NULL,
    TT.DATE, TT.TIME, TT.TIMESTAMP, TT.MONTH,
    TT.DOT,
    TT.NAME,
    TT.KW_SELECT, TT.KW_UPDATE, TT.KW_EXEC, TT.KW_DELETE,
    TT.KW_FROM, TT.KW_WHERE, TT.KW_BY,
    TT.RPAREN, TT.RBRACKET, TT.RBRACE,
})


@dataclass(slots=True)
class Token:
    type:  TT
    value: object   # parsed Python value (int, float, str, bool, None)
    line:  int
    col:   int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


# ── Regex patterns ────────────────────────────────────────────────────────────
#
# Each entry: (TT or None, regex_pattern)
# None → skip (whitespace, comments).
# Patterns are anchored implicitly by re.match at the current position.

_RAW_PATTERNS: list[tuple[TT | None, str]] = [
    # Comments — skip entire line after //
    (None,               r"//[^\n]*"),
    # Newlines
    (TT.NEWLINE,         r"\n"),
    # Whitespace (not newline) — skip
    (None,               r"[ \t]+"),
    # Bool must be checked before INT: 0b 1b
    (TT.BOOL,            r"[01]b(?![a-zA-Z0-9_.])"),
    # Typed nulls  0N  0n  0Nd  0Nt  0Np  0Nv  0Nu  0Nz  (plus generic ::)
    (TT.NULL,            r"0[Nn][a-z]?(?![a-zA-Z0-9_.])"),
    (TT.DCOLON,          r"::"),
    # Temporal literals — checked before FLOAT/INT to avoid partial matches.
    # Order: TIMESTAMP > MONTH > DATE > TIME  (most specific first).
    (TT.TIMESTAMP,       r"\d{4}\.\d{2}\.\d{2}D\d{2}:\d{2}:\d{2}\.\d+"),
    (TT.MONTH,           r"\d{4}\.\d{2}m(?![a-zA-Z0-9_.])"),
    (TT.DATE,            r"\d{4}\.\d{2}\.\d{2}(?![D\d.])"),
    (TT.TIME,            r"\d{2}:\d{2}:\d{2}\.\d+"),
    # Floats — various forms; optional negative handled separately in _lex()
    # Matches: 3.14  .5  3.  1.5e3  1.5e-3  3.14f  (no leading minus here)
    (TT.FLOAT,           r"[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?[ef]?"
                         r"|[0-9]+[eE][+-]?[0-9]+[ef]?"
                         r"|[0-9]+[ef](?![a-zA-Z0-9_.])"),
    # Integers — optional type suffix i j h N
    (TT.INT,             r"[0-9]+[ijhN]?(?![a-zA-Z0-9_.])"),
    # Strings
    (TT.STRING,          r'"(?:[^"\\]|\\.)*"'),
    # Symbols (backtick, optional name body; chain handled by parser)
    (TT.SYM,             r"`[a-zA-Z0-9_.]*"),
    # Adverbs — multi-character before single-character
    (TT.ADVERB_EACHRIGHT, r"/:"),
    (TT.ADVERB_EACHLEFT,  r"\\:"),
    (TT.ADVERB_EACHPRIOR, r"':"),
    (TT.ADVERB_OVER,      r"/"),
    (TT.ADVERB_SCAN,      r"\\"),
    (TT.ADVERB_EACH,      r"'"),
    # Single colon (after :: already matched above)
    (TT.COLON,           r":"),
    # Structural
    (TT.LPAREN,          r"\("),
    (TT.RPAREN,          r"\)"),
    (TT.LBRACE,          r"\{"),
    (TT.RBRACE,          r"\}"),
    (TT.LBRACKET,        r"\["),
    (TT.RBRACKET,        r"\]"),
    (TT.SEMI,            r";"),
    # Operators / verbs
    (TT.PLUS,    r"\+"),
    (TT.MINUS,   r"-"),
    (TT.STAR,    r"\*"),
    (TT.PERCENT, r"%"),
    (TT.BANG,    r"!"),
    (TT.HASH,    r"#"),
    (TT.UNDER,   r"_"),
    (TT.TILDE,   r"~"),
    (TT.AT,      r"@"),
    (TT.COMMA,   r","),
    (TT.CARET,   r"\^"),
    (TT.DOLLAR,  r"\$"),
    (TT.AMP,     r"&"),
    (TT.PIPE,    r"\|"),
    (TT.LE,      r"<="),
    (TT.GE,      r">="),
    (TT.LT,      r"<"),
    (TT.GT,      r">"),
    (TT.EQ,      r"="),
    # Names / keywords — also handles dotted namespace names like .z.p or .myns.foo
    (TT.NAME,    r"\.[a-zA-Z][a-zA-Z0-9_.]*|[a-zA-Z_][a-zA-Z0-9_.]*"),
    # Standalone dot — apply/index operator; must come after NAME so .foo matches NAME
    (TT.DOT,     r"\."),
]

# Compile all patterns once
_PATTERNS: list[tuple[TT | None, re.Pattern]] = [
    (tt, re.compile(pat)) for tt, pat in _RAW_PATTERNS
]

# Negative-number prefix pattern: minus immediately before digits (no space)
_NEG_NUM = re.compile(r"-([0-9]+(?:\.[0-9]*(?:[eE][+-]?[0-9]+)?[ef]?)?[ijhef]?)")


# ── Tokeniser ─────────────────────────────────────────────────────────────────

class Lexer:
    """
    Convert a q source string into a flat list of Token objects.

    Usage::

        tokens = Lexer(source).tokenize()
    """

    def __init__(self, source: str):
        self._src  = source
        self._pos  = 0
        self._line = 1
        self._col  = 1

    # ── Public ────────────────────────────────────────────────────────────────

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        last_sig: TT | None = None   # last non-whitespace, non-newline token type

        src = self._src
        pos = 0
        line = 1
        col  = 1

        while pos < len(src):
            matched = False

            # Negative number: `-` immediately before digits, only when the
            # previous significant token was NOT a value-like token.
            if src[pos] == "-" and pos + 1 < len(src) and src[pos + 1].isdigit():
                if last_sig not in _VALUE_CLOSE:
                    # Absorb the negative number as a single token
                    m = re.match(
                        r"-([0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?[ef]?"
                        r"|-[0-9]+[eE][+-]?[0-9]+[ef]?"
                        r"|[0-9]+[ijhN]?)",
                        src[pos:]
                    )
                    if m:
                        raw = src[pos : pos + len(m.group())]
                        tok = _make_num_token(raw, line, col)
                        tokens.append(tok)
                        last_sig = tok.type
                        col += len(raw)
                        pos += len(raw)
                        continue

            for tt, pat in _PATTERNS:
                m = pat.match(src, pos)
                if not m:
                    continue
                raw = m.group()
                if tt is not None:
                    tok = _make_token(tt, raw, line, col)
                    tokens.append(tok)
                    last_sig = tok.type
                # Update line/col
                newlines = raw.count("\n")
                if newlines:
                    line += newlines
                    col = len(raw) - raw.rfind("\n")
                else:
                    col += len(raw)
                pos += len(raw)
                matched = True
                break

            if not matched:
                raise SyntaxError(
                    f"Unexpected character {src[pos]!r} at line {line}, col {col}"
                )

        tokens.append(Token(TT.EOF, None, line, col))
        return tokens


# ── Token value helpers ───────────────────────────────────────────────────────

def _make_token(tt: TT, raw: str, line: int, col: int) -> Token:
    """Construct a Token with the appropriate parsed Python value."""
    if tt == TT.INT:
        suffix = raw[-1] if raw[-1] in "ijh" else "j"
        val = int(raw.rstrip("ijhN"))
        return Token(tt, (val, suffix), line, col)
    if tt == TT.FLOAT:
        val = float(raw.rstrip("fe"))
        return Token(tt, val, line, col)
    if tt == TT.BOOL:
        return Token(tt, raw[0] == "1", line, col)
    if tt == TT.SYM:
        return Token(tt, raw[1:], line, col)   # strip leading `
    if tt == TT.STRING:
        val = raw[1:-1].replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
        return Token(tt, val, line, col)
    if tt == TT.NULL:
        kind = raw[2] if len(raw) > 2 else ("f" if raw[1] == "n" else "j")
        return Token(tt, kind, line, col)
    if tt == TT.NAME:
        kw_tt = KEYWORDS.get(raw)
        if kw_tt:
            return Token(kw_tt, raw, line, col)
        return Token(tt, raw, line, col)
    # Everything else: value is the raw string
    return Token(tt, raw, line, col)


def _make_num_token(raw: str, line: int, col: int) -> Token:
    """Create INT or FLOAT token from a raw string (possibly negative)."""
    stripped = raw.rstrip("ijhNeEf")
    if "." in stripped or "e" in stripped.lower():
        return Token(TT.FLOAT, float(raw.rstrip("feE")), line, col)
    suffix = raw[-1] if raw[-1] in "ijh" else "j"
    return Token(TT.INT, (int(raw.rstrip("ijhN")), suffix), line, col)


# ── Convenience function ──────────────────────────────────────────────────────

def tokenize(source: str) -> list[Token]:
    """Tokenise q source and return a list of Token objects."""
    return Lexer(source).tokenize()
