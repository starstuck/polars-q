"""Exception hierarchy for polarq."""


class QError(Exception):
    """Base class for all q runtime and parse errors."""


class QTypeError(QError):
    """Type mismatch — wrong kind for the operation."""


class QLengthError(QError):
    """Vector length mismatch."""


class QRankError(QError):
    """Wrong number of arguments to a function or verb."""


class QDomainError(QError):
    """Value is out of the domain for this operation (e.g. divide by zero, bad temporal attr)."""


class QParseError(QError):
    """Syntax error in q source — includes line/column context when available."""

    def __init__(self, msg: str, line: int = 0, col: int = 0):
        super().__init__(f"parse error at {line}:{col} — {msg}")
        self.line = line
        self.col = col
