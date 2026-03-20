"""
QEnv — linked scope chain with dotted namespace support.

q has two kinds of assignment:
  x:42     local assignment (inside lambdas) / global at script level
  x::42    always global (amend the root scope)

Dotted namespaces:  .myns.foo:42  sets foo in namespace myns.
"""

from __future__ import annotations
from typing import Any


class QEnv:
    """
    A single scope frame.  Scopes are linked via _parent.

    Root scope:   QEnv()
    Lambda scope: env.child()   (inherits read access to parent)
    """

    def __init__(self, parent: QEnv | None = None):
        self._bindings: dict[str, Any] = {}
        self._parent = parent

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Any:
        """Walk the scope chain.  Raises KeyError if not found anywhere."""
        if name in self._bindings:
            return self._bindings[name]
        if self._parent is not None:
            return self._parent.get(name)
        raise KeyError(name)

    def __contains__(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except KeyError:
            return False

    # ── Write ─────────────────────────────────────────────────────────────────

    def set(self, name: str, value: Any) -> None:
        """Local assignment — always writes to *this* frame."""
        self._bindings[name] = value

    def set_global(self, name: str, value: Any) -> None:
        """Global assignment — walks up to root, then writes there."""
        root = self
        while root._parent is not None:
            root = root._parent
        root._bindings[name] = value

    # ── Dotted namespaces ─────────────────────────────────────────────────────

    def set_dotted(self, dotted: str, value: Any) -> None:
        """
        .myns.foo:42 — create/update a nested QEnv at .myns, bind foo inside it.
        Dots in the leading namespace prefix are stripped; the final segment is
        the name within that namespace frame.
        """
        parts = dotted.lstrip(".").split(".")
        if len(parts) == 1:
            self.set(parts[0], value)
            return
        ns_name, leaf = parts[0], ".".join(parts[1:])
        # Ensure the namespace frame exists in the root scope
        root = self
        while root._parent is not None:
            root = root._parent
        if ns_name not in root._bindings or not isinstance(root._bindings[ns_name], QEnv):
            root._bindings[ns_name] = QEnv(parent=self)
        root._bindings[ns_name].set_dotted(leaf, value)

    # ── Scope creation ────────────────────────────────────────────────────────

    def child(self) -> QEnv:
        """Create a child scope (used when entering a lambda)."""
        return QEnv(parent=self)

    # ── Introspection ─────────────────────────────────────────────────────────

    def keys(self) -> list[str]:
        """All names visible from this scope (local + ancestors, no duplicates)."""
        seen: set[str] = set()
        result = []
        env: QEnv | None = self
        while env is not None:
            for k in env._bindings:
                if k not in seen:
                    seen.add(k)
                    result.append(k)
            env = env._parent
        return result

    def local_keys(self) -> list[str]:
        return list(self._bindings.keys())

    def __repr__(self) -> str:  # pragma: no cover
        depth = 0
        e = self._parent
        while e:
            depth += 1
            e = e._parent
        return f"QEnv(depth={depth}, locals={list(self._bindings.keys())})"
