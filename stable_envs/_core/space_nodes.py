"""Walk nested Gymnasium spaces."""

from __future__ import annotations


def nodes(space, path=""):
    """Yield ``(path, space)`` for ``space`` and every nested child, depth first, with ``/``-joined keys."""
    yield path, space
    for key, child in getattr(space, "spaces", {}).items():
        yield from nodes(child, path + "/" + key)
