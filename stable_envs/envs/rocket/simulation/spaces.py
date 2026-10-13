from __future__ import annotations


def nodes(space, path=""):
    yield path, space
    for key, value in getattr(space, "spaces", {}).items():
        yield from nodes(value, path + "/" + key)
