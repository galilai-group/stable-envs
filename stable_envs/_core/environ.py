"""The registry of every ``STABLE_ENVS_*`` environment variable and the one function that reads them."""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PREFIX = "STABLE_ENVS_"


@dataclass(frozen=True)
class Variable:
    """One environment variable: its name, value type, default, documentation and parser."""

    name: str
    kind: str
    default: Any
    doc: str
    parse: Callable[[str], Any]


VARIABLES: dict[str, Variable] = {}

_SUFFIX = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_int(text: str) -> int | None:
    """Parse an integer with an optional K, M, G or T suffix (powers of 1024); ``auto`` or empty is None."""
    t = text.strip().replace("_", "")
    if t.lower() in ("", "auto"):
        return None
    m = re.fullmatch(r"(\d+)\s*([KMGT]?)I?B?", t.upper())
    if m is None:
        raise ValueError("expected an integer, optionally with a K, M, G or T suffix, or 'auto'")
    return int(m.group(1)) * _SUFFIX[m.group(2)]


def parse_flag(text: str) -> bool:
    """Parse ``1`` as True and ``0`` or empty as False."""
    t = text.strip()
    if t not in ("", "0", "1"):
        raise ValueError("expected 1 or 0")
    return t == "1"


def register(name: str, kind: str, default: Any, doc: str, parse: Callable[[str], Any]) -> Variable:
    """Add a variable to the registry; registering the same name twice keeps the first entry."""
    if not name.startswith(PREFIX):
        raise ValueError(f"{name} does not start with {PREFIX}")
    return VARIABLES.setdefault(name, Variable(name, kind, default, doc, parse))


@functools.lru_cache(maxsize=512)
def _parsed(name: str, raw: str) -> Any:
    try:
        return VARIABLES[name].parse(raw)
    except ValueError as e:
        raise ValueError(f"{name}={raw!r}: {e}") from None


def read(name: str) -> Any:
    """Return the parsed value of variable ``name``, or its default when unset.

    Raises:
        KeyError: ``name`` is not registered.
        ValueError: The value does not parse; the message names the variable.
    """
    var = VARIABLES[name]
    raw = os.environ.get(name)
    return var.default if raw is None else _parsed(name, raw)


def variables() -> dict[str, Variable]:
    """Return every registered variable, including those of modules not imported yet."""
    from .. import tuning  # noqa: F401  (registers the knob variables)

    return dict(sorted(VARIABLES.items()))


def set_variables(environ=None) -> tuple[dict[str, str], list[str]]:
    """Return the ``STABLE_ENVS_*`` variables set in ``environ`` and the names that are not registered."""
    environ = os.environ if environ is None else environ
    known = variables()
    found = {k: v for k, v in sorted(environ.items()) if k.startswith(PREFIX)}
    return found, [k for k in found if k not in known]


def parse_verbosity(text: str) -> str:
    """Parse a log level name such as ``INFO`` or ``SILENT``; an empty value is ``WARNING``."""
    from .config import _LEVELS

    t = text.strip().upper() or "WARNING"
    if t not in _LEVELS:
        raise ValueError(f"expected one of {sorted(_LEVELS)}")
    return t


register(
    "STABLE_ENVS_VERBOSE",
    "level",
    "WARNING",
    "Default log level of the library, with the same values as snv.set(verbose=...). snv.set overrides it.",
    parse_verbosity,
)
register(
    "STABLE_ENVS_OGB_UNMEASURED",
    "flag",
    False,
    "1 lets OGBench Cube and Scene cells without measured visual fidelity register, with placeholder "
    "values, while measuring them. Never set it for normal use.",
    parse_flag,
)
