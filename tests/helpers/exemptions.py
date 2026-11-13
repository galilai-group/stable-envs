"""The registry-wide rules every registered env is held to, and the one file that may waive them.

``tests/exemptions.toml`` lists ``(env, family, checks, reason)``; ``env`` may be a glob over class names.
An exempted check still runs and is expected to fail; once it passes, the entry is stale and the run
fails until it is removed.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

import pytest

from tests.helpers.families import FAMILIES, ROOT

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

FILE = ROOT / "tests" / "exemptions.toml"
CONFORMANCE = ("specs", "seeded", "lanes", "state", "variation", "batch_render", "native_size")
COVERAGE = ("owner", "original", "swm", "variation_map")
CHECKS = CONFORMANCE + COVERAGE
ENTRIES: list[dict] = tomllib.loads(FILE.read_text()).get("exemption", [])


def waiver(name: str, check: str) -> str | None:
    found = [e["reason"] for e in ENTRIES if check in e["checks"] and fnmatch.fnmatchcase(name, e["env"])]
    return found[0] if found else None


def source_owners(path: str) -> list[str]:
    """The ci/families.toml families whose env source paths hold ``path``."""
    return [
        family
        for family, spec in FAMILIES.items()
        if any(
            fnmatch.fnmatchcase(path, glob) for glob in spec.get("paths", ()) if glob.startswith("stable_envs/envs/")
        )
    ]


def owners(cls) -> list[str]:
    return source_owners(Path(sys.modules[cls.__module__].__file__).resolve().relative_to(ROOT).as_posix())


def family_of_env(cls) -> str:
    found = owners(cls)
    return found[0] if len(found) == 1 else "core"


# Envs whose CPU frames come from MuJoCo's OpenGL renderer.
EGL_RENDERED = ("OGBCube*", "OGBScene*")


def param(name: str, cls, check: str):
    """``name`` as a parameter owned by its env's family, expected to fail when ``check`` is waived."""
    family = family_of_env(cls)
    marks = [pytest.mark.family(family)]
    if check in CONFORMANCE and "gl" in FAMILIES[family].get("capabilities", ()):
        marks.append(pytest.mark.gl)
    if check in ("specs", "batch_render", "native_size") and any(
        fnmatch.fnmatchcase(name, glob) for glob in EGL_RENDERED
    ):
        marks.append(pytest.mark.egl)
    reason = waiver(name, check)
    if reason:
        marks.append(pytest.mark.xfail(reason=f"exempt: {reason}", strict=True))
    return pytest.param(name, marks=marks, id=name)
