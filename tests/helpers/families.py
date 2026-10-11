"""Family ownership, requirements and machine capabilities, as declared in ci/families.toml.

A test belongs to one family: an explicit ``@pytest.mark.family("<name>")``, else the first family
whose ``tests`` globs match its file name, else ``core``. A family whose ``requires`` are not
installed at the stated versions is absent: its failures are reported as skips with that reason,
and ``SE_REQUIRE_FAMILIES`` makes the absence a usage error. Capabilities (``gl``, ``swm``) are
probed once per process.
"""

from __future__ import annotations

import fnmatch
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from collections import Counter
from functools import cache
from pathlib import Path

from packaging.requirements import Requirement

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((ROOT / "ci" / "families.toml").read_text())
FAMILIES: dict[str, dict] = CONFIG["family"]

REQUIRE_FAMILIES = "SE_REQUIRE_FAMILIES"
REQUIRE_CAPABILITIES = "SE_REQUIRE_CAPABILITIES"

# dm_control's renderer after stable_envs has chosen the GL backend; it is disabled on macOS.
_GL_PROBE = """
import importlib.util, sys
if importlib.util.find_spec("dm_control") is None:
    sys.exit(1)
from stable_envs._core.mujoco_gl import configure_mujoco_gl
configure_mujoco_gl()
from dm_control import _render
sys.exit(0 if _render.BACKEND not in (None, "off", "disable") else 1)
"""

CAPABILITIES = {
    "gl": "dm_control has no OpenGL rendering backend here",
    "swm": "stable_worldmodel is not importable",
}


def required(variable: str) -> set[str]:
    return {name.strip() for name in os.environ.get(variable, "").split(",") if name.strip()}


def family_of(item) -> str:
    marker = item.get_closest_marker("family")
    if marker is not None:
        return marker.args[0]
    return family_of_file(Path(str(getattr(item, "path", item.fspath))).name)


def family_of_file(name: str) -> str:
    for family, spec in FAMILIES.items():
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in spec.get("tests", ())):
            return family
    return "core"


@cache
def unmet(family: str) -> tuple[str, ...]:
    """Requirements of ``family`` this interpreter does not satisfy, as short phrases."""
    missing = []
    for entry in FAMILIES[family].get("requires", ()):
        req = Requirement(entry)
        try:
            version = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(req.name)
            continue
        if req.specifier and not req.specifier.contains(version, prereleases=True):
            missing.append(f"{req.name} {version}, needs {req.specifier}")
    return tuple(missing)


def absent(family: str) -> str | None:
    """Why ``family`` cannot run here, or None when it is installed or this run requires it."""
    missing = unmet(family)
    if not missing or family in required(REQUIRE_FAMILIES):
        return None
    return f"family {family} not installed ({'; '.join(missing)})"


@cache
def have(capability: str) -> bool:
    cached = os.environ.get(f"SE_CAPABILITY_{capability.upper()}")
    if cached in ("0", "1"):
        return cached == "1"
    if capability == "swm":
        ok = importlib.util.find_spec("stable_worldmodel") is not None
    elif capability == "gl":
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
        try:
            ok = (
                subprocess.run([sys.executable, "-c", _GL_PROBE], env=env, capture_output=True, timeout=120).returncode
                == 0
            )
        except subprocess.TimeoutExpired:
            ok = False
    else:
        raise KeyError(f"unknown capability {capability!r}; known: {sorted(CAPABILITIES)}")
    os.environ[f"SE_CAPABILITY_{capability.upper()}"] = "1" if ok else "0"
    return ok


def check_required() -> list[str]:
    """Why the families and capabilities this run must have are absent; empty when all are present."""
    problems = []
    for family in sorted(required(REQUIRE_FAMILIES)):
        if family not in FAMILIES:
            problems.append(f"{REQUIRE_FAMILIES} names unknown family {family!r}; known: {', '.join(FAMILIES)}")
        elif unmet(family):
            problems.append(f"{REQUIRE_FAMILIES} requires family {family}, but: {'; '.join(unmet(family))}")
    for capability in sorted(required(REQUIRE_CAPABILITIES)):
        if capability not in CAPABILITIES:
            problems.append(f"{REQUIRE_CAPABILITIES} names unknown capability {capability!r}")
        elif not have(capability):
            problems.append(f"{REQUIRE_CAPABILITIES} requires {capability}, but {CAPABILITIES[capability]}")
    return problems


def skip_cause(report) -> str:
    """One line naming why a test was skipped, so equal causes group together."""
    longrepr = report.longrepr
    reason = longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)
    reason = reason.removeprefix("Skipped: ").strip()
    if reason.startswith("could not import "):
        module = reason.split("'")[1] if "'" in reason else reason.split()[3]
        return f"module {module} not installed"
    return reason.splitlines()[0][:160] if reason else "no reason given"


def grouped(reports, limit: int = 25) -> list[str]:
    counts = Counter(skip_cause(report) for report in reports)
    lines = [f"{count:6d}  {cause}" for cause, count in counts.most_common(limit)]
    rest = sum(count for _, count in counts.most_common()[limit:])
    if rest:
        lines.append(f"{rest:6d}  ({len(counts) - limit} other causes; run with -rs for each)")
    return lines
