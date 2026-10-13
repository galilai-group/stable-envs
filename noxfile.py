"""One command per test environment, built from uv.lock and ci/families.toml.

    nox -l                              # list sessions
    nox -s "test(core)"                 # core tests on floating dependencies
    nox -s "test(dmc)" -- -k reacher    # one family under its reference pins
    nox -s affected                     # the profiles CI selects for this branch
    SE_PYTHON=3.10 nox -s "test(core)"  # another interpreter for the core profile

Each profile syncs into ``.venvs/<profile>-py<version>`` with ``uv sync --frozen``. With ``CI`` set,
the profile's family and capabilities become required (``SE_REQUIRE_FAMILIES``,
``SE_REQUIRE_CAPABILITIES``), so a lost dependency fails instead of skipping. A profile with the ``swm``
capability also gets stable-worldmodel at ``[swm].commit``, cloned once into ``.venvs/`` (or the checkout
``SE_SWM_SOURCE`` names).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nox

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

nox.options.default_venv_backend = "none"
nox.options.sessions = ["test(core)"]

ROOT = Path(__file__).resolve().parent
CONFIG = tomllib.loads((ROOT / "ci" / "families.toml").read_text())
FAMILIES = CONFIG["family"]
SWM = CONFIG["swm"]
TEST_PATHS = ("tests/unit", "tests/integration", "tests/regression")
# Linux takes CPU-only torch; macOS wheels carry no CUDA.
TORCH = ["torch-cpu"] if sys.platform == "linux" else []


def _sync(session: nox.Session, name: str, *, extras=(), groups=(), python: str = "3.12") -> Path:
    venv = ROOT / ".venvs" / f"{name}-py{python}"
    args = ["uv", "sync", "--frozen", "--python", python, "--no-default-groups"]
    args += [arg for extra in extras for arg in ("--extra", extra)]
    args += [arg for group in groups for arg in ("--group", group)]
    session.run(*args, env={"UV_PROJECT_ENVIRONMENT": str(venv)}, external=True)
    return venv / "bin" / "python"


def _install_swm(session: nox.Session, python: Path, extra=()) -> Path:
    """Editable stable-worldmodel at the [swm] commit, without moving any locked version.

    The gates read the checkout's git HEAD, so it is a clone rather than a wheel. ``SE_SWM_SOURCE`` names an
    existing checkout instead. ``extra`` are distributions the profile's gates need beside it.
    """
    source = Path(os.environ.get("SE_SWM_SOURCE") or ROOT / ".venvs" / f"stable-worldmodel-{SWM['commit'][:12]}")
    if not (source / ".git").exists():
        session.run("git", "init", "-q", str(source), external=True)
        session.run("git", "-C", str(source), "fetch", "-q", "--depth", "1", SWM["url"], SWM["commit"], external=True)
        session.run("git", "-C", str(source), "checkout", "-q", "FETCH_HEAD", external=True)
    frozen = session.run("uv", "pip", "freeze", "--python", str(python), silent=True, external=True)
    constraints = python.parents[1] / "swm-constraints.txt"
    constraints.write_text("".join(f"{line}\n" for line in frozen.splitlines() if "==" in line))
    session.run(
        "uv",
        "pip",
        "install",
        "--python",
        str(python),
        "-c",
        str(constraints),
        # On Linux torch is the CPU build, and swm's torchvision has to come from the same index.
        *(["--extra-index-url", "https://download.pytorch.org/whl/cpu"] if TORCH else []),
        "-e",
        str(source),
        *extra,
        external=True,
    )
    return source


@nox.session
@nox.parametrize("profile", [nox.param(name, id=name) for name in FAMILIES])
def test(session: nox.Session, profile: str) -> None:
    """Unit, integration and CPU regression tests of one family under its profile."""
    spec = FAMILIES[profile]
    python = _sync(
        session,
        profile,
        extras=spec.get("extras", ()),
        groups=["test", *TORCH, *spec.get("groups", ())],
        python=os.environ.get("SE_PYTHON", "3.12"),
    )
    env = {"SE_SKIP_NODE_GATE": "1", "SDL_VIDEODRIVER": "dummy", "MPLBACKEND": "Agg"}
    if "swm" in spec.get("capabilities", ()):
        env["SWM_ROOT"] = str(_install_swm(session, python, spec.get("swm_with", ())))
    if sys.platform == "linux":
        env.update(spec.get("linux_env", {}))
    if os.environ.get("CI"):
        env["SE_REQUIRE_FAMILIES"] = profile
        env["SE_REQUIRE_CAPABILITIES"] = ",".join(spec.get("capabilities", ()))
    workers = os.environ.get("SE_WORKERS", "auto")
    session.run(
        str(python),
        "-m",
        "pytest",
        *TEST_PATHS,
        "-m",
        f"family_{profile} and not gpu and not slow",
        *(["-n", workers] if workers != "0" else []),
        *session.posargs,
        env=env,
        external=True,
    )


@nox.session
def affected(session: nox.Session) -> None:
    """Run test(<profile>) for every profile the diff against origin/main selects."""
    base = os.environ.get("SE_BASE", "origin/main")
    out = session.run(
        sys.executable, "scripts/ci/plan.py", "--base", base, "--profiles-only", silent=True, external=True
    )
    for profile in out.split():
        session.notify(f"test({profile})", posargs=session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """pre-commit over the whole tree, or over the given range: ``-- --from-ref A --to-ref B``."""
    session.run("uvx", "pre-commit", "run", *(session.posargs or ["--all-files"]), external=True)


@nox.session
def typecheck(session: nox.Session) -> None:
    """The adapter interface check."""
    python = _sync(session, "typecheck", groups=["test", "lint", *TORCH])
    session.run(str(python), "-m", "mypy", "--config-file", "tests/typing/mypy.ini", external=True)


@nox.session
def docs(session: nox.Session) -> None:
    """Strict documentation build."""
    python = _sync(session, "docs", groups=["docs", *TORCH])
    session.run(str(python), "-m", "mkdocs", "build", "--strict", "--config-file", "mkdocs.yaml", external=True)
