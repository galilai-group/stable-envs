"""Static and ``--help`` checks for the ``examples/`` scripts, without running a benchmark.

Every script compiles, does no work at import time, and (if it uses argparse) prints ``--help`` in a
subprocess. Scripts needing an optional dependency this interpreter lacks skip with the reason.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"

#: Directories under ``examples/`` that are outputs or vendored, not scripts to check.
SKIP_DIRS = {"results", "media", "logs", "tables", "figures", "config", "__pycache__", ".hydra"}


def _scripts() -> list[Path]:
    out = []
    for p in sorted(EXAMPLES.rglob("*.py")):
        if SKIP_DIRS & set(p.relative_to(EXAMPLES).parts):
            continue
        out.append(p)
    return out


SCRIPTS = _scripts()
IDS = [str(p.relative_to(EXAMPLES)) for p in SCRIPTS]


def test_there_are_scripts_to_check():
    """The script glob finds scripts, so the parametrized tests below are not empty."""
    assert len(SCRIPTS) >= 7, f"only {len(SCRIPTS)} scripts found under {EXAMPLES}"


@pytest.mark.parametrize("path", SCRIPTS, ids=IDS)
def test_every_example_script_parses(path: Path):
    """Each script compiles under the running interpreter."""
    src = path.read_text()
    compile(src, str(path), "exec")


@pytest.mark.parametrize("path", SCRIPTS, ids=IDS)
def test_a_runnable_script_does_its_work_behind_a_main_guard(path: Path):
    """No module-level ``main()``/``run()``/``sweep()`` call; ``test_benchmarks.py`` imports these scripts."""
    tree = ast.parse(path.read_text(), str(path))
    calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    named = [c for c in calls if isinstance(c.value.func, ast.Name) and c.value.func.id in {"main", "run", "sweep"}]
    assert not named, (
        f"{path.relative_to(ROOT)} calls {named[0].value.func.id}() at import time; put it behind "
        f'`if __name__ == "__main__":` — tests/unit/test_benchmarks.py imports scripts from here'
    )


ARGPARSE = [p for p in SCRIPTS if "argparse" in p.read_text()]
ARGPARSE_IDS = [str(p.relative_to(EXAMPLES)) for p in ARGPARSE]


def test_the_argparse_sweep_below_is_not_empty():
    assert len(ARGPARSE) >= 6, f"only {len(ARGPARSE)} scripts under examples/ use argparse"


@pytest.mark.parametrize("path", ARGPARSE, ids=ARGPARSE_IDS)
def test_every_argparse_script_can_print_its_own_help(path: Path):
    """``--help`` imports the module, builds the parser, prints usage and exits 0."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(EXAMPLES), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    p = subprocess.run(
        [sys.executable, str(path), "--help"], env=env, capture_output=True, text=True, timeout=300, cwd=str(ROOT)
    )
    if p.returncode != 0:
        tail = (p.stderr or p.stdout)[-2500:]
        last = [ln for ln in (p.stderr or "").strip().splitlines() if ln.strip()]
        if last and last[-1].startswith(("ModuleNotFoundError", "ImportError")):
            pytest.skip(f"{path.relative_to(ROOT)} needs a package this interpreter does not have: {last[-1]}")
        pytest.fail(f"{path.relative_to(ROOT)} --help exited {p.returncode}:\n{tail}")
    assert "usage" in (p.stdout + p.stderr).lower(), (
        f"{path.relative_to(ROOT)} --help exited 0 without printing a usage line"
    )


def test_the_readme_lists_the_scripts_that_exist():
    """``examples/README.md`` mentions every top-level script and points only at existing ones."""
    readme = (EXAMPLES / "README.md").read_text()
    top = sorted(p.name for p in EXAMPLES.glob("*.py"))
    missing = [n for n in top if n not in readme]
    assert not missing, f"examples/README.md does not mention {missing}"
    for line in readme.splitlines():
        for tok in line.replace("`", " ").replace("(", " ").replace(")", " ").split():
            if tok.startswith("examples/") and tok.endswith(".py"):
                assert (ROOT / tok).exists(), f"examples/README.md points at {tok}, which is not there"
