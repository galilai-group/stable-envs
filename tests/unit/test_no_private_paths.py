"""The benchmarks use only the public stable-envs API, and the repository holds no cluster or pod paths."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"

#: Private attributes that benchmarks/ may read, each with the reason. Attributes of ``self`` and ``cls`` are not
#: checked.
ALLOWED_PRIVATE = {
    "_ob_type": "OGBench's pixel task has no public switch to state observations",
    "_worker": "a hidden argparse flag of benchmarks/dmc_mjwarp.py",
}

#: Private stable-envs members named in shell code, such as ``python -c`` lines.
PRIVATE_IN_TEXT = re.compile(r"stable_envs\._|stable_envs\.envs\.|\._(render|pack_render_state|simulator)\b")

#: Paths and names of one cluster or pod. Pod scripts keep these outside the repository.
SITE_SPECIFIC = re.compile(
    r"(?<![\w./-])/(oscar|workspace)\b|/home/sboughan|\bPOD_NAME\b|sbatch\s+-p\s+qed|(?<![\w.-])fleet/"
)


def _private_imports(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            names: list[str] = []
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules = [node.module]
            names = [alias.name for alias in node.names]
        else:
            continue
        for module in modules:
            parts = module.split(".")
            if parts[0] != "stable_envs":
                continue
            if any(p.startswith("_") for p in parts) or parts[1:2] == ["envs"]:
                found.append(f"line {node.lineno}: import {module}")
            found += [f"line {node.lineno}: from {module} import {n}" for n in names if n.startswith("_")]
    return found


def _private_attributes(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        name = node.attr
        if not name.startswith("_") or name.startswith("__") or name in ALLOWED_PRIVATE:
            continue
        if isinstance(node.value, ast.Name) and node.value.id in ("self", "cls"):
            continue
        found.append(f"line {node.lineno}: .{name}")
    return found


@pytest.mark.parametrize("path", sorted(BENCHMARKS.rglob("*.py")), ids=lambda p: str(p.relative_to(ROOT)))
def test_benchmark_python_uses_only_public_stable_envs(path):
    tree = ast.parse(path.read_text())
    problems = _private_imports(tree) + _private_attributes(tree)
    assert not problems, f"{path.relative_to(ROOT)} reads private members: {problems}"


@pytest.mark.parametrize("path", sorted(BENCHMARKS.rglob("*.sh")), ids=lambda p: str(p.relative_to(ROOT)))
def test_benchmark_shell_uses_only_public_stable_envs(path):
    text = path.read_text().splitlines()
    lines = [f"line {i}: {s.strip()}" for i, s in enumerate(text, 1) if PRIVATE_IN_TEXT.search(s)]
    assert not lines, f"{path.relative_to(ROOT)} reads private members: {lines}"


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the repository is not a git checkout")
    return [ROOT / p for p in out.decode().split("\0") if p and p != "uv.lock"]


def test_repository_holds_no_cluster_or_pod_paths():
    here = Path(__file__).resolve()
    hits = []
    for path in _tracked_files():
        if path.resolve() == here or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for i, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if SITE_SPECIFIC.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:120]}")
    assert not hits, "cluster or pod paths; read them from environment variables:\n" + "\n".join(hits)
