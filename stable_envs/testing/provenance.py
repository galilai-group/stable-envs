from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypedDict

from .evidence import sha256


class CodeIdentity(TypedDict):
    mode: Literal["historical", "worktree"]
    commit: str | None
    files: dict[str, str]


def verify_recorded_code(
    files: Mapping[str, str],
    root: str | Path,
    *,
    revision: str | None = None,
    sources: Mapping[str, bytes] | None = None,
) -> CodeIdentity:
    """Verify recorded file hashes against the worktree, a Git commit (``revision``) or stored contents.

    A commit is read from Git objects, without checking out or running its code. ``sources`` maps each
    recorded path to the stored bytes of that file; ``revision`` then only names the commit they came from.
    """
    root = Path(root).resolve(strict=True)
    if not files:
        raise ValueError("Missing recorded code identity")
    commit = revision if sources is not None else None
    if revision is not None and sources is None:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
            text=True,
        ).strip()
    for name, expected in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != name or not name:
            raise ValueError(f"Invalid recorded code path: {name!r}")
        if sources is not None:
            if name not in sources:
                raise ValueError(f"Missing stored code: {name}")
            actual = hashlib.sha256(sources[name]).hexdigest()
        elif commit:
            content = subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{name}"])
            actual = hashlib.sha256(content).hexdigest()
        else:
            target = (root / path).resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError(f"Recorded code escapes root: {name}")
            actual = sha256(target)
        if actual != expected:
            raise ValueError(f"Recorded code hash mismatch: {name}")
    historical = commit is not None or sources is not None
    return dict(mode="historical" if historical else "worktree", commit=commit, files=dict(files))


# Unknown recorded versions cannot establish a declared match.
UNKNOWN = (None, "", "present")

# These checks depend on a versioned upstream renderer; self-consistency checks do not.
REFERENCE_JUDGED_CHECKS = ("check_source_mse",)


def normalize_package(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def pinned_source_identity(root, commit, versions) -> dict:
    """Hash a clean source checkout after verifying its commit and installed package pins.

    Platform and math-library settings are recorded for replay diagnostics.
    """
    if not commit or not versions or any(value in UNKNOWN for value in versions.values()):
        raise ValueError("Source comparisons require a commit and explicit package pins")
    root = Path(root).resolve(strict=True)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    actual = git("rev-parse", "HEAD")
    if actual != commit or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Source checkout must be clean and match the pinned commit")
    recorded = {}
    for package in versions:
        try:
            recorded[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            recorded[package] = None
    mismatch = reference_version_mismatch(versions, recorded)
    if mismatch:
        raise ValueError("Source profile blocked: " + "; ".join(mismatch))
    names = git("ls-files", "-z", "stable_worldmodel").split("\0")
    files = {name: sha256(root / name) for name in names if name and (root / name).is_file()}
    if not files:
        raise ValueError("Source checkout contains no tracked source files")
    import numpy as np

    return dict(
        root=str(root),
        commit=actual,
        files=files,
        versions=recorded,
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        numpy_config=np.show_config(mode="dicts"),
        math_environment={
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_CORETYPE",
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NPY_DISABLE_CPU_FEATURES",
            )
        },
    )


def reference_version_mismatch(declared: dict | None, recorded: dict | None) -> list[str]:
    """List missing, unknown or unequal declared reference versions; an empty declaration requires none."""
    if not declared:
        return []
    rec = {normalize_package(k): v for k, v in (recorded or {}).items()}
    out = []
    for pkg, want in sorted(declared.items()):
        got = rec.get(normalize_package(pkg))
        if got in UNKNOWN:
            out.append(f"{pkg}: declared {want}, version not recorded by the measuring interpreter")
        elif str(got).strip() != str(want).strip():
            out.append(f"{pkg}: declared {want}, interpreter had {got}")
    return out


def block_on_reference_mismatch(check: str, declared: dict | None, recorded: dict | None) -> dict | None:
    """Return blocked_environment details for reference-dependent checks with mismatched pins.

    Self-consistency checks are unaffected by reference versions.
    """
    if check not in REFERENCE_JUDGED_CHECKS:
        return None
    bad = reference_version_mismatch(declared, recorded)
    if not bad:
        return None
    rec_norm = {normalize_package(k): v for k, v in (recorded or {}).items()}
    return {
        "status": "blocked_environment",
        "why": (
            "reference version mismatch: the bound was measured against "
            + ", ".join(f"{k} {v}" for k, v in sorted((declared or {}).items()))
            + "; this interpreter has a different reference version, so the measurement cannot "
            "pass or fail the declaration (" + "; ".join(bad) + ")"
        ),
        "reference_version_mismatch": bad,
        "declared_reference_versions": dict(declared or {}),
        "recorded_reference_packages": {k: rec_norm.get(normalize_package(k)) for k in (declared or {})},
    }
