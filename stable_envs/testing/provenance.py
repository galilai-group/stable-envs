"""Was a fidelity bound measured against the reference this interpreter is running?

A declaration's numbers describe a pair: our renderer AND the official renderer at the version the
measurement was taken against. When the reference package moves, the official frame can move with it
and the declaration stops describing anything -- DM Control's trackcom cells were judged, declared
under mujoco 3.11.0, against mujoco 3.12.0, which re-anchors finite-plane textures (changelog 3.12.0
item 15, commit ``cc7fb98c``); the reference's floor checker had phase-shifted, ours had not, and three
renderer FAILs were recorded for a version drift (``tests/fixtures/regression/join/step6/texture-fidelity/findings.md``).

The rule this module implements is the one already applied to a reference that is missing: a
dependency failure is not a fidelity failure. A reference at the WRONG version is the same fact one
step later -- the comparison could not be made -- so the check is ``blocked_environment``, with the
mismatch recorded, and never ``fail``.

Library code so the certifier (``scripts/regression/fidelity/certify.py``) and anything else that judges a bound
share one definition of "mismatch".
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from .evidence import sha256

#: ``reference_packages`` values that mean "installed, version unknown" (``fidelity_step4.provenance``
#: writes ``"present"`` for a package with no ``__version__``). Unknown cannot confirm a match.
UNKNOWN = (None, "", "present")

#: The checks whose bound is a claim about OUR frame against the OFFICIAL one, and so about the
#: reference's version. Self-consistency checks (determinism, batch invariance, the 2x downsample)
#: never read the reference and are judged regardless.
REFERENCE_JUDGED_CHECKS = (
    "render_parity_with_official_renderer",
    "check_visual_fidelity",
    "check_visual_fidelity_under_variation",
)


def normalize_package(name: str) -> str:
    """``mujoco-warp`` and ``mujoco_warp`` are one package; compare them as one key."""
    return str(name).strip().lower().replace("-", "_")


def pinned_source_identity(root, commit, versions) -> dict:
    """New source profiles require explicit pins and a clean matching checkout.

    Historical undeclared profiles keep their old behavior below. Runtime math
    identity is recorded for replay diagnostics, not asserted portable by versions.
    """
    if not commit or not versions or any(value in UNKNOWN for value in versions.values()):
        raise ValueError("New source comparisons require a commit and explicit package pins")
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
    """Every declared reference version the measuring interpreter did not match, as sentences.

    ``declared`` is the env's ``reference_versions`` (``stable_envs.env.reference_versions_of``);
    ``recorded`` is the artifact's ``provenance.reference_packages``. Empty list means every declared
    version was recorded and equal -- or nothing was declared, which is not a mismatch: an
    undeclared version is the pre-field state of the world and is judged as before.

    A declared package the interpreter did not record (or recorded as ``"present"``) IS a mismatch:
    the comparison cannot be confirmed, and "could not confirm" is the blocked case, not the passing
    one.
    """
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
    """What a certifier records for ``check`` when the reference version does not match, or ``None``.

    Only for ``REFERENCE_JUDGED_CHECKS``; every other check is judged whatever the versions.
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
            + "; this interpreter's reference is not that version, so the measurement is a "
            "different comparison and cannot pass or fail the declaration (" + "; ".join(bad) + ")"
        ),
        "reference_version_mismatch": bad,
        "declared_reference_versions": dict(declared or {}),
        "recorded_reference_packages": {k: rec_norm.get(normalize_package(k)) for k in (declared or {})},
    }
