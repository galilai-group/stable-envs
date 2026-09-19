"""Was a fidelity bound measured against the reference this interpreter is running?

A declaration's numbers describe a pair: our renderer AND the official renderer at the version the
measurement was taken against. When the reference package moves, the official frame can move with it
and the declaration stops describing anything -- Step 4 judged DM Control's trackcom cells, declared
under mujoco 3.11.0, against mujoco 3.12.0, which re-anchors finite-plane textures (changelog 3.12.0
item 15, commit ``cc7fb98c``); the reference's floor checker had phase-shifted, ours had not, and three
renderer FAILs were recorded for a version drift (``examples/results/step6/texture-fidelity/findings.md``).

The rule this module implements is the one Step 4 already applies to a reference that is missing: a
dependency failure is not a fidelity failure. A reference at the WRONG version is the same fact one
step later -- the comparison could not be made -- so the check is ``blocked_environment``, with the
mismatch recorded, and never ``fail``.

Library code so the certifier (``scratch/step4/certify.py``) and anything else that judges a bound
share one definition of "mismatch".
"""
from __future__ import annotations

#: ``reference_packages`` values that mean "installed, version unknown" (``fidelity_step4.provenance``
#: writes ``"present"`` for a package with no ``__version__``). Unknown cannot confirm a match.
UNKNOWN = (None, "", "present")

#: The checks whose bound is a claim about OUR frame against the OFFICIAL one, and so about the
#: reference's version. Self-consistency checks (determinism, batch invariance, the 2x downsample)
#: never read the reference and are judged regardless.
REFERENCE_JUDGED_CHECKS = ("render_parity_with_official_renderer", "check_visual_fidelity",
                           "check_visual_fidelity_under_variation")


def normalize_package(name: str) -> str:
    """``mujoco-warp`` and ``mujoco_warp`` are one package; compare them as one key."""
    return str(name).strip().lower().replace("-", "_")


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
    return {"status": "blocked_environment",
            "why": ("reference version mismatch: the bound was measured against "
                    + ", ".join(f"{k} {v}" for k, v in sorted((declared or {}).items()))
                    + "; this interpreter's reference is not that version, so the measurement is a "
                      "different comparison and cannot pass or fail the declaration ("
                    + "; ".join(bad) + ")"),
            "reference_version_mismatch": bad,
            "declared_reference_versions": dict(declared or {}),
            "recorded_reference_packages": {k: rec_norm.get(normalize_package(k))
                                            for k in (declared or {})}}
