"""Which envs the release gate is required to cover, and the ones deliberately left out of it.

The gate is ``scripts/run_gpu_tests.sbatch``, and its contract is that a run FAILS if an env it was
launched to cover is not reachable — otherwise a lost dependency skips a whole family with a
per-env reason that is individually honest and collectively means nothing was tested. See
``tests/gpu/test_conformance.py::test_required_envs_are_all_reachable``.

That contract needs a list of what "covered" means, and it needs the exceptions to be DECLARED rather
than inferred from whatever happened to import. An env that is absent because nobody installed its
package and an env that is absent because it CANNOT be installed alongside the others are different
facts, and only the second one is allowed to be quiet.

``EXCLUDED`` is that second list. It is deliberately hard to add to: an entry needs a reason naming
the mechanism, and ``tests/certification/test_declarations.py`` asserts every entry has one. Being on it does
NOT remove an env from the library — the module, its parity numbers and its docs all stay, and the env
registers normally in an interpreter that can install it.
"""

from __future__ import annotations

#: ``env-name prefix -> why it cannot be in the same interpreter as the rest``.
#:
#: One entry. Resist adding a second without the same standard of evidence: "it did not import on my
#: machine" is a missing package, not an exclusion.
EXCLUDED: dict[str, str] = {
    "Overcooked": (
        "jaxmarl 0.1.0 requires scipy<=1.12 and jax 0.10.2 requires scipy>=1.14, which no resolution "
        "satisfies. Installing jaxmarl --no-deps to dodge that pin walks a chain into the rest of the "
        "2023 JAX ecosystem (brax, then jaxopt, ...) and arrives at an interpreter where nothing has "
        "verified that Overcooked still renders correctly on modern jax — which is worse than not "
        "running it, because the suite would go green either way. The other 191 envs share one "
        "interpreter; these 5 need their own, and stable_envs/envs/overcooked.py says so at the top. "
        "The permanent fix is to embed the atlas the way craftax/crafter/minihack embed theirs, which "
        "would make jaxmarl build-time-only; until then this is an honest carve-out rather than a "
        "silent skip."
    ),
}


def is_excluded(name: str) -> str | None:
    """The reason ``name`` is outside the gate, or ``None`` if it is inside it."""
    for prefix, why in EXCLUDED.items():
        if name.startswith(prefix):
            return why
    return None


def gate_envs(names) -> list[str]:
    """``names`` minus the declared exclusions, sorted. What one interpreter must be able to reach."""
    return sorted(n for n in names if not is_excluded(n))


if __name__ == "__main__":  # the launcher's way in: no imports, no GPU, one line
    import json
    import pathlib
    import sys

    cover = json.loads(
        (pathlib.Path(__file__).resolve().parents[2] / "scratch" / "census" / "assignment.json").read_text()
    )
    print(",".join(gate_envs({e for es in cover.values() for e in es})))
    sys.exit(0)
