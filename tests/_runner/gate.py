from __future__ import annotations

# Gate exclusions require an incompatible-dependency reason, not merely a missing installation.
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
    """Return the declared exclusion reason for a name prefix, or None."""
    for prefix, why in EXCLUDED.items():
        if name.startswith(prefix):
            return why
    return None


def gate_envs(names) -> list[str]:
    """Return sorted names after removing explicitly declared gate exclusions."""
    return sorted(n for n in names if not is_excluded(n))


if __name__ == "__main__":  # the launcher's way in: no imports, no GPU, one line
    import json
    import pathlib
    import sys

    cover = json.loads(
        (
            pathlib.Path(__file__).resolve().parents[2]
            / "tests"
            / "fixtures"
            / "regression"
            / "census"
            / "assignment.json"
        ).read_text()
    )
    print(",".join(gate_envs({e for es in cover.values() for e in es})))
    sys.exit(0)
