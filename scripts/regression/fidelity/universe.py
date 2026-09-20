"""What Step 4 validates: the render-distinct configurations the repo currently CLAIMS.

Computed from Step 1's committed artifacts rather than typed out, for the same reason Step 3 computed
its universe: a hand-maintained list of 52 names drifts from the matrix the moment either moves, and
the drift is silent -- a configuration quietly dropped from the list is indistinguishable, in the
report, from a configuration that passed.

Two inputs, both committed:

    tests/fixtures/regression/fidelity/swm_matrix.json     the 42 swm rows, of which 34 carry a stable-envs match, and the
                                per-row ``render_parameter_space_coverage`` that says which OTHER
                                render-distinct configurations each registered id can build and which
                                of them we render.
    tests/fixtures/regression/census/census_*.json  ``env -> reference_id``, so a covered configuration named by its
                                official id can be resolved to the env that draws it. Read from the
                                census and not from the live registry on purpose: no interpreter holds
                                the whole registry, so a universe built from ``se.REGISTRY`` would be
                                a different set in each shard and the shards could not be merged.

THE ONE COLLAPSE, AND WHY IT IS THE ONLY ONE. Step 1 measured that OGBench's dataset suffixes
(``Navigate``/``Stitch``/``Explore``) over one ``robot x maze`` cell compile to one rendered scene:
15 cells, 24 registered envs, 11 cells covered. Validating all 24 would be counting task names, which
section 6 of the brief forbids; so one env per CELL is validated in full and its siblings are
RE-MEASURED against it on the same frozen states (``sibling_digest_matches``) rather than assumed
equal. Everywhere else no collapse is applied at all -- ``Piecewise``/``PiecewiseZones`` and
``TwoRoom``/``TwoRoomTarget`` share a ``reference_id`` and are NOT assumed to share a picture, and
each of the twelve Fetch envs is validated on its own.

The result is 52 configurations against 34 swm rows and 37 matched envs. The excess is real coverage,
not double counting: 3 DM Control configurations (``reacher-easy``, ``finger-turn_easy``,
``finger-spin``) and 10 further OGBench cells are claimed by the parameter-space axis of a row whose
DEFAULT is a different configuration, and Step 1 counted them as covered.
"""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError("repo root not found")


ROOT = _repo_root()

#: ``reference_id`` prefix whose envs differ only by which offline DATASET they were built for, which
#: Step 1 measured to be one rendered scene per ``robot x maze`` cell. The only family that collapses.
CELL_COLLAPSE_PREFIX = "ogbench/"


def read_censuses(paths=None) -> dict:
    """``env name -> its declaration``, unioned over the committed censuses. Step 3's reader, verbatim."""
    paths = sorted(paths or (ROOT / "tests" / "fixtures" / "regression" / "census").glob("census_*.json"))
    out: dict[str, dict] = {}
    for p in paths:
        txt = Path(p).read_text()
        c = json.loads(txt.split("CENSUS ", 1)[1] if txt.startswith("CENSUS ") else txt)
        for k, v in (c.get("declarations") or {}).items():
            out.setdefault(k, v)
    return out


def _envs_for_reference(decls: dict, ref: str) -> list[str]:
    return sorted(n for n, d in decls.items() if d.get("reference_id") == ref)


def configurations(matrix_path: Path | None = None, decls: dict | None = None) -> list[dict]:
    """The Step 4 validation universe: one entry per render-distinct configuration we claim.

    Deterministically ordered by ``config_id``. Each entry carries where it came from, so a reader can
    tell a configuration that IS somebody's registered default from one that is only reachable through
    a constructor argument, and so the report can say which swm rows a result projects onto.
    """
    m = json.loads(
        Path(matrix_path or ROOT / "tests" / "fixtures" / "regression" / "fidelity" / "swm_matrix.json").read_text()
    )
    decls = decls if decls is not None else read_censuses()
    cfg: dict[str, dict] = {}

    def touch(env: str) -> dict:
        d = decls.get(env, {})
        return cfg.setdefault(
            env,
            {
                "config_id": env,
                "env": env,
                "siblings": [],
                "swm_ids": [],
                "roles": [],
                "reference_id": d.get("reference_id"),
                "fidelity_tier": d.get("fidelity_tier"),
                "backends": d.get("backends") or [],
                "default_size": d.get("default_size"),
                "render_decoder": d.get("render_decoder"),
                "module": d.get("module"),
                # The declared parity contract, so the report can print the floor each row was actually
                # held to instead of assuming one. A relaxed floor is legitimate and must carry a
                # ``parity_note`` naming the mechanism -- tests/gpu/test_conformance.py enforces that on
                # the class; Step 4 shows it.
                "parity_interior_min": d.get("parity_interior_min"),
                "parity_all_exact_min": d.get("parity_all_exact_min"),
                "parity_note": (d.get("parity_note") or "").strip(),
                "registered_default_coverage": None,
                "claimed_by": [],
            },
        )

    def representative(env: str) -> str:
        """The env Step 4 validates for ``env``'s configuration, and the siblings it stands for."""
        ref = (decls.get(env) or {}).get("reference_id") or ""
        if not ref.startswith(CELL_COLLAPSE_PREFIX):
            return env
        family = _envs_for_reference(decls, ref)
        return family[0] if family else env

    for r in m["rows"]:
        envs = r["stable_envs"]["envs"]
        if not envs:
            continue  # a renderer gap or a symbolic id: nothing to check
        covered = r["axes"]["registered_default_render_covered"]["covered"]
        coverage = "full" if covered is True else "partial"
        for env in envs:  # the REGISTERED DEFAULT configuration(s)
            rep = representative(env)
            c = touch(rep)
            c["swm_ids"].append(r["swm_id"])
            c["roles"].append("registered_default")
            c["registered_default_coverage"] = coverage
            c["claimed_by"].append(
                {
                    "swm_id": r["swm_id"],
                    "axis": "registered_default",
                    "detail": r["axes"]["registered_default_render_covered"]["kind"],
                }
            )
            if rep != env and env not in c["siblings"]:
                c["siblings"].append(env)

        par = r["axes"]["render_parameter_space_coverage"]
        for cell in par.get("configs_covered") or []:  # ADDITIONALLY claimed configurations
            # A parameterised family names a covered configuration by official id; the dm_control
            # branch names it by a render-equivalence CLASS of task names, whose members all resolve
            # to the same picture and so to at most one of our envs.
            refs = (
                [cell]
                if isinstance(cell, str)
                else [f"{(r['stable_envs'].get('reference_ids') or [''])[0].split('-', 1)[0]}-{t}" for t in cell]
            )
            hits = sorted({e for ref in refs for e in _envs_for_reference(decls, ref)})
            if not hits:
                continue
            rep = representative(hits[0])
            c = touch(rep)
            if r["swm_id"] not in c["swm_ids"]:
                c["swm_ids"].append(r["swm_id"])
            c["roles"].append("parameter_space")
            c["claimed_by"].append(
                {
                    "swm_id": r["swm_id"],
                    "axis": "render_parameter_space",
                    "detail": f"{par['level']} coverage of {par.get('configs_total')} configurations",
                }
            )
            for e in hits:
                if e != rep and e not in c["siblings"]:
                    c["siblings"].append(e)
            if c["registered_default_coverage"] is None:
                c["registered_default_coverage"] = "n/a"

    for c in cfg.values():
        c["swm_ids"] = sorted(set(c["swm_ids"]))
        c["siblings"] = sorted(set(c["siblings"]))
        c["role"] = "registered_default" if "registered_default" in c["roles"] else "parameter_space"
        c.pop("roles")
    return [cfg[k] for k in sorted(cfg)]


def shard_of(c: dict) -> str:
    """Which interpreter a configuration needs. Named so a shard file says what it was launched for."""
    mod = (c.get("module") or "").rsplit(".", 1)[-1]
    return {
        "dmcontrol": "dmc",
        "fetch": "fetch",
        "ogbench": "ogb",
        "craftax": "swm",
        "swm_nav": "swm",
        "simple_point_maze": "swm",
        "navigation": "swm",
    }.get(mod, "raster")


if __name__ == "__main__":
    from collections import Counter

    cs = configurations()
    print(
        f"{len(cs)} render-distinct configurations claimed, "
        f"{len({i for c in cs for i in c['swm_ids']})} swm rows, "
        f"{len({c['env'] for c in cs} | {s for c in cs for s in c['siblings']})} envs touched"
    )
    print(Counter(shard_of(c) for c in cs))
    for c in cs:
        sib = f"  +{len(c['siblings'])} sibling(s)" if c["siblings"] else ""
        print(
            f"  {shard_of(c):7s} {c['config_id']:26s} {c['role']:19s} "
            f"{c['fidelity_tier'] or '?':7s} cov={str(c['registered_default_coverage']):8s}"
            f" {','.join(c['swm_ids'])[:44]}{sib}"
        )
