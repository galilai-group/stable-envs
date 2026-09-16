"""The Step-8 certification universe: the Step-4 configurations plus the seven Step-7 cells.

Step 4 computed its universe from two committed Step-1 artifacts, ``scratch/swm_matrix.json`` and the
censuses, for a reason that still holds: a hand-typed list of configurations drifts from the matrix the
moment either moves, and the drift is silent. Step 8 keeps that discipline and changes only the input
that went stale.

``scratch/swm_matrix.json`` is a **Step-1** artifact. Six of its 42 rows still carry an empty
``stable_envs.envs`` -- CraftaxPixels, Manipulator, Quadruped, OGBCube, OGBScene, PFRocketLanding --
because those cells did not exist when it was written. Step 4's ``universe.configurations()`` skips a
row with no envs (``if not envs: continue``), so the seven Step-7 corpora cannot enter the universe
through it however correct their metadata is. That file is not this workstream's to rewrite: it is the
SWM-integration workstream's join, and regenerating it is a separate change with a separate review.

So this module reads the Step-7 closure matrix instead, which is the artifact that already records the
join at the Step-8 baseline:

    examples/results/step7/closure2/coverage_matrix.json   42 rows, 40 pixel-producing, 2 symbolic,
                                                           ``integration_head_verified``

and composes it with Step 4's own universe, which stays the authority for every configuration it
already produced. Step 4's 52 entries are passed through UNCHANGED: this module adds rows, it never
edits or removes one, and ``check_step4_universe_unchanged()`` asserts exactly that.

    python scratch/step8/certify/universe8.py --check        verify the universe and the corpora
    python scratch/step8/certify/universe8.py --stage DIR    stage the seven corpora into DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scratch" / "step4"))
sys.path.insert(0, str(ROOT))

#: The committed join at the Step-8 baseline. Read, never rewritten.
COVERAGE_MATRIX = ROOT / "examples" / "results" / "step7" / "closure2" / "coverage_matrix.json"

#: The committed record of the seven Step-7 corpora, written by this workstream: where each array
#: lives, which key holds it, how many rows are taken, and the sha256 of the result. Every number in
#: it was read off a committed artifact; nothing was drawn or re-measured.
CORPUS_INDEX = ROOT / "examples" / "results" / "step8" / "certification" / "corpus_index.json"

#: The corpus size every cell in this campaign is judged on, from Step 4 onwards.
CORPUS_N = 64


def corpus_index() -> dict:
    return json.loads(CORPUS_INDEX.read_text())["cells"]


def step7_cells() -> dict[str, dict]:
    """``config_id -> {"swm_ids": [...], "corpus": {...}}`` for the seven Step-7 cells.

    The swm ids come from the coverage matrix's own rows rather than from a table here, so a cell that
    is re-joined upstream is re-joined here too instead of being quietly mismatched.
    """
    rows = json.loads(COVERAGE_MATRIX.read_text())["rows"]
    idx = corpus_index()
    out: dict[str, dict] = {}
    for r in rows:
        for env in r.get("stable_envs_envs") or []:
            if env in idx:
                out.setdefault(env, {"config_id": env, "swm_ids": [], "corpus": idx[env]})
                out[env]["swm_ids"].append(r["swm_id"])
    missing = sorted(set(idx) - set(out))
    if missing:
        raise AssertionError(
            f"{missing} have a frozen Step-7 corpus but no row in {COVERAGE_MATRIX.name} claims them, "
            f"so they would be certified against nothing. Fix the join, not this list.")
    return out


def configurations() -> list[dict]:
    """Step 4's universe, unchanged, plus one entry per Step-7 cell it does not already contain.

    Deterministically ordered by ``config_id``, exactly as Step 4 orders its own.
    """
    import universe as u4                                                   # noqa: PLC0415

    base = {c["config_id"]: c for c in u4.configurations()}
    for cell, info in step7_cells().items():
        if cell in base:
            # CraftaxClassic is the one overlap: Step 4 already claims it, and Step 4's entry wins.
            # Its Step-7 corpus is recorded beside it rather than replacing it -- see
            # ``craftax_classic_corpus_conflict()``, which is the reason this is not a silent merge.
            base[cell].setdefault("step7_corpus", info["corpus"])
            base[cell]["step7_corpus_is_a_second_corpus"] = True
            continue
        base[cell] = {
            "config_id": cell, "env": cell, "siblings": [], "swm_ids": sorted(info["swm_ids"]),
            "roles": ["registered_default"], "registered_default_coverage": "full",
            "claimed_by": [{"swm_id": s, "axis": "registered_default",
                            "detail": "Step-7 renderer coverage; join from "
                                      + str(COVERAGE_MATRIX.relative_to(ROOT))}
                           for s in sorted(info["swm_ids"])],
            "added_in_step": 8, "corpus": info["corpus"],
        }
    return [base[k] for k in sorted(base)]


def check_step4_universe_unchanged() -> dict:
    """Step 4's own entries must come through byte-for-byte. Additive means additive."""
    import universe as u4                                                   # noqa: PLC0415

    before = {c["config_id"]: json.dumps(c, sort_keys=True) for c in u4.configurations()}
    after = {c["config_id"]: c for c in configurations()}
    changed = []
    for cid, blob in before.items():
        got = {k: v for k, v in after[cid].items()
               if k not in ("step7_corpus", "step7_corpus_is_a_second_corpus")}
        if json.dumps(got, sort_keys=True) != blob:
            changed.append(cid)
    return {"step4_configs": len(before), "step8_configs": len(after),
            "added": sorted(set(after) - set(before)), "changed": changed,
            "removed": sorted(set(before) - set(after))}


def stage(out: Path) -> dict:
    """Copy -- never redraw -- the seven committed arrays into the ``<Config>.npz`` schema.

    Same job as ``scratch/step7/closure/stage_corpora.py``, driven by the committed index instead of a
    table in the script, and with the sha256 RE-VERIFIED against the index rather than merely
    recorded. A staged corpus whose digest does not match the index is refused: the point of the index
    is that a later run can show it measured the same states, and a staging step that silently accepts
    a different array destroys exactly that.
    """
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for cell, e in sorted(corpus_index().items()):
        src = ROOT / e["source"]
        z = np.load(src, allow_pickle=False)
        st = np.ascontiguousarray(np.asarray(z[e["key"]], np.float32))
        if e["rows_taken"] != "all":
            st = np.ascontiguousarray(st[: int(e["rows_taken"])])
        sha = hashlib.sha256(st.tobytes()).hexdigest()
        if sha != e["array_sha256"]:
            raise AssertionError(f"{cell}: staged array sha256 {sha} != the committed index's "
                                 f"{e['array_sha256']}; the corpus is not the one that was frozen")
        meta = json.loads(str(z["meta"])) if "meta" in z.files else json.loads(
            (src.parent / f"{src.stem}.meta.json").read_text())
        meta = dict(meta, staged_by="scratch/step8/certify/universe8.py",
                    staged_from=e["source"], staged_key=e["key"], staged_rows=e["rows_taken"],
                    sha256=sha, env=cell, state_dim=int(st.shape[1]))
        np.savez(out / f"{cell}.npz", states=st, meta=np.array(json.dumps(meta, sort_keys=True)))
        report[cell] = {"shape": list(st.shape), "array_sha256": sha, "source": e["source"]}
    return report


def check_corpora(staged: Path) -> dict:
    """``corpus_for`` must return the frozen 64 for every cell, from the staged dir AND in place.

    Both, because they are different claims. The staged dir is what a Gate-A run walks; the in-place
    read is what proves the metadata sidecars this workstream wrote are the metadata ``corpus_for``
    actually reads, rather than something only the staging step understands.
    """
    sys.path.insert(0, str(ROOT / "examples"))
    import fidelity_step4 as F                                              # noqa: PLC0415

    rows = {}
    for cell, e in sorted(corpus_index().items()):
        r: dict = {"expected_sha256": e["array_sha256"], "expected_n": CORPUS_N}
        st, meta = F.corpus_for(None, cell, staged, CORPUS_N, 256)
        r["staged"] = {"n": int(len(st)), "state_dim": int(st.shape[1]),
                       "sha256": hashlib.sha256(np.ascontiguousarray(st, np.float32).tobytes()).hexdigest(),
                       "meta_keys": sorted(meta)}
        src = ROOT / e["source"]
        z = np.load(src, allow_pickle=False)
        in_place = F._corpus_meta(src, z)
        r["in_place_meta_keys"] = sorted(in_place)
        r["in_place_meta_from"] = in_place.get("meta_loaded_from_sidecar", "inside the npz")
        r["ok"] = (r["staged"]["n"] == CORPUS_N and r["staged"]["sha256"] == e["array_sha256"])
        rows[cell] = r
    return rows


def craftax_classic_corpus_conflict() -> dict:
    """The one cell with TWO frozen corpora, and why this workstream did not choose between them.

    ``examples/results/step4/states/CraftaxClassic.npz`` is 64 x 127 and frozen in Step 4.
    ``examples/results/step7/craftax-classic-full/corpus/states.npz`` is 64 x 149 and frozen in Step 7,
    because the cell's state grew when the full 9-row world window was implemented. The Step-7 closure
    replay records CraftaxClassic as ``NOT REPLAYED`` for exactly this reason, and that is the state of
    the world at the Step-8 baseline rather than anything Step 8 caused.

    Retiring a frozen Step-4 corpus is a RE-FREEZE of an authoritative oracle. Section 19 of the Step-8
    plan says that needs a documented mechanism, before/after evidence, independent review and Sami's
    approval. So both are recorded, neither is edited, and the choice is escalated.
    """
    s4 = ROOT / "examples/results/step4/states/CraftaxClassic.npz"
    s7 = ROOT / "examples/results/step7/craftax-classic-full/corpus/states.npz"
    out = {}
    for tag, p in (("step4_frozen", s4), ("step7_frozen", s7)):
        z = np.load(p, allow_pickle=False)
        st = np.asarray(z["states"], np.float32)
        out[tag] = {"path": str(p.relative_to(ROOT)), "shape": list(st.shape),
                    "sha256": hashlib.sha256(np.ascontiguousarray(st).tobytes()).hexdigest()}
    out["conflict"] = ("the two corpora have different state_dim, so at most one of them can be set on "
                       "the cell as it stands at the Step-8 baseline")
    out["resolution"] = "ESCALATED: retiring the Step-4 corpus is a re-freeze (plan section 19)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, help="stage the seven corpora into this directory")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", type=Path, help="write the check report here")
    a = ap.parse_args()
    report: dict = {"universe": check_step4_universe_unchanged(),
                    "craftax_classic_corpus_conflict": craftax_classic_corpus_conflict()}
    staged = a.stage
    if staged is None and a.check:
        # A scratch directory, deliberately, and never a committed one. Staging writes COPIES of
        # frozen arrays; committing them would put a second file with the same digest beside each
        # oracle, and "which of these two is the corpus" is a question a certification tree must not
        # be able to ask. The index records the digest, so a staged copy is verifiable, not precious.
        staged = Path(tempfile.mkdtemp(prefix="se8-states-"))
    if staged is not None:
        report["staged"] = stage(staged)
        report["staged_into"] = str(staged)
    if a.check:
        report["corpora"] = check_corpora(staged)
    print(json.dumps(report, indent=1, sort_keys=True))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
