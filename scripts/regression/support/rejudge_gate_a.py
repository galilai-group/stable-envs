"""Judge the saved Step-7 Gate-A measurements with the frozen judge, family_specific checks included.

    python scripts/regression/support/rejudge_gate_a.py

This is a REJUDGMENT of historical measurements, not a fresh certification. Nothing renders, no
threshold moves, and the measurement artifacts under ``tests/fixtures/regression/join/step7/closure2/gate_a/`` are
read by sha256 and never written. The judge is ``scripts/regression/fidelity/certify.py:verdict``, unchanged.

Why it exists: the Step-7 closure harness recorded ``family_specific`` checks without judging them
(``scratch/step7/closure/gate_a_new_cells.py``), so a cell could be recorded as PASSING Gate A while
carrying a FAILING Gate-A check -- which ``docs/research/certification.md:143,149`` says is part of
Gate A once the env declares the capability. Judging both groups is the fix.

Judging both groups also means honouring the two APPLICABILITY declarations the classes carry, and
this is the part to read carefully, because narrowing what is judged is exactly what a loophole looks
like. Three rules, and every one of them is checked rather than assumed:

1. **Applicability is READ off the class, never invented here.** ``type(env)`` only -- no env is
   constructed, so no PyFlyt or craftax install is needed to see what the cell declares. A cell with
   no declaration is judged exactly as it was.
2. **The declaration is only honoured where the OFFICIAL renderer's own evidence is attached.**
   ``conformance.color_exactness_applicability`` refuses a declaration missing any of its five
   required keys, and this file refuses a resolution join whose recorded bands disagree with the
   class's, or whose recorded aggregate is not this shard's aggregate.
3. **The measurement stays in the record.** A check judged ``not_applicable`` still carries the
   numbers it measured, so a wiring regression moves a number even where no bound is asserted.

The two joins, both recorded with the sha256 of every file they read:

``CraftaxClassic`` / ``check_resolution_consistency``
    The closure2 shard recorded only the AGGREGATE ``exact_frac`` over its 8-state protocol draw --
    there is no per-state array in it, so it cannot be re-subsetted here. The subset judgment was
    made on GPU hardware, on both devices, and committed to
    ``tests/fixtures/regression/join/step8/certification/oscar_paired_tests.json``; it is joined in by
    ``(cell, device)`` and only after its recorded bands are confirmed to be the class's bands and
    its ``all_bands`` numbers confirmed to be the shard's numbers. The mixed-band number is kept and
    reported as ``all_bands`` -- recorded, never judged -- exactly as ``_downsample_judge`` expects.

``PFRocketLanding`` / ``check_colors_apply_exactly``
    Every probe the check failed names one of the five leaves the class declares, so the check is
    recorded ``not_applicable`` with the declaration and its evidence. A probe naming a leaf the
    class does NOT declare would leave the check measured and failing, which is the narrow scoping
    the ruling requires.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError("repo root not found")


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))


def _frozen_cell_name(cls) -> str:
    """Regression-layer spelling: frozen oracle cells, not unpublished public aliases."""
    from tests.helpers.reference import frozen_from_canonical

    return frozen_from_canonical(cls.__name__)


GATE_A = ROOT / "tests/fixtures/regression/join/step7/closure2/gate_a"
S8 = ROOT / "tests/fixtures/regression/join/step8/certification"
OUTPUT = S8 / "gate_a_rejudged.json"
PAIRED = S8 / "oscar_paired_tests.json"
DOWNSAMPLE = S8 / "downsample_provenance.json"

#: Step-8 RE-SWEEPS: a shard measured in Step 8 on the same corpus and the same protocol as the
#: closure2 one, because the closure2 measurement had a provenance gap that only a new measurement
#: can close. A re-sweep SUPERSEDES the closure2 row for its ``(cell, device)`` and says so, with
#: the superseded shard's path and sha256 kept in the record. The closure2 shard is never deleted
#: and never edited: it stays the history of what was measured when.
RESWEEP = S8 / "gate_a_resweep"

BASELINE_SHA = "43ac95238f7472137619f3e69e7636e11dd088b1"

#: Keys of an applicability declaration that describe it rather than naming a band. Same set as
#: ``conformance._APPLICABILITY_METADATA``, read from there so the two cannot drift apart.


def _metadata_keys() -> frozenset:
    from stable_envs.testing.conformance import _APPLICABILITY_METADATA

    return _APPLICABILITY_METADATA


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_judge():
    spec = importlib.util.spec_from_file_location(
        "step8_frozen_judge", ROOT / "scripts/regression/fidelity/certify.py"
    )
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    return judge


def env_class(cell: str):
    """The CLASS for ``cell``, without constructing it -- ``type(env)`` is all any declaration needs.

    The registry first, because that is the authoritative name -> class map. A cell whose backend is
    not importable here is absent from it (PFRocketLanding needs nvdiffrast), so the fallback finds
    the one module under ``stable_envs/envs/`` that defines the class and imports only that module.
    Rocket's module imports cleanly without PyFlyt; a module that does not is reported, not guessed.
    """
    try:
        from tests.helpers.reference import canonical_from_frozen, class_for_frozen_cell

        cls = class_for_frozen_cell(cell)
        if cls is not None:
            return cls, f"REGISTRY[{cls.__name__!r}] (frozen cell {cell!r})"
        live = canonical_from_frozen(cell)
    except BaseException as e:  # noqa: BLE001
        return None, f"stable_envs could not be imported: {type(e).__name__}: {e}"
    names = (cell,) if live == cell else (cell, live)
    pattern = re.compile(rf"^class ({'|'.join(re.escape(n) for n in names)})\(", re.M)
    for path in sorted((ROOT / "stable_envs" / "envs").glob("*.py")):
        text = path.read_text()
        m = pattern.search(text)
        if not m:
            continue
        found = m.group(1)
        name = f"stable_envs.envs.{path.stem}"
        try:
            return getattr(importlib.import_module(name), found), f"{name}.{found} (module import only)"
        except BaseException as e:  # noqa: BLE001
            return None, f"{name} is not importable here: {type(e).__name__}: {e}"
    return None, f"no module under stable_envs/envs defines class {cell}"


# ----------------------------------------------------------- resolution applicability join


def _resolution_join(cell: str, device: str, evidence: dict, cls, paired: dict) -> tuple[dict, dict]:
    """``(evidence, join record)``. The evidence is rewritten ONLY when every condition below holds."""
    decl = getattr(cls, "resolution_consistency_applicability", None) if cls is not None else None
    if not isinstance(decl, dict) or not decl:
        return evidence, {}
    meta = _metadata_keys()
    declared_bands = {
        k: ("not_applicable" if str(v).strip().lower().startswith("not_applicable") else "applicable")
        for k, v in decl.items()
        if k not in meta
    }
    join = {
        "check": "check_resolution_consistency",
        "cell": cell,
        "device": device,
        "class_declaration": {k: str(v) for k, v in decl.items()},
        "declared_bands": declared_bands,
    }

    # If a shard ever carries per-state data, it is subsetted here rather than joined from elsewhere.
    if evidence.get("per_state") is not None:
        join["result"] = "the shard carries per-state data; not implemented, no shard has it yet"
        return evidence, join

    row = ((paired.get("applicability_on_gpu_hardware") or {}).get("cells") or {}).get(f"{cell}/{device}")
    if row is None:
        join["result"] = (
            "no committed subset judgment for this cell/device, so the AGGREGATE number "
            "stays judged; nothing is narrowed"
        )
        return evidence, join
    if row.get("bands") != declared_bands:
        join["result"] = (
            f"REFUSED: the committed subset judgment records bands {row.get('bands')}, "
            f"which are not the class's {declared_bands}; the aggregate stays judged"
        )
        return evidence, join
    all_bands = row.get("all_bands") or {}
    shard = {k: evidence.get(k) for k in ("exact_frac", "mean_abs", "max_abs")}
    if row.get("applicable") and any(all_bands.get(k) != v for k, v in shard.items()):
        join["result"] = (
            f"REFUSED: the subset judgment's all_bands {all_bands} is not this shard's "
            f"{shard}, so the two are not the same measurement; the aggregate stays judged"
        )
        return evidence, join
    if not row.get("applicable") or row.get("judged") is None:
        join["result"] = (
            "the class declares every band inapplicable; no bound is asserted and the measured number is recorded"
        )
        return {
            **evidence,
            "applicability": {"bands": row.get("bands"), "declaration": decl, "why": row.get("why"), "indices": []},
        }, join

    judged = row["judged"]
    join.update(
        {
            "result": "JOINED",
            "n_judged": row.get("n_judged"),
            "judged": {k: judged.get(k) for k in ("exact_frac", "mean_abs", "max_abs")},
            "all_bands_recorded_not_judged": all_bands,
            "source": str(PAIRED.relative_to(ROOT)),
            "source_sha256": _sha256(PAIRED),
            "measured_on": paired.get("host"),
            "head_sha": paired.get("head_sha"),
            "why": "the closure2 shard records only the aggregate over its 8-state protocol "
            "draw, so the applicable subset cannot be recomputed from it; this is the "
            "committed subset judgment for the same draw, on hardware, joined by "
            "(cell, device) after its bands and its all_bands numbers were confirmed "
            "against the class and against this shard",
        }
    )
    return (
        {
            **evidence,
            **{k: judged.get(k) for k in ("exact_frac", "mean_abs", "max_abs")},
            "applicability": {
                "bands": row.get("bands"),
                "declaration": decl,
                "why": row.get("why"),
                "indices": list(range(int(row.get("n_judged") or 0))),
            },
            "all_bands": all_bands,
            "joined_from": str(PAIRED.relative_to(ROOT)),
        },
        join,
    )


# ----------------------------------------------------------- colour applicability join


def _colour_join(cell: str, device: str, res: dict, cls) -> tuple[dict, dict]:
    """``(check result, join record)``. ``not_applicable`` ONLY where the class declares the leaf."""
    from stable_envs.testing.conformance import color_exactness_applicability

    if cls is None or res.get("status") != "measured":
        return res, {}
    evidence = res.get("evidence") or {}
    probes = list(evidence.get("colors_not_applied") or [])
    app = color_exactness_applicability(cls, probes)
    join = {
        "check": "check_colors_apply_exactly",
        "cell": cell,
        "device": device,
        "honoured": app["honoured"],
        "why": app["why"],
        "declared_leaves": app["declared_leaves"],
        "undeclared_leaves_still_judged": app["undeclared_leaves_still_judged"],
        "measured_colors_not_applied": probes,
        "measured_best_abs_diff": evidence.get("best_abs_diff"),
        "best_abs_diff_note": (
            "not captured by this measurement; conformance.check_colors_apply_exactly "
            "records the per-leaf distance on subsequent runs of this cell, so a knob "
            "coming off its primitive moves a number"
            if evidence.get("best_abs_diff") is None
            else None
        ),
    }
    if not app["honoured"]:
        join["result"] = "no honoured declaration; the check is judged exactly as before"
        return res, join
    if not probes:
        join["result"] = "the check recorded no failing probe; nothing to declare"
        return res, join
    outside = sorted(set(probes) - set(app["not_applicable_probes"]))
    if outside:
        join["result"] = (
            f"REFUSED: {outside} name leaves the class does not declare, so the check stays measured and stays failing"
        )
        return res, join
    join["result"] = "NOT_APPLICABLE (declared, evidence attached)"
    why = (
        f"declared not_applicable on {len(app['declared_leaves'])} named lit RGB leaves. "
        f"{app['why']} The measurement is kept: colors_not_applied={probes}, "
        f"best_abs_diff={evidence.get('best_abs_diff')}."
    )
    return (
        {
            **res,
            "status": "not_applicable",
            "why": why,
            "measured_evidence": evidence,
            "declared_by": f"{_frozen_cell_name(cls)}.color_exactness_applicability",
        },
        join,
    )


# ----------------------------------------------------------- build


def build() -> dict:
    judge_mod = _load_judge()
    judges = judge_mod.judges(judge_mod.B.load())
    paired = json.loads(PAIRED.read_text()) if PAIRED.is_file() else {}

    # (cell, device) -> the superseded closure2 shard, for every row a Step-8 re-sweep replaces.
    resweeps = sorted(RESWEEP.glob("*.json")) if RESWEEP.is_dir() else []
    superseded: dict[tuple, dict] = {}
    for path in sorted(GATE_A.glob("*.json")):
        doc = json.loads(path.read_text())
        for cell in doc.get("configs", {}):
            superseded[(cell, doc["device"])] = {
                "source": str(path.relative_to(ROOT)),
                "source_sha256": _sha256(path),
                "previous_verdict": doc.get("verdicts", {}).get(cell, {}).get("verdict"),
            }
    replaced = {
        (cell, json.loads(p.read_text())["device"])
        for p in resweeps
        for cell in json.loads(p.read_text()).get("configs", {})
    }

    records, joins = [], []
    classes: dict[str, tuple] = {}
    for path in sorted(GATE_A.glob("*.json")) + resweeps:
        raw = path.read_bytes()
        doc = json.loads(raw)
        device = doc["device"]
        is_resweep = path in resweeps
        for cell, row in sorted(doc.get("configs", {}).items()):
            if not is_resweep and (cell, device) in replaced:
                # A Step-8 re-sweep of this exact cell/device exists; its row is the one judged.
                # The closure2 row is not dropped -- it is cited as ``supersedes`` on that record.
                continue
            if cell not in classes:
                classes[cell] = env_class(cell)
            cls, how = classes[cell]
            row = json.loads(json.dumps(row))  # never mutate the loaded artifact
            fam = row.get("family_specific") or {}

            res = fam.get("check_resolution_consistency")
            if res is not None and res.get("status") == "measured":
                ev, join = _resolution_join(cell, device, res.get("evidence") or {}, cls, paired)
                if join:
                    joins.append(join)
                res["evidence"] = ev

            col = fam.get("check_colors_apply_exactly")
            if col is not None:
                new, join = _colour_join(cell, device, col, cls)
                if join:
                    joins.append(join)
                fam["check_colors_apply_exactly"] = new

            result = judge_mod.verdict(row, judges, doc.get("provenance"))
            # Put the measurement back beside a check the declaration made not_applicable, so the
            # record still carries the numbers a regression would move.
            col_res = fam.get("check_colors_apply_exactly") or {}
            rec_chk = result["checks"].get("check_colors_apply_exactly")
            if rec_chk is not None and col_res.get("measured_evidence") is not None:
                rec_chk["measured_evidence"] = col_res["measured_evidence"]
                rec_chk["declared_by"] = col_res["declared_by"]
            records.append(
                {
                    "cell": cell,
                    "device": device,
                    "class_resolved_by": how,
                    "measurement": "step8_resweep" if is_resweep else "step7_closure2",
                    "supersedes": superseded.get((cell, device)) if is_resweep else None,
                    "source": str(path.relative_to(ROOT)),
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "measurement_provenance": doc.get("provenance", {}),
                    "previous_mandatory_only_verdict": doc.get("verdicts", {}).get(cell, {}).get("verdict"),
                    "judgment": result,
                }
            )
    return {
        "what_this_is": "Gate A for the seven Step-7 cells, rejudged from the UNCHANGED closure2 "
        "measurements with the frozen Step-4 judge, mandatory AND family_specific. "
        "Historical rejudgment, not a fresh Step-8 certification: nothing rendered.",
        "baseline_sha": BASELINE_SHA,
        "judge": "scripts/regression/fidelity/certify.py:verdict",
        "judge_sha256": _sha256(ROOT / "scripts/regression/fidelity/certify.py"),
        "generated_at_head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "applicability_is_read_from": "the CLASS (type(env)), never from this file: "
        "resolution_consistency_applicability and "
        "color_exactness_applicability",
        "sources": {
            "gate_a": sorted(str(p.relative_to(ROOT)) for p in GATE_A.glob("*.json")),
            "step8_resweeps": sorted(str(p.relative_to(ROOT)) for p in resweeps),
            "step8_resweep_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in resweeps},
            "subset_judgment": str(PAIRED.relative_to(ROOT)),
            "subset_judgment_sha256": _sha256(PAIRED) if PAIRED.is_file() else None,
            "downsample_provenance": str(DOWNSAMPLE.relative_to(ROOT)),
            "downsample_provenance_sha256": _sha256(DOWNSAMPLE) if DOWNSAMPLE.is_file() else None,
        },
        "applicability_joins": joins,
        "records": records,
    }


if __name__ == "__main__":
    doc = build()
    OUTPUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    for join in doc["applicability_joins"]:
        print(f"  join {join['cell']}/{join['device']} {join['check']}: {join.get('result')}")
    for record in doc["records"]:
        v = record["judgment"]
        print(
            record["cell"],
            record["device"],
            v["gate_a"],
            "fail=",
            v["failures"],
            "blocked=",
            v["blocked"],
            "n/a=",
            v["not_applicable"],
        )
