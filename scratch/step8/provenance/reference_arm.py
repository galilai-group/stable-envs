"""Which ARM of the Step-7 sweep is the authoritative certification for a reference-judged cell.

    python scratch/step8/provenance/reference_arm.py

writes ``examples/results/step8/provenance/reference_arm.json``.

This is a SELECTION, not a measurement. Nothing renders, no threshold moves, no frozen artifact is
written. Every arm it reads is a committed Step-7 shard, read by sha256.

The problem it fixes, precisely
-------------------------------
The Step-7 closure2 sweep measured the four DM Control trackcom cells TWICE, in the same SLURM job,
on the same card, in two interpreters:

    examples/results/step7/closure2/replay/dmcvis.json   mujoco 3.11.0 + mujoco_warp 3.11.0   shard "adhoc"
    examples/results/step7/closure2/replay/dmc.json      mujoco 3.12.0 + mujoco_warp 3.12.0   shard "dmc"

The four classes DECLARE ``reference_versions = {"mujoco": "3.11.0"}`` (``_AT_VERSIONS`` in
``stable_envs/envs/dmcontrol.py``), so ``dmcvis`` is the declared reference and ``dmc`` is not.
``scratch/step4/certify.py:merge`` keeps, of two equally rich rows, the one whose SHARD TAG sorts
last -- and ``"adhoc"`` sorts before ``"dmc"``. The declared-reference arm was therefore dropped from
``replay/certify/certification.json`` by an alphabetical tiebreak that knows nothing about reference
versions, and the support matrix read the drifted arm.

That is the whole mechanism. It is a JOIN defect, not a renderer defect and not a threshold
question: the frozen judge already refuses to fail a reference-judged check measured against an
undeclared reference (``stable_envs/testing/provenance.py``), which is why the drifted arm reads
``blocked_environment`` rather than ``FAIL`` on ``check_visual_fidelity``.

What this file does, and the three rules it will not break
----------------------------------------------------------
1. **The declaration picks the arm; this file never does.** The authoritative arm is the one whose
   recorded ``provenance.reference_packages`` satisfies the CLASS's declared ``reference_versions``,
   decided by ``stable_envs.testing.provenance.reference_version_mismatch`` -- the same function the
   certifier uses. A cell whose declaration no arm satisfies gets ``UNMEASURED`` and the exact gap,
   never a quiet pass.
2. **The drifted arm is kept, judged, and reported.** Deleting the 3.12 finding would be the real
   damage here: it is the measurement of an upstream change that a user running mujoco >= 3.12 will
   see. Both arms are judged with the frozen judge and both verdicts are in the record.
3. **The judge is frozen.** ``scratch/step4/certify.py:verdict``, unmodified, with the shard's own
   provenance passed in, exactly as the Step-8 rejudgment does.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

REPLAY = ROOT / "examples/results/step7/closure2/replay"
OUT = ROOT / "examples/results/step8/provenance/reference_arm.json"
BASELINE_SHA = "43ac95238f7472137619f3e69e7636e11dd088b1"

#: The cells this selection covers: the four DM Control trackcom cells whose declaration names a
#: reference version and which the closure2 sweep measured in more than one interpreter. Listed
#: explicitly rather than discovered, because "every cell measured twice" is a different and much
#: larger claim than the one this file makes.
CELLS = ("DMCCheetahRun", "DMCHopperHop", "DMCHumanoidStand", "DMCWalkerWalk")

#: Candidate arms, as committed shard paths. Order is irrelevant: the declaration decides.
ARMS = ("dmcvis.json", "dmc.json")

#: Provenance fields copied into the record. Everything a reader needs to re-run the arm.
PROV_FIELDS = ("host", "slurm_job_id", "slurm_nodelist", "gpu", "driver", "cpu", "os", "python",
               "torch", "mujoco_gl", "device", "git_commit", "git_commit_source", "git_dirty",
               "timestamp", "check_seed", "state_seed", "n_states", "pool", "reference_packages")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _judge():
    spec = importlib.util.spec_from_file_location("step8_frozen_judge", ROOT / "scratch/step4/certify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def declared_reference_versions(cell: str) -> tuple[dict, str]:
    """The CLASS's declared reference versions, with the precedence ``describe()`` publishes.

    Read off ``type(env)`` without constructing anything, through the library's own accessor so this
    file and ``describe()`` cannot disagree about what a cell declares.
    """
    from stable_envs.core.env import reference_versions_of
    import stable_envs
    cls = stable_envs.REGISTRY.get(cell)
    if cls is None:
        return {}, f"{cell} is not in stable_envs.REGISTRY in this interpreter"
    return dict(reference_versions_of(cls) or {}), f"stable_envs.REGISTRY[{cell!r}] -> {cls.__name__}"


def build() -> dict:
    from stable_envs.testing.provenance import reference_version_mismatch

    judge_mod = _judge()
    judges = judge_mod.judges(judge_mod.B.load())

    shards = {}
    for name in ARMS:
        path = REPLAY / name
        doc = json.loads(path.read_bytes())
        shards[name] = (path, doc)

    cells = {}
    for cell in CELLS:
        declared, how = declared_reference_versions(cell)
        arms = []
        for name in ARMS:
            path, doc = shards[name]
            row = doc.get("configs", {}).get(cell)
            if row is None:
                continue
            prov = doc.get("provenance") or {}
            mismatch = reference_version_mismatch(declared, prov.get("reference_packages"))
            result = judge_mod.verdict(json.loads(json.dumps(dict(row, shard=doc.get("shard")))),
                                       judges, prov)
            arms.append({
                "arm": name,
                "shard_tag": doc.get("shard"),
                "source": str(path.relative_to(ROOT)),
                "source_sha256": _sha256(path),
                "reference_version_mismatch": mismatch,
                "satisfies_declaration": not mismatch,
                "provenance": {k: prov[k] for k in PROV_FIELDS if k in prov},
                "gate_a": result["gate_a"],
                "failures": result["failures"],
                "blocked": result["blocked"],
                "not_applicable": result["not_applicable"],
                "checks": {n: {k: v for k, v in c.items() if k != "measured_evidence"}
                           for n, c in result["checks"].items()},
                "physics": result.get("physics"),
            })
        matching = [a for a in arms if a["satisfies_declaration"]]
        if len(matching) == 1:
            authoritative, why = matching[0]["arm"], (
                "the one arm whose recorded reference_packages satisfy the class's declared "
                "reference_versions")
        elif not matching:
            authoritative, why = None, (
                "UNMEASURED: no committed arm was measured against the declared reference "
                + ", ".join(f"{k} {v}" for k, v in sorted(declared.items()))
                + ". The gap is exactly: " + "; ".join(
                    f"{a['arm']} -> " + "; ".join(a["reference_version_mismatch"]) for a in arms))
        else:
            authoritative, why = None, (
                "AMBIGUOUS: more than one arm satisfies the declaration ("
                + ", ".join(a["arm"] for a in matching)
                + "); a selection here would be this file choosing, which rule 1 forbids")
        for a in arms:
            a["role"] = ("authoritative_certification" if a["arm"] == authoritative
                         else "reference_drift_recorded_not_authoritative")
        cells[cell] = {
            "declared_reference_versions": declared,
            "declaration_read_from": how,
            "authoritative_arm": authoritative,
            "why": why,
            "gate_a": next((a["gate_a"] for a in arms if a["arm"] == authoritative), "BLOCKED"),
            "arms": arms,
        }

    return {
        "what_this_is":
            "Which measured ARM is the authoritative Gate-A certification for each DM Control "
            "trackcom cell, decided by the class's declared reference_versions. A SELECTION over "
            "committed Step-7 measurements: nothing rendered, no threshold moved, no frozen "
            "artifact written. Both arms are judged and both verdicts are kept.",
        "baseline_sha": BASELINE_SHA,
        "generated_at_head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                         text=True).strip(),
        "judge": "scratch/step4/certify.py:verdict",
        "judge_sha256": _sha256(ROOT / "scratch/step4/certify.py"),
        "selection_rule": "stable_envs.testing.provenance.reference_version_mismatch(class "
                          "reference_versions, shard provenance.reference_packages) == []",
        "why_the_closure2_certification_chose_the_other_arm":
            "scratch/step4/certify.py:merge keeps, between two equally rich rows, the one whose "
            "shard tag sorts last (_richness's third element). dmcvis.json is tagged 'adhoc' and "
            "dmc.json 'dmc', so the declared-reference arm lost an alphabetical tiebreak. The "
            "closure2 certification.json is not edited by this file and stays the record of that.",
        "cells": cells,
    }


if __name__ == "__main__":
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    for cell, rec in sorted(doc["cells"].items()):
        print(f"{cell}: authoritative={rec['authoritative_arm']} gate_a={rec['gate_a']}")
        for a in rec["arms"]:
            print(f"    {a['arm']:<12} {a['shard_tag']:<7} mujoco="
                  f"{(a['provenance'].get('reference_packages') or {}).get('mujoco')} "
                  f"gate_a={a['gate_a']:<8} fail={a['failures']} blocked={a['blocked']} "
                  f"[{a['role']}]")
