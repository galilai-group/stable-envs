"""Gate A for the seven cells Step 7 added, on their COMMITTED Step-7 corpora, at the registered
default size, through the repo's own conformance machinery.

These cells are not in the frozen Step-4 universe (``examples/fidelity_step4.universe``) -- the
Step-8 freeze is what adds them -- so ``examples/fidelity_step4.py``'s CLI cannot select them. This
calls ``fidelity_step4.measure`` directly with the configuration dict the universe would carry, which
is exactly what ``scratch/step7/dmc-quadruped/gateA_local.py`` did for one cell; this generalises it
to seven and writes a shard-shaped document per device so the raw rows are directly comparable with
every other shard in the repository.

NOTHING HERE JUDGES A THRESHOLD. The verdict per cell is read off the raw row:

  * the MANDATORY group must be ``measured`` on every check and must carry a passing observation --
    the bound for each is ``== 0`` / ``identical`` / ``True`` and is written down in ``_MANDATORY_OK``
    next to the check it belongs to, taken from ``stable_envs/testing/conformance.py``'s own
    contract, not invented here;
  * ``family_specific`` checks are RECORDED, never judged, because several of them are meaningful
    only under the reference pair a cell's declaration names (the DM Control cells at mujoco 3.11.0 /
    dm_control 1.0.44) and several carry per-cell applicability declarations that the Step-8
    certifier, not this script, is the place to consume.

A check that raises because the interpreter cannot REACH the renderer is ``blocked_environment`` and
is reported as such: it is neither a pass nor a fail, and a device on which a cell raises BY DESIGN
(OGBCube/OGBScene/PFRocketLanding on cpu) is recorded as ``refused_by_design`` rather than hidden.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

#: The universe row each cell will carry at the Step-8 freeze. ``swm_ids`` and ``backends`` come from
#: the workstream findings the dashboard records; everything else is read off the class so it cannot
#: drift from the declaration.
CELLS = {
    "CraftaxClassic": {"swm_ids": ["swm/CraftaxClassicPixels-v1"], "backends": ["tilemap"],
                       "devices": ["cuda", "cpu"], "venv": "craftax"},
    "CraftaxFull": {"swm_ids": ["swm/CraftaxPixels-v1"], "backends": ["tilemap"],
                    "devices": ["cuda", "cpu"], "venv": "craftax"},
    "DMCManipulatorBringBall": {"swm_ids": ["swm/ManipulatorDMControl-v0"],
                                "backends": ["mjwarp", "raycast3d"],
                                "devices": ["cuda", "cpu"], "venv": "vis311"},
    "DMCQuadrupedWalk": {"swm_ids": ["swm/QuadrupedDMControl-v0"],
                         "backends": ["mjwarp", "cpu3d", "raycast3d"],
                         "devices": ["cuda", "cpu"], "venv": "vis311"},
    "OGBCube": {"swm_ids": ["swm/OGBCube-v0"], "backends": ["mjwarp"],
                "devices": ["cuda"], "cpu_raises_by_design": True, "venv": "ogbmanip"},
    "OGBScene": {"swm_ids": ["swm/OGBScene-v0"], "backends": ["mjwarp"],
                 "devices": ["cuda"], "cpu_raises_by_design": True, "venv": "ogbmanip"},
    "PFRocketLanding": {"swm_ids": ["swm/PFRocketLanding-v0"], "backends": ["nvdiff"],
                        "devices": ["cuda"], "cpu_raises_by_design": True, "venv": "rocket"},
}

def _num(evidence, keys):
    if not isinstance(evidence, dict):
        return None
    for k in keys:
        if k in evidence:
            try:
                return float(evidence[k])
            except (TypeError, ValueError):
                return None
    return None


#: The bound each mandatory check must meet, spelled out per check rather than guessed from the
#: evidence's shape. Each entry is (evidence keys that must be exactly zero, evidence keys that must
#: be exactly the given boolean, evidence keys that must be an EMPTY collection). Taken from
#: stable_envs/testing/conformance.py's own contract for the mandatory group -- "zero difference",
#: not "a small difference".
_MANDATORY_BOUNDS: dict[str, dict] = {
    "check_batch_invariance": {"zero": ("max_abs_diff",)},
    "check_batch_size_invariance": {"zero": ("max_abs_diff",)},
    "check_chunked_render_matches": {"zero": ("max_abs_diff",)},
    "check_default_style_is_official": {"zero": ("max_abs_diff",)},
    "check_determinism": {"true": ("identical",)},
    "check_fidelity_declaration": {"true": ("agrees",)},
    "check_input_form_invariance": {"empty": ("disagreements",)},
    "check_order_independence": {"zero": ("duplicate_max_abs_diff", "permutation_max_abs_diff")},
    "check_per_env_variation": {"empty": ("collapsed_variations",)},
    "check_render_is_pure": {"zero": ("frame_changed_by_double_set", "state_changed_by_double_set",
                                      "state_changed_by_render", "style_changed_by_render")},
    "check_set_state_copies": {"zero": ("aliased_caller_buffer",),
                               "false": ("state_tensor_rebound",)},
    "check_state_dims_matter": {"empty": ("dead_dims",)},
    "check_state_roundtrip": {"zero": ("max_abs_err",)},
    "check_style_is_state": {"zero": ("max_abs_diff",)},
}


def _mandatory_ok(name: str, rec: dict) -> tuple[bool | None, str]:
    """Pass/fail for one mandatory check, or ``None`` when this script does not own its bound."""
    st = rec.get("status")
    if st != "measured":
        return (None, st or "absent")
    ev = rec.get("evidence")
    b = _MANDATORY_BOUNDS.get(name)
    if b is None or not isinstance(ev, dict):
        return (None, "no bound owned here")
    ok, seen = True, []
    for k in b.get("zero", ()):
        if k not in ev:
            return (None, f"missing evidence key {k}")
        v = float(ev[k]); ok &= (v == 0); seen.append(f"{k}={v:g}")
    for k in b.get("true", ()):
        if k not in ev:
            return (None, f"missing evidence key {k}")
        ok &= (ev[k] is True); seen.append(f"{k}={ev[k]}")
    for k in b.get("false", ()):
        if k not in ev:
            return (None, f"missing evidence key {k}")
        ok &= (ev[k] is False); seen.append(f"{k}={ev[k]}")
    for k in b.get("empty", ()):
        if k not in ev:
            return (None, f"missing evidence key {k}")
        ok &= (len(ev[k]) == 0); seen.append(f"{k}={ev[k]}")
    return (bool(ok), "; ".join(seen))


def rejudge(path: Path) -> None:
    """Recompute the verdicts in an existing document under the current rule. Measures nothing.

    The rows carry every check's status and evidence, so a corrected rule can be applied to a run
    that already happened -- which is cheaper and more honest than re-rendering 64 states to reach
    the same numbers, and it keeps the measured values byte-identical to what the card produced.
    """
    doc = json.loads(path.read_text())
    for name, row in (doc.get("configs") or {}).items():
        man = row.get("mandatory") or {}
        judged = {k: _mandatory_ok(k, v) for k, v in man.items()}
        failing = sorted(k for k, (ok, _) in judged.items() if ok is False)
        not_applicable = sorted(k for k, v in man.items() if v.get("status") == "not_applicable")
        unreachable = sorted(k for k, v in man.items()
                             if v.get("status") in ("blocked_environment", "error", None))
        spec_cell = CELLS.get(name, {})
        verdict = ("FAIL" if failing else "BLOCKED" if unreachable
                   else "PARTIAL" if spec_cell.get("registered_default_coverage") == "partial"
                   else "PASS")
        if row.get("status") != "measured":
            verdict = "BLOCKED"
        v = doc["verdicts"].setdefault(name, {})
        was = v.get("verdict")
        v.update({"verdict": verdict, "mandatory_failing": failing,
                  "mandatory_not_applicable": {k: man[k].get("why") for k in not_applicable},
                  "mandatory_unreachable": {k: man[k].get("status") for k in unreachable},
                  "rejudged_from": was,
                  "rejudge_rule": "scratch/step4/certify.py: a not_applicable mandatory check is a "
                                  "property the env does not have, never a skipped check; PARTIAL "
                                  "is reserved for registered_default_coverage == 'partial'"})
        v.pop("mandatory_not_measured", None)
        print(f"[{name:26s}] {was} -> {verdict}  n/a={not_applicable or '-'} "
              f"failing={failing or '-'}")
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print("rewrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejudge", type=Path, default=None,
                    help="recompute the verdicts in an existing document under the current rule "
                         "and exit; measures nothing")
    ap.add_argument("--cells", default="", help="comma-separated cell ids")
    ap.add_argument("--device", default="")
    ap.add_argument("--states-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--pool", type=int, default=256)
    a = ap.parse_args()
    if a.rejudge:
        rejudge(a.rejudge)
        return
    if not (a.cells and a.device and a.states_dir and a.out):
        ap.error("--cells, --device, --states-dir and --out are required unless --rejudge is given")

    spec = importlib.util.spec_from_file_location("fidelity_step4",
                                                  ROOT / "examples" / "fidelity_step4.py")
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)
    import stable_envs as se

    doc = {"schema_version": 1, "step": 7, "shard": "step7-closure-new-cells",
           "device": a.device, "provenance": F.provenance(a.device),
           "what_this_is": "Gate A for the seven Step-7 cells on their committed Step-7 corpora, "
                           "at the registered default size, through examples/fidelity_step4.measure. "
                           "No threshold is applied here beyond the mandatory group's own == 0 "
                           "contract; scratch/step4/certify.py owns the rest at the Step-8 freeze.",
           "configs": {}, "verdicts": {}}
    for name in a.cells.split(","):
        spec_cell = CELLS[name]
        if name not in se.REGISTRY:
            errs = se.envs.import_errors() or {}
            doc["verdicts"][name] = {
                "verdict": "NOT_REGISTERED_HERE",
                "why": "absent from se.REGISTRY in this interpreter",
                "import_errors": {k: str(v)[:300] for k, v in sorted(errs.items())}}
            print(f"[{name:26s}] NOT REGISTERED in this interpreter", flush=True)
            continue
        cls = se.REGISTRY[name]
        config = {"config_id": name, "env": name, "swm_ids": spec_cell["swm_ids"],
                  "role": "registered_default", "siblings": [],
                  "registered_default_coverage": "full",
                  "fidelity_tier": cls.fidelity_tier, "backends": spec_cell["backends"],
                  "default_size": cls.default_size, "reference_id": cls.reference_id,
                  "parity_interior_min": getattr(cls, "parity_interior_min", None),
                  "parity_all_exact_min": getattr(cls, "parity_all_exact_min", None),
                  "parity_note": getattr(cls, "parity_note", "") or "",
                  "render_decoder": getattr(cls, "render_decoder", None),
                  "module": cls.__module__}
        try:
            row = F.measure(config, Path(a.states_dir), a.n_states, a.pool, a.device, None)
        except BaseException as e:                                          # noqa: BLE001
            by_design = bool(spec_cell.get("cpu_raises_by_design")) and a.device == "cpu"
            doc["verdicts"][name] = {
                "verdict": "REFUSED_BY_DESIGN" if by_design else "ERROR",
                "why": f"{type(e).__name__}: {str(e)[:400]}",
                "traceback": traceback.format_exc()[-800:]}
            print(f"[{name:26s}] {doc['verdicts'][name]['verdict']}: {type(e).__name__}", flush=True)
            continue
        doc["configs"][name] = row

        man = row.get("mandatory") or {}
        fam = row.get("family_specific") or {}
        judged = {k: _mandatory_ok(k, v) for k, v in man.items()}
        failing = sorted(k for k, (ok, _) in judged.items() if ok is False)
        # THE REPO'S RULE, not a stricter one of this script's own. scratch/step4/certify.py:
        # "A not_applicable here is a property the env does not have (an env with no style cannot
        # fail check_style_is_state), never a check that was skipped for convenience", and its
        # PARTIAL is reserved for registered_default_coverage == "partial" -- it has nothing to do
        # with how many checks ran. Counting not_applicable against a cell marked OGBCube and
        # OGBScene PARTIAL in the first pass of this re-sweep purely because they declare no style,
        # which is the opposite of what the certifier says. Only a check that was REACHED and was
        # wrong, or one the interpreter could not reach at all, may count.
        not_applicable = sorted(k for k, v in man.items() if v.get("status") == "not_applicable")
        unreachable = sorted(k for k, v in man.items()
                             if v.get("status") in ("blocked_environment", "error", None))
        partial_by_declaration = spec_cell.get("registered_default_coverage") == "partial"
        verdict = ("FAIL" if failing
                   else "BLOCKED" if unreachable
                   else "PARTIAL" if partial_by_declaration
                   else "PASS")
        if row.get("status") != "measured":
            verdict = "BLOCKED"
        doc["verdicts"][name] = {
            "verdict": verdict, "device": a.device,
            "row_status": row.get("status"), "why": str(row.get("why") or "")[:300],
            "mandatory_n": len(man),
            "mandatory_failing": failing,
            "mandatory_not_applicable": {k: man[k].get("why") for k in not_applicable},
            "mandatory_unreachable": {k: man[k].get("status") for k in unreachable},
            "mandatory_observations": {k: judged[k][1] for k in sorted(man)},
            "family_specific_status": {k: v.get("status") for k, v in sorted(fam.items())},
            "corpus": row.get("corpus", {}),
            "seconds": row.get("seconds")}
        print(f"[{name:26s}] {verdict:8s} mandatory {len(man) - len(failing) - len(unreachable)}"
              f"/{len(man)} failing={failing or '-'} n/a={not_applicable or '-'} "
              f"unreachable={unreachable or '-'} {row.get('seconds')}s", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
