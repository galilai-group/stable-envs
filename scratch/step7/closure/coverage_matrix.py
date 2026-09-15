"""Build ``examples/results/step7/closure/coverage_matrix.json``: one row per registered
``stable_worldmodel`` id, rebuilt from the registries and the committed artifacts at the Step-7
integration HEAD.

WHAT IT READS, AND WHAT IT REFUSES TO INVENT

  * the 42 live ``swm/`` ids, their frame status and their ``variation_space`` leaves come from
    ``scratch/swm_matrix.json`` -- the committed output of ``scratch/swm_registry.py`` run in an
    interpreter where ``stable_worldmodel`` is installed. The swm registry is built at import time
    and a family can vanish silently, so the id list is never taken from directory names.
  * the join swm id -> stable-envs cells comes from that same file's evidence ladder for the 34 ids
    that already had one, and from ``STEP7`` below for the six Step 7 closed, each of which names the
    workstream and the merge commit that made it.
  * every declaration (tier, declared fidelity numbers, default size, reference id) is read LIVE off
    the class in this interpreter where the class is importable, and falls back to the matrix's
    recorded declaration where it is not, with ``declaration_source`` saying which.
  * Gate A comes from a measured artifact: the closure sweep's own Gate-A documents for the seven
    Step-7 cells, and the closure replay's ``certification.json`` (or, if that has not landed,
    ``examples/results/step4/certification.json``) for everything frozen in Step 4.
  * CPU and GPU tiers are COPIED, never recomputed: ``performance_record_cpu`` /
    ``performance_record_gpu`` where the class carries them, otherwise the generated Step-6 tier
    table in ``docs/research/step6_optimization.md``, otherwise the Step-7 dashboard row. The closure
    sweep opens no optimisation work and re-measures no tier, so a tier with no committed source is
    ``not_recorded`` and says so.

A cell whose value cannot be sourced is ``null`` with a ``*_source`` of ``none``. Nothing here
averages, rounds or reconciles two disagreeing sources: where they disagree the artifact wins and
the disagreement is recorded, which is ``scratch/step6/report.py``'s rule and is kept.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

#: The six ids Step 7 closed plus the one it took from partial to full, with every value quoted from
#: the workstream row of ``examples/results/step7/dashboard.yaml`` and the merge commit that landed
#: it. This table is the ONLY hand-entered data in this file and each field names its origin.
STEP7 = {
    "swm/CraftaxClassicPixels-v1": {
        "ws": "W1 craftax-classic-full", "envs": ["CraftaxClassic"],
        "coverage_change": "partial -> full (inventory strip, night, sleeping)",
        "merged_sha": "9ca318f", "tier_merge_sha": "b1f61b1",
        "cpu_tier": "CPU_FASTER", "cpu_qualifiers": ["BELOW_G4"],
        "cpu_note": "S_cpu 2.80 paired / 5.39 sweep; cores_cpu 0.263; official jitted 213.44 f/s, "
                    "node2335 cgroup affinity [8], job 6409933. NO_REFERENCE reading retained as a "
                    "diagnostic: CPU_REGRESSION on cores alone, the coverage change (night now "
                    "represented), not a slower renderer",
        "gpu_tier": "GPU_FASTER", "gpu_qualifiers": ["BATCH_SCALES"],
        "gpu_note": "R 316.0 paired / 304.5 sweep; cores 15.40; official jitted 151.68 f/s on "
                    "gpu2109 from the se7-craftax-gpu venv, job 6409934",
        "variation_in": [], "variation_out": [],
        "variation_note": "the swm wrapper declares NO variation space (0 leaves), so there is no "
                          "Gate-F variation gap to close for this id",
        "gate_a_note": "all mandatory checks pass; check_resolution_consistency declared "
                       "not_applicable below light_level 0.5 on the OFFICIAL renderer's own "
                       "evidence (official 0.1634 == ours 0.1634); the Step-8 certifier must honour "
                       "resolution_consistency_applicability or judge on the light >= 0.5 subset"},
    "swm/CraftaxPixels-v1": {
        "ws": "W2 craftax-full", "envs": ["CraftaxFull"],
        "coverage_change": "renderer gap -> full (9 levels, mobs, projectiles, light map)",
        "merged_sha": "d024c1a", "tier_merge_sha": "00c7c8d",
        "cpu_tier": "CPU_FASTER", "cpu_qualifiers": ["BELOW_G4"],
        "cpu_note": "S_cpu 2.395, cores_cpu 0.342; 1,469.1 f/s at B=64 vs official jitted 613.4 "
                    "(decode 4,294) - job 6409963 node2339 --cpus-per-task=1, paired USABLE",
        "gpu_tier": "GPU_FASTER", "gpu_qualifiers": [],
        "gpu_note": "R 100.96, cores 21.82; 94,343.9 f/s at B=4096 vs official jitted 934.5 on the "
                    "same 3090 in the same job 6410052. Measuring the GPU reference demoted this "
                    "from GPU_FAST (cores only) to GPU_FASTER",
        "variation_in": [], "variation_out": [],
        "variation_note": "the swm wrapper declares NO variation space (0 leaves)",
        "gate_a_note": "18 checks clean; check_resolution_consistency declared not_applicable in "
                       "ALL bands on the official's own evidence (official fails the frozen 0.995 "
                       "in every band: world 0.4817 day / 0.0991 night)"},
    "swm/ManipulatorDMControl-v0": {
        "ws": "W3 dmc-manipulator", "envs": ["DMCManipulatorBringBall"],
        "coverage_change": "renderer gap -> covered (plate + mover pass under a directional "
                           "castshadow light; per-episode target via _episode_slots)",
        "merged_sha": "1d9dccb",
        "cpu_tier": "CPU_REGRESSION", "cpu_qualifiers": [],
        "cpu_note": "15.50 f/s at B=1 (224 px) vs official 22.23; S_cpu 0.697, cores_cpu 0.0071; "
                    "job 6396735 node2315 se-vis311 one thread, gates USABLE. A tier NAME, not a "
                    "slowdown: no CPU number existed for this cell before. Mechanism: the _fk "
                    "Python loop per env plus three full-frame ray passes at 50,176 rays",
        "gpu_tier": "GPU_FAST", "gpu_qualifiers": ["HOST_BOUND"],
        "gpu_note": "R 32.04, cores 3.72; 8,396.35 f/s at B=4096 vs official dm_control 262.06 on "
                    "gpu2107 in one process, job 6395157; cpu_frac 1.603 > 1.5 so HOST_BOUND "
                    "applies per Step-2 6.7",
        "variation_in": ["visibility.target_ball", "target_ball.shape", "light.diffuse_scale"],
        "variation_out": ["floor.color"],
        "variation_note": "the registered SWM default (hidden target, .015 sphere, light 0.7) is "
                          "reached THROUGH declared leaves and asserted by Gate F; floor.color is a "
                          "Gate-F RANGE gap (the leaf exists, its range is not covered), recorded by "
                          "W3's reviewer as a range gap and not a default gap",
        "gate_a_note": "every mandatory check passes; check_state_dims_matter passes with three "
                       "REPORTED exclusions (ball_y, target_ball_quat_qw, target_ball_quat_qy), "
                       "approved by Sami as a narrow per-class render-contract exception validated "
                       "by the check itself against the frozen geom_type"},
    "swm/QuadrupedDMControl-v0": {
        "ws": "W4 dmc-quadruped", "envs": ["DMCQuadrupedWalk"],
        "coverage_change": "renderer gap -> covered (DMControlTrackCom cell + shared ellipsoid "
                           "intersector; the walk model's HFIELD is deleted so no hfield was needed)",
        "merged_sha": "0a029bd", "tier_merge_sha": "e0a416c",
        "cpu_tier": "CPU_UNMEASURED", "cpu_qualifiers": ["LOAD_QUARANTINED"],
        "cpu_note": "job 6398084 node2301: 73 polls over 6 h, load1 34.1-43.6, never below the "
                    "load-12 guard, so nothing was measured. Local M5 numbers exist and are NOT "
                    "certification numbers. The CPU path itself runs: 25/25 CPU identity",
        "gpu_tier": "GPU_EXCEPTIONAL", "gpu_qualifiers": ["CENSUS_ENTRY_MISSING"],
        "gpu_note": "class-S ladder: S 129.9, cores 13.6, BATCH_SCALES pass; 44,374 f/s at B=4096 "
                    "vs same-node official 341.5, job 6398085 gpu2109. The repo evaluator prints "
                    "'unknown' only because the Step-3 census has no entry for this cell yet - a "
                    "Step-8 freeze item, not a missing measurement",
        "variation_in": [], "variation_out": [],
        "variation_note": "certified at the registered SWM default (camera 2, 224 px, SWM "
                          "appearance: material self rgba .6 .3 .3, foot_back_left density 1000); "
                          "no Gate-F leaf gap recorded by the workstream",
        "gate_a_note": "all mandatory checks pass on both devices and both reference arms on the "
                       "new corpus; ELLIPSOID stays out of _BOUNDED_KINDS (kind-cost ratio "
                       "0.69-1.43 measured only on a load-34 node: undecided, not admitted silently)"},
    "swm/OGBCube-v0": {
        "ws": "W5 ogb-manipulation", "envs": ["OGBCube"],
        "coverage_change": "renderer gap -> covered (mjwarp-backed cell on the Fetch plate + mover "
                           "pattern; goal/object placement as state)",
        "merged_sha": "24d1054",
        "cpu_tier": "CPU_MISSING", "cpu_qualifiers": ["RAISES_BY_DESIGN"],
        "cpu_note": "device='cpu' raises honestly rather than falling back to CUDA. CPU3D-OGB-1: "
                    "cores_cpu short 56-72x at full rate and 9.9-12.9x even at the 24 ns/tri "
                    "vertex-setup floor, so CPU_FAST is a demonstrated blocker; S_cpu ~6-7 vs the "
                    "llvmpipe official means CPU_FASTER is reachable with a shared span rasteriser, "
                    "deferred to Step 8",
        "gpu_tier": "GPU_PARITY", "gpu_qualifiers": ["BATCH_SCALES_FAIL"],
        "gpu_note": "R 3.20, cores 1.88, BATCH_SCALES 5.36x against a bar of 20 -> capped at "
                    "GPU_PARITY per Step-2 6.7. fps(1) 1,200, best 6,025; official on hardware EGL "
                    "2,012 f/s (NOT the 6.31 f/s llvmpipe number, which is the CPU comparator)",
        "variation_in": [], "variation_out": [],
        "variation_note": "certified at the registered default (224 px, camera front_pixels, "
                          "visualize_info=True recorded as a parity_note); the registered-default "
                          "variation set was taken first and appearance leaves are declared per env",
        "gate_a_note": "determinism, batch invariance, batch-size invariance, order independence, "
                       "state_dims_matter (dead_dims []) all pass"},
    "swm/OGBScene-v0": {
        "ws": "W5 ogb-manipulation", "envs": ["OGBScene"],
        "coverage_change": "renderer gap -> covered (as OGBCube, plus drawer/window/button and the "
                           "two button state bits)",
        "merged_sha": "24d1054",
        "cpu_tier": "CPU_MISSING", "cpu_qualifiers": ["RAISES_BY_DESIGN"],
        "cpu_note": "as OGBCube (CPU3D-OGB-1)",
        "gpu_tier": "GPU_PARITY", "gpu_qualifiers": ["BATCH_SCALES_FAIL"],
        "gpu_note": "R 3.62, cores 2.05, BATCH_SCALES 6.50x -> capped at GPU_PARITY. fps(1) 873, "
                    "best 5,498; official on hardware EGL 1,526 f/s",
        "variation_in": [], "variation_out": [],
        "variation_note": "as OGBCube; the two button bits are state dims, not style",
        "gate_a_note": "as OGBCube. Ships with a DECLARED residual: the btn_metal cube-map cannot be "
                       "sampled by mujoco_warp <= 3.12.0 (fixed upstream in 3.13.0) and is baked to "
                       "its mean colour on two small geoms - 0.139% of pixels, max 37/255. Sami "
                       "ruled: keep the reviewed declared bake"},
    "swm/PFRocketLanding-v0": {
        "ws": "W6 rocket-landing", "envs": ["PFRocketLanding"],
        "coverage_change": "NO-GO (structural) -> covered at the visual tier (URDF primitives "
                           "through the nvdiffrast raster backend under PyBullet's TinyRenderer law)",
        "merged_sha": "fd2b8ef",
        "cpu_tier": "CPU_MISSING", "cpu_qualifiers": ["RAISES_BY_DESIGN"],
        "cpu_note": "device='cpu' raises honestly. Measured feasibility: a naive single-thread "
                    "torch rasteriser draws 486 triangles at 480 px at 1.92 f/s = 0.056x the "
                    "official 34.25 f/s (S_cpu bar 171) and 0.0016 cores (bar 1,191) - both bars "
                    "missed by two to three orders. A shared CPU triangle backend is deferred to "
                    "Step 8",
        "gpu_tier": "GPU_PARITY", "gpu_qualifiers": ["BATCH_SCALES_FAIL"],
        "gpu_note": "R 241.5, cores 2.61, NO_LARGE_BATCH_REGRESSION 1.253; BATCH_SCALES 14.17 "
                    "against a bar of 20, which fails because fps(1)=238 is unusually high for a "
                    "class-S cell (3,374 f/s at B=4096) - correct application of Step-2 6.7 and a "
                    "definitional edge case recorded for Step 8",
        "variation_in": ["5 RGB material colours", "flame ignition bit"],
        "variation_out": ["environment.sky_color"],
        "variation_note": "sky_color is a DEAD leaf in the swm wrapper (it reaches no pixel) and is "
                          "deliberately not offered; the 5 colours and the ignition bit are the "
                          "style dims by path",
        "gate_a_note": "all mandatory checks PASS, including batch-size invariance after the "
                       "batch-invariant mvp. Two conformance tests fail with a stated mechanism: "
                       "check_colors_apply_exactly is structurally unsatisfiable under "
                       "TinyRenderer's multiplicative law (max factor 0.993567, so 200 can never "
                       "render above 198) and fails for the OFFICIAL frames too - OPEN escalation, "
                       "Sami did not rule; and DOWNSAMPLE_EXACT_FRAC has no table entry for this "
                       "cell (measured 0.95614, to be added at the Step-8 freeze since tests/ is "
                       "frozen in Step 7)"},
}

#: The two ids that produce no frame, with the place the claim is DECLARED. Not our judgement: the
#: upstream env raises it, and the live probe recorded the raise verbatim.
SYMBOLIC_EXPECTED = {"swm/CraftaxClassicSymbolic-v1", "swm/CraftaxSymbolic-v1"}


def _step6_tiers() -> dict:
    """Parse the generated Step-6 tier tables out of docs/research/step6_optimization.md.

    That page is itself generated by ``scratch/step6/report.py`` from the Step-6 artifacts and every
    row carries the artifact key it came from, so reading it copies a tier rather than deciding one.
    Two tables are read: the registered-default table (CPU tier / GPU tier / meets the bar) and the
    parameter-space table (CPU tier reported / GPU tier reported).
    """
    p = ROOT / "docs" / "research" / "step6_optimization.md"
    if not p.is_file():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        m = re.fullmatch(r"`([A-Za-z0-9_]+)`", cells[0])
        if not m:
            continue
        name = m.group(1)
        tiers = [c.replace("`", "") for c in cells[1:]]
        cpu = next((t for t in tiers if t.startswith("CPU_")), None)
        gpu = next((t for t in tiers if t.startswith("GPU_")), None)
        if cpu or gpu:
            prev = out.setdefault(name, {"cpu": None, "gpu": None})
            prev["cpu"] = prev["cpu"] or cpu
            prev["gpu"] = prev["gpu"] or gpu
    return out


def _gate_a_from_closure(closure: Path) -> dict:
    """Gate-A verdicts for the seven Step-7 cells, out of the closure sweep's own documents."""
    out = {}
    for p in sorted((closure / "gate_a").glob("*.json")) if (closure / "gate_a").is_dir() else []:
        doc = json.loads(p.read_text())
        dev = doc.get("device", "?")
        for cell, v in (doc.get("verdicts") or {}).items():
            verdict, why = v.get("verdict"), str(v.get("why") or "")
            # A cell that declares no CPU backend returns a BLOCKED row on `cpu` carrying the raise
            # it performed. That is not the same thing as a check that could not be reached, and
            # collapsing the two would read as "we could not measure it" where the answer is "it
            # refused, which is what it must do". The relabel is driven by the RECORDED MESSAGE, so
            # it cannot promote a cell that failed for some other reason.
            if dev == "cpu" and verdict == "BLOCKED" and "has no CPU backend" in why:
                verdict = "REFUSED_BY_DESIGN"
            out.setdefault(cell, {})[dev] = {
                "verdict": verdict,
                "failing_checks": v.get("mandatory_failing") or [],
                "not_measured": v.get("mandatory_not_measured") or {},
                "family_specific": v.get("family_specific_status") or {},
                "why": v.get("why") or "",
                "source": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)}
    return out


def _gate_a_frozen(closure: Path) -> tuple[dict, dict, str]:
    """Gate A for the Step-4 universe, per config, with the artifact each verdict came from.

    The closure replay's own certification wins where it has a row -- it is this HEAD's verdict on
    this HEAD's frames -- and the frozen Step-4 certification fills in only where the closure run
    produced no row at all. Which of the two answered is recorded per configuration rather than
    stated once for the table, because a reader has to be able to tell a re-measured PASS from a
    carried-forward one.
    """
    frozen_p = ROOT / "examples" / "results" / "step4" / "certification.json"
    verdicts, source = {}, {}
    if frozen_p.is_file():
        for c, v in json.loads(frozen_p.read_text())["configs"].items():
            verdicts[c] = v.get("gate_a")
            source[c] = str(frozen_p.relative_to(ROOT)) + " (frozen Step-4, carried forward: the "
            source[c] += "closure replay produced no row for this configuration)"
    n_closure = 0
    for cand in sorted(closure.glob("replay/certify*/certification.json")):
        for c, v in json.loads(cand.read_text())["configs"].items():
            verdicts[c] = v.get("gate_a")
            source[c] = str(cand.relative_to(ROOT)) + " (this sweep)"
            n_closure += 1
        break
    return (verdicts, source, f"{n_closure} configuration(s) re-judged by this sweep; the rest "
                              f"carried forward from the frozen Step-4 certification")


def _live_declarations() -> dict:
    try:
        import stable_envs as se
    except Exception as e:                                                  # noqa: BLE001
        print(f"  (no live registry in this interpreter: {type(e).__name__}: {e})")
        return {}
    out = {}
    for name, cls in se.REGISTRY.items():
        out[name] = {
            "fidelity_tier": getattr(cls, "fidelity_tier", None),
            "default_size": getattr(cls, "default_size", None),
            "reference_id": getattr(cls, "reference_id", None),
            "render_decoder": getattr(cls, "render_decoder", None),
            "visual_fidelity": getattr(cls, "visual_fidelity", None) or {},
            "parity_interior_min": getattr(cls, "parity_interior_min", None),
            "parity_all_exact_min": getattr(cls, "parity_all_exact_min", None),
            "performance_record_cpu": getattr(cls, "performance_record_cpu", None),
            "performance_record_gpu": getattr(cls, "performance_record_gpu", None),
            "module": cls.__module__}
    return out


def _backends(module: str) -> list:
    """The renderer family a cell's module implies. Read off the module, never guessed per env."""
    return {
        "stable_envs.envs.classic_control": ["gfxdraw"],
        "stable_envs.envs.pusht": ["raster2d"],
        "stable_envs.envs.swm_nav": ["raster2d", "swm_nav"],
        "stable_envs.envs.craftax": ["tilemap"],
        "stable_envs.envs.craftax_full": ["tilemap"],
        "stable_envs.envs.dmcontrol": ["raycast3d", "cpu3d", "mjwarp"],
        "stable_envs.envs.ogbench": ["raycast3d"],
        "stable_envs.envs.ogb_manip": ["mjwarp"],
        "stable_envs.envs.fetch": ["mjwarp"],
        "stable_envs.envs.rocket_landing": ["nvdiff"],
        "stable_envs.envs.mujoco_locomotion": ["nvdiff"],
    }.get(module, [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-declarations", type=Path, default=None,
                    help="write this interpreter's live declarations and exit. No single "
                         "interpreter can import every cell (jax, nvdiffrast, PyFlyt and "
                         "mujoco_warp do not coexist), so the matrix is built from the UNION of "
                         "several dumps rather than from whichever subset one venv happens to have")
    ap.add_argument("--declarations", type=Path, action="append", default=[],
                    help="a declaration dump to merge in, repeatable; later dumps never overwrite "
                         "an earlier one, so the first interpreter to declare a cell owns it")
    ap.add_argument("--closure", type=Path,
                    default=ROOT / "examples" / "results" / "step7" / "closure")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "examples" / "results" / "step7" / "closure" / "coverage_matrix.json")
    a = ap.parse_args()

    if a.dump_declarations:
        d = _live_declarations()
        a.dump_declarations.parent.mkdir(parents=True, exist_ok=True)
        a.dump_declarations.write_text(json.dumps(d, indent=1, sort_keys=True, default=str))
        print(f"{len(d)} declarations -> {a.dump_declarations}")
        return

    matrix = json.loads((ROOT / "scratch" / "swm_matrix.json").read_text())
    live = _live_declarations()
    merged_from = ["live import in this interpreter"] if live else []
    for extra in a.declarations:
        if not extra.is_file():
            print(f"declaration dump missing, skipped: {extra}")
            continue
        got = json.loads(extra.read_text())
        added = [k for k in got if k not in live]
        live.update({k: got[k] for k in added})
        merged_from.append(f"{extra.name} (+{len(added)} cells)")
    s6 = _step6_tiers()
    ga7 = _gate_a_from_closure(a.closure)
    ga4, ga4_src_per_config, ga4_src = _gate_a_frozen(a.closure)
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or None
    # The SHA a row's "merged" field must name is the INTEGRATION head this sweep verified, not the
    # closure branch's own tip, which moves with every artifact commit and is not where anything was
    # merged. It is the merge-base with step7/integration; if that branch is not present the value
    # is left null rather than silently falling back to HEAD.
    integ = subprocess.run(["git", "-C", str(ROOT), "merge-base", "HEAD", "step7/integration"],
                           capture_output=True, text=True).stdout.strip() or None

    rows, counts = [], {"pixel_producing": 0, "pixel_covered": 0, "symbolic": 0,
                        "gate_a": {}, "cpu_tier": {}, "gpu_tier": {}}
    for r in matrix["rows"]:
        sid = r["swm_id"]
        s7 = STEP7.get(sid)
        pixel = r["swm"]["frame_status"] == "pixels"
        envs = list(s7["envs"]) if s7 else list(r["match"]["envs"])
        row = {
            "swm_id": sid,
            "pixel_producing": pixel,
            "stable_envs_envs": envs,
            "join": {"rung": "step7_workstream" if s7 else r["match"]["rung"],
                     "evidence": (f"{s7['ws']}, merged {s7['merged_sha']}" if s7
                                  else r["match"]["evidence"])},
            "closed_in_step7": bool(s7),
            "merged": {"yes": bool(envs), "sha": (s7 or {}).get("merged_sha"),
                       "tier_merge_sha": (s7 or {}).get("tier_merge_sha"),
                       "integration_head_verified": integ,
                       "closure_branch_head": sha},
        }
        if not pixel:
            counts["symbolic"] += 1
            row["symbolic"] = {
                "renderer_free": True,
                "declared_by": r["swm"]["entry_point"],
                "declaration": r["swm"]["frame_status_why"],
                "observed_by": "scratch/swm_registry.py (live swm registry probe), recorded in "
                               "scratch/swm_matrix.json",
                "expected": sid in SYMBOLIC_EXPECTED}
            rows.append(row)
            continue

        counts["pixel_producing"] += 1
        if envs:
            counts["pixel_covered"] += 1

        primary = next((e for e in envs if live.get(e)), envs[0] if envs else None)
        d = live.get(primary) or {}
        src = "live class at the integration HEAD"
        if not d:
            # No interpreter on this machine can import the cell (Fetch needs mujoco_warp at the
            # 3.11 pin, for instance). Fall back to the declaration the matrix RECORDED from an
            # interpreter that could, and say so: a recorded declaration is a weaker source than a
            # live one and the difference has to be visible.
            rec = ((r.get("stable_envs") or {}).get("declarations") or {}).get(primary) or {}
            if rec:
                d = dict(rec)
                d.setdefault("module", rec.get("module"))
                src = ("scratch/swm_matrix.json (declaration recorded by an interpreter that could "
                       "import the cell); no interpreter here can")
            else:
                src = "none"
        row["declaration_source"] = src
        row["renderer"] = {
            "family": _backends(d.get("module", "")) or None,
            "module": d.get("module"),
            "default_size": d.get("default_size"),
            "render_decoder": d.get("render_decoder"),
            "reference_id": d.get("reference_id")}
        tier = d.get("fidelity_tier")
        row["fidelity"] = {
            "tier": tier,
            "parity_interior_min": d.get("parity_interior_min"),
            "parity_all_exact_min": d.get("parity_all_exact_min"),
            "visual_fidelity": d.get("visual_fidelity") or None,
            "registered_default_coverage": "full"}

        # ---- Gate A ----
        g = {"result": None, "failing_checks": [], "per_device": {}, "source": "none"}
        if primary in ga7:
            g["per_device"] = ga7[primary]
            verdicts = {dev: v["verdict"] for dev, v in ga7[primary].items()}
            # Gate A is reported from the CUDA arm where one exists, because that is the device every
            # one of these cells supports; the cpu arm is kept beside it in per_device rather than
            # averaged in. A refusal on cpu is device coverage, not a Gate-A verdict.
            fails = sorted({c for dev, v in ga7[primary].items() for c in v["failing_checks"]
                            if v["verdict"] != "REFUSED_BY_DESIGN"})
            cuda = verdicts.get("cuda")
            g["result"] = cuda or next(iter(verdicts.values()), None)
            g["devices"] = verdicts
            g["failing_checks"] = fails
            g["source"] = "; ".join(sorted({v["source"] for v in ga7[primary].values()}))
        elif primary in ga4:
            g["result"] = ga4[primary]
            g["source"] = ga4_src_per_config.get(primary, ga4_src)
        if s7 and s7.get("gate_a_note"):
            g["note"] = s7["gate_a_note"]
        row["gate_a"] = g
        counts["gate_a"][str(g["result"])] = counts["gate_a"].get(str(g["result"]), 0) + 1

        # ---- tiers: copied, never recomputed ----
        def _tier(dev: str) -> dict:
            rec = d.get(f"performance_record_{dev}")
            if isinstance(rec, dict) and rec.get("tier"):
                q = rec.get("qualifier")
                return {"tier": rec["tier"],
                        "qualifiers": [q] if isinstance(q, str) and q else list(q or []),
                        "source": f"{primary}.performance_record_{dev} (class declaration)",
                        "detail": {k: v for k, v in rec.items() if k not in ("tier", "qualifier")}}
            if s7 and s7.get(f"{dev}_tier"):
                return {"tier": s7[f"{dev}_tier"], "qualifiers": s7.get(f"{dev}_qualifiers") or [],
                        "source": "examples/results/step7/dashboard.yaml (workstream row)",
                        "detail": {"note": s7.get(f"{dev}_note")}}
            t = (s6.get(primary) or {}).get(dev)
            if t:
                return {"tier": t, "qualifiers": [],
                        "source": "docs/research/step6_optimization.md (generated Step-6 tier table)",
                        "detail": {}}
            return {"tier": None, "qualifiers": [], "source": "none",
                    "detail": {"why": "no committed tier artifact names this cell; the closure "
                                      "sweep re-measures no tier"}}
        row["cpu_tier"] = _tier("cpu")
        row["gpu_tier"] = _tier("gpu")
        counts["cpu_tier"][str(row["cpu_tier"]["tier"])] = \
            counts["cpu_tier"].get(str(row["cpu_tier"]["tier"]), 0) + 1
        counts["gpu_tier"][str(row["gpu_tier"]["tier"])] = \
            counts["gpu_tier"].get(str(row["gpu_tier"]["tier"]), 0) + 1

        # ---- variation space ----
        leaves = r["swm"].get("variation_names") or []
        vmaps = []
        try:
            from stable_envs.compat import worldmodel as W
            vmaps = [e for e in envs if e in (getattr(W, "VARIATION_MAPS", {}) or {})]
        except Exception:                                                   # noqa: BLE001
            vmaps = []
        row["variation_space"] = {
            "swm_status": r["swm"]["variation_space_status"],
            "swm_leaves": leaves,
            "n_swm_leaves": len(leaves),
            "declared_in": (s7 or {}).get("variation_in", []),
            "declared_out": (s7 or {}).get("variation_out", []),
            "variation_map_present_for": vmaps,
            "variation_map_status": "complete" if vmaps else ("n/a" if not leaves else "missing"),
            "note": (s7 or {}).get("variation_note",
                     "pre-Step-7 coverage; no Step-7 workstream re-examined this id's leaves")}
        rows.append(row)

    doc = {"schema_version": 1, "step": 7,
           "what_this_is": __doc__.strip().splitlines()[0],
           "integration_head_verified": integ,
           "closure_branch_head": sha,
           "sources": {
               "swm_registry": "scratch/swm_matrix.json (live swm registry probe)",
               "declarations": "the stable-envs classes at the integration HEAD, unioned over "
                               + "; ".join(merged_from),
               "gate_a_step7_cells": "examples/results/step7/closure/gate_a/*.json",
               "gate_a_step4_universe": ga4_src,
               "tiers": "performance_record_* where declared, else the generated Step-6 tier table, "
                        "else the Step-7 dashboard workstream row; nothing re-measured here"},
           "counts": counts, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    print(f"{len(rows)} swm ids; pixel-producing {counts['pixel_producing']}, of which covered "
          f"{counts['pixel_covered']}; symbolic {counts['symbolic']}")
    print("gate A:", counts["gate_a"])
    print("cpu tiers:", counts["cpu_tier"])
    print("gpu tiers:", counts["gpu_tier"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
