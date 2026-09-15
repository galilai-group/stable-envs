"""M6: the correctness re-freeze measurement for the six locomotion nvdiff cells.

Sami's ruling (2026-09-15): mandatory batch-size invariance takes precedence over accidental float
bit identity, and the six cells are re-baselined as a CORRECTNESS RE-FREEZE. This script produces the
evidence the ruling asks for, baseline (76dc511) vs branch, IN ONE JOB ON ONE CARD:

  --phase render --tag {baseline,branch}
      per cell, three measurements on a state draw that is written down rather than inherited:
        A. frames at the identity draw (``sample_states(64, rng=default_rng(0))``, the draw
           ``nvdiff_identity.py`` uses) -> npz + sha256, for the per-pixel delta table;
        B. ``check_visual_fidelity`` against the OFFICIAL renderer (gymnasium MuJoCo through EGL),
           with ``env.rng`` reseeded so both legs see the same states -- this is the number that must
           be unchanged or improved, because it is the one the cells' declarations are made of;
        C. ``check_resolution_consistency`` on the downsample protocol's own eight states
           (``conformance.downsample_protocol_states``). This is the ONLY Step-4 artifact that names
           these six cells: ``examples/results/step4/certification.json``'s
           ``bounds.downsample_exact_frac.value``, mirrored from
           ``tests/gpu/test_reproducibility.py:DOWNSAMPLE_EXACT_FRAC``. The six have NO frozen frame
           digest anywhere in ``examples/results/step4`` -- they are not among its 49 configs, have no
           ``states/<config>.npz`` and appear in no ``conformance/<shard>.json`` -- so this bound is
           the whole of what a re-freeze can touch there, and whether it moves is measured, not assumed.

  --phase gatea
      the full mandatory set on the BRANCH for each cell: determinism, batch invariance, batch-size
      invariance, order independence, chunk invariance, render purity, state_dims_matter.

  --phase compare  (and --phase rejudge, which re-labels a committed comparison without a GPU)
      per cell: max_abs_pixel_delta, fraction of pixels changed, fraction changed by MORE THAN ONE
      level, the two digests, fidelity before/after with the delta and the cell's declared thresholds,
      and the downsample fraction before/after against its frozen bound and margin. Each cell is then
      LABELLED -- IDENTICAL / ULP_LEVEL / SUB_NOISE_SHIFT / MEANINGFUL_CHANGE -- under the written-down
      rule in ``_judge``, and ``refreeze_ok`` is the conjunction of the ruling's nine conditions, each
      answered from a committed file rather than from this script's opinion.

Usage (one GPU node, se7-rocket venv; the sbatch does this):
    PYTHONPATH=$BASE python refreeze_locomotion.py --phase render --tag baseline --out DIR
    PYTHONPATH=$WS   python refreeze_locomotion.py --phase render --tag branch   --out DIR
    PYTHONPATH=$WS   python refreeze_locomotion.py --phase gatea                 --out DIR
    PYTHONPATH=$WS   python refreeze_locomotion.py --phase compare               --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np

CELLS = ("MujocoHalfCheetah", "MujocoHopper", "MujocoWalker2d", "MujocoAnt", "MujocoHumanoid",
         "MujocoSwimmer")

#: The identity draw: the same one ``nvdiff_identity.py`` uses, so the M5 numbers and these are the
#: same states. Written down here rather than inherited from an env's generator.
IDENTITY_SEED = 0
IDENTITY_N = 64
RENDER_CHUNK = 16


def _code_root() -> str:
    """The checkout the MEASURED ``stable_envs`` came from -- not the one this script lives in.

    The baseline leg runs this same file with ``PYTHONPATH=$SCR/se7/baseline``, so ``__file__`` points
    at the branch while the package under test is the baseline. Everything provenance-bearing has to
    be resolved from the package, or the baseline leg would report the branch's commit.
    """
    import stable_envs as se
    return os.path.dirname(os.path.dirname(os.path.realpath(se.__file__)))


def _prov(tag: str) -> dict:
    import torch
    import stable_envs as se
    root = _code_root()
    sha_file = os.path.join(root, ".git_sha")
    return {"tag": tag, "host": platform.node(), "gpu": torch.cuda.get_device_name(0),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "threads": torch.get_num_threads(),
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "stable_envs_file": os.path.realpath(se.__file__),
            "sha": (open(sha_file).read().strip() if os.path.exists(sha_file) else "unknown")}


def phase_render(out: str, tag: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import stable_envs as se
    from stable_envs.testing import conformance as C

    R = {"provenance": _prov(tag), "cells": {}}
    for name in CELLS:
        if name not in se.REGISTRY:
            R["cells"][name] = {"missing": se.envs.import_errors().get("mujoco_locomotion",
                                                                       "not registered")}
            print(tag, name, "NOT REGISTERED", flush=True)
            continue
        cell: dict = {}
        env = se.make(name, batch=16, device="cuda")
        cell["size"] = int(env.size)
        cell["supersample"] = int(getattr(env, "ss", getattr(env, "supersample", 1)))
        cell["declared"] = {k: v for k, v in (getattr(type(env), "visual_fidelity", {}) or {}).items()
                            if k in ("mean_abs", "within8_frac", "ssim", "mean_abs_max",
                                     "within8_min", "ssim_min")}

        # A. the identity draw -> frames
        st = env.sample_states(IDENTITY_N, rng=np.random.default_rng(IDENTITY_SEED))
        frames = np.concatenate([env.render(st[lo:lo + RENDER_CHUNK]).cpu().numpy()
                                 for lo in range(0, IDENTITY_N, RENDER_CHUNK)])
        np.savez_compressed(os.path.join(out, f"{tag}_{name}.npz"), frames=frames, states=st)
        cell["frames"] = {"sha256": C.frames_digest(frames), "shape": list(frames.shape),
                          "states_sha256": C.frames_digest(np.ascontiguousarray(st, np.float32)),
                          "mean": float(frames.mean())}

        # B. official-reference visual fidelity -- the number the declaration is made of.
        #    env.rng is reseeded so both legs draw the SAME states (check_visual_fidelity draws
        #    internally from env.rng); a leg that inherited a consumed generator would compare two
        #    different state sets and the "before/after" would be meaningless.
        env.rng = np.random.default_rng(IDENTITY_SEED)
        try:
            cell["visual_fidelity"] = C.check_visual_fidelity(env, n=IDENTITY_N)
        except Exception as e:                                                      # pragma: no cover
            cell["visual_fidelity"] = {"blocked": f"{type(e).__name__}: {e}"}

        # C. the downsample protocol -- the only Step-4 bound that names these cells
        try:
            ds = C.downsample_protocol_states(env)
            cell["downsample"] = C.check_resolution_consistency(env, ds)
            cell["downsample_protocol"] = C.downsample_protocol(ds)
        except Exception as e:                                                      # pragma: no cover
            cell["downsample"] = {"blocked": f"{type(e).__name__}: {e}"}

        R["cells"][name] = cell
        print(tag, name, json.dumps({"sha256": cell["frames"]["sha256"][:16],
                                     "vf": cell["visual_fidelity"].get("mean_abs"),
                                     "ds": cell.get("downsample", {}).get("exact_frac")}), flush=True)
        del env
        torch.cuda.empty_cache()
    _dump(os.path.join(out, f"{tag}.json"), R)


def phase_gatea(out: str) -> None:
    """The full mandatory set on the branch, per cell. Every one of these must PASS."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import stable_envs as se
    from stable_envs.testing import conformance as C

    R = {"provenance": _prov("branch"), "cells": {}}
    for name in CELLS:
        if name not in se.REGISTRY:
            R["cells"][name] = {"missing": "not registered"}
            continue
        env = se.make(name, batch=16, device="cuda")
        st = env.sample_states(256, rng=np.random.default_rng(11))
        checks: dict = {}
        def run(key, fn):
            try:
                checks[key] = fn()
            except Exception as e:                                                  # pragma: no cover
                checks[key] = {"error": f"{type(e).__name__}: {e}"}
        run("check_determinism", lambda: C.check_determinism(env))
        run("check_batch_invariance", lambda: C.check_batch_invariance(env))
        run("check_batch_size_invariance",
            lambda: C.check_batch_size_invariance(env, st, sizes=(1, 3, 7, 8, 64, 256)))
        run("check_order_independence", lambda: C.check_order_independence(env, st[:64]))
        run("check_chunked_render_matches",
            lambda: C.check_chunked_render_matches(env, st[:64], chunks=(1, 7, 64)))
        run("check_render_is_pure", lambda: C.check_render_is_pure(env, st[:64]))
        env.rng = np.random.default_rng(IDENTITY_SEED)
        run("check_state_dims_matter", lambda: C.check_state_dims_matter(env))
        R["cells"][name] = {"checks": checks, "pass": _gatea_ok(checks)}
        print("gateA", name, R["cells"][name]["pass"], json.dumps(checks, default=str)[:300], flush=True)
        del env
        torch.cuda.empty_cache()
    R["all_pass"] = all(c.get("pass") for c in R["cells"].values() if "checks" in c) and \
        len([c for c in R["cells"].values() if "checks" in c]) == len(CELLS)
    _dump(os.path.join(out, "gatea_branch.json"), R)
    print("gateA all_pass", R["all_pass"], flush=True)


def _gatea_ok(checks: dict) -> bool:
    """Every mandatory check's own pass condition, transcribed from what the suite asserts."""
    ok = True
    for key, r in checks.items():
        if not isinstance(r, dict) or "error" in r:
            return False
        if key == "check_determinism":
            ok &= bool(r.get("identical"))
        elif key == "check_state_dims_matter":
            ok &= not r.get("dead_dims")
        elif key == "check_render_is_pure":
            ok &= (float(r["state_changed_by_render"]) == 0.0
                   and float(r["style_changed_by_render"]) == 0.0
                   and float(r["state_changed_by_double_set"]) == 0.0
                   and int(r["frame_changed_by_double_set"]) == 0)
        elif key == "check_order_independence":
            ok &= (int(r["permutation_max_abs_diff"]) == 0
                   and int(r["duplicate_max_abs_diff"]) == 0)
        else:                            # batch invariance, batch-size invariance, chunk invariance
            ok &= int(r.get("max_abs_diff", 1)) == 0
    return bool(ok)


#: SSIM's sliding window, from ``conformance.ssim``'s default. One changed pixel can only be inside
#: this many windows, which is what bounds how far one flipped pixel can move the SSIM average.
SSIM_WIN = 7

#: A cell is allowed to be called sub-noise only if the pixels that moved by MORE than one level are
#: under this fraction of the frame. Stated here rather than discovered per run; the observed maximum
#: over the six cells in job 6410105 is 1.36e-06, so this is a bound with an order of magnitude in hand
#: and it is the number a later run has to beat rather than a number fitted to this one.
GT1_FRAC_MAX = 1e-5

#: An aggregate move is IMMATERIAL when it is under this fraction of the cell's own
#: declaration-vs-measurement headroom -- the distance between what the cell measures and the
#: threshold it promises. 1 % is the tolerance: a change that eats one hundredth of the margin a
#: declaration was frozen with cannot alter any judgement that declaration supports.
HEADROOM_FRAC = 0.01


def _budget(metric: str, px_changed: int, max_abs: int, total_px: int) -> float:
    """The most the aggregate ``metric`` could move GIVEN the pixels that were counted as changed.

    This is the tolerance the reviewer asked to be tied to the measurement rather than to an
    arbitrary epsilon, and it is an arithmetic bound rather than an estimate: if a metric moved by
    less than this, the move is fully accounted for by the pixel flips already in the table and is
    not a second effect that needs its own explanation.

    * ``mean_abs`` averages ``|ours - official|`` over every CHANNEL of every pixel. Only
      ``px_changed`` pixels moved, each channel of each by at most ``max_abs``, so the average moves
      by at most ``3 * px_changed * max_abs / (3 * total_px)``.
    * ``within8_frac`` is the fraction of pixels within 8/255. A pixel that did not move cannot
      cross the boundary, so at most ``px_changed`` of them can, and the fraction moves by at most
      ``px_changed / total_px``.
    * ``ssim`` is an average over sliding windows. One changed pixel lies in at most ``SSIM_WIN**2``
      of them and each window's value is in [-1, 1], so the average moves by at most
      ``2 * SSIM_WIN**2 * px_changed / total_px``. Loose by construction -- it is a sufficient bound,
      not a tight one -- and it still discriminates: a real change is orders of magnitude above it.
    """
    if total_px <= 0:
        return 0.0
    if metric == "mean_abs":
        return max_abs * px_changed / total_px
    if metric == "within8_frac":
        return px_changed / total_px
    if metric == "ssim":
        return 2.0 * (SSIM_WIN ** 2) * px_changed / total_px
    raise ValueError(metric)


def _headroom(metric: str, value: float, decl: dict) -> float | None:
    """How far this cell's measurement sits from the threshold it declares, for this metric."""
    key = {"mean_abs": "mean_abs_max", "within8_frac": "within8_min", "ssim": "ssim_min"}[metric]
    if decl.get(key) is None or value is None:
        return None
    return abs(float(decl[key]) - float(value))


def _judge(cell: dict) -> dict:
    """Turn a cell's measured deltas into a LABELLED verdict, under a written-down rule.

    The rule this replaces was a zero-tolerance sign test on three floats, which is not a test of
    anything a renderer does: it called a -2.9e-09 move of SSIM a regression and put Walker2d (4
    changed pixels, none by more than one level) in the same bucket as a real change, while Swimmer
    (180 changed pixels) passed because its three deltas happened to fall the other way. A sign test
    has no tolerance, and a measurement without a tolerance cannot answer "did this change anything".

    The rule now has two conditions per metric, both stated and both derived from the measurement:

      EXPLAINED  ``|delta| <= _budget(...)``  -- the move is arithmetically accounted for by the
                 pixel flips already counted, so it is not a separate effect;
      IMMATERIAL ``|delta| <= HEADROOM_FRAC * headroom`` -- the move is under 1 % of the distance
                 between this cell's measurement and the threshold it declares, so it cannot change
                 any judgement that declaration supports.

    and three labels:

      IDENTICAL         no pixel moved.
      ULP_LEVEL         every changed pixel moved by exactly ONE level, and every metric move is
                        EXPLAINED and IMMATERIAL. This is a renderer whose output is the same picture
                        with a different last bit.
      SUB_NOISE_SHIFT   some pixels moved by more than one level, but they are under GT1_FRAC_MAX of
                        the frame and every metric move is still EXPLAINED and IMMATERIAL. A handful
                        of silhouette pixels flipped across a high-contrast edge; the picture did not
                        change.
      MEANINGFUL_CHANGE anything else -- including any cell that left its declaration, whose
                        downsample bound no longer clears, or whose fidelity could not be measured.

    ``regressed_sign`` (the old zero-tolerance test) is kept in the output, because the direction of
    each move is evidence worth reading even when its size is not. It is not what the label is made of.
    """
    fid, ds = cell.get("fidelity", {}), cell.get("downsample_exact_frac", {})
    total, P, M = cell["total_px"], cell["px_changed"], cell["max_abs_pixel_delta"]
    if "blocked" in fid or "blocked" in ds:
        cell["verdict"] = "MEANINGFUL_CHANGE"
        cell["verdict_why"] = "fidelity or downsample could not be measured on one of the legs"
        return cell
    decl = fid.get("declared", {})
    judged, all_explained, all_immaterial = {}, True, True
    for metric in ("mean_abs", "within8_frac", "ssim"):
        delta = abs(float(fid[metric]["delta"]))
        budget = _budget(metric, P, M, total)
        head = _headroom(metric, fid[metric]["after"], decl)
        explained = delta <= budget
        immaterial = head is None or delta <= HEADROOM_FRAC * head
        judged[metric] = {"abs_delta": delta, "pixel_budget": budget, "explained_by_pixels": explained,
                          "headroom": head, "headroom_frac_used": None if not head else delta / head,
                          "immaterial_vs_headroom": immaterial}
        all_explained &= explained
        all_immaterial &= immaterial
    fid["judged"] = judged
    fid["regressed_beyond_tolerance"] = not (all_explained and all_immaterial)
    ok_core = (all_explained and all_immaterial and fid["still_inside_declaration"]
               and not ds.get("bound_needs_update"))
    if cell["identical"]:
        cell["verdict"] = "IDENTICAL"
    elif ok_core and cell["frac_px_changed_gt1"] == 0:
        cell["verdict"] = "ULP_LEVEL"
    elif ok_core and cell["frac_px_changed_gt1"] <= GT1_FRAC_MAX:
        cell["verdict"] = "SUB_NOISE_SHIFT"
    else:
        cell["verdict"] = "MEANINGFUL_CHANGE"
    cell["verdict_why"] = (f"gt1 frac {cell['frac_px_changed_gt1']:.3e} (bar {GT1_FRAC_MAX:.0e}); "
                           f"every metric explained by the counted pixels: {all_explained}; "
                           f"every metric under {HEADROOM_FRAC:.0%} of its declaration headroom: "
                           f"{all_immaterial}; inside declaration: {fid['still_inside_declaration']}; "
                           f"downsample bound holds: {not ds.get('bound_needs_update')}")
    return cell


ACCEPTABLE = ("IDENTICAL", "ULP_LEVEL", "SUB_NOISE_SHIFT")


def _conditions(R: dict, out: str) -> dict:
    """The nine conditions the ruling attaches to the re-freeze, each answered from a committed file.

    ``refreeze_ok`` used to be one boolean made of the cell verdicts alone, which under-reported: it
    could not see Gate A, the rocket's gates or the Step-5 surfaces even though those are what the
    ruling asks for. It is now the conjunction of these nine, each carrying the evidence it was
    answered from, so a reviewer can check any one of them without rerunning anything.
    """
    cells = R["cells"]
    C: dict = {}
    C["1_max_abs_pixel_delta_recorded"] = {
        "pass": all("max_abs_pixel_delta" in c for c in cells.values()),
        "evidence": {k: c.get("max_abs_pixel_delta") for k, c in cells.items()}}
    C["2_frac_px_changed_recorded"] = {
        "pass": all("frac_px_changed" in c for c in cells.values()),
        "evidence": {k: c.get("frac_px_changed") for k, c in cells.items()}}
    C["3_frac_px_changed_gt1_under_bar"] = {
        "pass": all(c.get("frac_px_changed_gt1", 1.0) <= GT1_FRAC_MAX for c in cells.values()),
        "bar": GT1_FRAC_MAX,
        "evidence": {k: c.get("frac_px_changed_gt1") for k, c in cells.items()}}
    C["4_official_fidelity_unchanged_or_improved"] = {
        "pass": all(not c.get("fidelity", {}).get("regressed_beyond_tolerance", True)
                    and c.get("fidelity", {}).get("still_inside_declaration") for c in cells.values()),
        "rule": "explained by the counted pixels AND under 1% of the declaration headroom; "
                "and the cell still inside every declared threshold",
        "evidence": {k: {"regressed_sign": c.get("fidelity", {}).get("regressed"),
                         "regressed_beyond_tolerance": c.get("fidelity", {}).get("regressed_beyond_tolerance"),
                         "inside_declaration": c.get("fidelity", {}).get("still_inside_declaration")}
                     for k, c in cells.items()}}
    C["5_gate_a_mandatory_all_pass"] = _read_flag(out, "gatea_branch.json", "all_pass")
    job = R["branch_provenance"].get("slurm_job")
    C["6_rocket_batch_size_invariance_pass"] = _rocket_batch_size(out, job)
    C["7_rocket_seven_gates"] = _rocket_gates(out, job)
    C["8_step5_surfaces_untouched"] = _read_flag(out, "step5_surfaces_compare.json", "all_untouched")
    C["9_both_legs_same_card_same_job"] = {
        "pass": bool(R["same_gpu"] and R["same_host"]
                     and R["baseline_provenance"].get("slurm_job") == R["branch_provenance"].get("slurm_job")),
        "evidence": {"gpu": R["branch_provenance"]["gpu"], "host": R["branch_provenance"]["host"],
                     "slurm_job": R["branch_provenance"].get("slurm_job"),
                     "baseline_stable_envs": R["baseline_provenance"]["stable_envs_file"],
                     "branch_stable_envs": R["branch_provenance"]["stable_envs_file"]}}
    C["all_cells_acceptable"] = {"pass": all(c.get("verdict") in ACCEPTABLE for c in cells.values()),
                                 "evidence": {k: c.get("verdict") for k, c in cells.items()}}
    return C


def _read_flag(out: str, fname: str, key: str) -> dict:
    p = os.path.join(out, fname)
    if not os.path.exists(p):
        return {"pass": None, "evidence": f"{fname} not present beside this comparison"}
    d = json.load(open(p))
    return {"pass": bool(d.get(key)), "evidence": {fname: {key: d.get(key)}}}


def _tier_dirs(out: str, job: str | None) -> list:
    """Where this job's rocket artifacts are, most specific first.

    In the SLURM job the measurement writes straight into ``tier/``; in the worktree that same
    directory holds an EARLIER run (the M4 run of record) and this job's copy lives in
    ``tier/job<id>/``. Answering condition 6 or 7 out of the wrong directory would silently report
    another job's numbers, so the job-tagged directory is searched first and the file that answered
    is named in the evidence.
    """
    tier = os.path.join(os.path.dirname(out.rstrip("/")), "tier")
    dirs = []
    if job:
        dirs.append(os.path.join(tier, f"job{job}"))
    dirs += [tier, out]
    return [d for d in dirs if os.path.isdir(d)]


def _rocket_batch_size(out: str, job: str | None = None) -> dict:
    """PFRocketLanding's mandatory batch-size invariance, from the tier JSON the same job wrote."""
    for root in _tier_dirs(out, job):
        for f in sorted(os.listdir(root)):
            if not f.endswith(".json"):
                continue
            try:
                d = json.load(open(os.path.join(root, f)))
            except Exception:                                                       # pragma: no cover
                continue
            if job and str(d.get("slurm_job")) != str(job):
                continue
            r = (d.get("mandatory") or {}).get("check_batch_size_invariance")
            if isinstance(r, dict) and "max_abs_diff" in r:
                return {"pass": int(r["max_abs_diff"]) == 0,
                        "evidence": {os.path.join(os.path.basename(root), f): r}}
    return {"pass": None, "evidence": f"no tier JSON for job {job} with check_batch_size_invariance"}


def _rocket_gates(out: str, job: str | None = None) -> dict:
    """The rocket's gate lines, from ``tier/gates.txt``; BATCH_SCALES is a declared, recorded FAIL."""
    p = ""
    for root in _tier_dirs(out, job):
        cand = os.path.join(root, "gates.txt")
        if os.path.exists(cand) and (not job or f"job {job}" in open(cand).read()):
            p = cand
            break
    if not p:
        return {"pass": None, "evidence": f"no tier/gates.txt for job {job} beside this run"}
    rows = [ln.split("  ")[0:2] for ln in open(p).read().splitlines() if ln.strip()]
    got = {r[1].strip(): r[0].strip() for r in rows if len(r) > 1}
    failing = {k: v for k, v in got.items() if v != "PASS"}
    return {"pass": set(failing) <= {"BATCH_SCALES"},
            "rule": "every gate PASS except BATCH_SCALES, which is a declared, recorded FAIL "
                    "(fps(4096)/fps(1) under the bar of 20) and is carried as the tier's qualifier",
            "evidence": {"from": os.path.join(os.path.basename(os.path.dirname(p)), "gates.txt"),
                         "failing": failing, "n_gates": len(got)}}


def phase_compare(out: str, rejudge: bool = False) -> None:
    """The per-cell delta table and the re-freeze verdict.

    ``rejudge=True`` re-derives the labels from the COMMITTED comparison rather than from the frame
    ``.npz`` files, which the job deletes once the digests and counts are taken (they are 11 MB per
    cell per leg and the counts are what the judgement uses). The pixel statistics are read back from
    the previous ``compare_refreeze.json`` unchanged -- nothing is recomputed, only re-labelled -- so
    the rule can be corrected without a GPU and without pretending the run happened again.
    """
    base = json.load(open(os.path.join(out, "baseline.json")))
    brch = json.load(open(os.path.join(out, "branch.json")))
    prev = {}
    if rejudge:
        prev = json.load(open(os.path.join(out, "compare_refreeze.json")))["cells"]
    R = {"baseline_provenance": base["provenance"], "branch_provenance": brch["provenance"],
         "same_gpu": base["provenance"]["gpu"] == brch["provenance"]["gpu"],
         "same_host": base["provenance"]["host"] == brch["provenance"]["host"], "cells": {}}
    if rejudge:
        R["relabelled_from"] = {
            "file": "compare_refreeze.json", "why": "reviewer item RD-1: the verdict rule was a "
            "zero-tolerance sign test; the pixel statistics below are the same measured numbers, "
            "carried over unchanged, and only the labels and refreeze_ok are re-derived"}
    frozen = _frozen_downsample_bound()
    for name in CELLS:
        b, n = base["cells"].get(name, {}), brch["cells"].get(name, {})
        if "frames" not in b or "frames" not in n:
            R["cells"][name] = {"blocked": {"baseline": b.get("missing"), "branch": n.get("missing")}}
            continue
        if rejudge:
            p = prev[name]
            cell = {k: p[k] for k in ("identical", "total_px", "max_abs_pixel_delta", "px_changed",
                                      "frac_px_changed", "px_changed_gt1", "frac_px_changed_gt1",
                                      "frames_per_cell", "sha256_baseline", "sha256_branch",
                                      "states_sha256_match")}
        else:
            fb = np.load(os.path.join(out, f"baseline_{name}.npz"))["frames"].astype(np.int16)
            fn = np.load(os.path.join(out, f"branch_{name}.npz"))["frames"].astype(np.int16)
            d = np.abs(fn - fb).max(-1)                   # per-pixel, over the three channels
            total = int(d.size)
            changed = int((d > 0).sum())
            gt1 = int((d > 1).sum())
            cell = {"identical": bool(changed == 0), "total_px": total,
                    "max_abs_pixel_delta": int(d.max()),
                    "px_changed": changed, "frac_px_changed": changed / total,
                    "px_changed_gt1": gt1, "frac_px_changed_gt1": gt1 / total,
                    "frames_per_cell": int(fb.shape[0]),
                    "sha256_baseline": b["frames"]["sha256"], "sha256_branch": n["frames"]["sha256"],
                    "states_sha256_match": b["frames"]["states_sha256"] == n["frames"]["states_sha256"]}

        vb, vn = b.get("visual_fidelity", {}), n.get("visual_fidelity", {})
        if "mean_abs" in vb and "mean_abs" in vn:
            fid = {k: {"before": vb[k], "after": vn[k], "delta": vn[k] - vb[k]}
                   for k in ("mean_abs", "within8_frac", "ssim", "exact_frac", "interior_exact_frac")
                   if k in vb and k in vn}
            # the ZERO-TOLERANCE sign test, kept for transparency: it is the direction of each move,
            # and it is deliberately not what the verdict is made of (see _judge).
            fid["regressed"] = bool(fid["mean_abs"]["delta"] > 0
                                    or fid["within8_frac"]["delta"] < 0
                                    or fid["ssim"]["delta"] < 0)
            decl = n.get("declared", {})
            fid["still_inside_declaration"] = bool(
                (decl.get("mean_abs_max") is None or vn["mean_abs"] <= decl["mean_abs_max"])
                and (decl.get("within8_min") is None or vn["within8_frac"] >= decl["within8_min"])
                and (decl.get("ssim_min") is None or vn["ssim"] >= decl["ssim_min"]))
            fid["declared"] = decl
            cell["fidelity"] = fid
        else:
            cell["fidelity"] = {"blocked": {"baseline": vb.get("blocked"), "branch": vn.get("blocked")}}

        db, dn = b.get("downsample", {}), n.get("downsample", {})
        if "exact_frac" in db and "exact_frac" in dn:
            bound = frozen["value"].get(name)
            cell["downsample_exact_frac"] = {
                "before": db["exact_frac"], "after": dn["exact_frac"],
                "delta": dn["exact_frac"] - db["exact_frac"],
                "frozen_bound": bound, "margin": frozen["margin"],
                "before_clears_bound": None if bound is None else db["exact_frac"] >= bound - frozen["margin"],
                "after_clears_bound": None if bound is None else dn["exact_frac"] >= bound - frozen["margin"],
                "bound_needs_update": None if bound is None else dn["exact_frac"] < bound - frozen["margin"]}
        else:
            cell["downsample_exact_frac"] = {"blocked": {"baseline": db.get("blocked"),
                                                         "branch": dn.get("blocked")}}

        R["cells"][name] = _judge(cell)
        print("compare", name, cell["verdict"],
              json.dumps({k: cell[k] for k in ("max_abs_pixel_delta", "frac_px_changed",
                                               "frac_px_changed_gt1")}), flush=True)
    R["rule"] = {"labels": list(ACCEPTABLE) + ["MEANINGFUL_CHANGE"],
                 "gt1_frac_max": GT1_FRAC_MAX, "headroom_frac": HEADROOM_FRAC, "ssim_win": SSIM_WIN,
                 "where": "scratch/step7/rocket-landing/refreeze_locomotion.py:_judge / _budget"}
    R["conditions"] = _conditions(R, out)
    R["refreeze_ok"] = bool(all(v["pass"] for v in R["conditions"].values()))
    R["frozen_downsample_bound_source"] = frozen["source"]
    _dump(os.path.join(out, "compare_refreeze.json"), R)
    for k, v in R["conditions"].items():
        print(f"  condition {k}: {v['pass']}", flush=True)
    print("REFREEZE_OK", R["refreeze_ok"], flush=True)


def _frozen_downsample_bound() -> dict:
    """``DOWNSAMPLE_EXACT_FRAC`` and its margin, read from the test module that declares them.

    ``tests/`` is frozen, so the branch's copy and the baseline's are the same file; it is read from
    the measured checkout anyway, by the same rule as every other provenance field here.
    """
    root = _code_root()
    ns: dict = {}
    src = open(os.path.join(root, "tests", "gpu", "test_reproducibility.py")).read()
    import ast
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in ("DOWNSAMPLE_EXACT_FRAC", "DOWNSAMPLE_MARGIN"):
                    ns[t.id] = ast.literal_eval(node.value)
    return {"value": ns.get("DOWNSAMPLE_EXACT_FRAC", {}), "margin": ns.get("DOWNSAMPLE_MARGIN", 0.005),
            "source": "tests/gpu/test_reproducibility.py:DOWNSAMPLE_EXACT_FRAC / DOWNSAMPLE_MARGIN"}


def _dump(path: str, obj) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", path, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("render", "gatea", "compare", "rejudge"))
    ap.add_argument("--tag", default="branch", choices=("baseline", "branch"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # NOTHING is inserted into sys.path: PYTHONPATH decides which checkout is imported, and the
    # baseline leg must not pick up the branch this script lives in.
    if a.phase == "render":
        phase_render(a.out, a.tag)
    elif a.phase == "gatea":
        phase_gatea(a.out)
    else:
        phase_compare(a.out, rejudge=a.phase == "rejudge")


if __name__ == "__main__":
    main()
