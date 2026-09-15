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

  --phase compare
      per cell: max_abs_pixel_delta, fraction of pixels changed, fraction changed by MORE THAN ONE
      level, the two digests, fidelity before/after with the delta and the cell's declared thresholds,
      and the downsample fraction before/after against its frozen bound and margin. Emits a verdict
      per cell and a single ``refreeze_ok``.

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


def phase_compare(out: str) -> None:
    """The per-cell delta table and the re-freeze verdict."""
    base = json.load(open(os.path.join(out, "baseline.json")))
    brch = json.load(open(os.path.join(out, "branch.json")))
    R = {"baseline_provenance": base["provenance"], "branch_provenance": brch["provenance"],
         "same_gpu": base["provenance"]["gpu"] == brch["provenance"]["gpu"],
         "same_host": base["provenance"]["host"] == brch["provenance"]["host"], "cells": {}}
    frozen = _frozen_downsample_bound()
    all_ok = True
    for name in CELLS:
        b, n = base["cells"].get(name, {}), brch["cells"].get(name, {})
        if "frames" not in b or "frames" not in n:
            R["cells"][name] = {"blocked": {"baseline": b.get("missing"), "branch": n.get("missing")}}
            all_ok = False
            continue
        fb = np.load(os.path.join(out, f"baseline_{name}.npz"))["frames"].astype(np.int16)
        fn = np.load(os.path.join(out, f"branch_{name}.npz"))["frames"].astype(np.int16)
        d = np.abs(fn - fb).max(-1)                       # per-pixel, over the three channels
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
            # "unchanged or improved": mean_abs must not rise, within8/ssim must not fall
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

        fid_bad = bool(cell["fidelity"].get("regressed") or cell["fidelity"].get("blocked"))
        ds_bad = bool(cell["downsample_exact_frac"].get("bound_needs_update"))
        cell["verdict"] = ("IDENTICAL" if cell["identical"] else
                           "ULP_ONLY" if not fid_bad and not ds_bad else "MEANINGFUL_CHANGE")
        all_ok &= cell["verdict"] in ("IDENTICAL", "ULP_ONLY")
        R["cells"][name] = cell
        print("compare", name, cell["verdict"],
              json.dumps({k: cell[k] for k in ("max_abs_pixel_delta", "frac_px_changed",
                                               "frac_px_changed_gt1")}), flush=True)
    R["refreeze_ok"] = bool(all_ok and R["same_gpu"] and R["same_host"])
    R["frozen_downsample_bound_source"] = frozen["source"]
    _dump(os.path.join(out, "compare_refreeze.json"), R)
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
    ap.add_argument("--phase", required=True, choices=("render", "gatea", "compare"))
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
        phase_compare(a.out)


if __name__ == "__main__":
    main()
