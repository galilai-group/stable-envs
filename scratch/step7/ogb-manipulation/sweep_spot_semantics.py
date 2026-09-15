"""Registry-wide proof for the spot correction: only out-of-range spot lights may change.

`neutralise_illegal_spots` runs inside `prepare_model`, which every MJWarp-backed family calls, so
the obligation is the same one `texture_gaps` carried: show that nothing else moves. The check here
is stronger than a set comparison, because this correction WRITES to the model rather than only
reporting -- for every registered family it enumerates the lights, classifies each, and asserts that
the set the correction touches is exactly the set of spot lights whose cutoff MuJoCo itself would
reject. A family with no such light must come back with `neutralised_spots == []` and a
bit-identical `light_exponent`.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _provenance() -> dict:
    """Host, GPU, package versions and the tree's git SHA, in the artifact itself.

    A sweep result with no provenance is a number without a machine. Recorded in the JSON rather
    than only in a SLURM log, because logs are rotated and artifacts are committed.
    """
    import importlib.metadata as md
    import platform, socket, subprocess
    out = {"host": socket.gethostname(), "platform": platform.platform(),
           "python": sys.version.split()[0], "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "mujoco_gl": os.environ.get("MUJOCO_GL")}
    try:
        out["git_sha"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        out["git_sha"] = None
    for pkg in ("mujoco", "mujoco-warp", "warp-lang", "torch", "numpy", "ogbench", "dm-control"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out["gpu"] = None
    return out


def models_from_env(env):
    u = getattr(env, "unwrapped", env)
    for attr in ("model", "_model", "mj_model"):
        mm = getattr(u, attr, None)
        if mm is not None and hasattr(mm, "ngeom"):
            return mm
    for acc in ("_scene", "scene"):
        f = getattr(u, acc, None)
        if callable(f):
            try:
                sc = f()
            except Exception:
                continue
            mm = getattr(sc, "model", None)
            if mm is not None and hasattr(mm, "ngeom"):
                return mm
    for k, v in list(vars(u).items()):
        if hasattr(v, "ngeom"):
            return v
        mm = getattr(v, "model", None)
        if mm is not None and hasattr(mm, "ngeom"):
            return mm
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bake", action="store_true",
                    help="also sweep bake_unsamplable_textures (see sweep_bake below)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import mujoco
    import stable_envs as se
    from stable_envs.render.mjwarp import (GL_SPOT_CUTOFF_MAX, prepare_model, restore_model)

    rows, touched = [], []
    for name in sorted(se.REGISTRY):
        try:
            env = se.make(name, batch=1, device="cpu")
        except Exception as e:
            rows.append({"env": name, "status": "skip", "reason": f"{type(e).__name__}"[:60]})
            continue
        m = models_from_env(env)
        if m is None:
            rows.append({"env": name, "status": "no-mjmodel"})
            continue
        lights = []
        for i in range(int(m.nlight)):
            is_spot = (int(m.light_type[i]) == 0 if hasattr(m, "light_type")
                       else not int(m.light_directional[i]))
            lights.append({"i": i, "spot": bool(is_spot), "cutoff": float(m.light_cutoff[i]),
                           "exponent": float(m.light_exponent[i])})
        expected = [l["i"] for l in lights
                    if l["spot"] and not (0.0 <= l["cutoff"] <= GL_SPOT_CUTOFF_MAX)]
        before = np.array(m.light_exponent, copy=True)
        p = prepare_model(m)
        after = np.array(m.light_exponent, copy=True)
        changed = sorted(np.nonzero(before != after)[0].tolist())
        restore_model(m, p)
        restored = np.array_equal(np.array(m.light_exponent), before)
        r = {"env": name, "status": "ok", "nlight": int(m.nlight),
             "n_spot": sum(1 for l in lights if l["spot"]),
             "expected_neutralised": expected, "reported": sorted(p.neutralised_spots),
             "actually_changed": changed, "restored_exactly": bool(restored),
             "correct": bool(sorted(p.neutralised_spots) == expected == changed and restored),
             "lights": lights}
        rows.append(r)
        if changed:
            touched.append(name)
        try:
            env.close()
        except Exception:
            pass

    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in ok if not r["correct"]]
    doc = {"built": len(ok), "correct": len(ok) - len(bad), "incorrect": len(bad),
           "families_touched": touched, "incorrect_rows": bad,
           "provenance": _provenance(),
           # The denominator, stated rather than implied. "40 identical" is only meaningful next to
           # how many ids exist and how many could not be reached in THIS interpreter.
           "registry_ids_total": len(rows),
           "n_with_reachable_mjmodel": len(ok),
           "n_skipped_not_constructible": sum(1 for r in rows if r.get("status") == "skip"),
           "n_no_mjmodel": sum(1 for r in rows if r.get("status") == "no-mjmodel"),
           "coverage_note": (
               "Families that need a package this interpreter lacks, or that expose no MjModel "
               "(the 2-D families), are counted here rather than dropped. Fetch in particular is "
               "cuda-and-mujoco-3.11 only and is covered by its CUDA byte-identity replays, not by "
               "this sweep."),
           "rows": rows}
    (out / "spot_semantics_sweep.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"built {len(ok)}  correct {len(ok)-len(bad)}  incorrect {len(bad)}")
    print(f"families whose light_exponent changed at all: {touched or 'NONE'}")
    nspot = [(r['env'], r['n_spot']) for r in ok if r['n_spot']]
    print(f"families carrying a spot light at all: {nspot or 'none'}")
    for r in bad:
        print("  INCORRECT", r["env"], r["expected_neutralised"], r["reported"], r["actually_changed"])
    if a.bake:
        sweep_bake(a.out)




# --- appended: the same no-change obligation for `bake_unsamplable_textures` ------------------
# Run with --bake. The helper is opt-in (nothing calls it automatically), so the obligation is
# weaker than for prepare_model -- but "which registered families would it touch if called?" is
# exactly what a reviewer needs, and the answer must be NONE.
def sweep_bake(out):
    import json
    import numpy as np
    import stable_envs as se
    from stable_envs.render.mjwarp import (bake_unsamplable_textures, restore_primitive_textures,
                                           unsamplable_texture_materials)
    rows, touched = [], []
    for name in sorted(se.REGISTRY):
        try:
            env = se.make(name, batch=1, device="cpu")
        except Exception:
            continue
        m = models_from_env(env)
        if m is None:
            continue
        targets = unsamplable_texture_materials(m)
        rgba, texid = np.array(m.mat_rgba, copy=True), np.array(m.mat_texid, copy=True)
        b = bake_unsamplable_textures(m)
        changed = not (np.array_equal(m.mat_rgba, rgba) and np.array_equal(m.mat_texid, texid))
        restore_primitive_textures(m, b)
        restored = np.array_equal(m.mat_rgba, rgba) and np.array_equal(m.mat_texid, texid)
        rows.append({"env": name, "n_unsamplable_materials": len(targets),
                     "n_baked": b["n_materials"], "changed": bool(changed),
                     "restored_exactly": bool(restored)})
        if changed:
            touched.append(name)
        try:
            env.close()
        except Exception:
            pass
    doc = {"built": len(rows), "families_touched": touched, "provenance": _provenance(),
           "all_restored": all(r["restored_exactly"] for r in rows), "rows": rows}
    (Path(out) / "bake_unsamplable_sweep.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"bake sweep: built {len(rows)}  would touch: {touched or 'NONE'}  "
          f"all restored exactly: {doc['all_restored']}")

if __name__ == "__main__":
    main()
