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

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


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
           "families_touched": touched, "incorrect_rows": bad, "rows": rows}
    (out / "spot_semantics_sweep.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"built {len(ok)}  correct {len(ok)-len(bad)}  incorrect {len(bad)}")
    print(f"families whose light_exponent changed at all: {touched or 'NONE'}")
    nspot = [(r['env'], r['n_spot']) for r in ok if r['n_spot']]
    print(f"families carrying a spot light at all: {nspot or 'none'}")
    for r in bad:
        print("  INCORRECT", r["env"], r["expected_neutralised"], r["reported"], r["actually_changed"])


if __name__ == "__main__":
    main()
