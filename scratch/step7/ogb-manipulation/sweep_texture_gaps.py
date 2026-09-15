"""M2 proof obligation: `texture_gaps` must not change its answer for any existing family.

The fix adds one condition (a texture whose `tex_type` is not 2-D is a gap, whatever the geom type).
That is only safe if it is *additive*: every geom the old rule flagged must still be flagged, and no
registered family may acquire a flag it did not have — unless that family really does wear a cube
map on a geom, in which case the new flag is the point and must be shown explicitly rather than
absorbed.

So: build every registered env that will build here, run BOTH implementations over its MjModel, and
print the two sets. The OLD implementation is reproduced verbatim below rather than imported from
git, so the comparison is against the code that actually shipped and not against a paraphrase.
"""
from __future__ import annotations

import argparse, json, sys, traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

TEXTURED_OK = (0, 7)


def texture_gaps_old(m, groups=(0, 1, 2)):
    """Verbatim pre-Step-7 implementation (stable_envs/render/mjwarp.py @ 76dc511)."""
    import mujoco
    out = []
    for g in range(m.ngeom):
        if int(m.geom_group[g]) not in groups or int(m.geom_type[g]) in TEXTURED_OK:
            continue
        mid = int(m.geom_matid[g])
        if mid < 0:
            continue
        t = m.mat_texid[mid]
        t = int(np.max(t)) if np.ndim(t) else int(t)
        if t >= 0:
            out.append((g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}",
                        int(m.geom_type[g])))
    return out


def models_from_env(env):
    """The env's MjModel, however this repo happens to hold it.

    Families differ: the OGBench/DM Control cells keep it on a cached `_Scene` behind `_scene()`,
    Fetch builds one in `reference_env`, others expose `.model` directly. Rather than guess, try the
    direct attributes, then the scene accessor, then walk one level of `__dict__` for anything with
    an `ngeom`.
    """
    u = getattr(env, "unwrapped", env)
    for attr in ("model", "_model", "mj_model"):
        mm = getattr(u, attr, None)
        if mm is not None and hasattr(mm, "ngeom"):
            return {attr: mm}
    for acc in ("_scene", "scene"):
        f = getattr(u, acc, None)
        if callable(f):
            try:
                sc = f()
            except Exception:
                continue
            mm = getattr(sc, "model", None)
            if mm is not None and hasattr(mm, "ngeom"):
                return {f"{acc}().model": mm}
    for k, v in list(vars(u).items()):
        if hasattr(v, "ngeom"):
            return {k: v}
        mm = getattr(v, "model", None)
        if mm is not None and hasattr(mm, "ngeom"):
            return {f"{k}.model": mm}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from stable_envs.render.mjwarp import texture_gaps as texture_gaps_new
    import stable_envs as se

    names = a.only
    if not names:
        try:
            names = sorted(se.REGISTRY)
        except Exception:
            names = sorted(se.REGISTRY)
    rows, built, skipped = [], 0, 0
    for name in names:
        try:
            env = se.make(name, batch=1, device="cpu")
        except Exception as e:
            skipped += 1
            rows.append({"env": name, "status": "not-constructible",
                         "reason": f"{type(e).__name__}: {e}"[:160]})
            continue
        mm = models_from_env(env)
        if not mm:
            skipped += 1
            rows.append({"env": name, "status": "no-mjmodel"})
            continue
        m = list(mm.values())[0]
        try:
            old = {g for g, _, _ in texture_gaps_old(m)}
            new = {g for g, _, _ in texture_gaps_new(m)}
        except Exception as e:
            rows.append({"env": name, "status": "error",
                         "reason": f"{type(e).__name__}: {e}"[:160]})
            continue
        built += 1
        rows.append({"env": name, "status": "ok", "ngeom": int(m.ngeom), "ntex": int(m.ntex),
                     "n_old": len(old), "n_new": len(new),
                     "identical": bool(old == new),
                     "lost": sorted(old - new), "gained": sorted(new - old),
                     "tex_types": sorted({int(m.tex_type[i]) for i in range(m.ntex)})})
        try:
            env.close()
        except Exception:
            pass

    # POSITIVE CONTROL. A sweep that only ever prints "identical" cannot distinguish a correct
    # additive fix from a fix that does nothing. `swm/OGBScene-v0` is the model that crashed the
    # renderer while the old rule returned an empty list, so it is the case the new condition must
    # catch -- and it is NOT in this registry, which is why it is loaded separately.
    control = {"env": "swm/OGBScene-v0 (positive control, not a stable-envs registry entry)"}
    try:
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        ce = gym.make("swm/OGBScene-v0")
        ce.reset(seed=0)
        cm = ce.unwrapped.model
        old = {g for g, _, _ in texture_gaps_old(cm)}
        new = {g for g, _, _ in texture_gaps_new(cm)}
        import mujoco
        control.update({
            "status": "ok", "n_old": len(old), "n_new": len(new),
            "gained": sorted(new - old), "lost": sorted(old - new),
            "gained_names": [mujoco.mj_id2name(cm, mujoco.mjtObj.mjOBJ_GEOM, g) for g in sorted(new - old)],
            "fires": bool(new - old)})
        ce.close()
    except Exception as e:
        control.update({"status": "unavailable", "reason": f"{type(e).__name__}: {e}"[:160]})
    print(f"positive control swm/OGBScene-v0: old {control.get('n_old')} -> new {control.get('n_new')} "
          f"gained {control.get('gained_names')}")

    same = [r for r in rows if r.get("status") == "ok" and r["identical"]]
    diff = [r for r in rows if r.get("status") == "ok" and not r["identical"]]
    doc = {"built": built, "skipped": skipped, "identical": len(same),
           "changed": len(diff), "changed_rows": diff, "positive_control": control, "rows": rows}
    (out / "texture_gaps_sweep.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"built {built}  skipped {skipped}  identical {len(same)}  changed {len(diff)}")
    for r in diff:
        print(f"  CHANGED {r['env']}: old {r['n_old']} -> new {r['n_new']} "
              f"lost={r['lost']} gained={r['gained']} tex_types={r['tex_types']}")
    if not diff:
        print("  no registered family that builds here changes its texture_gaps answer")


if __name__ == "__main__":
    main()
