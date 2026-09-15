"""Step 7 / W5 phase 0 -- split the non-plate pixels into MOVER BODY and MOVER SHADOW.

`probe_plate_shadow.py` shows the plate is static once the movers are hidden, so the
Fetch pattern's cheap half exists.  What it cannot separate is WHY the remaining pixels
move: a mover silhouette (which a ray-caster draws) or the shadow the `global`
directional light casts from that mover onto the floor (which needs a shadow ray, and
which lands on pixels the plate would otherwise own).

Uses MuJoCo's own segmentation buffer for the silhouette, so the split is measured and
not inferred from a colour difference.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def movers(m):
    out = []
    for i in range(m.ngeom):
        b = int(m.geom_bodyid[i]); moving = False
        while b > 0:
            if int(m.body_dofnum[b]) > 0 or int(m.body_mocapid[b]) >= 0:
                moving = True; break
            b = int(m.body_parentid[b])
        if moving:
            out.append(i)
    return out


def main():
    import gymnasium as gym
    import mujoco
    import stable_worldmodel  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", nargs="+", default=["swm/OGBCube-v0", "swm/OGBScene-v0"])
    ap.add_argument("--states", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for eid in a.envs:
        env = gym.make(eid); u = env.unwrapped
        env.reset(seed=5); m, d = u.model, u.data
        h, w = u._render_height, u._render_width
        mv = set(movers(m))
        nb = getattr(u, "_num_buttons", 0)
        kw = {f"button_state_{i}": 0 for i in range(nb)} if nb else {}
        rec = {"env_id": eid, "h": h, "w": w, "states": a.states,
               "rows": [], "light_castshadow": [int(x) for x in m.light_castshadow]}

        for s in range(a.states):
            env.reset(seed=300 + s); m, d = u.model, u.data
            mv = set(movers(m))
            for _ in range(int(np.random.default_rng(s).integers(0, 20))):
                env.step(env.action_space.sample())
            f_on = u.render().copy()
            seg = u.render(segmentation=True).copy()
            gid = seg[..., 0]
            mover_px = np.isin(gid, list(mv))
            li = [i for i in range(m.nlight) if int(m.light_castshadow[i])]
            for i in li:
                m.light_castshadow[i] = 0
            f_off = u.render().copy()
            for i in li:
                m.light_castshadow[i] = 1
            diff = (f_on != f_off).any(-1)
            mag = np.abs(f_on.astype(np.int16) - f_off.astype(np.int16)).max(-1)
            rec["rows"].append({
                "state": s,
                "mover_silhouette_share": float(mover_px.mean()),
                "shadow_changed_share_all": float(diff.mean()),
                "shadow_changed_share_off_mover": float((diff & ~mover_px).mean()),
                "shadow_changed_max_abs": int(mag.max()),
                "shadow_changed_share_off_mover_gt8": float(((mag > 8) & ~mover_px).mean()),
                "shadow_changed_share_off_mover_gt32": float(((mag > 32) & ~mover_px).mean()),
                "shadow_mean_abs_off_mover": float(mag[~mover_px].mean()),
            })

        def avg(k):
            return float(np.mean([r[k] for r in rec["rows"]]))
        rec["mean"] = {k: avg(k) for k in rec["rows"][0] if k != "state"}
        tag = eid.split("/")[-1].replace("-v0", "")
        (out / f"shadow_split_{tag}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        r = rec["mean"]
        print(f"{eid}: mover silhouette {r['mover_silhouette_share']*100:.1f}%  "
              f"shadow-off-mover any {r['shadow_changed_share_off_mover']*100:.1f}%  "
              f">8/255 {r['shadow_changed_share_off_mover_gt8']*100:.2f}%  "
              f">32/255 {r['shadow_changed_share_off_mover_gt32']*100:.2f}%  "
              f"max {r['shadow_changed_max_abs']:.0f}  mean|d| {r['shadow_mean_abs_off_mover']:.2f}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
