"""How much of the fixed-camera Adroit frame is geometry mujoco_warp cannot draw?

The deciding number for the family. probe6's attempt reported background=1.0 because it decoded
mujoco's segmentation buffer as (type, id); it is (H, W, 2) with channel 0 = OBJECT ID and channel
1 = OBJECT TYPE -- ch1 carries only {-1, 5, 6} (background, mjOBJ_GEOM, mjOBJ_SITE) while ch0 carries
the geom ids. Decoded correctly here.

Categories, over states the env actually reaches (reset + 0-25 random actions):

  moving_undrawable   a geom that MOVES and that mujoco_warp cannot draw -- either textured with a
                      type whose UVs it cannot derive (anything but PLANE/MESH), or translucent
                      (its shader packs a constant alpha of 255). This is the number that decides
                      the family: a static geom is baked through MuJoCo itself and is exact.
  moving_ok           a geom that moves and that mujoco_warp CAN draw (the hand's meshes).
  static_*            same split for geometry welded to the world and not touched at reset.
  site                mujoco draws sites, mujoco_warp draws none; reported separately because each
                      one has to be re-expressed as a geom or shown to reach no pixel.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

REF_ID = {"Door": "AdroitHandDoor-v1", "Hammer": "AdroitHandHammer-v1",
          "Pen": "AdroitHandPen-v1", "Relocate": "AdroitHandRelocate-v1"}
PER_EPISODE = {"Door": "frame", "Hammer": "nail_board", "Pen": "target", "Relocate": "Object"}
SIZE = 480
TMP = "/mnt/vast/home/randall.balestriero/se_logs/adroit"


def main():
    import mujoco
    import gymnasium, gymnasium_robotics                                        # noqa: F401
    from stable_envs.render import mjwarp as MW
    gymnasium.register_envs(gymnasium_robotics)

    out = {}
    for scene, ref in REF_ID.items():
        env = gymnasium.make(ref)
        u = env.unwrapped
        m = u.model
        u.reset(seed=0)
        env.action_space.seed(0)

        gname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"
        pe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PER_EPISODE[scene])
        subtree = set()
        for b in range(m.nbody):
            x = b
            while x > 0:
                if x == pe:
                    subtree.add(b); break
                x = int(m.body_parentid[x])

        groups = MW.DEFAULT_GROUPS
        visible = [g for g in range(m.ngeom) if int(m.geom_group[g]) in groups]
        alpha = MW.effective_alpha(m, "geom")
        gaps = {g for g, _, _ in MW.texture_gaps(m, groups)}
        translucent = {g for g in visible if 0.0 < alpha[g] < 1.0}
        invisible = {g for g in visible if alpha[g] <= 0.0}
        undrawable = (gaps | translucent) - invisible
        moves = [g for g in visible
                 if int(m.body_weldid[m.geom_bodyid[g]]) != 0 or int(m.geom_bodyid[g]) in subtree
                 or int(m.body_mocapid[m.geom_bodyid[g]]) >= 0]
        static = [g for g in visible if g not in moves]

        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "fixed")
        rend = mujoco.Renderer(m, width=SIZE, height=SIZE)
        keys = ("moving_undrawable", "moving_ok", "static_undrawable", "static_ok", "site",
                "background")
        cov = {k: 0.0 for k in keys}
        per_geom = {}
        n = 12
        for i in range(n):
            for _ in range(int(np.random.default_rng(i).integers(0, 26))):
                u.step(env.action_space.sample())
            rend.enable_segmentation_rendering()
            rend.update_scene(u.data, camera=cid)
            seg = rend.render()
            rend.disable_segmentation_rendering()
            oid, otype = seg[..., 0].astype(np.int64), seg[..., 1].astype(np.int64)
            gpix = otype == int(mujoco.mjtObj.mjOBJ_GEOM)
            spix = otype == int(mujoco.mjtObj.mjOBJ_SITE)
            tot = float(oid.size)
            sets = {"moving_undrawable": [g for g in moves if g in undrawable],
                    "moving_ok": [g for g in moves if g not in undrawable],
                    "static_undrawable": [g for g in static if g in undrawable],
                    "static_ok": [g for g in static if g not in undrawable]}
            for k, sel in sets.items():
                a = np.asarray(sel, np.int64) if sel else np.zeros(0, np.int64)
                cov[k] += float((np.isin(oid, a) & gpix).sum() / tot) / n
            cov["site"] += float(spix.sum() / tot) / n
            cov["background"] += float((~(gpix | spix)).sum() / tot) / n
            for g in undrawable:
                if g in moves:
                    per_geom[gname(g)] = per_geom.get(gname(g), 0.0) + float(
                        ((oid == g) & gpix).sum() / tot) / n
            u.reset(seed=300 + i)

        out[scene] = {
            "coverage": {k: round(v, 5) for k, v in cov.items()},
            "moving_undrawable_per_geom": {k: round(v, 5) for k, v in
                                           sorted(per_geom.items(), key=lambda x: -x[1])},
            "why_undrawable": {
                gname(g): ("translucent alpha=%.2f" % alpha[g] if g in translucent
                           else "textured type=%d" % int(m.geom_type[g]))
                for g in sorted(undrawable) if g in moves},
        }
        print(f"\n===== {scene} =====")
        print(json.dumps(out[scene], indent=2))

    with open(os.path.join(TMP, "probe8.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
