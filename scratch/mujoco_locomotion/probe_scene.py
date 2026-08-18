"""What MuJoCo actually hands its rasterizer for these scenes, and what blocks the tracking family.

Part 1 (flat family: Reacher / Pusher / InvertedPendulum / InvertedDoublePendulum): the resolved
lights and per-geom material constants, read off ``MjvScene`` rather than guessed from the XML, plus
the background colour where no geom is drawn.

Part 2 (tracking family: cheetah / hopper / walker2d / ant / humanoid / swimmer): how many distinct
colours the reference paints on the FLOOR. A nearest-texel sampler can produce two (the checker's two
colours) times the shading; anything more is GL texture filtering, which no analytic caster
reproduces. Also: how much of the frame the floor's ``reflectance=0.5`` mirror is worth.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import gymnasium as gym
import mujoco

FLAT = ["Reacher-v5", "Pusher-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]
TRACK = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5", "Swimmer-v5"]


def _env(name, size):
    env = gym.make(name, render_mode="rgb_array", width=size, height=size)
    u = env.unwrapped
    env.reset(seed=0)
    u.render()
    return env, u


def _seg(m, d, cam, size, r=None):
    r = r or mujoco.Renderer(m, height=size, width=size)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam)
    s = r.render()
    r.disable_segmentation_rendering()
    return np.where(s[..., 1].astype(np.int64) == int(mujoco.mjtObj.mjOBJ_GEOM),
                    s[..., 0].astype(np.int64), -1), r


def flat_facts(name, size=128):
    env, u = _env(name, size)
    m, d = u.model, u.data
    mr = u.mujoco_renderer
    cam = mr.camera_id if mr.camera_id >= 0 else mr.viewer.cam
    r = mujoco.Renderer(m, height=size, width=size)
    r.update_scene(d, camera=cam)
    scn = r.scene
    out = {"env": name, "scn_nlight": int(scn.nlight), "lights": [],
           "flags": {str(f): int(scn.flags[f]) for f in range(len(scn.flags))},
           "offsamples": int(m.vis.quality.offsamples),
           "headlight": {"ambient": [float(x) for x in m.vis.headlight.ambient],
                         "diffuse": [float(x) for x in m.vis.headlight.diffuse],
                         "specular": [float(x) for x in m.vis.headlight.specular],
                         "active": int(m.vis.headlight.active)},
           "geoms": []}
    for i in range(int(scn.nlight)):
        L = scn.lights[i]
        out["lights"].append({
            "pos": [float(x) for x in L.pos], "dir": [float(x) for x in L.dir],
            "type": int(L.type), "headlight": int(L.headlight), "castshadow": int(L.castshadow),
            "ambient": [float(x) for x in L.ambient], "diffuse": [float(x) for x in L.diffuse],
            "specular": [float(x) for x in L.specular],
            "attenuation": [float(x) for x in L.attenuation],
            "cutoff": float(L.cutoff), "exponent": float(L.exponent)})
    for i in range(int(scn.ngeom)):
        g = scn.geoms[i]
        out["geoms"].append({
            "i": i, "objid": int(g.objid), "type": int(g.type), "matid": int(g.matid),
            "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g.objid)) or "",
            "rgba": [float(x) for x in g.rgba], "specular": float(g.specular),
            "shininess": float(g.shininess), "emission": float(g.emission),
            "reflectance": float(g.reflectance), "texcoord": int(g.texcoord),
            "size": [float(x) for x in g.size], "category": int(g.category)})
    # background colour where nothing is drawn
    seg, _ = _seg(m, d, cam, size, r)
    frame = u.render()
    bg = frame[seg < 0]
    out["background_unique"] = [list(map(int, c)) for c in np.unique(bg.reshape(-1, 3), axis=0)[:6]]
    out["background_n_unique"] = int(len(np.unique(bg.reshape(-1, 3), axis=0))) if len(bg) else 0
    out["background_frac"] = float((seg < 0).mean())
    env.close()
    return out


def track_facts(name, size=128, n=6):
    """How badly the checker floor and its mirror block an analytic caster."""
    env, u = _env(name, size)
    m, d = u.model, u.data
    mr = u.mujoco_renderer
    cam = mr.camera_id if mr.camera_id >= 0 else mr.viewer.cam
    rng = np.random.default_rng(0)
    q0 = np.asarray(u.init_qpos, np.float64)
    Q = np.tile(q0, (n, 1)) + rng.uniform(-0.2, 0.2, (n, m.nq))
    Q[:, 0] += rng.uniform(-2, 2, n)

    def render(q):
        u.set_state(q, np.zeros(m.nv))
        return u.render().copy()

    seg, r = _seg(m, d, cam, size)
    floor = seg == 0                                   # geom 0 is the floor in every one of these
    fr = render(Q[0])
    fl = fr[floor]
    out = {"env": name, "floor_frac": float(floor.mean()),
           "floor_n_unique_colors": int(len(np.unique(fl.reshape(-1, 3), axis=0))),
           "floor_pixels": int(floor.sum())}

    # the mirror: reflectance 0.5 on MatPlane
    matid = int(m.geom_matid[0])
    keep = float(m.mat_reflectance[matid]) if matid >= 0 else 0.0
    if matid >= 0:
        m.mat_reflectance[matid] = 0.0
        norefl = render(Q[0])
        m.mat_reflectance[matid] = keep
        dmax = np.abs(fr.astype(np.int16) - norefl.astype(np.int16)).max(-1)
        out["reflectance"] = keep
        out["mirror_frac_pixels_changed"] = float((dmax > 0).mean())
        out["mirror_mean_abs"] = float(dmax.mean())
        out["mirror_max_abs"] = int(dmax.max())

    # the skybox under a TRANSLATING camera: does the sky half of the frame move at all?
    sky = seg < 0
    frames = np.stack([render(q) for q in Q])
    sk = frames[:, sky]
    out["sky_frac"] = float(sky.mean())
    out["sky_maxdiff_across_states"] = int(
        np.abs(sk.astype(np.int16) - sk[:1].astype(np.int16)).max()) if sky.any() else -1
    fld = frames[:, floor]
    out["floor_maxdiff_across_states"] = int(
        np.abs(fld.astype(np.int16) - fld[:1].astype(np.int16)).max()) if floor.any() else -1
    env.close()
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    out = {"flat": [], "track": []}
    if which in ("both", "flat"):
        for n in FLAT:
            try:
                out["flat"].append(flat_facts(n))
            except Exception as e:                                       # noqa: BLE001
                import traceback
                traceback.print_exc()
                out["flat"].append({"env": n, "error": str(e)})
            print(json.dumps(out["flat"][-1])[:4000], flush=True)
    if which in ("both", "track"):
        for n in TRACK:
            try:
                out["track"].append(track_facts(n))
            except Exception as e:                                       # noqa: BLE001
                import traceback
                traceback.print_exc()
                out["track"].append({"env": n, "error": str(e)})
            print(json.dumps(out["track"][-1]), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_scene.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
