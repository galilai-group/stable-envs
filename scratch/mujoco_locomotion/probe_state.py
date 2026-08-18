"""Is the Gymnasium MuJoCo locomotion frame a function of qpos alone, and what is in the scene?

Answers, per env, by measurement:
  * model facts: nq/nv/ngeom/nmesh/ntex, geom type histogram, which geoms are textured
  * which camera the OFFICIAL renderer uses, and what its parameters end up being
  * state sufficiency: re-render / qvel x100 / different history
  * per-qpos-dim liveness
  * is there a STATIC PLATE: hide the movers, render at N different qpos, diff
  * texture coverage of the frame, from MuJoCo's own segmentation buffer
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import gymnasium as gym
import mujoco

ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5", "Swimmer-v5",
        "Reacher-v5", "Pusher-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]

GEOM_NAMES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder",
              6: "box", 7: "mesh", 8: "sdf"}


def maxdiff(a, b):
    return int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())


def sample_qpos(u, rng, n):
    """States around the env's own reset distribution, widened so the body actually moves."""
    q0 = np.asarray(u.init_qpos, np.float64)
    out = np.tile(q0, (n, 1))
    out += rng.uniform(-0.4, 0.4, out.shape)
    # move the root along x so a tracking camera has to track something
    if u.model.nq >= 2:
        out[:, 0] += rng.uniform(-2.0, 2.0, n)
    return out


def probe(name, size=128, n=8):
    rng = np.random.default_rng(0)
    env = gym.make(name, render_mode="rgb_array", width=size, height=size)
    u = env.unwrapped
    env.reset(seed=0)
    m, d = u.model, u.data
    f0 = u.render()                                   # freezes the free camera, if any
    mr = u.mujoco_renderer
    cam = mr.viewer.cam
    rec = {
        "env": name, "nq": int(m.nq), "nv": int(m.nv), "ngeom": int(m.ngeom),
        "nmesh": int(m.nmesh), "ntex": int(m.ntex), "nmat": int(m.nmat), "ncam": int(m.ncam),
        "nlight": int(m.nlight), "nsite": int(m.nsite), "ntendon": int(m.ntendon),
        "default_size": [int(u.width), int(u.height)],
        "camera_id": int(mr.camera_id),
        "cam_type": int(cam.type), "cam_fixedcamid": int(cam.fixedcamid),
        "cam_lookat": [float(x) for x in cam.lookat], "cam_distance": float(cam.distance),
        "cam_azimuth": float(cam.azimuth), "cam_elevation": float(cam.elevation),
        "stat_extent": float(m.stat.extent),
        "geom_types": {}, "textured_geoms": [], "joint_names": [], "reflectance": [],
        "skybox": False, "fog": bool(m.vis.map.fogend != m.vis.map.fogstart),
    }
    for g in range(m.ngeom):
        t = GEOM_NAMES.get(int(m.geom_type[g]), str(int(m.geom_type[g])))
        rec["geom_types"][t] = rec["geom_types"].get(t, 0) + 1
        mid = int(m.geom_matid[g])
        if mid >= 0:
            tid = m.mat_texid[mid]
            tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
            if tid >= 0:
                rec["textured_geoms"].append(
                    [g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}", t])
            if float(m.mat_reflectance[mid]) > 0:
                rec["reflectance"].append([g, float(m.mat_reflectance[mid])])
    for t in range(m.ntex):
        if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            rec["skybox"] = True
    for j in range(m.njnt):
        rec["joint_names"].append([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}",
                                   int(m.jnt_type[j])])

    Q = sample_qpos(u, rng, n)
    nv = m.nv

    def render_at(q, qv=None):
        u.set_state(np.asarray(q, np.float64), np.zeros(nv) if qv is None else np.asarray(qv))
        return u.render().copy()

    fwd = np.stack([render_at(q) for q in Q])
    again = np.stack([render_at(q) for q in Q])
    rec["rerender_maxdiff"] = maxdiff(fwd, again)

    rng2 = np.random.default_rng(1)
    fastv = np.stack([render_at(q, rng2.uniform(-100, 100, nv)) for q in Q])
    rec["qvel_x100_maxdiff"] = maxdiff(fwd, fastv)

    rev = np.stack([render_at(q) for q in Q[::-1]])[::-1]
    rec["history_maxdiff"] = maxdiff(fwd, rev)

    # per-dim liveness: substitute dim i from another sampled state
    dead = []
    base = render_at(Q[0])
    for i in range(m.nq):
        q = Q[0].copy()
        q[i] = Q[1][i] if Q[1][i] != Q[0][i] else Q[2][i]
        if maxdiff(base, render_at(q)) == 0:
            dead.append(i)
    rec["dead_qpos_dims"] = dead

    # STATIC PLATE: hide every geom that moves (i.e. is not a child of the world body), render N states
    movers = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) != 0]
    rec["n_mover_geoms"] = len(movers)
    rec["n_static_geoms"] = int(m.ngeom) - len(movers)
    keep_g = np.array(m.geom_rgba, copy=True)
    keep_s = np.array(m.site_rgba, copy=True)
    m.geom_rgba[movers, 3] = 0.0
    m.site_rgba[:, 3] = 0.0
    plates = np.stack([render_at(q) for q in Q])
    m.geom_rgba[:] = keep_g
    m.site_rgba[:] = keep_s
    rec["plate_maxdiff_across_states"] = maxdiff(plates[:1].repeat(len(plates), 0), plates)
    rec["plate_frac_pixels_differing"] = float(
        (np.abs(plates.astype(np.int16) - plates[:1].astype(np.int16)).max(-1) > 0).mean())

    # how much of the frame do the MOVERS cover, and how much is textured/floor/sky
    seg = _segmentation(u, mr, Q[0], size)
    tex_ids = {g for g, _, _ in rec["textured_geoms"]}
    mover_ids = set(movers)
    tot = seg.size
    rec["frac_sky"] = float((seg < 0).mean())
    rec["frac_movers"] = float(np.isin(seg, list(mover_ids)).mean()) if mover_ids else 0.0
    rec["frac_textured"] = float(np.isin(seg, list(tex_ids)).mean()) if tex_ids else 0.0
    rec["frac_static_geoms"] = float(((seg >= 0) & ~np.isin(seg, list(mover_ids))).mean())
    assert tot
    env.close()
    return rec


def _segmentation(u, mr, q, size):
    """geom id per pixel (-1 = background), through MuJoCo's own segmentation buffer.

    Own ``mujoco.Renderer`` rather than gymnasium's ``segmentation=True`` path: that one does
    ``rgb_img[:, :, 1] * (2**8)`` on a uint8 array, which NumPy 2 refuses outright.
    """
    u.set_state(np.asarray(q, np.float64), np.zeros(u.model.nv))
    r = mujoco.Renderer(u.model, height=size, width=size)
    r.enable_segmentation_rendering()
    cam = mr.camera_id if mr.camera_id >= 0 else mr.viewer.cam
    r.update_scene(u.data, camera=cam)
    seg = r.render()                                   # (H,W,2): objid, objtype
    objid, objtype = seg[..., 0].astype(np.int64), seg[..., 1].astype(np.int64)
    return np.where(objtype == int(mujoco.mjtObj.mjOBJ_GEOM), objid, -1)


if __name__ == "__main__":
    names = sys.argv[1:] or ENVS
    out = []
    for n in names:
        try:
            r = probe(n)
        except Exception as e:                                              # noqa: BLE001
            import traceback; traceback.print_exc()
            r = {"env": n, "error": f"{type(e).__name__}: {e}"}
        out.append(r)
        print(json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_state.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
