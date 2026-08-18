"""Does dm_control load, what is in each scene, and is the frame a function of qpos?

Three questions, in the order that makes the later ones worth asking:

  1. does ``dm_control.suite.load`` work at all on this pin (the shared conda env fails with
     ``'MjModel' object has no attribute 'flex_bandwidth'``, a dm_control/mujoco version skew);
  2. what does each domain's compiled model actually contain -- geom kinds, meshes, cameras,
     lights, materials, textures -- because a static reading of the XML misses everything the
     compiler adds;
  3. is the rendered frame a pure function of ``qpos``? Measured the way ``docs/research/ogbench.md``
     measured it: render twice, scale ``qvel`` by 100, and reach the same ``qpos`` through a
     different history.

Run under SLURM with MUJOCO_GL=egl.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

DOMAINS = [
    ("cartpole", "swingup"), ("cartpole", "balance"),
    ("pendulum", "swingup"),
    ("acrobot", "swingup"),
    ("point_mass", "easy"),
    ("ball_in_cup", "catch"),
    ("reacher", "easy"),
    ("finger", "spin"),
    ("cheetah", "run"),
    ("walker", "walk"),
    ("hopper", "hop"),
    ("swimmer", "swimmer6"),
    ("fish", "swim"),
    ("humanoid", "stand"),
    ("manipulator", "bring_ball"),
    ("stacker", "stack_2"),
    ("quadruped", "walk"),
]


def inventory(physics) -> dict:
    import mujoco
    m = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    kinds = {}
    for g in range(m.ngeom):
        kinds[int(m.geom_type[g])] = kinds.get(int(m.geom_type[g]), 0) + 1
    cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) or f"cam{c}" for c in range(m.ncam)]
    tex = []
    for t in range(m.ntex):
        tex.append({"type": int(m.tex_type[t]), "w": int(m.tex_width[t]), "h": int(m.tex_height[t]),
                    "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t)})
    mats = []
    for i in range(m.nmat):
        tid = m.mat_texid[i]
        tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
        mats.append({"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i),
                     "rgba": [round(float(x), 4) for x in m.mat_rgba[i]],
                     "texid": tid, "reflectance": float(m.mat_reflectance[i]),
                     "specular": float(m.mat_specular[i]), "shininess": float(m.mat_shininess[i]),
                     "emission": float(m.mat_emission[i])})
    lights = [{"directional": int(m.light_directional[i]) if hasattr(m, "light_directional") else -1,
               "pos": [round(float(x), 4) for x in m.light_pos[i]],
               "dir": [round(float(x), 4) for x in m.light_dir[i]],
               "castshadow": int(m.light_castshadow[i]),
               "ambient": [round(float(x), 3) for x in m.light_ambient[i]],
               "diffuse": [round(float(x), 3) for x in m.light_diffuse[i]],
               "specular": [round(float(x), 3) for x in m.light_specular[i]]}
              for i in range(m.nlight)]
    return {
        "nq": int(m.nq), "nv": int(m.nv), "ngeom": int(m.ngeom), "nmesh": int(m.nmesh),
        "nhfield": int(m.nhfield), "nskin": int(getattr(m, "nskin", 0)), "nsite": int(m.nsite),
        "ntendon": int(m.ntendon), "nflex": int(getattr(m, "nflex", 0)),
        "geom_kinds": {str(k): v for k, v in sorted(kinds.items())},
        "geom_groups": sorted({int(g) for g in m.geom_group}),
        "site_groups": sorted({int(g) for g in m.site_group}),
        "cameras": cams, "ncam": int(m.ncam),
        "textures": tex, "materials": mats, "lights": lights,
        "headlight": {"ambient": [round(float(x), 3) for x in m.vis.headlight.ambient],
                      "diffuse": [round(float(x), 3) for x in m.vis.headlight.diffuse],
                      "specular": [round(float(x), 3) for x in m.vis.headlight.specular],
                      "active": int(m.vis.headlight.active)},
        "offsamples": int(m.vis.quality.offsamples),
        "shadowsize": int(m.vis.quality.shadowsize),
        "fovy": float(m.vis.global_.fovy), "ipd": float(m.vis.global_.ipd),
        "cam_fovy": [round(float(x), 4) for x in m.cam_fovy],
        "jnt_types": [int(x) for x in m.jnt_type],
    }


def state_sufficiency(env, size: int, cam: int, n: int = 6, seed: int = 0) -> dict:
    """Is the frame f(qpos)? Same three adversarial conditions as the OGBench study."""
    p = env.physics
    m = p.model.ptr if hasattr(p.model, "ptr") else p.model
    rng = np.random.default_rng(seed)

    # states the env actually reaches: roll the real task out and record qpos
    ts = env.reset()
    qs = []
    for i in range(n * 12):
        env.step(rng.uniform(env.action_spec().minimum, env.action_spec().maximum))
        if i % 12 == 0:
            qs.append(np.array(p.data.qpos, np.float64))
    qs = np.stack(qs[:n])

    def render(q, qvel_scale=0.0):
        with p.reset_context():
            p.data.qpos[:] = q
            p.data.qvel[:] = qvel_scale * rng.standard_normal(m.nv)
        return p.render(height=size, width=size, camera_id=cam).copy()

    a = np.stack([render(q, 0.0) for q in qs])
    b = np.stack([render(q, 0.0) for q in qs])
    c = np.stack([render(q, 100.0) for q in qs])
    # different history: walk the states in reverse, then re-render forwards
    _ = [render(q, 3.0) for q in qs[::-1]]
    d = np.stack([render(q, 0.0) for q in qs])

    def mx(x, y):
        return int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max())

    # which qpos entries move pixels
    live = 0
    base = render(qs[0], 0.0)
    for j in range(m.nq):
        q = qs[0].copy()
        q[j] += 0.3
        if mx(render(q, 0.0), base) > 0:
            live += 1
    return {"render_twice": mx(a, b), "qvel_x100": mx(a, c), "different_history": mx(a, d),
            "live_qpos_dims": live, "nq": int(m.nq)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--out", default="")
    ap.add_argument("--domains", default="")
    args = ap.parse_args()

    import dm_control
    import mujoco
    from dm_control import suite

    out = {"mujoco": mujoco.__version__,
           "dm_control": getattr(dm_control, "__version__", "?"),
           "domains": {}}
    want = set(args.domains.split(",")) if args.domains else None
    for dom, task in DOMAINS:
        if want and dom not in want:
            continue
        key = f"{dom}-{task}"
        try:
            env = suite.load(dom, task, task_kwargs={"random": 0})
            rec = inventory(env.physics)
            rec["sufficiency"] = state_sufficiency(env, args.size, 0)
            out["domains"][key] = rec
            print(f"{key}: OK nq={rec['nq']} ngeom={rec['ngeom']} nmesh={rec['nmesh']} "
                  f"suff={rec['sufficiency']}", flush=True)
        except Exception as e:                                              # noqa: BLE001
            out["domains"][key] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{key}: FAIL {type(e).__name__}: {e}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps({"mujoco": out["mujoco"], "dm_control": out["dm_control"]}))


if __name__ == "__main__":
    main()
