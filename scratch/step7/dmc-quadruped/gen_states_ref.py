"""Phase 0, step 1: 64 states from quadruped-walk's own visited distribution, and the official frames.

States: ``initialize_episode`` plus a 0-40 step random-action rollout (the family's ``sample_states``
protocol, seed 100 as ``visual_calib.py``'s first draw). Official frames: ``physics.render`` under the
interpreter this runs in (mujoco 3.12.0 here; ``ref_render.py`` re-renders the SAME qpos under the
declared 3.11.0 reference), at the SWM camera (2, ``y``) and the family's declared 84 px, plus 224 px
(SWM's ``render()`` default) and camera 0 for the record. The segmentation buffer (objtype, objid) is
saved per frame so residuals can be localised per surface.

Usage: MUJOCO_GL=glfw python gen_states_ref.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")


def rollout_states(n: int, seed: int) -> np.ndarray:
    from dm_control import suite
    env = suite.load("quadruped", "walk", task_kwargs={"random": seed})
    p = env.physics
    spec = env.action_spec()
    r2 = np.random.default_rng(seed)
    out = np.empty((n, p.model.nq), np.float64)
    for i in range(n):
        env.reset()
        for _ in range(int(r2.integers(0, 40))):
            env.step(r2.uniform(spec.minimum, spec.maximum))
        out[i] = np.asarray(p.data.qpos, np.float64)
    return out


def render_official(qpos: np.ndarray, size: int, cam: int, seg: bool = True):
    """``physics.render`` at each qpos with qvel = 0, round trip asserted (as ``reference_render``)."""
    import mujoco
    from dm_control import suite
    env = suite.load("quadruped", "walk", task_kwargs={"random": 0})
    env.reset()
    p = env.physics
    m = p.model.ptr
    frames, segs = [], []
    rs = None
    if seg:
        rs = mujoco.Renderer(m, height=size, width=size)
        rs.enable_segmentation_rendering()
    for q in qpos:
        with p.reset_context():
            p.data.qpos[:] = q
            p.data.qvel[:] = 0
        assert np.abs(np.asarray(p.data.qpos) - q).max() <= 1e-9
        frames.append(p.render(height=size, width=size, camera_id=cam).copy())
        if rs is not None:
            rs.update_scene(p.data.ptr, camera=cam)
            segs.append(rs.render().copy())
    return np.stack(frames), (np.stack(segs) if seg else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--states", default="", help="reuse these qpos (npz with 'qpos') instead of sampling")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-seg", action="store_true")
    a = ap.parse_args()
    import mujoco
    import dm_control
    import importlib.metadata as md
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"mj{mujoco.__version__.replace('.', '')}"
    if a.states:
        qpos = np.load(a.states)["qpos"]
    else:
        qpos = rollout_states(a.n, a.seed)
        np.savez_compressed(os.path.join(a.out, "states.npz"), qpos=qpos)
    prov = {"mujoco": mujoco.__version__, "dm_control": md.version("dm_control"), "n": int(len(qpos)),
            "seed": a.seed, "MUJOCO_GL": os.environ.get("MUJOCO_GL")}
    for cam in (2, 0):
        for size in (84, 224):
            fr, sg = render_official(qpos, size, cam, seg=not a.no_seg)
            name = f"official_{tag}_cam{cam}_{size}px.npz"
            np.savez_compressed(os.path.join(a.out, name), frames=fr, **({"segs": sg} if sg is not None else {}))
            print("wrote", name, fr.shape, flush=True)
    json.dump(prov, open(os.path.join(a.out, f"provenance_{tag}.json"), "w"), indent=1)
    print(json.dumps(prov))


if __name__ == "__main__":
    main()
