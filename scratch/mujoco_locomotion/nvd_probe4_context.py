"""Does an nvdiffrast context break a live EGL viewer? One process, one question at a time.

Found the hard way: a probe that built three visual-tier envs in one process got a BLACK official
frame from the second one on, while its own frames stayed right — i.e. the reference renderer, not
the renderer under test, was the thing that broke. This walks the interleaving one step at a time and
prints the official frame's mean after each step, so the step that breaks it is named rather than
guessed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")


def emit(tag, obj):
    print(f"@@{tag} " + json.dumps(obj, default=float), flush=True)


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium
    import mujoco
    import torch

    import nvdiffrast.torch as dr

    S = 240
    env = gymnasium.make("Hopper-v5", render_mode="rgb_array", width=S, height=S)
    u = env.unwrapped
    u.model.vis.global_.offwidth = max(int(u.model.vis.global_.offwidth), S)
    u.model.vis.global_.offheight = max(int(u.model.vis.global_.offheight), S)
    env.reset(seed=0)
    u.render()

    def shot(tag):
        u.mujoco_renderer.viewer.make_context_current()
        f = u.render()
        emit(tag, {"mean": float(np.asarray(f).mean()), "max": int(np.asarray(f).max())})
        return np.asarray(f).copy()

    base = shot("0.fresh")
    ctxs = []
    for k in range(3):
        ctxs.append(dr.RasterizeCudaContext(device="cuda"))
        got = shot(f"1.after_ctx{k + 1}")
        emit(f"1.diff{k + 1}", {"max_abs": int(np.abs(got.astype(int) - base.astype(int)).max())})

    # a rasterization through one of them, not merely its construction
    pos = torch.tensor([[[-0.8, -0.8, 0.0, 1.0], [0.8, -0.8, 0.0, 1.0], [0.0, 0.8, 0.0, 1.0]]],
                       device="cuda")
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
    dr.rasterize(ctxs[0], pos, tri, resolution=[64, 64])
    got = shot("2.after_rasterize")
    emit("2.diff", {"max_abs": int(np.abs(got.astype(int) - base.astype(int)).max())})

    del ctxs[0]
    import gc
    gc.collect()
    got = shot("3.after_del_ctx")
    emit("3.diff", {"max_abs": int(np.abs(got.astype(int) - base.astype(int)).max())})

    # and a full env-shaped build: another gymnasium env of the same id, as the suite does
    env2 = gymnasium.make("Hopper-v5", render_mode="rgb_array", width=S, height=S)
    env2.reset(seed=0)
    env2.unwrapped.render()
    got = shot("4.after_second_gym_env")
    emit("4.diff", {"max_abs": int(np.abs(got.astype(int) - base.astype(int)).max())})

    # a mujoco.Renderer, which is what a plain reference build makes
    ren = mujoco.Renderer(u.model, height=S, width=S)
    d = mujoco.MjData(u.model)
    mujoco.mj_forward(u.model, d)
    ren.update_scene(d, camera=mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_CAMERA, "track"))
    ren.render()
    got = shot("5.after_mujoco_Renderer")
    emit("5.diff", {"max_abs": int(np.abs(got.astype(int) - base.astype(int)).max())})


if __name__ == "__main__":
    main()
