"""Does a shadow map pay, and at what resolution? The ablation the shadow pass is declared from.

``docs/research/nvdiffrast.md`` Section 4.2 names shadows as the largest remaining term after the
minified floor, from the other side: turning them off on the REFERENCE moves HalfCheetah's near floor
band from 0.427 to 0.753 exact. That measures what MuJoCo's shadow costs a renderer that has none; it
does not measure what OURS buys, because "the reference with shadows disabled" is not "the reference
minus its shadows" — on Swimmer, ``mjRND_SHADOW = 0`` changes 54 % of the frame by a mean of 45/255
and moves the reference AWAY from us.

So this measures the only thing that settles it: the env, against the official frame, with the shadow
pass on and off, at two shadow-map resolutions, plus what it costs in frames per second.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdloco"


def emit(tag, obj):
    print(f"@@{tag} " + json.dumps(obj, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="MujocoHalfCheetah")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--states", type=int, default=64)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch

    from stable_envs.envs import mujoco_locomotion as M
    from stable_envs.testing import conformance as C

    env_id = M.VISUAL_ENVS[args.cls]
    out = {}
    ref = None
    states = None
    for tag, shadows, res in (("off", False, 0), ("mip256", True, 256), ("mip512", True, 512)):
        M.NVD_SHADOW_RES = int(res) if res else M.NVD_SHADOW_RES
        cls = type(args.cls, (M.GymMujocoVisual,), dict(
            ENV_ID=env_id, name=M._hyphen_name(env_id), reference_id=f"gymnasium/{env_id}",
            SHADOWS=bool(shadows), __doc__="VISUAL TIER shadow ablation"))
        M._NVD_CACHE.clear()
        env = cls(batch=8, size=args.size, device="cuda")
        if states is None:
            states = env.sample_states(int(args.states))
            ref = C.reference_env(cls, env.size)
        official = cls.reference_render(ref, states)
        ours = env.render(states).cpu().numpy()
        r = C.pixel_agreement(ours, official)
        r["ssim"] = C.ssim(ours, official)
        x = np.tile(states, (8, 1))[:256]
        env.render(x[:8])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            env.render(x)
        torch.cuda.synchronize()
        r["fps_b256"] = 256 / ((time.perf_counter() - t0) / 5)
        out[tag] = r
        emit(f"shadow.{tag}", r)
        if tag == "mip256":
            np.save(f"{OUT}/shadow_{args.cls}_ours.npy", ours[:3])
            np.save(f"{OUT}/shadow_{args.cls}_off.npy", official[:3])
        del env

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/shadow_{args.cls}.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
