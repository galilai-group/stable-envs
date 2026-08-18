"""Sweep the shadow bias against the OFFICIAL frame. One process per env, one env per process.

The first shadow pass used a constant slope-scaled depth bias and put 80 % of the pixels it changed
on the body rather than on the floor, which is what self-shadow acne looks like from the outside. The
fix is a normal offset, and how far to push is a measurement rather than a taste: too little leaves
the acne, too much peels the shadow away from the thing casting it. Nothing is compared against
``mjRND_SHADOW = 0`` here, because that flag does not mean "the same frame without its shadows" —
on Swimmer it changes 54 % of the frame by a mean of 45/255 — so the only honest reference is the
official one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdloco"


def emit(tag, obj):
    print(f"@@{tag} " + json.dumps(obj, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="MujocoHopper")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--states", type=int, default=32)
    ap.add_argument("--offsets", default="0,1,2,3,5")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    from stable_envs.envs import mujoco_locomotion as M
    from stable_envs.testing import conformance as C

    env_id = M.VISUAL_ENVS[args.cls]

    def build(shadows):
        cls = type(args.cls, (M.GymMujocoVisual,), dict(
            ENV_ID=env_id, name=M._hyphen_name(env_id), reference_id=f"gymnasium/{env_id}",
            SHADOWS=bool(shadows), __doc__="VISUAL TIER shadow bias sweep"))
        M._NVD_CACHE.clear()
        return cls, cls(batch=8, size=args.size, device="cuda")

    cls, env = build(True)
    states = env.sample_states(int(args.states))
    ref = C.reference_env(cls, env.size)
    official = cls.reference_render(ref, states)

    def score(ours):
        r = C.pixel_agreement(ours, official)
        r["ssim"] = C.ssim(ours, official)
        return {k: r[k] for k in ("mean_abs", "exact_frac", "within8_frac", "ssim",
                                  "interior_exact_frac", "interior_mean_abs")}

    cls0, env0 = build(False)
    emit("bias.none", score(env0.render(states).cpu().numpy()))
    del env0

    for k in [float(x) for x in args.offsets.split(",")]:
        M.NVD_SHADOW_OFFSET_TEXELS = k
        _, e = build(True)
        emit(f"bias.n{k}", score(e.render(states).cpu().numpy()))
        del e


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
