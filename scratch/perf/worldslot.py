"""Does the SAME state render the same picture in every world slot of one launch?

``docs/research/humanoidbench.md`` §5.2 records 2-3 of ~660 tests failing per suite run, always
``test_determinism`` or ``test_frame_i_is_independent_of_batch_size``, by 1-2/255, on a different cell
each time -- and gates the whole family behind an environment variable because of it. Both of those
tests compare a state rendered in one launch against the same state rendered in another, and the only
thing that differs between the two is WHERE IN THE LAUNCH it sat.

``mujoco_warp`` builds ONE scene BVH over ``nworld * ngeom`` primitives with a per-world root, so a
world's subtree is its own geometry but its internal SHAPE is decided by a build over every world at
once, and closest-hit is resolved by strict ``<`` on the ray parameter as nodes are visited. Two
exactly coincident surfaces -- a foot on the floor, a ramp laid on the floor, a stair tread -- are a
tie, and a tie is broken by visit order.

So: fill every world with the identical ``qpos``, render once, and compare every world against world
0. Any disagreement is that, and nothing else: same scene, same state, same kernel, same launch.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se


def module_for(name):
    import importlib
    for f, m in (("MetaWorld", "metaworld"), ("HumanoidBench", "humanoidbench"),
                 ("Fetch", "fetch"), ("RoboCasa", "robocasa")):
        if name.startswith(f):
            return importlib.import_module("stable_envs.envs." + m)
    raise SystemExit(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--states", type=int, default=6)
    a = ap.parse_args()
    out = {}
    for name in a.envs.split(","):
        try:
            mod = module_for(name)
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()
            env = se.make(name, batch=a.nworld, size=a.size, device="cuda")
            pool = env.sample_states(a.states, rng=np.random.default_rng(0))
            worst = {"npix": 0, "max": 0, "state": -1, "slots": []}
            for si, st in enumerate(pool):
                # THE SAME STATE IN EVERY WORLD. Through the env's own render, so the pre-pass, the
                # appearance and the padding are the shipped ones.
                s = np.repeat(st[None], a.nworld, 0)
                f = env.render(s)
                d = (f.to(torch.int16) - f[0:1].to(torch.int16)).abs().amax(-1)
                npix = int((d > 0).sum())
                if npix:
                    slots = [int(i) for i in (d.reshape(a.nworld, -1).amax(1) > 0).nonzero().ravel()]
                    if npix > worst["npix"]:
                        worst = {"npix": npix, "max": int(d.max()), "state": si,
                                 "slots": slots[:12], "n_slots": len(slots)}
            out[name] = worst
            print(f"{name:32s} worst over {a.states} states: {worst['npix']} px differ from world 0, "
                  f"max {worst['max']}/255, in {worst.get('n_slots', 0)}/{a.nworld} slots "
                  f"{worst['slots']}", flush=True)
            del env
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
