"""How much of a Meta-World frame does the ray-cast actually decide?

The composite takes MJWarp's frame in exactly two places: where the segmentation says MOVER, and where
MJWarp's movers-on frame differs from its movers-off frame (a mover's shadow). Everywhere else the
output is the MuJoCo bake, bit for bit, whatever the ray-cast produced. So rays outside the union of
those two sets are pure waste, and this measures how big that union's bounding box is over the states
the env actually reaches -- per state, and unioned over a chunk, which is the number that decides
whether a windowed launch can pay.

The masks are taken from inside the env's own render by wrapping ``_movers_off``, so they are the
masks the composite really uses rather than a reconstruction of them.
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

CAP = {}


def bbox(mask: torch.Tensor):
    rows = mask.any(1).nonzero()
    cols = mask.any(0).nonzero()
    if not len(rows) or not len(cols):
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="MetaWorldReach,MetaWorldAssembly,MetaWorldSoccer,"
                                      "MetaWorldWindowOpen,MetaWorldPickPlace,MetaWorldHammer")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()

    from stable_envs.envs import metaworld as MWE

    orig = MWE.MetaWorldBase._movers_off

    def spy(self, c, n, plates, g):
        # BEFORE, not after: ``_movers_off`` replays the graph with the movers removed and leaves
        # ``c.bgra``/``c.seg`` holding THAT frame, so a snapshot taken afterwards reports rgb == off
        # and an empty mover mask -- which is exactly the trap the render's own comment warns about.
        CAP["rgb"] = c.bgra[:n][..., [2, 1, 0]].clone()
        CAP["seg"] = c.seg[:n].clone()
        CAP["off"] = orig(self, c, n, plates, g)
        return CAP["off"]
    MWE.MetaWorldBase._movers_off = spy

    out = {}
    for name in a.envs.split(","):
        try:
            env = se.make(name, batch=a.n, size=a.size, device="cuda")
            st = env.sample_states(a.n)
            env.render(st)
            rgb, off, seg = CAP["rgb"], CAP["off"], CAP["seg"]
            movers = torch.as_tensor(env._b.movers, device=seg.device, dtype=seg.dtype)
            mv = torch.isin(seg, movers)
            shad = (rgb != off).any(-1)
            used = mv | shad
            print(f"  [{name}] nmovers={len(env._b.movers)} casting={len(env._b.casting)} "
                  f"seg[{seg.dtype}] uniq={seg.unique()[:12].tolist()} "
                  f"mv={int(mv.sum())} shadow={int(shad.sum())} used={int(used.sum())} "
                  f"shapes rgb{tuple(rgb.shape)} off{tuple(off.shape)} seg{tuple(seg.shape)}",
                  flush=True)
            k = used.shape[0]
            fr = used.reshape(k, -1).float().mean(1)
            boxes = [bbox(used[i]) for i in range(k)]
            areas = [((b[1] - b[0]) * (b[3] - b[2])) / (a.size * a.size) for b in boxes if b] or [0.0]
            ub = bbox(used.any(0))
            uarea = ((ub[1] - ub[0]) * (ub[3] - ub[2])) / (a.size * a.size) if ub else 0.0
            out[name] = {"n": k, "mask_frac_mean": float(fr.mean()), "mask_frac_max": float(fr.max()),
                         "bbox_frac_mean": float(np.mean(areas)), "bbox_frac_max": float(np.max(areas)),
                         "union_bbox_frac": float(uarea), "union_bbox": ub}
            print(f"{name:28s} mask {fr.mean():.3f}  bbox/state {np.mean(areas):.3f} "
                  f"(max {np.max(areas):.3f})  union-bbox {uarea:.3f} {ub}", flush=True)
            del env
            MWE._CTX_CACHE.clear()
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
