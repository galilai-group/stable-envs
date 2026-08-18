"""The two window tasks, which the resolution lever alone cannot rescue, and the one lever that can.

``window-open-v3`` and ``window-close-v3`` render at 3,17x f/s against a ~15,600 f/s stored-frame bar
-- 4.9x, the widest gap in the family -- and the frontier says why they are different: at 256 px the
captured graph is 7,348 f/s but the env delivers 3,162, i.e. **the frame pays for TWO graph replays**.
Meta-World's ``_blend`` renders the scene a second time with the translucent geoms' BVH slots
re-pointed at a static geom, because ``window_glass`` (``rgba="0 .3 .4 .1"``) has to be blended
against what is behind it and what is behind it is another MOVER, which the plate does not contain.
It is worth mean 2.33 -> 1.11/255 on the parity path and it is 100 % of the reason these two tasks
are the family's slowest.

So the fast path's question here is not resolution, it is: **what does the frame cost, and what does
it look like, if the second layer is not rendered at all?** That is one line -- ``_blend`` returns its
input, which is exactly what ``mujoco_warp``'s shader does on its own (it reads ``color[0..2]`` and
packs a constant 255, so the pane is painted opaque) -- and it turns a defect that is invisible to a
throughput knob into a LOCAL, STRUCTURED, always-in-the-same-place defect, which is the kind a
corrector can be asked to fix.

Nothing here is written back into the env: the patch is applied to the class object inside this
process and removed afterwards. It is a prototype measurement, and it is the measurement that says
whether an opt-in fast path is reachable for these two tasks at all.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import common as C
from probe_frontier import bake_cost, graph_only, throughput


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--res", default="256,128,112,96,80")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lightspec", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import stable_envs as se
    cls = se.REGISTRY[a.env]
    M = C.mw()

    probe = se.make(a.env, batch=1, size=a.size, device="cuda", seed=0)
    states = probe.sample_states(a.states)
    del probe
    C.clear_caches()
    official = C.official_frames(cls, states, a.size)
    res = {"env": a.env, "task": cls.TASK, "batch": a.batch, "size": a.size,
           "gpu": torch.cuda.get_device_name(0), "node": os.environ.get("SLURMD_NODENAME", "?"),
           "rows": []}

    orig_blend = M.MetaWorldBase._blend
    for blend in (True, False):
        M.MetaWorldBase._blend = (orig_blend if blend else
                                  (lambda self, c, rgb, plate, n, alpha: rgb))
        for r in [int(x) for x in a.res.split(",") if x]:
            for ls in ({False, a.lightspec} if True else {False}):
                name = f"r{r}" + ("" if blend else "_noblend") + ("_lightspec" if ls else "")
                prev = os.environ.get("STABLE_ENVS_LIGHTSPEC")
                os.environ["STABLE_ENVS_LIGHTSPEC"] = "1" if ls else "0"
                C.clear_caches()
                try:
                    env = se.make(a.env, batch=1, size=a.size, device="cuda", seed=0)
                    ours = C.upsample(env.render(states, size=r), a.size).cpu().numpy()
                    row = {"config": name, "render_size": r, "blend": blend, "lightspec": ls}
                    row.update(C.metrics(ours, official))
                    del ours
                    torch.cuda.empty_cache()
                    row.update(throughput(env, states, a.batch, r, a.size))
                    row.update(graph_only(cls.TASK, r))
                    row.update(bake_cost(env, states, r))
                    del env
                except Exception as e:                                    # noqa: BLE001
                    row = {"config": name, "error": f"{type(e).__name__}: {e}"}
                finally:
                    if prev is None:
                        os.environ.pop("STABLE_ENVS_LIGHTSPEC", None)
                    else:
                        os.environ["STABLE_ENVS_LIGHTSPEC"] = prev
                    C.clear_caches()
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
    M.MetaWorldBase._blend = orig_blend

    p = a.out or os.path.join(C.LOGS, f"window_{a.env}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
