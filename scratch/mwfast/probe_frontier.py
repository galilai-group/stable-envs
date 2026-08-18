"""The resolution frontier for one Meta-World task, and where the frame's time goes.

Two questions, one process, because the answer to the second decides whether the first is worth
spending a corrector on.

**The frontier.** ``env.render(states, size=r)`` then bilinear to 256, with the upsample INSIDE the
timed region because a consumer asked for 256 px and must be given 256 px. Fidelity is against
``mujoco.Renderer`` at 256 -- the official frame, the same target the shipped parity numbers are
measured against -- in blocks of 64 frames (``common.metrics``).

**The decomposition.** Meta-World's frame is not one kernel. It is a per-episode MuJoCo bake (cached),
one captured-graph MJWarp replay per chunk of 256 worlds, and a torch composite that selects between
the two through the segmentation buffer. Resolution scales the ray-cast and the composite and leaves
the bake alone, so the share those three take at 256 px is what says how much of the frame the lever
can reach. Measured here by replaying the graph on its own, which is the only part of the render that
can be timed without reimplementing the render.

``jpeg_q95`` is the storage baseline: its error is the yardstick a re-declaration is scored against,
and its 8-worker Lance read rate is the throughput bar this whole exercise exists to clear.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import common as C


def throughput(env, states, batch: int, r: int, size: int, reps: int = 7) -> dict:
    """End-to-end f/s at ``batch``, plates warm, upsample inside the timed region."""
    st = torch.as_tensor(states, device="cuda", dtype=torch.float32)
    idx = np.random.default_rng(0).integers(0, len(states), batch)
    st = st[torch.as_tensor(idx, device="cuda")]
    low = torch.empty((batch, r, r, 3), dtype=torch.uint8, device="cuda")

    def step():
        env.render(st, size=r, out=low)
        return C.upsample(low, size)

    step()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    dt = float(np.median(ts))
    del low, st
    torch.cuda.empty_cache()
    return {"fps": batch / dt, "seconds_per_batch": dt, "batch": batch,
            "fps_iqr_frac": float((np.percentile(ts, 75) - np.percentile(ts, 25)) / dt)}


def graph_only(task: str, r: int, nworld: int = 256, reps: int = 20) -> dict:
    """Bare captured-graph replay: the ray-cast alone, with no upload and no composite."""
    M = C.mw()
    c = M._context(task, int(r), int(nworld))
    c.wp.capture_launch(c.graph)
    c.wp.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        c.wp.capture_launch(c.graph)
        c.wp.synchronize()
        ts.append(time.perf_counter() - t0)
    dt = float(np.median(ts))
    return {"graph_ms": 1e3 * dt, "graph_fps": nworld / dt, "nworld": nworld}


def bake_cost(env, states, r: int, reps: int = 3) -> dict:
    """One MuJoCo plate bake at ``r``: the per-EPISODE cost the resolution lever does not remove."""
    M = C.mw()
    row = np.asarray(states[0], np.float32)
    ts = []
    for _ in range(reps):
        M._PLATE_CACHE.clear()
        t0 = time.perf_counter()
        env._bake(int(r), None, row)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    M._PLATE_CACHE.clear()
    return {"bake_ms": 1e3 * float(np.median(ts))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--res", default="256,224,192,176,160,144,128,112,96")
    ap.add_argument("--lightspec-res", default="256,176,160,144,128",
                    help="resolutions also measured with STABLE_ENVS_LIGHTSPEC=1")
    ap.add_argument("--no-jpeg", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import stable_envs as se
    cls = se.REGISTRY[a.env]
    M = C.mw()
    task = cls.TASK

    # The evaluation states are the env's OWN draw -- the one a declaration is measured on.
    probe = se.make(a.env, batch=1, size=a.size, device="cuda", seed=0)
    states = probe.sample_states(a.states)
    del probe
    C.clear_caches()
    official = C.official_frames(cls, states, a.size)
    res = {"env": a.env, "task": task, "size": a.size, "batch": a.batch, "n_states": a.states,
           "gpu": torch.cuda.get_device_name(0), "node": os.environ.get("SLURMD_NODENAME", "?"),
           "rows": []}

    if not a.no_jpeg:
        jo, nbytes = C.jpeg_roundtrip(official, 95)
        row = {"config": "jpeg_q95_of_official", "kind": "storage_baseline",
               "bytes_per_frame": nbytes}
        row.update(C.metrics(jo, official))
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        del jo

    plan = [(int(r), False) for r in a.res.split(",") if r]
    plan += [(int(r), True) for r in a.lightspec_res.split(",") if r]
    for r, lightspec in plan:
        name = f"r{r}" + ("_lightspec" if lightspec else "")
        prev = os.environ.get("STABLE_ENVS_LIGHTSPEC")
        os.environ["STABLE_ENVS_LIGHTSPEC"] = "1" if lightspec else "0"
        C.clear_caches()
        try:
            env = se.make(a.env, batch=1, size=a.size, device="cuda", seed=0)
            low = env.render(states, size=r)
            ours = C.upsample(low, a.size).cpu().numpy()
            del low
            torch.cuda.empty_cache()
            row = {"config": name, "kind": "render", "render_size": r, "lightspec": lightspec}
            row.update(C.metrics(ours, official))
            del ours
            torch.cuda.empty_cache()
            row.update(throughput(env, states, a.batch, r, a.size))
            row.update(graph_only(task, r))
            row.update(bake_cost(env, states, r))
            del env
        except Exception as e:                                            # noqa: BLE001
            row = {"config": name, "error": f"{type(e).__name__}: {e}"}
        finally:
            if prev is None:
                os.environ.pop("STABLE_ENVS_LIGHTSPEC", None)
            else:
                os.environ["STABLE_ENVS_LIGHTSPEC"] = prev
            C.clear_caches()
        res["rows"].append(row)
        print(json.dumps(row), flush=True)

    p = a.out or os.path.join(C.LOGS, f"frontier_{a.env}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
