"""Per-task calibration for ``stable_envs.envs.metaworld``: parity, throughput, and the state slice.

ONE TASK PER PROCESS, and that is not a style choice. A CUDA 700 out of ``mujoco_warp`` poisons the
context for everything after it — ``probe_backend2.py`` measured one real fault followed by seven
fabricated "Failed to allocate 4 bytes" — so a loop over 50 tasks in one interpreter reports a real
failure once and then invents the rest. Two live MuJoCo envs in one process is the second reason
(``docs/research/adroit.md`` §2.1).

    python scratch/metaworld/calibrate.py reach-v3 [--size 480] [--states 64] [--dims]

Writes ``$OUT/cal_<task>.json`` and prints one ``RESULT {...}`` line, so a driver can collect the
table by grepping the array job's logs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OUT = "/mnt/vast/home/randall.balestriero/se_logs/metaworld"


def run(task: str, size: int, n_states: int, batches, dims: bool) -> dict:
    import torch

    import stable_envs as se
    from stable_envs.envs.metaworld import _build_bundle, _class_name
    from stable_envs.testing import conformance as C

    cls = se.REGISTRY[_class_name(task)]
    rec: dict = {"task": task, "cls": cls.__name__, "size": int(size)}

    t0 = time.perf_counter()
    env = cls(batch=8, size=int(size), device="cuda")
    rec["build_s"] = round(time.perf_counter() - t0, 2)
    b = _build_bundle(task)
    rec |= {"state_dim": int(env.state_dim), "nq": int(b.nq),
            "slots": [f"{a}[{f}]" for a, f in b.slots],
            "slot_names": list(env.state_names[b.nq:]),
            "n_movers": int(len(b.movers)), "n_static": int(len(b.static_geoms)),
            "n_mover_sites": int(len(b.mover_sites)),
            "knobs": {k: [int(len(v["geoms"])), int(len(v["mats"]))] for k, v in b.knobs.items()},
            "style_names": env.style_names}

    # ---- parity against the official renderer ---------------------------------------------------
    states = env.sample_states(int(n_states), rng=np.random.default_rng(0))
    rec["distinct_episodes"] = int(len(np.unique(states[:, b.nq:], axis=0)))
    ref = C.reference_env(cls, int(size))
    t0 = time.perf_counter()
    official = cls.reference_render(ref, states)
    rec["mujoco_state_and_render_fps"] = round(len(states) / (time.perf_counter() - t0), 1)
    ours = env.render(states).cpu().numpy()
    rec["parity"] = {k: (round(v, 6) if isinstance(v, float) else v)
                     for k, v in C.pixel_agreement(ours, official).items()}

    # The official renderer alone, with no state restore, so the speedup is not flattered by MuJoCo's
    # own mj_forward. Both numbers are reported for the same reason fetch.py reports both.
    import mujoco
    u = ref.unwrapped
    t0 = time.perf_counter()
    for i in range(20):
        u.render()
    rec["mujoco_render_only_fps"] = round(20 / (time.perf_counter() - t0), 1)

    # ---- throughput, per batch size, with the CUDA graph replayed -------------------------------
    fps = {}
    for bs in batches:
        st = env.sample_states(int(bs), rng=np.random.default_rng(1))
        out = torch.empty((int(bs), int(size), int(size), 3), dtype=torch.uint8, device="cuda")
        env.render(st, out=out)                                   # warm: bakes plates, builds context
        torch.cuda.synchronize()
        reps = max(2, min(20, int(2000 / max(bs, 1))))
        t0 = time.perf_counter()
        for _ in range(reps):
            env.render(st, out=out)
        torch.cuda.synchronize()
        fps[str(bs)] = round(bs * reps / (time.perf_counter() - t0), 1)
    rec["fps"] = fps
    rec["speedup_vs_render_only"] = {k: round(v / rec["mujoco_render_only_fps"], 2)
                                     for k, v in fps.items()}

    # ---- the checks the conformance suite will run, cheaply, here -------------------------------
    rec["determinism"] = C.check_determinism(env)
    rec["batch_invariance"] = C.check_batch_invariance(env)
    rec["variations"] = C.check_variations_do_something(env)
    rec["per_env_variation"] = C.check_per_env_variation(env)
    rec["default_style_is_official"] = C.check_default_style_is_official(env)
    if dims:
        t0 = time.perf_counter()
        rec["dead_dims"] = C.check_state_dims_matter(env)["dead_dims"]
        rec["dims_s"] = round(time.perf_counter() - t0, 1)
    fa = cls.reference_render(ref, states[:8])
    fb = cls.reference_render(ref, states[:8][::-1])[::-1]
    rec["state_is_sufficient_max"] = int(np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max())
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 256])
    ap.add_argument("--dims", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    try:
        r = run(a.task, a.size, a.states, a.batches, a.dims)
    except Exception as e:                                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        r = {"task": a.task, "error": f"{type(e).__name__}: {e}"}
    print("RESULT " + json.dumps(r), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"cal_{a.task}.json"), "w") as f:
        json.dump(r, f, indent=1)
