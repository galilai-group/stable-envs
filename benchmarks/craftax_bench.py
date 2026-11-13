"""Time Craftax steps: the upstream jitted vmap against stable-envs, on the device JAX and torch both use.

Arms per class and batch size:
  upstream            ``jit(vmap(env.step))`` with auto-reset and random actions; its observation is the class's own
                      (pixels for *Pixels, the symbolic vector for *Symbolic).
  snv-step            stable-envs ``env.step`` without frames.
  snv-step-render     stable-envs ``env.step`` with frames (the pixel renderer, also for *Symbolic).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks.offline_sweep import _host

N_ACTIONS = {"CraftaxClassicPixels": 17, "CraftaxClassicSymbolic": 17, "CraftaxPixels": 43, "CraftaxSymbolic": 43}


def _median_rate(fn, steps_per_call, repeats, calls):
    fn()
    rates = []
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(calls):
            fn()
        rates.append(steps_per_call * calls / (time.perf_counter() - t))
    return {"steps_per_s": float(np.median(rates)), "samples": rates}


def upstream(name, batch, repeats, calls):
    import jax
    from craftax.craftax_env import make_craftax_env_from_name

    import stable_envs as snv

    env = make_craftax_env_from_name(snv.info(name)["reference_id"], auto_reset=True)
    params = env.default_params
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
    box = {"state": state, "key": key}

    def call():
        box["key"], k1, k2 = jax.random.split(box["key"], 3)
        actions = jax.random.randint(k1, (batch,), 0, N_ACTIONS[name])
        o, box["state"], r, d, _ = step(jax.random.split(k2, batch), box["state"], actions, params)
        jax.block_until_ready(o)

    return _median_rate(call, batch, repeats, calls)


def ours(name, batch, device, frames, repeats, calls):
    import torch

    import stable_envs as snv

    env = snv.make(name, num_envs=batch, device=device)
    env.reset(seed=0)
    g = torch.Generator().manual_seed(0)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    def call():
        a = torch.randint(0, N_ACTIONS[name], (batch, 1, 1), generator=g).to(device)
        env.step(a, states=None, frames=slice(None) if frames else None)
        sync()

    try:
        return _median_rate(call, batch, repeats, calls)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", default=list(N_ACTIONS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batches", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    res = {"host": _host(), "device": args.device, "envs": {}}
    for name in args.envs:
        rows = []
        for b in args.batches:
            for arm, fn in (
                ("upstream", lambda: upstream(name, b, args.repeats, args.calls)),
                ("snv-step", lambda: ours(name, b, args.device, False, args.repeats, args.calls)),
                ("snv-step-render", lambda: ours(name, b, args.device, True, args.repeats, args.calls)),
            ):
                try:
                    r = {"arm": arm, "batch": b, "status": "ok", **fn()}
                except Exception as e:  # noqa: BLE001
                    r = {"arm": arm, "batch": b, "status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
                print(
                    f"{name} {args.device} b={b} {arm} {r['status']} {r.get('steps_per_s', 0):.0f} steps/s", flush=True
                )
                rows.append(r)
        res["envs"][name] = rows
    Path(args.output).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
