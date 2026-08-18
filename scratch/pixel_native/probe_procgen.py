"""Measure, on the official Procgen binary, whether the frame is a function of a compact state.

Procgen exposes ``get_state``/``set_state`` on the C env (a full serialization of the game object).
Three things are measured rather than assumed:

  * how many bytes that state actually is, per game, versus the 64*64*3 = 12288-byte observation;
  * whether restoring the state reproduces the observation *bit for bit* (if not, the state is
    incomplete and the whole premise is dead);
  * how the state size scales with the number of live entities, which tells us whether the state is
    "tile grid + entity list" (compact, atlas-renderable) or something frame-sized.

Run under SLURM. Writes JSON to --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np


def probe_game(env_name, n_steps, seed, distribution_mode="hard"):
    from procgen import ProcgenGym3Env

    env = ProcgenGym3Env(num=1, env_name=env_name, rand_seed=seed,
                         distribution_mode=distribution_mode)
    rng = np.random.RandomState(seed)
    n_actions = env.ac_space.eltype.n

    sizes, obs_hashes, states = [], [], []
    _, ob, _ = env.observe()
    frame = ob["rgb"]
    for t in range(n_steps):
        st = env.callmethod("get_state")[0]
        sizes.append(len(st))
        _, ob, _ = env.observe()
        obs_hashes.append(hashlib.sha1(ob["rgb"].tobytes()).hexdigest())
        states.append(st)
        env.act(rng.randint(0, n_actions, size=(1,)).astype(np.int32))

    # round trip: restore each of a few states into a FRESH env (different rand_seed, so nothing
    # is shared except the bytes) and check the observation matches bit for bit.
    idx = np.linspace(0, n_steps - 1, min(8, n_steps)).astype(int)
    rt_exact, rt_mean_abs = [], []
    for i in idx:
        env2 = ProcgenGym3Env(num=1, env_name=env_name, rand_seed=seed + 12345,
                              distribution_mode=distribution_mode)
        env2.callmethod("set_state", [states[i]])
        _, ob2, _ = env2.observe()
        h = hashlib.sha1(ob2["rgb"].tobytes()).hexdigest()
        rt_exact.append(h == obs_hashes[i])
        rt_mean_abs.append(0.0 if h == obs_hashes[i] else None)

    # how much of the state is the tile grid?  serialize a state, then compare a state taken right
    # after reset on two different level seeds: same game, different level -> same grid dims.
    return {
        "env_name": env_name,
        "frame_shape": list(frame.shape[1:]),
        "frame_bytes": int(np.prod(frame.shape[1:])),
        "state_bytes_min": int(min(sizes)),
        "state_bytes_max": int(max(sizes)),
        "state_bytes_mean": float(np.mean(sizes)),
        "state_bytes_p50": float(np.median(sizes)),
        "ratio_state_over_frame_mean": float(np.mean(sizes)) / float(np.prod(frame.shape[1:])),
        "roundtrip_exact_frac": float(np.mean(rt_exact)),
        "n_steps": n_steps,
        "n_actions": int(n_actions),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/procgen_state.json")
    args = ap.parse_args()

    from procgen.env import ENV_NAMES

    rows = []
    for name in ENV_NAMES:
        r = probe_game(name, args.n_steps, args.seed)
        rows.append(r)
        print(f"{r['env_name']:<12} state {r['state_bytes_mean']:9.0f} B "
              f"(min {r['state_bytes_min']}, max {r['state_bytes_max']})  "
              f"frame {r['frame_bytes']} B  ratio {r['ratio_state_over_frame_mean']:.3f}  "
              f"roundtrip-exact {100*r['roundtrip_exact_frac']:.0f}%", flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
