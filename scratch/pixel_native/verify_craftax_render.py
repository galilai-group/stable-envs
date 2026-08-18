"""Does the Craftax renderer, called on the state alone, reproduce the env's own pixel observation?

The renderer is already a JAX function of ``EnvState``, so this is the pixel-diff step: roll the env,
call ``make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_AGENT)`` on the stored state, and compare against the
observation the env returned, bit for bit.  Also times a vmapped batch, since a renderer that is not
batchable is not useful here.

Run under SLURM.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=40)
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 64, 256])
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/craftax_render.json")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name

    print("jax devices:", jax.devices())
    out = {}
    for tag, env_name, mod in [
        ("craftax_full", "Craftax-Pixels-v1", "craftax.craftax"),
        ("craftax_classic", "Craftax-Classic-Pixels-v1", "craftax.craftax_classic"),
    ]:
        consts = __import__(f"{mod}.constants", fromlist=["*"])
        renderer = __import__(f"{mod}.renderer", fromlist=["*"])
        render = jax.jit(renderer.make_craftax_pixel_renderer(consts.BLOCK_PIXEL_SIZE_AGENT))

        env = make_craftax_env_from_name(env_name, auto_reset=False)
        params = env.default_params
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)

        get_obs = jax.jit(env.get_obs)
        exact, worst, states, exact_u8 = [], [], [], []
        eq_ab, eq_bc = [], []
        for t in range(args.n_steps):
            # stay in float32: get_obs is float32(raw)/float32(255), and doing the divide
            # in float64 would manufacture a 1e-7 disagreement that is not the renderer's
            r = (np.asarray(render(state), dtype=np.float32) / np.float32(255.0))
            o = np.asarray(obs)
            exact.append(bool(np.array_equal(r, o)))
            worst.append(float(np.abs(r - o).max()))
            exact_u8.append(bool(np.array_equal((r * 255).round().astype(np.uint8),
                                                (o * 255).round().astype(np.uint8))))
            # A = renderer called on the state; B = the env's own get_obs, compiled on its own;
            # C = the obs step() returned (get_obs fused inside the jitted step_env).
            b = np.asarray(get_obs(state), dtype=np.float32)
            eq_ab.append(bool(np.array_equal(r, b)))
            eq_bc.append(bool(np.array_equal(b, o)))
            states.append(state)
            key, k = jax.random.split(key)
            a = int(jax.random.randint(k, (), 0, env.num_actions))
            obs, state, *_ = env.step(k, state, a, params)

        timings = {}
        vf = jax.jit(jax.vmap(render))
        for B in args.batches:
            st = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + jnp.shape(x)), states[-1])
            r = vf(st); r.block_until_ready()
            t0 = time.perf_counter()
            for _ in range(5):
                r = vf(st); r.block_until_ready()
            dt = (time.perf_counter() - t0) / 5
            timings[B] = {"ms": dt * 1000, "frames_per_s": B / dt}

        out[tag] = {
            "env_name": env_name,
            "block_pixel_size": int(consts.BLOCK_PIXEL_SIZE_AGENT),
            "frame_shape": list(np.asarray(obs).shape),
            "n_steps": args.n_steps,
            "frac_bit_exact_vs_env_obs": float(np.mean(exact)),
            "worst_abs_diff": float(max(worst)),
            "frac_bit_exact_as_uint8": float(np.mean(exact_u8)),
            "frac_renderer_equals_get_obs": float(np.mean(eq_ab)),
            "frac_get_obs_equals_step_obs": float(np.mean(eq_bc)),
            "batch_timings": {str(k): v for k, v in timings.items()},
            "device": str(jax.devices()[0]),
        }
        d = out[tag]
        print(f"\n== {tag} ==")
        print(f"  frame {d['frame_shape']}  bit-exact vs env obs on "
              f"{100*d['frac_bit_exact_vs_env_obs']:.1f}% of {args.n_steps} steps "
              f"(worst |diff| {d['worst_abs_diff']:g}); bit-exact as uint8 on {100*d['frac_bit_exact_as_uint8']:.1f}%")
        print(f"  renderer(state) == env.get_obs(state): {100*d['frac_renderer_equals_get_obs']:.1f}%   "
              f"env.get_obs(state) == obs from step(): {100*d['frac_get_obs_equals_step_obs']:.1f}%")
        for B, v in timings.items():
            print(f"  vmap batch {B:5d}: {v['ms']:8.2f} ms  -> {v['frames_per_s']:10.0f} frames/s")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
