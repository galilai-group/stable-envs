"""Where does the 1-ulp gap between the Craftax renderer and ``get_obs`` come from?

``get_obs`` is ``self._render_fn(state) / 255.0``.  Recomputing that division outside XLA disagrees with
it on every step, at the 1-ulp level.  The obvious suspect is XLA contracting ``x / 255.0`` into
``x * (1/255.0)``, which is a different floating-point operation.  That is testable: compute both forms
in numpy float32 and see which one matches ``get_obs`` exactly.

Run under SLURM.
"""
from __future__ import annotations

import json

import numpy as np


def main():
    import jax
    from craftax.craftax_env import make_craftax_env_from_name

    out = {}
    for tag, env_name, mod in [
        ("craftax_full", "Craftax-Pixels-v1", "craftax.craftax"),
        ("craftax_classic", "Craftax-Classic-Pixels-v1", "craftax.craftax_classic"),
    ]:
        consts = __import__(f"{mod}.constants", fromlist=["*"])
        renderer = __import__(f"{mod}.renderer", fromlist=["*"])
        render_raw = jax.jit(renderer.make_craftax_pixel_renderer(consts.BLOCK_PIXEL_SIZE_AGENT))

        env = make_craftax_env_from_name(env_name, auto_reset=False)
        params = env.default_params
        get_obs = jax.jit(env.get_obs)
        key = jax.random.PRNGKey(0)
        _obs, state = env.reset(key, params)

        div, mul = [], []
        for _ in range(30):
            raw = np.asarray(render_raw(state), dtype=np.float32)
            ref = np.asarray(get_obs(state), dtype=np.float32)
            div.append(bool(np.array_equal(raw / np.float32(255.0), ref)))
            mul.append(bool(np.array_equal(raw * np.float32(1.0 / 255.0), ref)))
            key, k = jax.random.split(key)
            _obs, state, *_ = env.step(k, state, int(jax.random.randint(k, (), 0, env.num_actions)),
                                       params)

        out[tag] = {"frac_matches_divide": float(np.mean(div)),
                    "frac_matches_reciprocal_multiply": float(np.mean(mul)),
                    "n_steps": len(div)}
        print(f"{tag:<16} raw/255 == get_obs: {100*np.mean(div):5.1f}%   "
              f"raw*(1/255) == get_obs: {100*np.mean(mul):5.1f}%")

    p = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/craftax_ulp.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
