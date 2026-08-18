"""Craftax: how big is the state that determines the frame, and is the frame an atlas blit?

Craftax already renders in JAX from an ``EnvState``, so the interesting questions are not "is it
possible" but "at what size":

  * the full ``EnvState`` carries every level of the world, most of which is off-screen -- measured
    against the frame it produces, because if the state is bigger than the frame then storing states
    is not a saving;
  * the *render-relevant* slice is the OBS_DIM window plus the player/inventory scalars -- measured
    separately, because that is what a stable-envs renderer would take as input;
  * the texture set is enumerated (count, tile size, total bytes) to check that the atlas is finite;
  * the renderer is checked to be a deterministic pure function of the state (same state twice -> same
    pixels, bit for bit) and to agree with the observation the env itself returns.

Run under SLURM.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def leaf_bytes(tree):
    import jax
    tot, rows = 0, []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a = np.asarray(leaf)
        n = int(a.nbytes)
        tot += n
        rows.append((".".join(str(p) for p in path), list(a.shape), str(a.dtype), n))
    return tot, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/craftax_state.json")
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    from craftax.craftax_env import make_craftax_env_from_name

    out = {}

    for tag, env_name, mod in [
        ("craftax_full", "Craftax-Pixels-v1", "craftax.craftax"),
        ("craftax_classic", "Craftax-Classic-Pixels-v1", "craftax.craftax_classic"),
    ]:
        consts = __import__(f"{mod}.constants", fromlist=["*"])
        renderer = __import__(f"{mod}.renderer", fromlist=["*"])

        env = make_craftax_env_from_name(env_name, auto_reset=False)
        params = env.default_params
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        for i in range(30):
            key, k = jax.random.split(key)
            obs, state, *_ = env.step(k, state, int(jax.random.randint(k, (), 0, env.num_actions)), params)

        obs = np.asarray(obs)
        frame_bytes_u8 = int(np.prod(obs.shape))
        tot, rows = leaf_bytes(state)

        # the renderer itself, called directly on the state
        rname = [n for n in dir(renderer) if n.startswith("render_") and "pixel" in n]
        maker = [n for n in dir(renderer) if n.startswith("make_") and "pixel" in n]
        px = None
        if maker:
            px = getattr(renderer, maker[0])(consts.BLOCK_PIXEL_SIZE_IMG)
        elif rname:
            px = getattr(renderer, rname[0])

        det = eq_env = None
        batch_ms = None
        if px is not None:
            f = jax.jit(px)
            a = np.asarray(f(state))
            b = np.asarray(f(state))
            det = bool(np.array_equal(a, b))
            eq_env = bool(np.array_equal(a, obs)) if a.shape == obs.shape else None

            # batched render, the whole point of a GPU renderer
            states = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (args.batch,) + jnp.shape(x)), state)
            vf = jax.jit(jax.vmap(px))
            r = vf(states)
            r.block_until_ready()
            t0 = time.perf_counter()
            for _ in range(5):
                r = vf(states)
                r.block_until_ready()
            batch_ms = (time.perf_counter() - t0) / 5 * 1000

        # texture atlas
        adir = os.path.join(os.path.dirname(consts.__file__), "assets")
        pngs = sorted(f for f in os.listdir(adir) if f.endswith(".png"))
        atlas_bytes = sum(os.path.getsize(os.path.join(adir, f)) for f in pngs)

        # render-relevant slice: the OBS_DIM window of each map layer + player/inventory scalars
        od = consts.OBS_DIM
        n_layers = sum(1 for name, *_ in rows if name.endswith("map") or "map" in name.split("."))
        window_cells = od[0] * od[1]
        window_bytes = window_cells * 4 * max(n_layers, 1)
        scalar_bytes = sum(n for name, shape, dt, n in rows if not any(k in name for k in ("map",)))

        out[tag] = {
            "env_name": env_name,
            "obs_shape": list(obs.shape),
            "obs_dtype": str(obs.dtype),
            "frame_bytes_uint8": frame_bytes_u8,
            "frame_bytes_as_returned": int(obs.nbytes),
            "envstate_total_bytes": tot,
            "envstate_over_frame_uint8": tot / frame_bytes_u8,
            "n_state_leaves": len(rows),
            "biggest_leaves": sorted(rows, key=lambda r: -r[3])[:8],
            "OBS_DIM": list(od),
            "BLOCK_PIXEL_SIZE_IMG": int(consts.BLOCK_PIXEL_SIZE_IMG),
            "n_map_layers": n_layers,
            "render_window_bytes": window_bytes,
            "non_map_scalar_bytes": scalar_bytes,
            "render_state_bytes": window_bytes + scalar_bytes,
            "render_state_over_frame": (window_bytes + scalar_bytes) / frame_bytes_u8,
            "n_texture_pngs": len(pngs),
            "atlas_png_bytes": atlas_bytes,
            "render_deterministic": det,
            "render_equals_env_obs": eq_env,
            "batch": args.batch,
            "vmap_render_ms_cpu": batch_ms,
        }
        d = out[tag]
        print(f"\n== {tag} ({env_name}) ==")
        print(f"  obs {d['obs_shape']} {d['obs_dtype']}  -> {d['frame_bytes_uint8']} B as uint8")
        print(f"  EnvState total {d['envstate_total_bytes']} B  = {d['envstate_over_frame_uint8']:.2f}x the frame")
        print(f"  biggest leaves: " + ", ".join(f"{n}{s}={b}B" for n, s, _, b in d["biggest_leaves"][:5]))
        print(f"  render-relevant slice ~{d['render_state_bytes']} B = {d['render_state_over_frame']:.3f}x the frame")
        print(f"  atlas {d['n_texture_pngs']} pngs, {d['atlas_png_bytes']/1024:.0f} kB on disk; "
              f"tile {d['BLOCK_PIXEL_SIZE_IMG']}px, window {d['OBS_DIM']}")
        print(f"  deterministic={d['render_deterministic']}  equals-env-obs={d['render_equals_env_obs']}  "
              f"vmap({args.batch}) {d['vmap_render_ms_cpu']:.1f} ms on CPU")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
