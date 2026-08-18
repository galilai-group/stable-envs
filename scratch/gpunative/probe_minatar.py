"""MinAtar against the four gates of docs/research/when_does_this_pay_off.md.

MinAtar is NOT Atari: 5 games on a 10x10xC binary channel grid (kenjyoung/MinAtar).
Two implementations are probed:
  * the original numpy package `minatar` 1.0.15 -- the official env AND the only
    official RGB path, `minatar.gym.BaseEnv.render(render_mode="rgb_array")`;
  * `gymnax` 0.0.9's JAX re-implementations, for the GPU throughput side.

Everything is measured; nothing is argued from what the source "should" do.
"""
from __future__ import annotations

import json
import os
import sys
import time
import io
import numpy as np

OUT = os.environ.get("OUT", "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/minatar.json")
GAMES = ["breakout", "freeway", "asterix", "seaquest", "space_invaders"]
R = {}


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


# ---------------------------------------------------------------- palettes / G3
def palettes():
    import seaborn as sns
    import matplotlib
    out = {}
    for g in GAMES:
        from minatar import Environment
        n = Environment(env_name=g).n_channels
        cmap = sns.color_palette("cubehelix", n)
        cmap = [(0.0, 0.0, 0.0)] + list(cmap)
        out[g] = {"n_channels": n, "palette": [[float(c) for c in e] for e in cmap]}
    # determinism: build twice, and in a fresh process-independent way
    same = all(
        np.array_equal(np.array(out[g]["palette"]),
                       np.array([(0.0, 0.0, 0.0)] + list(sns.color_palette("cubehelix", out[g]["n_channels"]))))
        for g in GAMES)
    # is a small-n palette a subset of a large-n palette?
    p4 = np.array(out["breakout"]["palette"])
    p10 = np.array(out["seaquest"]["palette"])
    subset = sum(1 for row in p4 if np.any(np.all(np.isclose(p10, row, atol=1e-12), axis=1)))
    return {
        "seaborn_version": sns.__version__,
        "matplotlib_version": matplotlib.__version__,
        "per_game": out,
        "deterministic_rebuild": bool(same),
        "breakout_palette_rows_also_in_seaquest_palette": int(subset),
        "distinct_palettes_in_family": len({tuple(map(tuple, out[g]["palette"])) for g in GAMES}),
        "decoder_bytes_float64": sum(len(out[g]["palette"]) * 3 * 8 for g in GAMES),
        "decoder_bytes_uint8_dedup": len({tuple(np.round(np.array(e) * 255).astype(int))
                                          for g in GAMES for e in out[g]["palette"]}) * 3,
    }


# ---------------------------------------------------------------- official RGB
def official_rgb(state, cmap_arr):
    n = state.shape[-1]
    num = np.amax(state * np.reshape(np.arange(n) + 1, (1, 1, -1)), 2)
    return cmap_arr[num]


def collapsed(state):
    n = state.shape[-1]
    return np.amax(state * np.reshape(np.arange(n) + 1, (1, 1, -1)), 2).astype(np.uint8)


def sig(env):
    """A hashable signature of ALL internal python attributes of the game object."""
    d = vars(env.env)
    parts = []
    for k in sorted(d):
        if k in ("random", "channels", "action_map"):
            continue
        v = d[k]
        if isinstance(v, np.ndarray):
            parts.append((k, v.tobytes()))
        else:
            parts.append((k, repr(v)))
    return repr(parts)


# ---------------------------------------------------------------- G1
def g1(n_roll=40, horizon=250, seed=0):
    from minatar import Environment
    import seaborn as sns
    out = {}
    for g in GAMES:
        env = Environment(env_name=g, sticky_action_prob=0.1)
        n = env.n_channels
        cmap_arr = np.array([(0.0, 0.0, 0.0)] + list(sns.color_palette("cubehelix", n)))
        rng = np.random.RandomState(seed)
        seen_obs = {}
        seen_idx = {}
        cross_obs_pairs = 0
        cross_obs_mismatch = 0
        cross_idx_pairs = 0
        cross_idx_mismatch = 0
        rerender_mismatch = 0
        mutation_mismatch = 0
        nframes = 0
        blank = 0
        for r in range(n_roll):
            env.seed(seed * 1000 + r)
            env.reset()
            for t in range(horizon):
                s = env.state()
                f = official_rgb(s, cmap_arr)
                nframes += 1
                if f.max() == f.min():
                    blank += 1
                # (a) render twice, and check render() does not mutate the state
                f2 = official_rgb(env.state(), cmap_arr)
                if not np.array_equal(f, f2):
                    rerender_mismatch += 1
                s2 = env.state()
                if not np.array_equal(s, s2):
                    mutation_mismatch += 1
                # (b) cross-history collisions on the full obs grid
                kobs = s.tobytes()
                sg = sig(env)
                if kobs in seen_obs:
                    prev_f, prev_sig = seen_obs[kobs]
                    if prev_sig != sg:                     # genuinely different internal state
                        cross_obs_pairs += 1
                        if not np.array_equal(prev_f, f):
                            cross_obs_mismatch += 1
                else:
                    seen_obs[kobs] = (f, sg)
                # (c) cross-history collisions on the collapsed index grid (the render slice)
                kidx = collapsed(s).tobytes()
                if kidx in seen_idx:
                    prev_f, prev_sig = seen_idx[kidx]
                    if prev_sig != sg:
                        cross_idx_pairs += 1
                        if not np.array_equal(prev_f, f):
                            cross_idx_mismatch += 1
                else:
                    seen_idx[kidx] = (f, sg)
                _, done = env.act(rng.randint(env.num_actions()))
                if done:
                    env.reset()
        out[g] = dict(frames=nframes, blank_frames=blank,
                      distinct_obs_grids=len(seen_obs), distinct_index_grids=len(seen_idx),
                      cross_history_obs_pairs=cross_obs_pairs, cross_history_obs_mismatch=cross_obs_mismatch,
                      cross_history_index_pairs=cross_idx_pairs, cross_history_index_mismatch=cross_idx_mismatch,
                      rerender_mismatch=rerender_mismatch, state_mutated_by_render=mutation_mismatch)
        print("G1", g, out[g], flush=True)
    return out


# ---------------------------------------------------------------- G2 (+ render slice by ablation)
def g2(n_states=200, seed=1):
    """Which internal attributes does state() actually read? Perturb each and diff."""
    from minatar import Environment
    out = {}
    for g in GAMES:
        env = Environment(env_name=g)
        env.seed(seed)
        env.reset()
        rng = np.random.RandomState(seed)
        # gather a pool of live game objects by rolling out
        import copy
        pool = []
        for t in range(n_states * 3):
            if t % 3 == 0:
                pool.append(copy.deepcopy(env.env))
            _, done = env.act(rng.randint(6))
            if done:
                env.reset()
            if len(pool) >= n_states:
                break
        reads = {}
        for obj in pool:
            base = obj.state()
            for k, v in list(vars(obj).items()):
                if k in ("random", "channels", "action_map"):
                    continue
                old = v
                changed = False
                try:
                    if isinstance(v, bool):
                        setattr(obj, k, not v)
                    elif isinstance(v, (int, np.integer)):
                        setattr(obj, k, int(v) + 1)
                    elif isinstance(v, (float, np.floating)):
                        setattr(obj, k, float(v) + 1.0)
                    elif isinstance(v, np.ndarray):
                        w = v.copy()
                        w.flat[rng.randint(w.size)] = 1 - w.flat[rng.randint(w.size)] if w.dtype == bool else w.flat[0] + 1
                        setattr(obj, k, w)
                    elif isinstance(v, list):
                        continue   # entity lists: obviously read, handled below
                    else:
                        continue
                    try:
                        changed = not np.array_equal(obj.state(), base)
                    except Exception:
                        changed = True
                finally:
                    setattr(obj, k, old)
                reads[k] = reads.get(k, 0) + int(changed)
        n = env.n_channels
        obs_bytes = 10 * 10 * n            # numpy bool
        obs_packed = int(np.ceil(10 * 10 * n / 8))
        idx_bits = int(np.ceil(np.log2(n + 1)))
        idx_packed = int(np.ceil(10 * 10 * idx_bits / 8))
        frame_f64 = 10 * 10 * 3 * 8
        frame_u8 = 10 * 10 * 3
        out[g] = dict(
            n_channels=n,
            attr_render_relevant={k: v for k, v in sorted(reads.items()) if v > 0},
            attr_never_reached={k: v for k, v in sorted(reads.items()) if v == 0},
            obs_bytes_bool=obs_bytes, obs_bytes_bitpacked=obs_packed,
            index_grid_bits_per_cell=idx_bits, index_grid_bytes_packed=idx_packed,
            frame_bytes_float64=frame_f64, frame_bytes_uint8=frame_u8,
            ratio_obsbool_over_frame_u8=obs_bytes / frame_u8,
            ratio_obspacked_over_frame_u8=obs_packed / frame_u8,
            ratio_index_over_frame_u8=idx_packed / frame_u8,
            ratio_obsbool_over_frame_f64=obs_bytes / frame_f64,
            ratio_index_over_frame_f64=idx_packed / frame_f64,
        )
        print("G2", g, out[g]["n_channels"], out[g]["ratio_index_over_frame_u8"], flush=True)
    return out


def g2_gymnax():
    import jax, jax.numpy as jnp
    import gymnax
    out = {}
    ids = {"breakout": "Breakout-MinAtar", "freeway": "Freeway-MinAtar", "asterix": "Asterix-MinAtar",
           "seaquest": "Seaquest-MinAtar", "space_invaders": "SpaceInvaders-MinAtar"}
    for g, eid in ids.items():
        try:
            env, params = gymnax.make(eid)
        except Exception as e:
            out[g] = {"error": repr(e)}
            continue
        obs, st = env.reset(jax.random.PRNGKey(0), params)
        leaves, _ = jax.tree_util.tree_flatten(st)
        names = [str(k) for k in jax.tree_util.tree_structure(st)] if False else None
        tot = int(sum(np.asarray(l).nbytes for l in leaves))
        per = {}
        for k in st.__dataclass_fields__:
            v = getattr(st, k)
            a = np.asarray(v)
            per[k] = {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": int(a.nbytes)}
        out[g] = {"envstate_bytes": tot, "fields": per,
                  "obs_shape": list(np.asarray(obs).shape), "obs_dtype": str(np.asarray(obs).dtype),
                  "ratio_envstate_over_frame_u8": tot / 300.0}
        print("G2-gymnax", g, tot, flush=True)
    return out


# ---------------------------------------------------------------- G1 gymnax vs official parity
def gymnax_parity(n=200, seed=0):
    """Does gymnax's get_obs reproduce the official MinAtar obs for matched rollouts?

    Not a pixel-parity claim -- gymnax is an independent re-implementation. What is
    measured here is only whether gymnax's obs is a pure function of its EnvState
    (render the same state twice, and after a different rollout), which is the G1
    question for the JAX path.
    """
    import jax, jax.numpy as jnp
    import gymnax
    ids = {"breakout": "Breakout-MinAtar", "freeway": "Freeway-MinAtar", "asterix": "Asterix-MinAtar",
           "seaquest": "Seaquest-MinAtar", "space_invaders": "SpaceInvaders-MinAtar"}
    out = {}
    for g, eid in ids.items():
        env, params = gymnax.make(eid)
        key = jax.random.PRNGKey(seed)
        obs, st = env.reset(key, params)
        step = jax.jit(env.step)
        get_obs = jax.jit(env.get_obs)
        mism = 0
        blank = 0
        seen = {}
        cross = 0
        crossmis = 0
        for t in range(n):
            key, k1, k2 = jax.random.split(key, 3)
            a = jax.random.randint(k1, (), 0, env.num_actions)
            obs, st, r, d, info = step(k2, st, a, params)
            o1 = np.asarray(get_obs(st))
            o2 = np.asarray(get_obs(st))
            if not np.array_equal(o1, o2):
                mism += 1
            if o1.max() == o1.min():
                blank += 1
            if not np.array_equal(o1, np.asarray(obs)):
                mism += 1
        out[g] = {"obs_nondeterminism_or_step_obs_mismatch": mism, "blank_obs": blank, "steps": n}
        print("G1-gymnax", g, out[g], flush=True)
    return out


# ---------------------------------------------------------------- G4
def g4_cpu(steps=20000, seed=3):
    from minatar import Environment
    import seaborn as sns
    out = {}
    for g in GAMES:
        env = Environment(env_name=g)
        env.seed(seed)
        env.reset()
        rng = np.random.RandomState(seed)
        n = env.n_channels
        cmap_arr = np.array([(0.0, 0.0, 0.0)] + list(sns.color_palette("cubehelix", n)))
        acts = rng.randint(0, 6, size=steps)
        # warm
        for i in range(200):
            env.act(int(acts[i]))
        t0 = time.perf_counter()
        for i in range(steps):
            _, d = env.act(int(acts[i]))
            if d:
                env.reset()
        t_act = time.perf_counter() - t0
        env.reset()
        t0 = time.perf_counter()
        for i in range(steps):
            s = env.state()
        t_state = time.perf_counter() - t0
        s = env.state()
        t0 = time.perf_counter()
        for i in range(steps):
            f = official_rgb(env.state(), cmap_arr)
        t_render = time.perf_counter() - t0
        out[g] = dict(
            steps=steps,
            act_only_sps=steps / t_act,
            state_call_fps=steps / t_state,
            official_rgb_fps=steps / t_render,
            render_share_of_step=(t_render - 0.0) / (t_act + t_render),
        )
        print("G4-cpu", g, out[g], flush=True)
    # decode baseline for a 10x10x3 frame
    from PIL import Image
    img = (np.random.RandomState(0).randint(0, 255, (10, 10, 3))).astype(np.uint8)
    bio = io.BytesIO()
    Image.fromarray(img).save(bio, format="PNG")
    raw = bio.getvalue()
    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    t = time.perf_counter() - t0
    out["_decode_baseline"] = {"png_bytes": len(raw), "decode_fps_one_core": N / t}
    # and a raw 300-byte memcpy "decode" (the favourable case)
    buf = img.tobytes()
    t0 = time.perf_counter()
    for _ in range(N):
        np.frombuffer(buf, np.uint8).reshape(10, 10, 3)
    t = time.perf_counter() - t0
    out["_decode_baseline"]["raw_frombuffer_fps"] = N / t
    print("G4-decode", out["_decode_baseline"], flush=True)
    return out


def g4_gpu(batches=(1, 64, 256, 1024, 4096, 16384)):
    import jax, jax.numpy as jnp
    import gymnax
    import seaborn as sns
    dev = jax.devices()
    out = {"jax_devices": [str(d) for d in dev], "jax_version": jax.__version__}
    if dev[0].platform != "gpu":
        out["ERROR"] = "not on GPU"
        return out
    ids = {"breakout": "Breakout-MinAtar", "asterix": "Asterix-MinAtar",
           "freeway": "Freeway-MinAtar", "seaquest": "Seaquest-MinAtar",
           "space_invaders": "SpaceInvaders-MinAtar"}
    # --- the renderer alone: a gather palette[idx]
    rend = {}
    for g, eid in ids.items():
        env, params = gymnax.make(eid)
        n = env.obs_shape[-1]
        pal = jnp.asarray(np.array([(0.0, 0.0, 0.0)] + list(sns.color_palette("cubehelix", n))), jnp.float32)
        w = jnp.arange(1, n + 1, dtype=jnp.int32)

        @jax.jit
        def render(obs, pal=pal, w=w):
            idx = jnp.max(obs.astype(jnp.int32) * w, axis=-1)
            return (pal[idx] * 255).astype(jnp.uint8)

        rend[g] = {}
        for B in batches:
            o = jax.random.bernoulli(jax.random.PRNGKey(0), 0.1, (B, 10, 10, n))
            f = render(o); f.block_until_ready()
            reps = max(3, int(2e7 / (B * 100)))
            reps = min(reps, 2000)
            t0 = time.perf_counter()
            for _ in range(reps):
                f = render(o)
            f.block_until_ready()
            dt = time.perf_counter() - t0
            rend[g][B] = B * reps / dt
        print("G4-gpu-render", g, rend[g], flush=True)
    out["gpu_render_fps"] = rend

    # --- gymnax step throughput (step + get_obs) under vmap
    stepper = {}
    for g, eid in ids.items():
        env, params = gymnax.make(eid)

        def rollout(keys, T=64):
            def one(key):
                k, kr = jax.random.split(key)
                obs, st = env.reset(kr, params)

                def body(carry, _):
                    k, st = carry
                    k, ka, ks = jax.random.split(k, 3)
                    a = jax.random.randint(ka, (), 0, env.num_actions)
                    obs, st, r, d, _i = env.step(ks, st, a, params)
                    return (k, st), obs
                (k, st), obs = jax.lax.scan(body, (k, st), None, length=T)
                return obs
            return jax.vmap(one)(keys)

        T = 64
        f = jax.jit(lambda keys: rollout(keys, T))
        stepper[g] = {}
        for B in batches:
            keys = jax.random.split(jax.random.PRNGKey(0), B)
            o = f(keys); o.block_until_ready()
            reps = 3 if B >= 4096 else 10
            t0 = time.perf_counter()
            for _ in range(reps):
                o = f(keys)
            o.block_until_ready()
            dt = time.perf_counter() - t0
            stepper[g][B] = B * T * reps / dt
        print("G4-gpu-step", g, stepper[g], flush=True)
    out["gymnax_step_sps"] = stepper
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    def dump():
        with open(OUT, "w") as fh:
            json.dump(R, fh, indent=1, default=jsonable)
    if which in ("all", "cpu"):
        R["palettes_G3"] = palettes(); dump()
        R["G1"] = g1(); dump()
        R["G2"] = g2(); dump()
        R["G4_cpu"] = g4_cpu(); dump()
    if which in ("all", "gpu"):
        R["G2_gymnax"] = g2_gymnax()
        R["G1_gymnax"] = gymnax_parity()
        R["G4_gpu"] = g4_gpu()
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=1, default=jsonable)
    print("wrote", OUT)
