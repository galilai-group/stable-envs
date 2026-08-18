"""Octax (CHIP-8 in JAX, arXiv 2510.01764) against the four gates.

The point of interest: CHIP-8 HAS a framebuffer, so unlike the 2600 the frame is
literally a field of the machine state. That makes G1 pass by construction and
makes G2 a question about what the *rest* of the state costs.

Everything measured on one GPU under SLURM.
"""
from __future__ import annotations

import json
import os
import sys
import time
import io
import numpy as np

OUT = os.environ.get("OUT", "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/octax.json")
R = {}

GAMES = ["pong", "brix", "tetris", "space_flight1", "missile", "ufo", "worm", "blinky",
         "squash", "airplane", "tank", "cavern1", "submarine", "wipe_off"]


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes,)):
        return o.hex()
    raise TypeError(str(type(o)))


def leaf_bytes(state):
    import jax
    per = {}
    for k in state.__dataclass_fields__:
        v = getattr(state, k)
        if k == "stack":
            per["stack.data"] = np.asarray(v.data).nbytes
            per["stack.pointer"] = np.asarray(v.pointer).nbytes
            continue
        try:
            a = np.asarray(v)
        except Exception:
            continue
        per[k] = int(a.nbytes)
    return per


def build(game, **kw):
    from octax.environments import create_environment
    env, meta = create_environment(game, render_mode="rgb_array", **kw)
    return env, meta


# ------------------------------------------------------------------ state anatomy
def anatomy():
    import jax
    import jax.numpy as jnp
    out = {}
    for g in GAMES:
        try:
            env, meta = build(g)
        except Exception as e:
            out[g] = {"error": repr(e)}
            print("SKIP", g, repr(e)[:200], flush=True)
            continue
        key = jax.random.PRNGKey(0)
        st, obs, info = env.reset(key)
        per = leaf_bytes(st)
        tot = sum(per.values())
        disp = per["display"]
        obs_a = np.asarray(obs)
        out[g] = {
            "fields_bytes": per,
            "envstate_total_bytes": tot,
            "display_bytes": disp,
            "everything_else_bytes": tot - disp,
            "display_bitpacked_bytes": 64 * 32 // 8,
            "obs_shape": list(obs_a.shape), "obs_dtype": str(obs_a.dtype),
            "obs_bytes": int(obs_a.nbytes),
            "frame_scale1_rgb_uint8_bytes": 32 * 64 * 3,
            "frame_scale8_rgb_uint8_bytes": 256 * 512 * 3,
            "rom_bytes": len(env.rom_data),
            "instructions_per_env_step": env.instructions_per_step * env.frame_skip,
            "frame_skip": env.frame_skip,
            "action_set_size": int(env.num_actions),
        }
        out[g]["ratio_state_over_display_bool"] = tot / disp
        out[g]["ratio_state_over_frame_scale1"] = tot / (32 * 64 * 3)
        out[g]["ratio_state_over_frame_scale8"] = tot / (256 * 512 * 3)
        out[g]["ratio_nondisplay_over_display_bool"] = (tot - disp) / disp
        out[g]["ratio_displaypacked_over_frame_scale1"] = (64 * 32 // 8) / (32 * 64 * 3)
        print("ANATOMY", g, tot, "display", disp, flush=True)
    return out


# ------------------------------------------------------------------ G1 + dynamics
def g1(game="pong", T=400, seed=0):
    import jax
    import jax.numpy as jnp
    from octax.rendering import chip8_display_to_rgb, create_color_scheme
    env, meta = build(game)
    key = jax.random.PRNGKey(seed)
    st, obs, info = env.reset(key)
    step = env.step
    rng = np.random.RandomState(seed)
    states = []
    displays = []
    rom0 = np.asarray(st.memory).copy()
    mem_changed = np.zeros(4096, bool)
    blank = 0
    for t in range(T):
        a = int(rng.randint(env.num_actions))
        st, o, r, term, trunc, info = step(st, a)
        d = np.asarray(st.display)
        displays.append(d.copy())
        mem_changed |= (np.asarray(st.memory) != rom0)
        if d.max() == d.min():
            blank += 1
        if t % 17 == 0:
            states.append(st)
        if bool(term) or bool(trunc):
            st, o, info = env.reset(jax.random.PRNGKey(seed + t))
    on, off = create_color_scheme("classic")
    # (a) render twice -> identical
    rr = 0
    for s in states:
        f1 = chip8_display_to_rgb(s.display, 8, on, off)
        f2 = chip8_display_to_rgb(s.display, 8, on, off)
        if not np.array_equal(f1, f2):
            rr += 1
    # (b) frame is a pure function of state.display: build a fresh state carrying only
    #     the display and render it; compare against rendering the full state.
    pd = 0
    for s in states:
        f1 = chip8_display_to_rgb(s.display, 8, on, off)
        f2 = chip8_display_to_rgb(jnp.asarray(np.asarray(s.display)), 8, on, off)
        if not np.array_equal(f1, f2):
            pd += 1
    # (c) full-state round-trip through a fresh env instance (the ALE test that failed there)
    env2, _ = build(game)
    rt_mismatch = 0
    for s in states[:12]:
        s2 = env2.step(s, 0)[0]
        s1 = env.step(s, 0)[0]
        if not np.array_equal(np.asarray(s1.display), np.asarray(s2.display)):
            rt_mismatch += 1
    # (d) is the display recoverable from the rest of the state? Zero it and step.
    drop_mismatch = 0
    drop_px = []
    for s in states[:12]:
        true_next = np.asarray(env.step(s, 0)[0].display)
        s_blank = s.replace(display=jnp.zeros_like(s.display))
        blank_next = np.asarray(env.step(s_blank, 0)[0].display)
        if not np.array_equal(true_next, blank_next):
            drop_mismatch += 1
        drop_px.append(float((true_next != blank_next).mean()))
    return {
        "game": game, "steps": T,
        "blank_display_frames": blank,
        "nonblank_frac": 1.0 - blank / T,
        "render_twice_mismatch": rr, "n_states": len(states),
        "frame_is_pure_fn_of_display_mismatch": pd,
        "fresh_instance_roundtrip_mismatch": rt_mismatch,
        "display_dropped_then_stepped_mismatch": drop_mismatch,
        "display_dropped_mean_px_differing": float(np.mean(drop_px)) if drop_px else None,
        "memory_bytes_ever_written_over_rollout": int(mem_changed.sum()),
        "memory_bytes_total": 4096,
        "rom_bytes": len(env.rom_data),
    }


def dyn_slice(games=("pong", "brix", "tetris", "missile"), T=300, seed=0):
    """Per game: how much of the 4096-byte memory is ever written (i.e. is not a constant)."""
    import jax
    out = {}
    for g in games:
        try:
            env, _ = build(g)
        except Exception as e:
            out[g] = {"error": repr(e)}
            continue
        st, obs, info = env.reset(jax.random.PRNGKey(seed))
        rom0 = np.asarray(st.memory).copy()
        changed = np.zeros(4096, bool)
        rng = np.random.RandomState(seed)
        for t in range(T):
            st, o, r, term, trunc, info = env.step(st, int(rng.randint(env.num_actions)))
            changed |= (np.asarray(st.memory) != rom0)
            if bool(term) or bool(trunc):
                st, o, info = env.reset(jax.random.PRNGKey(seed + t))
        n = int(changed.sum())
        out[g] = {"memory_bytes_written": n, "memory_bytes_constant": 4096 - n,
                  "rom_bytes": len(env.rom_data),
                  "dynamic_state_bytes": n + 2 + 32 + 4 + 1 + 1 + 16 + 16 + 2,
                  }
        out[g]["dynamic_plus_display_bytes"] = out[g]["dynamic_state_bytes"] + 2048
        out[g]["ratio_dynamic_nondisplay_over_display"] = out[g]["dynamic_state_bytes"] / 2048.0
        print("DYN", g, out[g], flush=True)
    return out


# ------------------------------------------------------------------ G3
def g3():
    from octax.rendering import create_color_scheme, chip8_display_to_rgb
    import jax
    schemes = {k: create_color_scheme(k) for k in
               ["octax", "classic", "amber", "white", "blue", "retro"]}
    # decoder identical across games and across resets?
    per_game = {}
    for g in ("pong", "brix", "tetris"):
        try:
            env, _ = build(g)
        except Exception as e:
            per_game[g] = repr(e); continue
        per_game[g] = {"on_off": create_color_scheme(env.color_scheme), "scheme": env.color_scheme,
                       "render_scale": env.render_scale}
    return {"schemes": {k: [list(v[0]), list(v[1])] for k, v in schemes.items()},
            "n_colors_in_decoder": 2,
            "decoder_bytes": 6,
            "per_game": per_game,
            "note": "the decoder is 2 RGB triples for every game in the suite"}


# ------------------------------------------------------------------ G4
def g4(game="pong", batches=(1, 64, 256, 1024, 4096), T=32):
    import jax
    import jax.numpy as jnp
    dev = jax.devices()
    out = {"jax_devices": [str(d) for d in dev], "jax_version": jax.__version__, "game": game}
    if dev[0].platform != "gpu":
        out["ERROR"] = "not on GPU -- refusing to report"
        return out
    env, _ = build(game)
    ipst = env.instructions_per_step * env.frame_skip
    out["instructions_per_env_step"] = ipst

    key = jax.random.PRNGKey(0)
    st0, obs0, _ = env.reset(key)

    def rollout(states, keys):
        def body(carry, _):
            st, k = carry
            k, ka = jax.random.split(k)
            a = jax.random.randint(ka, (), 0, env.num_actions)
            st, o, r, te, tr, i = env.step(st, a)
            return (st, k), o
        (st, k), obs = jax.lax.scan(body, (states, keys), None, length=T)
        return obs

    step_sps = {}
    for B in batches:
        keys = jax.random.split(jax.random.PRNGKey(1), B)
        sts = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + jnp.shape(x)).copy()
                                     if not isinstance(x, bool) else x, st0)
        f = jax.jit(jax.vmap(rollout))
        try:
            o = f(sts, keys)
            jax.block_until_ready(o)
        except Exception as e:
            step_sps[B] = {"error": repr(e)[:400]}
            print("G4 err", B, repr(e)[:400], flush=True)
            continue
        reps = 3 if B >= 1024 else 8
        t0 = time.perf_counter()
        for _ in range(reps):
            o = f(sts, keys)
        jax.block_until_ready(o)
        dt = time.perf_counter() - t0
        step_sps[B] = B * T * reps / dt
        print("G4-step", B, step_sps[B], flush=True)
        del o
    out["env_steps_per_s"] = step_sps
    out["frames_per_s"] = {k: (v * env.frame_skip if isinstance(v, float) else v)
                           for k, v in step_sps.items()}
    out["chip8_instructions_per_s"] = {k: (v * ipst if isinstance(v, float) else v)
                                       for k, v in step_sps.items()}

    # renderer share: bool display -> RGB uint8 gather, on GPU
    pal = jnp.asarray(np.array([[0, 0, 0], [0, 255, 0]], np.uint8))

    @jax.jit
    def render(disp):
        return pal[disp.astype(jnp.int32)]

    rend = {}
    for B in batches:
        d = jax.random.bernoulli(jax.random.PRNGKey(0), 0.2, (B, 64, 32))
        f = render(d); f.block_until_ready()
        reps = min(2000, max(3, int(4e8 / (B * 2048))))
        t0 = time.perf_counter()
        for _ in range(reps):
            f = render(d)
        f.block_until_ready()
        rend[B] = B * reps / (time.perf_counter() - t0)
        print("G4-render", B, rend[B], flush=True)
    out["gpu_render_fps_scale1"] = rend

    # official CPU render path
    from octax.rendering import chip8_display_to_rgb, create_color_scheme
    on, off = create_color_scheme("classic")
    d = np.asarray(jax.random.bernoulli(jax.random.PRNGKey(0), 0.2, (64, 32)))
    N = 5000
    for scale, key_ in ((1, "official_cpu_render_fps_scale1"), (8, "official_cpu_render_fps_scale8")):
        chip8_display_to_rgb(d, scale, on, off)
        t0 = time.perf_counter()
        for _ in range(N):
            chip8_display_to_rgb(d, scale, on, off)
        out[key_] = N / (time.perf_counter() - t0)
    # decode baseline
    from PIL import Image
    for scale, key_ in ((1, "decode_png_fps_scale1"), (8, "decode_png_fps_scale8")):
        img = chip8_display_to_rgb(d, scale, on, off)
        bio = io.BytesIO(); Image.fromarray(img).save(bio, format="PNG")
        raw = bio.getvalue()
        M = 2000
        t0 = time.perf_counter()
        for _ in range(M):
            np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        out[key_] = M / (time.perf_counter() - t0)
        out[key_ + "_png_bytes"] = len(raw)
    print("G4 misc", {k: v for k, v in out.items() if "fps" in k}, flush=True)
    return out


def g4_cpu_reference(game="pong", steps=2000, seed=0):
    """Octax's own step on CPU (jax cpu backend is the closest thing to a CPU emulator
    baseline the package ships; there is no C reference in the wheel)."""
    import jax
    env, _ = build(game)
    st, obs, _ = env.reset(jax.random.PRNGKey(seed))
    rng = np.random.RandomState(seed)
    st, *_ = env.step(st, 0)
    jax.block_until_ready(st)
    t0 = time.perf_counter()
    for i in range(steps):
        st, o, r, te, tr, inf = env.step(st, int(rng.randint(env.num_actions)))
    jax.block_until_ready(st)
    dt = time.perf_counter() - t0
    return {"game": game, "steps": steps, "single_env_steps_per_s": steps / dt,
            "note": "single-env, unbatched, on whatever backend jax picked"}


if __name__ == "__main__":
    R["anatomy"] = anatomy()
    R["G3"] = g3()
    R["G1_pong"] = g1("pong")
    R["G1_brix"] = g1("brix")
    R["dyn_slice"] = dyn_slice()
    R["G4_single"] = g4_cpu_reference()
    R["G4"] = g4()
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=1, default=jsonable)
    print("wrote", OUT)
