"""KAGE-Bench (arXiv 2601.14232) against the four gates."""
from __future__ import annotations
import json, os, time
import numpy as np

OUT = os.environ.get("OUT", "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/kage.json")
R = {}


def jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def state_bytes(st):
    per, tot = {}, 0
    for k in st.__dataclass_fields__:
        v = getattr(st, k)
        if v is None:
            per[k] = {"bytes": 0, "shape": None, "dtype": None}; continue
        try:
            a = np.asarray(v)
        except Exception:
            per[k] = {"bytes": None}; continue
        per[k] = {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": int(a.nbytes)}
        tot += int(a.nbytes)
    return per, tot


def main():
    import jax, jax.numpy as jnp
    from kage_bench import KAGE_Env, EnvConfig
    dev = jax.devices()
    R["jax_devices"] = [str(d) for d in dev]
    R["jax_version"] = jax.__version__
    R["gpu"] = dev[0].platform == "gpu"
    if not R["gpu"]:
        R["ERROR"] = "not on GPU"
        return

    cfg = EnvConfig()
    env = KAGE_Env(cfg)
    frame_bytes = cfg.H * cfg.W * 3
    R["config"] = {"H": cfg.H, "W": cfg.W, "frame_bytes_uint8": frame_bytes,
                   "background_mode": cfg.background.mode,
                   "world_length": getattr(cfg.layout, "length", None)}

    key = jax.random.PRNGKey(0)
    obs, info = env.reset(key)
    st = info["state"]
    per, tot = state_bytes(st)
    R["G2"] = {"envstate_bytes": tot, "fields": per,
               "frame_bytes": frame_bytes,
               "ratio_envstate_over_frame": tot / frame_bytes,
               "obs_shape": list(np.asarray(obs).shape),
               "obs_dtype": str(np.asarray(obs).dtype),
               "biggest_fields": sorted(((v["bytes"] or 0, k) for k, v in per.items()),
                                        reverse=True)[:8]}
    print("G2", tot, "vs frame", frame_bytes, "=", tot / frame_bytes, flush=True)
    print("biggest", R["G2"]["biggest_fields"], flush=True)

    # ---------------- G3: is the decoder a constant of the family?
    # the sprite atlases are instance attributes; bg_image is a STATE field.
    atlas = {}
    for name in ("character_sprites", "npc_sprite_sets", "sticky_npc_sprite_sets",
                 "character_skin_sets", "background_images"):
        v = getattr(env, name, None)
        if v is None:
            atlas[name] = None; continue
        def walk(x):
            if isinstance(x, (list, tuple)):
                return sum(walk(y) for y in x)
            try:
                return int(np.asarray(x).nbytes)
            except Exception:
                return 0
        def count(x):
            if isinstance(x, (list, tuple)):
                return sum(count(y) for y in x)
            return 1
        atlas[name] = {"bytes": walk(v), "n_arrays": count(v)}
    R["G3"] = {"atlas": atlas}

    # bg_image across resets: is it content resampled per episode?
    bgs = []
    for s in range(6):
        _o, i2 = env.reset(jax.random.PRNGKey(s))
        b = getattr(i2["state"], "bg_image", None)
        bgs.append(None if b is None else np.asarray(b).copy())
    if bgs[0] is None:
        R["G3"]["bg_image_present"] = False
    else:
        R["G3"]["bg_image_present"] = True
        R["G3"]["bg_image_bytes"] = int(bgs[0].nbytes)
        R["G3"]["bg_image_distinct_across_6_resets"] = len({b.tobytes() for b in bgs})
        R["G3"]["bg_image_blank"] = bool(bgs[0].max() == bgs[0].min())
    # layout across resets
    lay = []
    for s in range(6):
        _o, i2 = env.reset(jax.random.PRNGKey(s))
        lay.append(np.asarray(i2["state"].layout_solid_mask).copy())
    R["G3"]["layout_distinct_across_6_resets"] = len({l.tobytes() for l in lay})
    R["G3"]["layout_shape"] = list(lay[0].shape)
    R["G3"]["layout_bytes"] = int(lay[0].nbytes)
    print("G3", R["G3"]["atlas"], R["G3"].get("bg_image_distinct_across_6_resets"),
          R["G3"]["layout_distinct_across_6_resets"], flush=True)

    # ---------------- G1: purity
    render = jax.jit(env.render)
    step = jax.jit(env.step)
    _o, i0 = env.reset(jax.random.PRNGKey(7))
    st = i0["state"]
    frames = []
    states = []
    rng = np.random.RandomState(0)
    blank = 0
    for t in range(200):
        f = np.asarray(render(st))
        if f.max() == f.min():
            blank += 1
        if t % 13 == 0:
            states.append(st); frames.append(f.copy())
        o, r, te, tr, inf = step(st, int(rng.randint(8)))
        st = inf["state"]
    rr = sum(0 if np.array_equal(np.asarray(render(s)), f) else 1
             for s, f in zip(states, frames))
    # re-render after an intervening rollout of a DIFFERENT trajectory
    st2 = states[0]
    for t in range(37):
        o, r, te, tr, inf = step(st2, int(rng.randint(8)))
        st2 = inf["state"]
    cross = sum(0 if np.array_equal(np.asarray(render(s)), f) else 1
                for s, f in zip(states, frames))
    # does render() agree with the obs the env returns?
    _o2, i2 = env.reset(jax.random.PRNGKey(11))
    s2 = i2["state"]
    obs_mism = 0
    for t in range(50):
        o, r, te, tr, inf = step(s2, int(rng.randint(8)))
        s2 = inf["state"]
        if not np.array_equal(np.asarray(o), np.asarray(render(s2))):
            obs_mism += 1
    R["G1"] = {"frames": 200, "blank_frames": blank, "checkpoints": len(states),
               "rerender_mismatch": rr, "rerender_after_other_rollout_mismatch": cross,
               "step_obs_vs_render_mismatch": obs_mism, "obs_checks": 50}
    print("G1", R["G1"], flush=True)

    # ---------------- G4
    def rollout(key, T=64):
        o, info = env.reset(key)
        s = info["state"]

        def body(carry, _):
            s, k = carry
            k, ka = jax.random.split(k)
            a = jax.random.randint(ka, (), 0, 8)
            o, r, te, tr, inf = env.step(s, a)
            return (inf["state"], k), o
        (s, k), obs = jax.lax.scan(body, (s, key), None, length=T)
        return obs

    g4 = {}
    T = 64
    f = jax.jit(jax.vmap(lambda k: rollout(k, T)))
    for B in (1, 64, 256, 1024, 4096, 16384, 32768):
        keys = jax.random.split(jax.random.PRNGKey(0), B)
        try:
            o = f(keys); o.block_until_ready()
        except Exception as e:
            g4[B] = {"error": repr(e)[:250]}
            print("G4 err", B, repr(e)[:250], flush=True)
            break
        reps = 3 if B >= 4096 else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            o = f(keys)
        o.block_until_ready()
        g4[B] = B * T * reps / (time.perf_counter() - t0)
        print("G4 step+render", B, g4[B], flush=True)
        del o
    R["G4_step_plus_render_sps"] = g4

    # render alone
    g4r = {}
    vrender = jax.jit(jax.vmap(env.render))
    for B in (1, 64, 256, 1024, 4096):
        sts = jax.tree_util.tree_map(
            lambda x: np.broadcast_to(np.asarray(x), (B,) + np.asarray(x).shape).copy(), st)
        sts = jax.tree_util.tree_map(jax.numpy.asarray, sts)
        try:
            fr = vrender(sts); fr.block_until_ready()
        except Exception as e:
            g4r[B] = {"error": repr(e)[:250]}; break
        reps = 3 if B >= 1024 else 10
        t0 = time.perf_counter()
        for _ in range(reps):
            fr = vrender(sts)
        fr.block_until_ready()
        g4r[B] = B * reps / (time.perf_counter() - t0)
        print("G4 render only", B, g4r[B], flush=True)
        del fr
    R["G4_render_only_fps"] = g4r

    # decode baseline at 128x128
    import io
    from PIL import Image
    img = np.asarray(render(st))
    bio = io.BytesIO(); Image.fromarray(img).save(bio, format="JPEG", quality=90)
    raw = bio.getvalue()
    N = 3000
    t0 = time.perf_counter()
    for _ in range(N):
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    R["decode_jpeg_fps_one_core_128px"] = N / (time.perf_counter() - t0)
    R["decode_jpeg_bytes"] = len(raw)
    print("decode", R["decode_jpeg_fps_one_core_128px"], flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(OUT, "w") as fh:
            json.dump(R, fh, indent=1, default=jsonable)
        print("wrote", OUT)
