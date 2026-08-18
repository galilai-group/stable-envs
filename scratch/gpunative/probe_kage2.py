"""KAGE-Bench, pass 2: the shipped rich config, the render slice, honest throughput.

Pass 1 (`probe_kage.py`) used `EnvConfig()`, whose python defaults are the *plain*
env (black background, no NPCs, no distractors, no filters). The package's own
`configs/default_config.yaml` is the rich one (image backgrounds + 27 sprite skins
+ NPCs + distractors + filters + lights), and that is what the benchmark uses.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

REPO = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/kage-bench"
OUT = os.environ.get("OUT", "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/kage2.json")
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
            per[k] = {"bytes": 0}; continue
        try:
            a = np.asarray(v)
        except Exception:
            per[k] = {"bytes": None}; continue
        per[k] = {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": int(a.nbytes)}
        tot += int(a.nbytes)
    return per, tot


def count_yaml_knobs(path):
    import yaml
    with open(path) as fh:
        d = yaml.safe_load(fh)
    leaves = []

    def walk(x, pre=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{pre}.{k}" if pre else k)
        else:
            leaves.append(pre)
    walk(d)
    physics_pref = ("physics", "episode_length", "forward_reward_scale", "jump_penalty",
                    "timestep_penalty", "idle_penalty", "dist_to_success")
    visual = [l for l in leaves if not l.startswith(physics_pref)]
    return {"total_leaves": len(leaves), "visual_leaves": len(visual),
            "physics_or_task_leaves": len(leaves) - len(visual),
            "top_level_sections": sorted(d.keys()),
            "visual_leaf_names": visual}


def bench(env, tag, batches=(1, 64, 256, 1024, 4096, 16384, 32768, 65536)):
    """Step throughput WITHOUT stacking the frames (pass 1 OOMed on its own output)."""
    import jax, jax.numpy as jnp
    T = 64

    def rollout(key):
        o, info = env.reset(key)
        s = info["state"]

        def body(carry, _):
            s, k = carry
            k, ka = jax.random.split(k)
            a = jax.random.randint(ka, (), 0, 8)
            o, r, te, tr, inf = env.step(s, a)
            return (inf["state"], k), r
        (s, k), rs = jax.lax.scan(body, (s, key), None, length=T)
        return rs.sum()

    f = jax.jit(jax.vmap(rollout))
    out = {}
    for B in batches:
        keys = jax.random.split(jax.random.PRNGKey(0), B)
        try:
            o = f(keys); o.block_until_ready()
        except Exception as e:
            out[B] = {"error": repr(e)[:200]}
            print(tag, "OOM/err", B, repr(e)[:160], flush=True)
            break
        reps = 3 if B >= 4096 else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            o = f(keys)
        o.block_until_ready()
        out[B] = B * T * reps / (time.perf_counter() - t0)
        print(tag, B, out[B], flush=True)
        del o
    return out


def main():
    os.chdir(REPO)                      # the YAML uses repo-relative asset paths
    import jax, jax.numpy as jnp
    from kage_bench import KAGE_Env, EnvConfig, load_config_from_yaml
    R["jax_devices"] = [str(d) for d in jax.devices()]
    R["gpu"] = jax.devices()[0].platform == "gpu"
    if not R["gpu"]:
        R["ERROR"] = "not on GPU"; return
    R["gpu_kind"] = jax.devices()[0].device_kind

    ycfg = os.path.join(REPO, "src/kage_bench/configs/default_config.yaml")
    R["yaml_knobs"] = count_yaml_knobs(ycfg)
    print("knobs", {k: v for k, v in R["yaml_knobs"].items() if k != "visual_leaf_names"}, flush=True)

    for tag, cfg in (("python_defaults", EnvConfig()),
                     ("shipped_yaml_rich", load_config_from_yaml(ycfg))):
        entry = {}
        try:
            env = KAGE_Env(cfg)
        except Exception as e:
            entry["error"] = repr(e)[:400]; R[tag] = entry
            print(tag, "BUILD FAILED", repr(e)[:300], flush=True); continue
        frame = cfg.H * cfg.W * 3
        key = jax.random.PRNGKey(0)
        obs, info = env.reset(key)
        st = info["state"]
        per, tot = state_bytes(st)
        entry["background_mode"] = cfg.background.mode
        entry["frame_bytes"] = frame
        entry["envstate_bytes"] = tot
        entry["ratio_envstate_over_frame"] = tot / frame
        entry["fields"] = per
        entry["biggest_fields"] = sorted(((v.get("bytes") or 0, k) for k, v in per.items()),
                                         reverse=True)[:8]
        # ---- the render slice: what a stable-envs env would actually store
        lay = np.asarray(st.layout_solid_mask)
        H, W = cfg.H, cfg.W
        window_bytes = H * W                      # the on-screen part of the mask, bool
        window_packed = H * W // 8
        scalars = sum(v.get("bytes") or 0 for k, v in per.items()
                      if k not in ("layout_solid_mask", "layout_ground_top",
                                   "layout_ground_bottom", "bg_image"))
        bg = per.get("bg_image", {}).get("bytes") or 0
        entry["render_slice"] = {
            "layout_window_bool_bytes": window_bytes,
            "layout_window_bitpacked_bytes": window_packed,
            "non_layout_non_bg_scalars_bytes": scalars,
            "bg_image_bytes": bg,
            "slice_bytes_with_bg": window_packed + scalars + bg,
            "slice_bytes_without_bg": window_packed + scalars,
            "ratio_slice_with_bg_over_frame": (window_packed + scalars + bg) / frame,
            "ratio_slice_without_bg_over_frame": (window_packed + scalars) / frame,
        }
        # ---- atlas / decoder
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
        atlas = {}
        for name in ("character_sprites", "npc_sprite_sets", "sticky_npc_sprite_sets",
                     "character_skin_sets", "background_images"):
            v = getattr(env, name, None)
            atlas[name] = None if v is None else {"bytes": walk(v), "n_arrays": count(v)}
        entry["atlas"] = atlas
        entry["atlas_total_bytes"] = sum((a or {}).get("bytes", 0) for a in atlas.values())
        # ---- across resets: is the decoder content?
        bgs, lays = [], []
        for s in range(8):
            _o, i2 = env.reset(jax.random.PRNGKey(s))
            b = getattr(i2["state"], "bg_image", None)
            bgs.append(None if b is None else np.asarray(b).copy())
            lays.append(np.asarray(i2["state"].layout_solid_mask).copy())
        entry["bg_image_present"] = bgs[0] is not None
        if bgs[0] is not None:
            entry["bg_image_bytes"] = int(bgs[0].nbytes)
            entry["bg_image_distinct_across_8_resets"] = len({b.tobytes() for b in bgs})
            entry["bg_image_blank"] = bool(bgs[0].max() == bgs[0].min())
        entry["layout_distinct_across_8_resets"] = len({l.tobytes() for l in lays})
        entry["atlas_identical_across_resets"] = True   # instance attrs, not resampled
        # ---- G1
        render = jax.jit(env.render)
        step = jax.jit(env.step)
        _o, i0 = env.reset(jax.random.PRNGKey(7))
        s0 = i0["state"]
        rng = np.random.RandomState(0)
        keep, frames, blank = [], [], 0
        s = s0
        for t in range(150):
            fimg = np.asarray(render(s))
            if fimg.max() == fimg.min():
                blank += 1
            if t % 11 == 0:
                keep.append(s); frames.append(fimg.copy())
            o, r, te, tr, inf = step(s, int(rng.randint(8)))
            s = inf["state"]
        s2 = keep[0]
        for t in range(29):
            o, r, te, tr, inf = step(s2, int(rng.randint(8)))
            s2 = inf["state"]
        entry["G1"] = {
            "frames": 150, "blank_frames": blank,
            "rerender_after_other_rollout_mismatch": sum(
                0 if np.array_equal(np.asarray(render(k), dtype=np.uint8), fr) else 1
                for k, fr in zip(keep, frames)),
            "mean_distinct_pixel_values": float(np.mean([len(np.unique(fr)) for fr in frames])),
        }
        entry["G4_step_sps"] = bench(env, tag)
        R[tag] = entry
        print(tag, {k: v for k, v in entry.items()
                    if k not in ("fields", "G4_step_sps", "atlas")}, flush=True)
        del env

    # decode baseline for a 128x128 frame
    import io
    from PIL import Image
    img = np.random.RandomState(0).randint(0, 255, (128, 128, 3)).astype(np.uint8)
    bio = io.BytesIO(); Image.fromarray(img).save(bio, format="JPEG", quality=90)
    raw = bio.getvalue()
    N = 3000
    t0 = time.perf_counter()
    for _ in range(N):
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    R["decode_jpeg_fps_one_core_128px"] = N / (time.perf_counter() - t0)
    print("decode", R["decode_jpeg_fps_one_core_128px"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        R["FATAL"] = traceback.format_exc()[-3000:]
        print(R["FATAL"], flush=True)
    finally:
        with open(OUT, "w") as fh:
            json.dump(R, fh, indent=1, default=jsonable)
        print("wrote", OUT)
