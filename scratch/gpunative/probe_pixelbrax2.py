"""PixelBrax pass 2: block on the PIXELS (pass 1 blocked on the proprio obs, and
PixelEnv.step dispatches the renderer as a separate jitted call), pin the render
share, and check the frame is not blank."""
from __future__ import annotations
import json, os, sys, time, io
import numpy as np

HERE = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative"
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.abspath(HERE)]
sys.path.insert(0, os.path.join(HERE, "pixelbrax"))
sys.path.insert(0, os.path.join(HERE, "pixelbrax", "pixelbrax", "brax"))
OUT = os.path.join(HERE, "pixelbrax2.json")
R = {}


def jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def main():
    import jax, jax.numpy as jnp
    import brax
    import pixelbrax
    R["brax_version"] = brax.__version__
    R["jax_version"] = jax.__version__
    R["gpu_kind"] = jax.devices()[0].device_kind
    assert jax.devices()[0].platform == "gpu"
    HW = 84
    frame_bytes = HW * HW * 3

    # ---- backends that actually build
    backends = {}
    for env_name in ("halfcheetah", "walker2d", "ant", "hopper", "humanoid", "reacher"):
        for backend in ("spring", "positional", "generalized"):
            try:
                e, _s, _r = pixelbrax.make(backend=backend, env_name=env_name, n_envs=2,
                                           seed=0, hw=HW, distractor=None, video_path=None,
                                           video_set="train", return_float32=False)
                backends.setdefault(env_name, []).append(backend)
                del e
            except Exception as ex:
                backends.setdefault(env_name + "_errors", []).append(
                    f"{backend}: {repr(ex)[:120]}")
    R["backends_that_build"] = backends
    print("backends", backends, flush=True)

    # ---- G1 + frame sanity on halfcheetah
    envs, st0, _r = pixelbrax.make(backend="spring", env_name="halfcheetah", n_envs=8,
                                   seed=0, hw=HW, distractor=None, video_path=None,
                                   video_set="train", return_float32=False)
    px = np.asarray(st0.pixels)
    R["pixels_shape"] = list(px.shape)
    R["pixels_dtype"] = str(px.dtype)
    R["pixels_blank_env0"] = bool(px[0].max() == px[0].min())
    R["pixels_unique_values_env0"] = int(len(np.unique(px[0])))
    R["pixels_bytes_per_env_as_shipped"] = int(px[0].nbytes)
    R["single_frame_bytes"] = frame_bytes
    print("pixels", R["pixels_shape"], R["pixels_dtype"], "blank", R["pixels_blank_env0"],
          "uniq", R["pixels_unique_values_env0"], flush=True)

    acts = jax.random.uniform(jax.random.PRNGKey(3), (8, envs.action_size), minval=-1, maxval=1)
    a1 = np.asarray(envs.step(st0, acts).pixels).copy()
    a2 = np.asarray(envs.step(st0, acts).pixels).copy()
    R["G1_render_twice_pixel_mismatch"] = int(not np.array_equal(a1, a2))
    # same pipeline_state reached through a different history
    s = st0
    for _ in range(7):
        s = envs.step(s, acts)
    b1 = np.asarray(envs.step(st0, acts).pixels).copy()
    R["G1_render_after_other_rollout_mismatch"] = int(not np.array_equal(a1, b1))
    print("G1", R["G1_render_twice_pixel_mismatch"], R["G1_render_after_other_rollout_mismatch"], flush=True)
    del envs

    # ---- G4: step+render blocking on PIXELS, and physics-only for the same env
    g4 = {}
    for n in (1, 16, 64, 256, 1024, 2048):
        try:
            envs, st, _r = pixelbrax.make(backend="spring", env_name="halfcheetah", n_envs=n,
                                          seed=0, hw=HW, distractor=None, video_path=None,
                                          video_set="train", return_float32=False)
            a = jax.random.uniform(jax.random.PRNGKey(1), (n, envs.action_size), minval=-1, maxval=1)
            s2 = envs.step(st, a); jax.block_until_ready(s2.pixels)
            reps = 3 if n >= 256 else 10
            t0 = time.perf_counter()
            for _ in range(reps):
                s2 = envs.step(st, a)
            jax.block_until_ready(s2.pixels)
            g4[n] = n * reps / (time.perf_counter() - t0)
            print("step+render (block on pixels)", n, g4[n], flush=True)
            del envs
        except Exception as e:
            g4[n] = {"error": repr(e)[:250]}
            print("err", n, repr(e)[:200], flush=True)
            break
    R["G4_step_plus_render_sps"] = g4

    from brax import envs as bxenvs
    be = bxenvs.create("halfcheetah", backend="spring", batch_size=None)
    jstep = jax.jit(jax.vmap(be.step))
    jreset = jax.jit(jax.vmap(be.reset))
    phys = {}
    for n in (1, 16, 64, 256, 1024, 2048):
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        s = jreset(keys)
        a = jax.random.uniform(jax.random.PRNGKey(1), (n, be.action_size), minval=-1, maxval=1)
        s2 = jstep(s, a); jax.block_until_ready(s2.obs)
        reps = 5 if n >= 256 else 20
        t0 = time.perf_counter()
        for _ in range(reps):
            s2 = jstep(s, a)
        jax.block_until_ready(s2.obs)
        phys[n] = n * reps / (time.perf_counter() - t0)
        print("physics only", n, phys[n], flush=True)
    R["G4_physics_only_sps"] = phys
    R["render_share_of_step"] = {n: (1 / g4[n] - 1 / phys[n]) / (1 / g4[n])
                                 for n in phys if isinstance(g4.get(n), float)}
    print("render share", R["render_share_of_step"], flush=True)

    # ---- decode baseline at 84x84 on this node
    from PIL import Image
    img = np.random.RandomState(0).randint(0, 255, (HW, HW, 3)).astype(np.uint8)
    bio = io.BytesIO(); Image.fromarray(img).save(bio, format="JPEG", quality=90)
    raw = bio.getvalue()
    N = 4000
    t0 = time.perf_counter()
    for _ in range(N):
        np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    dec = N / (time.perf_counter() - t0)
    R["decode_jpeg_fps_one_core_84px"] = dec
    R["decode_jpeg_bytes"] = len(raw)
    peak = max(v for v in g4.values() if isinstance(v, float))
    R["cores_of_decode_replaced"] = peak / dec
    print("decode", dec, "cores replaced", R["cores_of_decode_replaced"], flush=True)


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
