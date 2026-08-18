"""Render share of an env step, without the stacking artifact.

Two variants of the same rollout, differing only in whether the observation is
consumed:
  * `obs_live`   -- the scan accumulates `obs.sum()`, so XLA must materialise every
                    frame, but only a scalar is written out;
  * `obs_dce`    -- the scan returns the reward only, so the renderer is dead code.
The gap is the renderer. Stacking the frames (what a naive harness does) also pays
for writing B*T*frame bytes to HBM and overstates the renderer.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

WHICH = sys.argv[1]
OUT = f"/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/rendershare_{WHICH}.json"
R = {}


def jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def timeit(f, args, B, T, reps):
    import jax
    o = f(*args); jax.block_until_ready(o)
    t0 = time.perf_counter()
    for _ in range(reps):
        o = f(*args)
    jax.block_until_ready(o)
    return B * T * reps / (time.perf_counter() - t0)


def kage():
    os.chdir("/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/kage-bench")
    import jax, jax.numpy as jnp
    from kage_bench import KAGE_Env, EnvConfig, load_config_from_yaml
    R["gpu_kind"] = jax.devices()[0].device_kind
    cfgs = {"python_defaults": EnvConfig(),
            "shipped_yaml_rich": load_config_from_yaml(
                "src/kage_bench/configs/default_config.yaml")}
    T = 64
    for tag, cfg in cfgs.items():
        env = KAGE_Env(cfg)

        def mk(live):
            def rollout(key):
                o, info = env.reset(key)
                s = info["state"]

                def body(carry, _):
                    s, k, acc = carry
                    k, ka = jax.random.split(k)
                    a = jax.random.randint(ka, (), 0, 8)
                    o, r, te, tr, inf = env.step(s, a)
                    acc = acc + (o.astype(jnp.float32).sum() if live else r)
                    return (inf["state"], k, acc), None
                (s, k, acc), _ = jax.lax.scan(body, (s, key, jnp.float32(0)), None, length=T)
                return acc
            return jax.jit(jax.vmap(rollout))

        live, dce = mk(True), mk(False)
        per = {}
        for B in (256, 1024, 4096, 16384, 32768):
            keys = jax.random.split(jax.random.PRNGKey(0), B)
            reps = 3 if B >= 4096 else 5
            try:
                a = timeit(live, (keys,), B, T, reps)
                b = timeit(dce, (keys,), B, T, reps)
            except Exception as e:
                per[B] = {"error": repr(e)[:200]}; break
            per[B] = {"obs_live_sps": a, "obs_dce_sps": b,
                      "render_share_of_step": (1 / a - 1 / b) / (1 / a)}
            print(tag, B, per[B], flush=True)
        R[tag] = per
        del env


def octax():
    import jax, jax.numpy as jnp
    from octax.environments import create_environment
    R["gpu_kind"] = jax.devices()[0].device_kind
    T = 32
    for game in ("pong", "brix"):
        env, _ = create_environment(game)
        st0, _o, _i = env.reset(jax.random.PRNGKey(0))

        def mk(live):
            def rollout(st, key):
                def body(carry, _):
                    s, k, acc = carry
                    k, ka = jax.random.split(k)
                    a = jax.random.randint(ka, (), 0, env.num_actions)
                    s, o, r, te, tr, i = env.step(s, a)
                    acc = acc + (o.astype(jnp.float32).sum() if live else r)
                    return (s, k, acc), None
                (s, k, acc), _ = jax.lax.scan(body, (st, key, jnp.float32(0)), None, length=T)
                return acc
            return jax.jit(jax.vmap(rollout))

        live, dce = mk(True), mk(False)
        per = {}
        for B in (256, 1024, 4096, 16384, 65536):
            keys = jax.random.split(jax.random.PRNGKey(1), B)
            sts = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (B,) + jnp.shape(x)).copy()
                if not isinstance(x, bool) else x, st0)
            reps = 3 if B >= 4096 else 6
            try:
                a = timeit(live, (sts, keys), B, T, reps)
                b = timeit(dce, (sts, keys), B, T, reps)
            except Exception as e:
                per[B] = {"error": repr(e)[:200]}; break
            per[B] = {"obs_live_sps": a, "obs_dce_sps": b,
                      "render_share_of_step": (1 / a - 1 / b) / (1 / a)}
            print(game, B, per[B], flush=True)
            del sts
        R[game] = per


if __name__ == "__main__":
    try:
        {"kage": kage, "octax": octax}[WHICH]()
    except Exception:
        import traceback
        R["FATAL"] = traceback.format_exc()[-2500:]
        print(R["FATAL"], flush=True)
    finally:
        with open(OUT, "w") as fh:
            json.dump(R, fh, indent=1, default=jsonable)
        print("wrote", OUT)
