"""Octax pass 2: push the batch, and split the render out of the step."""
from __future__ import annotations
import json, os, time
import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/octax2.json"
R = {}


def jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def main():
    import jax, jax.numpy as jnp
    from octax.environments import create_environment
    R["jax_devices"] = [str(d) for d in jax.devices()]
    R["gpu_kind"] = jax.devices()[0].device_kind
    assert jax.devices()[0].platform == "gpu"

    out = {}
    for game in ("pong", "brix", "tetris"):
        env, _ = create_environment(game)
        T = 32
        st0, _o, _i = env.reset(jax.random.PRNGKey(0))

        def rollout(st, key):
            def body(carry, _):
                s, k = carry
                k, ka = jax.random.split(k)
                a = jax.random.randint(ka, (), 0, env.num_actions)
                s, o, r, te, tr, i = env.step(s, a)
                return (s, k), r          # do NOT stack frames: that OOMs on its own output
            (s, k), rs = jax.lax.scan(body, (st, key), None, length=T)
            return rs.sum()

        f = jax.jit(jax.vmap(rollout))
        per = {}
        for B in (1, 64, 256, 1024, 4096, 16384, 65536):
            keys = jax.random.split(jax.random.PRNGKey(1), B)
            sts = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (B,) + jnp.shape(x)).copy()
                if not isinstance(x, bool) else x, st0)
            try:
                o = f(sts, keys); jax.block_until_ready(o)
            except Exception as e:
                per[B] = {"error": repr(e)[:200]}
                print(game, "err", B, repr(e)[:160], flush=True)
                break
            reps = 3 if B >= 4096 else 6
            t0 = time.perf_counter()
            for _ in range(reps):
                o = f(sts, keys)
            jax.block_until_ready(o)
            per[B] = B * T * reps / (time.perf_counter() - t0)
            print(game, B, per[B], flush=True)
            del o, sts
        ipst = env.instructions_per_step * env.frame_skip
        out[game] = {"env_steps_per_s": per,
                     "instructions_per_env_step": ipst,
                     "frames_per_s": {k: (v * env.frame_skip if isinstance(v, float) else v)
                                      for k, v in per.items()},
                     "chip8_instructions_per_s": {k: (v * ipst if isinstance(v, float) else v)
                                                  for k, v in per.items()},
                     "peak_env_steps_per_s": max((v for v in per.values() if isinstance(v, float)),
                                                 default=None)}
    R["throughput"] = out
    print(json.dumps({g: v["peak_env_steps_per_s"] for g, v in out.items()}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        R["FATAL"] = traceback.format_exc()[-2500:]
        print(R["FATAL"], flush=True)
    finally:
        with open(OUT, "w") as fh:
            json.dump(R, fh, indent=1, default=jsonable)
        print("wrote", OUT)
