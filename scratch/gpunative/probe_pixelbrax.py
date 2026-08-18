"""PixelBrax (arXiv 2502.00021) against the four gates."""
from __future__ import annotations
import json, os, sys, time
import numpy as np

HERE = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative"
# the repo root is itself named "pixelbrax" and has no __init__.py, so having the
# scratch dir on sys.path makes `import pixelbrax` resolve to an empty namespace
# package. Drop it, then add the repo root (for the real package) and the vendored
# brax fork (the package imports top-level `brax`, and the wheel does not ship it).
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.abspath(HERE)]
sys.path.insert(0, os.path.join(HERE, "pixelbrax"))
sys.path.insert(0, os.path.join(HERE, "pixelbrax", "pixelbrax", "brax"))
OUT = os.environ.get("OUT", os.path.join(HERE, "pixelbrax.json"))
R = {}


def jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)


def pytree_bytes(t):
    import jax
    leaves, _ = jax.tree_util.tree_flatten(t)
    return int(sum(np.asarray(l).nbytes for l in leaves))


def main():
    import jax, jax.numpy as jnp
    import brax
    from brax import base
    R["brax_version"] = brax.__version__
    R["jax_version"] = jax.__version__
    R["jax_devices"] = [str(d) for d in jax.devices()]
    R["gpu"] = jax.devices()[0].platform == "gpu"
    if not R["gpu"]:
        R["ERROR"] = "not on GPU"
        return
    import pixelbrax

    HW = 84
    for env_name in ("halfcheetah", "hopper", "walker2d", "ant"):
        entry = {}
        try:
            envs, _a, _b = pixelbrax.make(backend="spring", env_name=env_name, n_envs=16,
                                          seed=0, hw=HW, distractor=None,
                                          video_path=None, video_set="train",
                                          return_float32=False)
        except Exception as e:
            entry["error"] = repr(e)[:600]
            R[env_name] = entry
            print("MAKE FAILED", env_name, repr(e)[:600], flush=True)
            continue
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 16)
        try:
            st = envs.reset(keys)
        except Exception as e:
            entry["reset_error"] = repr(e)[:600]
            R[env_name] = entry
            print("RESET FAILED", env_name, repr(e)[:600], flush=True)
            continue
        obs = np.asarray(st.obs)
        entry["obs_shape"] = list(obs.shape)
        entry["obs_dtype"] = str(obs.dtype)
        entry["obs_blank"] = bool(obs[0].max() == obs[0].min())
        entry["obs_unique_values_env0"] = int(len(np.unique(obs[0])))
        # ---- G3: geom counts / decoder stability across resets
        sys_ = envs.sys if hasattr(envs, "sys") else None
        counts = []
        for s in range(4):
            k = jax.random.split(jax.random.PRNGKey(100 + s), 16)
            st2 = envs.reset(k)
            g = envs.sys.geoms if hasattr(envs, "sys") else []
            counts.append([(type(b).__name__, int(len(b.friction))) for b in g])
        entry["geom_batches_per_reset"] = counts
        entry["decoder_stable_across_resets"] = all(c == counts[0] for c in counts)
        entry["n_geoms_total"] = sum(n for _t, n in counts[0])
        entry["n_mesh_geoms"] = sum(n for t, n in counts[0] if t in ("Mesh", "Convex"))
        # ---- G2: pipeline_state bytes vs frame
        ps = st.pipeline_state
        per = {}
        for f in ps.__dataclass_fields__ if hasattr(ps, "__dataclass_fields__") else []:
            v = getattr(ps, f, None)
            try:
                per[f] = pytree_bytes(v)
            except Exception:
                per[f] = None
        tot = pytree_bytes(ps)
        frame = HW * HW * 3
        entry["pipeline_state_bytes_batch16"] = tot
        entry["pipeline_state_bytes_per_env"] = tot / 16
        entry["pipeline_state_fields_bytes_batch16"] = per
        entry["frame_bytes_uint8"] = frame
        entry["ratio_pipeline_state_over_frame"] = (tot / 16) / frame
        try:
            q = np.asarray(ps.q); qd = np.asarray(ps.qd)
            entry["q_qd_bytes_per_env"] = int((q[0].nbytes + qd[0].nbytes))
            entry["ratio_q_qd_over_frame"] = (q[0].nbytes + qd[0].nbytes) / frame
        except Exception as e:
            entry["q_qd_error"] = repr(e)[:200]
        # ---- G1: render twice, and after a different history
        try:
            rp = envs.render_pixels if hasattr(envs, "render_pixels") else None
        except Exception:
            rp = None
        acts = jax.random.uniform(jax.random.PRNGKey(3), (16,) + tuple(envs.action_size
                                                                      if isinstance(envs.action_size, tuple)
                                                                      else (envs.action_size,)),
                                  minval=-1, maxval=1)
        st_a = envs.step(st, acts)
        f1 = np.asarray(st_a.obs).copy()
        st_b = envs.step(st, acts)
        f2 = np.asarray(st_b.obs).copy()
        entry["render_determinism_mismatch"] = int(not np.array_equal(f1, f2))
        entry["step_obs_blank"] = bool(f1[0].max() == f1[0].min())
        entry["step_obs_unique_values_env0"] = int(len(np.unique(f1[0])))
        R[env_name] = entry
        print(env_name, {k: v for k, v in entry.items()
                         if k not in ("pipeline_state_fields_bytes_batch16", "geom_batches_per_reset")},
              flush=True)
        del envs

    # ---- G4: throughput of step (physics + render) vs physics alone
    g4 = {}
    for n in (1, 16, 64, 256, 1024):
        try:
            envs, _a, _b = pixelbrax.make(backend="spring", env_name="halfcheetah", n_envs=n,
                                          seed=0, hw=HW, distractor=None, video_path=None,
                                          video_set="train", return_float32=False)
            keys = jax.random.split(jax.random.PRNGKey(0), n)
            st = envs.reset(keys)
            acts = jax.random.uniform(jax.random.PRNGKey(1), (n, envs.action_size), minval=-1, maxval=1)
            st2 = envs.step(st, acts); jax.block_until_ready(st2.obs)
            reps = 3 if n >= 256 else 10
            t0 = time.perf_counter()
            for _ in range(reps):
                st2 = envs.step(st, acts)
            jax.block_until_ready(st2.obs)
            g4[n] = n * reps / (time.perf_counter() - t0)
            print("G4 step+render", n, g4[n], flush=True)
            del envs
        except Exception as e:
            g4[n] = {"error": repr(e)[:300]}
            print("G4 err", n, repr(e)[:300], flush=True)
            break
    R["G4_step_plus_render_sps_halfcheetah_84px"] = g4

    # physics-only baseline: stock brax
    try:
        from brax import envs as bxenvs
        be = bxenvs.create("halfcheetah", backend="spring", batch_size=None)
        jstep = jax.jit(jax.vmap(be.step))
        jreset = jax.jit(jax.vmap(be.reset))
        phys = {}
        for n in (1, 16, 64, 256, 1024):
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
            print("G4 physics only", n, phys[n], flush=True)
        R["G4_physics_only_sps_halfcheetah"] = phys
        R["render_share_of_step"] = {
            n: (1.0 / g4[n] - 1.0 / phys[n]) / (1.0 / g4[n])
            for n in phys if isinstance(g4.get(n), float) and g4[n] > 0}
    except Exception as e:
        R["G4_physics_only_error"] = repr(e)[:400]


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        R["FATAL"] = traceback.format_exc()[-3000:]
        print(R["FATAL"], flush=True)
    finally:
        with open(OUT, "w") as fh:
            json.dump(R, fh, indent=1, default=jsonable)
        print("wrote", OUT)
