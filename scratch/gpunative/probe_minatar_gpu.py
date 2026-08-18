"""MinAtar, GPU side + gymnax fidelity. Companion to probe_minatar.py."""
from __future__ import annotations
import json, os, time, sys
import numpy as np

OUT = os.environ.get("OUT", "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/minatar_gpu.json")
R = {}
GAMES = ["breakout", "freeway", "asterix", "seaquest", "space_invaders"]
IDS = {"breakout": "Breakout-MinAtar", "freeway": "Freeway-MinAtar",
       "asterix": "Asterix-MinAtar", "space_invaders": "SpaceInvaders-MinAtar"}


def jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))


def gmake(g):
    import gymnax
    if g in IDS:
        return gymnax.make(IDS[g])
    if g == "seaquest":
        from gymnax.environments.minatar.seaquest import MinSeaquest
        e = MinSeaquest()
        return e, e.default_params
    raise KeyError(g)


# ---------------------------------------------------------- registration coverage
def coverage():
    import gymnax.registration as reg
    from importlib.metadata import version as _v
    return {"gymnax_version": _v("gymnax"), "minatar_version": _v("minatar"),
            "registered_minatar_ids": [e for e in reg.registered_envs if "MinAtar" in e],
            "modules_present": sorted(os.listdir(os.path.dirname(
                __import__("gymnax.environments.minatar", fromlist=["x"]).__file__))),
            "seaquest_registered": "Seaquest-MinAtar" in reg.registered_envs}


# ---------------------------------------------------------- G2 gymnax state bytes
def g2_gymnax():
    import jax, jax.numpy as jnp
    out = {}
    for g in GAMES:
        try:
            env, params = gmake(g)
            obs, st = env.reset(jax.random.PRNGKey(0), params)
        except Exception as e:
            out[g] = {"error": repr(e)[:300]}
            print("G2gx SKIP", g, repr(e)[:160], flush=True); continue
        per = {}
        tot = 0
        for k in st.__dataclass_fields__:
            a = np.asarray(getattr(st, k))
            per[k] = {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": int(a.nbytes)}
            tot += int(a.nbytes)
        oa = np.asarray(obs)
        out[g] = {"envstate_bytes": tot, "fields": per,
                  "obs_shape": list(oa.shape), "obs_dtype": str(oa.dtype),
                  "obs_bytes_as_shipped": int(oa.nbytes),
                  "frame_bytes_uint8_10x10x3": 300,
                  "ratio_envstate_over_frame_u8": tot / 300.0}
        print("G2gx", g, tot, flush=True)
    return out


# ---------------------------------------------------------- gymnax fidelity
def breakout_fidelity(T=400, trials=8):
    """Exact obs-grid parity between gymnax MinBreakout and the official numpy MinAtar,
    with sticky actions off and the ball start aligned. Breakout is the one MinAtar game
    whose post-reset dynamics are fully deterministic, so this can be an exact test."""
    import jax, jax.numpy as jnp
    from minatar import Environment
    from gymnax.environments.minatar.breakout import MinBreakout, EnvState
    genv, gparams = gmake("breakout")
    # gymnax uses the MINIMAL action set by default: ['n','l','r'] -> full ids [0,1,3]
    gact = np.asarray(genv.action_set)
    res = []
    for trial in range(trials):
        oenv = Environment(env_name="breakout", sticky_action_prob=0.0)
        oenv.seed(trial)
        oenv.reset()
        e = oenv.env
        # align gymnax state to the official one, keeping gymnax's own dtypes so that
        # its auto-reset lax.select does not trip on a dtype mismatch
        _obs0, st0 = genv.reset(jax.random.PRNGKey(trial), gparams)
        def like(name, val):
            ref = getattr(st0, name)
            return jnp.asarray(val, jnp.asarray(ref).dtype)
        st = st0.replace(ball_y=like("ball_y", e.ball_y), ball_x=like("ball_x", e.ball_x),
                         ball_dir=like("ball_dir", e.ball_dir), pos=like("pos", e.pos),
                         brick_map=like("brick_map", np.asarray(e.brick_map)),
                         strike=like("strike", e.strike), last_y=like("last_y", e.last_y),
                         last_x=like("last_x", e.last_x), time=like("time", 0),
                         terminal=like("terminal", False))
        rng = np.random.RandomState(1000 + trial)
        step = jax.jit(genv.step)
        get_obs = jax.jit(genv.get_obs)
        mism = 0
        first_mism = None
        n = 0
        key = jax.random.PRNGKey(trial)
        for t in range(T):
            ai = int(rng.randint(len(gact)))
            o_ref = oenv.state()
            o_gx = np.asarray(get_obs(st)).astype(bool)
            n += 1
            if not np.array_equal(o_ref, o_gx):
                mism += 1
                if first_mism is None:
                    first_mism = t
            _, d = oenv.act(int(gact[ai]))
            key, k = jax.random.split(key)
            _o, st, r, dd, _i = step(k, st, ai, gparams)
            if d or bool(dd):
                break
        res.append({"trial": trial, "compared": n, "mismatch": mism, "first_mismatch_step": first_mism})
    tot = sum(r["compared"] for r in res)
    bad = sum(r["mismatch"] for r in res)
    return {"per_trial": res, "frames_compared": tot, "frames_mismatched": bad,
            "exact_frac": (tot - bad) / tot if tot else None}


# ---------------------------------------------------------- G4 GPU
def g4_gpu(batches=(1, 64, 256, 1024, 4096, 16384, 65536)):
    import jax, jax.numpy as jnp
    import seaborn as sns
    dev = jax.devices()
    out = {"jax_devices": [str(d) for d in dev], "jax_version": jax.__version__,
           "gpu_name": None}
    try:
        out["gpu_name"] = dev[0].device_kind
    except Exception:
        pass
    if dev[0].platform != "gpu":
        out["ERROR"] = "jax is NOT on a GPU -- numbers refused"
        return out

    # --- reference: the OFFICIAL numpy renderer, batched (the fair CPU baseline)
    npref = {}
    for g in GAMES:
        from minatar import Environment
        n = Environment(env_name=g).n_channels
        cm = np.array([(0.0, 0.0, 0.0)] + list(sns.color_palette("cubehelix", n)))
        w = np.arange(n).reshape(1, 1, 1, -1) + 1
        npref[g] = {}
        for B in (1, 64, 256, 1024, 4096):
            s = np.random.RandomState(0).rand(B, 10, 10, n) < 0.1
            f = cm[np.amax(s * w, 3)]
            reps = max(3, min(2000, int(2e6 / (B * 100))))
            t0 = time.perf_counter()
            for _ in range(reps):
                f = cm[np.amax(s * w, 3)]
            npref[g][B] = B * reps / (time.perf_counter() - t0)
        print("npref", g, npref[g], flush=True)
    out["official_numpy_render_fps_batched"] = npref
    out["official_numpy_render_peak_fps"] = {g: max(v.values()) for g, v in npref.items()}

    # --- GPU gather renderer
    rend = {}
    for g in GAMES:
        from minatar import Environment
        n = Environment(env_name=g).n_channels
        pal = jnp.asarray(np.array([(0.0, 0.0, 0.0)] + list(sns.color_palette("cubehelix", n))) * 255,
                          jnp.float32)
        w = jnp.arange(1, n + 1, dtype=jnp.int32)

        @jax.jit
        def render(obs, pal=pal, w=w):
            return pal[jnp.max(obs.astype(jnp.int32) * w, axis=-1)].astype(jnp.uint8)

        rend[g] = {}
        for B in batches:
            o = jax.random.bernoulli(jax.random.PRNGKey(0), 0.1, (B, 10, 10, n))
            f = render(o); f.block_until_ready()
            reps = max(3, min(3000, int(6e7 / (B * 100))))
            t0 = time.perf_counter()
            for _ in range(reps):
                f = render(o)
            f.block_until_ready()
            rend[g][B] = B * reps / (time.perf_counter() - t0)
        print("gpurender", g, rend[g], flush=True)
    out["gpu_render_fps"] = rend
    out["crossover_batch_vs_official_peak"] = {
        g: next((B for B in sorted(rend[g]) if rend[g][B] > out["official_numpy_render_peak_fps"][g]), None)
        for g in GAMES}

    # --- gymnax step throughput
    stepper = {}
    for g in GAMES:
        try:
            env, params = gmake(g)
            env.reset(jax.random.PRNGKey(0), params)
        except Exception as e:
            stepper[g] = {"error": repr(e)[:200]}
            print("gymnaxstep SKIP", g, repr(e)[:160], flush=True); continue
        T = 64

        def rollout(key):
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

        f = jax.jit(jax.vmap(rollout))
        stepper[g] = {}
        for B in batches:
            if B > 16384:
                continue
            keys = jax.random.split(jax.random.PRNGKey(0), B)
            try:
                o = f(keys); o.block_until_ready()
            except Exception as e:
                stepper[g][B] = {"error": repr(e)[:200]}; continue
            reps = 3 if B >= 4096 else 8
            t0 = time.perf_counter()
            for _ in range(reps):
                o = f(keys)
            o.block_until_ready()
            stepper[g][B] = B * T * reps / (time.perf_counter() - t0)
            del o
        print("gymnaxstep", g, stepper[g], flush=True)
    out["gymnax_step_plus_obs_sps"] = stepper

    # --- official CPU step throughput for the same games (one core)
    cpu = {}
    from minatar import Environment
    for g in GAMES:
        e = Environment(env_name=g); e.seed(0); e.reset()
        rng = np.random.RandomState(0)
        N = 20000
        for i in range(200):
            e.act(int(rng.randint(6)))
        t0 = time.perf_counter()
        for i in range(N):
            _, d = e.act(int(rng.randint(6)))
            s = e.state()
            if d:
                e.reset()
        cpu[g] = N / (time.perf_counter() - t0)
    out["official_cpu_step_plus_state_sps_one_core"] = cpu
    return out


if __name__ == "__main__":
    R["coverage"] = coverage()
    R["G2_gymnax"] = g2_gymnax()
    R["gymnax_breakout_fidelity"] = breakout_fidelity()
    print("FIDELITY", R["gymnax_breakout_fidelity"]["frames_compared"],
          R["gymnax_breakout_fidelity"]["frames_mismatched"], flush=True)
    R["G4_gpu"] = g4_gpu()
    with open(OUT, "w") as fh:
        json.dump(R, fh, indent=1, default=jsonable)
    print("wrote", OUT)
