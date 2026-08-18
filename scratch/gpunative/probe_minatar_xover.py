"""MinAtar: pin the GPU-vs-official crossover, and widen the gymnax fidelity test."""
from __future__ import annotations
import json, os, time
import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/gpunative/minatar_xover.json"
GAMES = ["breakout", "freeway", "asterix", "seaquest", "space_invaders"]
R = {}


def jsonable(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))


def main():
    import jax, jax.numpy as jnp
    import seaborn as sns
    from minatar import Environment
    R["jax_devices"] = [str(d) for d in jax.devices()]
    R["gpu_kind"] = jax.devices()[0].device_kind
    assert jax.devices()[0].platform == "gpu", "not on GPU"

    BS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)
    xo = {}
    for g in GAMES:
        n = Environment(env_name=g).n_channels
        cm_np = np.array([(0.0, 0.0, 0.0)] + list(sns.color_palette("cubehelix", n)))
        w_np = np.arange(n).reshape(1, 1, 1, -1) + 1
        pal = jnp.asarray(cm_np * 255, jnp.float32)
        w = jnp.arange(1, n + 1, dtype=jnp.int32)

        @jax.jit
        def render(obs, pal=pal, w=w):
            return pal[jnp.max(obs.astype(jnp.int32) * w, axis=-1)].astype(jnp.uint8)

        gpu, cpu = {}, {}
        for B in BS:
            s = np.random.RandomState(0).rand(B, 10, 10, n) < 0.1
            f = cm_np[np.amax(s * w_np, 3)]
            reps = max(3, min(3000, int(3e6 / (B * 100))))
            t0 = time.perf_counter()
            for _ in range(reps):
                f = cm_np[np.amax(s * w_np, 3)]
            cpu[B] = B * reps / (time.perf_counter() - t0)

            o = jnp.asarray(s)
            fj = render(o); fj.block_until_ready()
            reps = max(3, min(3000, int(6e7 / (B * 100))))
            t0 = time.perf_counter()
            for _ in range(reps):
                fj = render(o)
            fj.block_until_ready()
            gpu[B] = B * reps / (time.perf_counter() - t0)
        peak_cpu = max(cpu.values())
        xover = next((B for B in BS if gpu[B] > peak_cpu), None)
        xover_pairwise = next((B for B in BS if gpu[B] > cpu[B]), None)
        xo[g] = {"cpu_official_numpy_fps": cpu, "gpu_gather_fps": gpu,
                 "cpu_peak_fps": peak_cpu,
                 "crossover_batch_vs_cpu_peak": xover,
                 "crossover_batch_pairwise": xover_pairwise,
                 "peak_gpu_fps": max(gpu.values()),
                 "peak_ratio_vs_cpu_peak": max(gpu.values()) / peak_cpu}
        print(g, "cpu_peak", round(peak_cpu), "xover", xover, "pairwise", xover_pairwise,
              "peak_ratio", round(max(gpu.values()) / peak_cpu, 1), flush=True)
    R["crossover"] = xo

    # ---- widened gymnax breakout fidelity
    import gymnax
    genv, gparams = gymnax.make("Breakout-MinAtar")
    gact = np.asarray(genv.action_set)
    step = jax.jit(genv.step)
    get_obs = jax.jit(genv.get_obs)
    tot = bad = 0
    eps = 0
    lens = []
    for trial in range(120):
        oenv = Environment(env_name="breakout", sticky_action_prob=0.0)
        oenv.seed(trial); oenv.reset()
        e = oenv.env
        _o0, st0 = genv.reset(jax.random.PRNGKey(trial), gparams)

        def like(name, val):
            return jnp.asarray(val, jnp.asarray(getattr(st0, name)).dtype)
        st = st0.replace(ball_y=like("ball_y", e.ball_y), ball_x=like("ball_x", e.ball_x),
                         ball_dir=like("ball_dir", e.ball_dir), pos=like("pos", e.pos),
                         brick_map=like("brick_map", np.asarray(e.brick_map)),
                         strike=like("strike", e.strike), last_y=like("last_y", e.last_y),
                         last_x=like("last_x", e.last_x), time=like("time", 0),
                         terminal=like("terminal", False))
        rng = np.random.RandomState(9000 + trial)
        key = jax.random.PRNGKey(trial)
        n = 0
        for t in range(1000):
            ai = int(rng.randint(len(gact)))
            if not np.array_equal(oenv.state(), np.asarray(get_obs(st)).astype(bool)):
                bad += 1
            n += 1
            _, d = oenv.act(int(gact[ai]))
            key, k = jax.random.split(key)
            _o, st, r, dd, _i = step(k, st, ai, gparams)
            if d or bool(dd):
                break
        tot += n; eps += 1; lens.append(n)
    R["gymnax_breakout_fidelity"] = {"episodes": eps, "frames_compared": tot,
                                     "frames_mismatched": bad,
                                     "exact_frac": (tot - bad) / tot,
                                     "mean_episode_len": float(np.mean(lens))}
    print("FIDELITY", R["gymnax_breakout_fidelity"], flush=True)


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
