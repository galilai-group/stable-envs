"""Fidelity and throughput of the batched GPU ``step``, per env, against the official CPU simulator.

    python scratch/step_bench.py --what fidelity|graph|bench|ulp|all

Everything here is measurement, not assertion: the numbers it prints are the ones quoted in the env
docstrings, so they have to be reproducible from one command.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

CARRY_NONE_CODE = 1                #: stable_envs.envs.minigrid.CARRY_NONE — "empty hands"
CLASSIC = ["CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous"]
GRID = ["MiniGridEmpty", "MiniGridDoorKey", "MiniGridFourRooms", "MiniGridLavaCrossing",
        "MiniGridKeyCorridor", "BabyAIGoToLocal"]


def fidelity(names, steps: int, episodes: int) -> dict:
    out = {}
    for n in names:
        env = se.make(n, batch=4, device="cuda")
        t0 = time.perf_counter()
        r = C.check_step_matches_reference(env, steps=steps, episodes=episodes)
        r["wall_s"] = time.perf_counter() - t0
        out[n] = r
        print(f"\n{n}: {json.dumps(r, indent=None)}", flush=True)
        del env
        torch.cuda.empty_cache()
    return out


def graph_vs_eager(names, steps: int = 40, batch: int = 64) -> dict:
    """Both paths run the same ``_advance``; this checks that the graph really replays it."""
    out = {}
    for n in names:
        a = se.make(n, batch=batch, device="cuda", seed=0)
        b = se.make(n, batch=batch, device="cuda", seed=0)
        b.use_step_graph = False
        st = a.sample_states(batch, np.random.default_rng(0))
        a.set_state(st)
        b.set_state(st)
        rng = np.random.default_rng(1)
        worst = 0.0
        for _ in range(steps):
            act = torch.as_tensor(rng.uniform(a.action_low, a.action_high, (batch, a.action_dim)),
                                  dtype=torch.float32, device="cuda")
            sa = a.step(act).clone()
            sb = b.step(act).clone()
            worst = max(worst, float((sa - sb).abs().max()))
        g = getattr(a, "_step_graph", None)
        err = getattr(a, "_step_graph_error", "")
        out[n] = {"captured": g is not None, "error": err,
                  "max_abs_diff": worst, "identical": worst == 0.0}
        print(f"{n:<26s} graph_captured={out[n]['captured']} identical={out[n]['identical']} "
              f"{out[n]['error']}", flush=True)
        del a, b
        torch.cuda.empty_cache()
    return out


def cpu_env_steps_per_s(name: str, steps: int = 2000) -> float:
    """One official CPU env, stepped in a loop. The honest per-env-step baseline."""
    cls = se.REGISTRY[name]
    ref = cls.reference_env()
    rng = np.random.default_rng(0)
    acts = cls.step_reference_actions(rng, steps, 1) if hasattr(cls, "step_reference_actions") else \
        C._default_actions(cls, rng, steps, 1)
    reset = getattr(cls, "step_reference_reset", None)
    if reset is not None:
        reset(ref, 0)
    else:
        ref.reset(seed=0)
    advance = getattr(cls, "reference_step", None) or (lambda r, a: r.step(a))
    advance(ref, acts[0])                                   # warm up
    t0 = time.perf_counter()
    for a in acts:
        advance(ref, a)
    dt = time.perf_counter() - t0
    if hasattr(ref, "close"):
        ref.close()
    return steps / dt


def bench(names, batches, reps: int = 30, cpu: bool = True) -> dict:
    out = {}
    for n in names:
        rows = []
        for B in batches:
            e = se.make(n, batch=B, device="cuda")
            e.set_state(e.sample_states(B, np.random.default_rng(0)))
            act = torch.as_tensor(
                np.random.default_rng(0).uniform(e.action_low, e.action_high, (B, e.action_dim)),
                dtype=torch.float32, device="cuda")

            def timed(env, r):
                for _ in range(3):
                    env.step(act)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(r):
                    env.step(act)
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / r

            dt = timed(e, reps)
            eager = se.make(n, batch=B, device="cuda")
            eager.use_step_graph = False
            eager.set_state(eager.sample_states(B, np.random.default_rng(0)))
            dte = timed(eager, max(3, reps // 4))
            rows.append({"batch": B, "step_ms": dt * 1e3, "env_steps_per_s": B / dt,
                         "eager_step_ms": dte * 1e3, "graph_speedup": dte / dt})
            print(f"{n:<26s} B={B:<6d} {dt*1e3:8.3f} ms  {B/dt:14,.0f} env-steps/s   "
                  f"eager {dte*1e3:8.3f} ms ({dte/dt:.1f}x)", flush=True)
            del e, eager
            torch.cuda.empty_cache()
        row = {"scale": rows}
        if cpu:
            row["cpu_env_steps_per_s"] = cpu_env_steps_per_s(n)
            print(f"{n:<26s} official CPU env: {row['cpu_env_steps_per_s']:,.0f} env-steps/s",
                  flush=True)
        out[n] = row
    return out


def rollout(names, steps: int = 50, batch: int = 256) -> dict:
    """The headline call — ``env.sample(steps, batch, policy)`` — on every env that now has a step."""
    out = {}
    for n in names:
        env = se.make(n, batch=batch, device="cuda", seed=0)
        env.sample(steps=2, batch=batch)                                    # warm up + capture
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = env.sample(steps=steps, batch=batch, policy=se.RandomPolicy(seed=0))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        moved = float((r["states"][:, -1] - r["states"][:, 0]).abs().max())
        out[n] = {"states": list(r["states"].shape), "actions": list(r["actions"].shape),
                  "sample_s": dt, "env_steps_per_s": batch * steps / dt, "max_state_change": moved}
        print(f"{n:<26s} sample(steps={steps}, batch={batch}) -> states{tuple(r['states'].shape)} "
              f"actions{tuple(r['actions'].shape)}  {batch*steps/dt:12,.0f} env-steps/s  "
              f"max|ds|={moved:.3g}", flush=True)
        del env
        torch.cuda.empty_cache()
    return out


def _mg_write(u, cls, sim) -> None:
    """Put one full MiniGrid dynamical state into the official env: grid, agent, carried object."""
    from minigrid.core.grid import Grid

    from stable_envs.render import tilemap as tm
    n = cls.grid_w * cls.grid_h
    codes = np.round(np.asarray(sim[:n])).astype(np.int64) % tm.N_CODE
    grid = Grid(cls.grid_w, cls.grid_h)
    for k, c in enumerate(codes):
        grid.set(k // cls.grid_h, k % cls.grid_h, tm.make_obj(*tm.REACHABLE[c]))
    u.grid, u.width, u.height = grid, cls.grid_w, cls.grid_h
    u.agent_pos = (int(round(sim[n])), int(round(sim[n + 1])))
    u.agent_dir = int(round(sim[n + 2]))
    u.carrying = tm.make_obj(*tm.REACHABLE[int(round(sim[n + 3])) % tm.N_CODE])


def grid_enumerate(names, n_states: int = 256, seed: int = 0) -> dict:
    """EVERY action from every one of ``n_states`` states, against the official ``MiniGridEnv.step``.

    A random rollout is a weak test of a decision table: it visits the transitions the policy happens
    to reach and, with nothing in the agent's hands, never once exercises ``drop`` or a locked door. So
    half the states here are given a randomly chosen carried object, and all seven actions are applied
    to each state — the whole cross product, in one batched GPU step per env.
    """
    from stable_envs.render import tilemap as tm
    pickable = [i for i, (t, c, s) in enumerate(tm.REACHABLE)
                if t in (tm.T_KEY, tm.T_BALL, tm.T_BOX)]
    out = {}
    for n in names:
        cls = se.REGISTRY[n]
        rng = np.random.default_rng(seed)
        proto = se.make(n, batch=1, device="cuda")
        pool = proto.sample_states(n_states, rng)                     # (n_states, state_dim)
        carry = np.where(rng.random(n_states) < 0.5,
                         rng.choice(pickable, n_states), CARRY_NONE_CODE)
        sim = np.concatenate([pool, carry[:, None]], 1).astype(np.float64)
        acts = np.arange(cls.N_ACTIONS)
        S, A = n_states, cls.N_ACTIONS
        batch = np.repeat(sim, A, axis=0)                             # (S*A, D)
        env = se.make(n, batch=S * A, device="cuda")
        env.set_sim_state(torch.as_tensor(batch, device="cuda"))
        env.step(torch.as_tensor(np.tile(acts, S)[:, None], dtype=torch.float32, device="cuda"))
        got = env.sim_state().cpu().numpy()

        ref = cls.reference_env()
        u = ref.unwrapped
        want = np.empty_like(got)
        for i in range(S):
            for j, a in enumerate(acts):
                _mg_write(u, cls, sim[i])
                u.step(int(a))
                want[i * A + j] = cls.reference_sim_state(ref)
        ref.close()
        bad = int((got != want).any(1).sum())
        out[n] = {"transitions": S * A, "states": S, "actions": int(A), "mismatched": bad,
                  "exact_frac": float((got == want).mean()),
                  "carrying_states": int((carry != CARRY_NONE_CODE).sum())}
        print(f"{n:<26s} {S*A:6d} (state, action) pairs, {bad} mismatched, "
              f"exact_frac={out[n]['exact_frac']}", flush=True)
        del env, proto
        torch.cuda.empty_cache()
    return out


def mcc_regime(steps: int = 100, episodes: int = 16) -> dict:
    """MountainCarContinuous has TWO dtype regimes and only one of them is its steady state.

    ``step`` writes ``np.array([position, velocity], dtype=np.float32)``, but ``reset`` leaves a float64
    state behind, so the first step after a reset accumulates in float64 and every step after it in
    float32. This measures the same env from a reference state that has already been through one step,
    which is the regime any rollout is made of.
    """
    class _Warm(se.REGISTRY["MountainCarContinuous"]):
        @classmethod
        def step_reference_reset(cls, ref, seed):
            ref.reset(seed=seed)
            ref.step(np.zeros(1, np.float32))       # one step, so the state is float32 from here on
            return cls.reference_sim_state(ref)

    r = C.check_step_matches_reference(_Warm(batch=4, device="cuda"), steps=steps, episodes=episodes)
    print(f"\nMountainCarContinuous from a float32 (already-stepped) reference state: "
          f"exact_frac={r['exact_frac']} max_err={r['max_err']:.3e}", flush=True)
    return r


def ulp_probe(n: int = 1_000_000) -> dict:
    """Where does a float64 GPU port of a float64 CPU integrator actually lose exactness?

    Only in the transcendentals: ``+ - * /`` are IEEE-exact and identical on both, so any deviation
    below has to come from ``sin``/``cos``. Measured rather than asserted, because it is the whole
    explanation for the classic-control numbers.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(-4 * np.pi, 4 * np.pi, n)
    out = {}
    for fn in ("sin", "cos"):
        g = getattr(torch, fn)(torch.as_tensor(x, device="cuda")).cpu().numpy()
        c = getattr(np, fn)(x)
        d = np.abs(g - c)
        ulp = np.spacing(np.abs(c))
        out[fn] = {"differ_frac": float((d > 0).mean()), "max_abs": float(d.max()),
                   "max_ulp": float((d / ulp).max())}
        print(f"torch.{fn} vs np.{fn} on float64: differs on {out[fn]['differ_frac']*100:.2f}% of "
              f"1e6 inputs, max {out[fn]['max_abs']:.3e} ({out[fn]['max_ulp']:.1f} ulp)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all")
    ap.add_argument("--envs", default="")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--batches", default="256,4096,16384")
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    names = args.envs.split(",") if args.envs else CLASSIC + GRID + ["PushT"]
    batches = [int(b) for b in args.batches.split(",") if b]
    res = {}
    if args.what in ("all", "ulp"):
        res["ulp"] = ulp_probe()
    if args.what in ("all", "rollout"):
        res["rollout"] = rollout(names)
    if args.what in ("all", "enumerate"):
        res["enumerate"] = grid_enumerate([n for n in names if n in GRID], args.states)
    if args.what in ("all", "mcc"):
        res["mcc"] = mcc_regime(args.steps, args.episodes)
    if args.what in ("all", "graph"):
        res["graph"] = graph_vs_eager(names)
    if args.what in ("all", "fidelity"):
        res["fidelity"] = fidelity(names, args.steps, args.episodes)
    if args.what in ("all", "bench"):
        res["bench"] = bench(names, batches)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
