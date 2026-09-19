"""``env.sample(steps=100, batch=256, policy=...)`` — native on PushT, and the envs that cannot.

PushT ships a batched GPU simulator, so the whole rollout is device-resident: states are generated,
stepped and (optionally) rendered without anything crossing PCIe. The primary path in this script is
that one call.

**Most envs here do not ship a simulator, and that is deliberate.** Rendering and simulation are
separable and only the renderer can be bit-exact against a reference, so an env is free to ship a
renderer alone; ``sample`` then raises with the reason and the alternative rather than failing somewhere
deep. This script prints that message verbatim from a renderer-only env, and then demonstrates the
alternative it names — ``ReferencePushT`` adds ``step`` by driving B copies of the official CPU env,
which is the shape of the answer for any env whose dynamics you have but whose solver you do not want
to port. It doubles as the CPU baseline the native path is measured against.

The two parities are different claims and this script keeps them apart:

  * frames rendered during collection are **bit-identical** to frames re-rendered from the stored
    states. That is the repo's thesis and it is an exactness claim.
  * the GPU solver tracks pymunk to a **median 0.68 px over 100 steps** and is not bit-exact, because
    contact is chaotic. ``check_step_matches_reference`` measures it here, with the kinematic agent as
    the control (no contact acts on it, so the integrators must agree to float precision).

Neither policy solves PushT, and the printed numbers say so rather than implying otherwise: the honest
readout is the command tracking error (does the policy control the agent at all) alongside the block's
distance to the goal (is it any good at the task).

    python examples/rollout_with_policy.py
    python examples/rollout_with_policy.py --batch 4096 --steps 100 --skip-cpu
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing.conformance import check_step_matches_reference

GOAL_XY = np.array([256.0, 256.0])

#: The claim in the PushT docstring, as a number this script refuses to pass without. Set from the
#: measured distribution (median 0.68 px over 16 x 100 steps) with room for the run-to-run spread of a
#: chaotic system, not from what happened to come out today.
MEDIAN_TOLERANCE_PX = 2.0
#: The agent is kinematic, so this is an integrator check and has no chaos in it: it must stay tiny.
AGENT_TOLERANCE_PX = 1e-3


class ReferencePushT(se.registry.REGISTRY["PushT"]):
    """PushT stepped by B copies of the official ``gym_pusht`` env, on the CPU.

    Kept, now that PushT has a native ``step``, for two reasons. It is the CPU baseline the GPU solver
    is timed against, and it is the worked example of what to do for an env that ships a renderer and no
    simulator: implement ``step`` on a subclass, and ``stable_envs.rollout.sample`` works unchanged.

    ``set_state`` writes through to the reference bodies, so the two paths start from the same state.
    """

    use_step_graph = False                # the state is written by pymunk each step, not by a kernel

    def __init__(self, num_envs: int = 256, **kw):
        super().__init__(num_envs=num_envs, **kw)
        self._refs = [self.reference_env(size=self.size) for _ in range(self.B)]
        for i, e in enumerate(self._refs):
            e.reset(seed=i)

    def set_state(self, states, style=None):
        super().set_state(states, style=style)
        for e, s in zip(self._refs, self._state.detach().cpu().numpy()):
            self.reference_set_state(e.unwrapped, s)
        return self

    def step(self, actions) -> torch.Tensor:
        a = np.asarray(torch.as_tensor(actions).detach().cpu().numpy(), np.float64).reshape(self.B, 2)
        out = np.empty((self.B, self.state_dim), np.float32)
        for i, (e, ai) in enumerate(zip(self._refs, a)):
            u = e.unwrapped
            u.step(ai)
            out[i] = (*u.agent.position, *u.block.position, u.block.angle)
        self._state.copy_(torch.as_tensor(out, device=self.device))
        return self._state

    def close(self) -> None:
        for e in self._refs:
            e.close()


#: Approach offset behind the block, the radius at which "behind" counts as reached, how far past the
#: block to aim while pushing, the sideways clearance when swinging around, and the per-step command
#: limit. Chosen by measurement, not by taste: over 64 x 100 steps the unlimited version leaves the
#: block 149 px from the goal and the rate-limited one 130 px (random: 200 px), because gym_pusht's
#: action is an absolute target and an unlimited command makes the PD controller arrive at speed and
#: scatter the T. Limits of 30 and 15 track their commands more tightly (20 px, 10 px) and push less.
SWING = dict(behind=70.0, tol=30.0, push=60.0, clear=110.0, max_step=60.0)


def push_towards_goal(states: torch.Tensor, t: int) -> torch.Tensor:
    """Swing behind the block, then push through it towards the goal, at a bounded speed.

    This is NOT a solution to PushT — see the numbers the script prints. It exists so the rollout has a
    second, non-random policy whose effect is measurable, and it is written to make the geometry
    explicit: pushing a T towards a target means standing on the far side of it, approaching that side
    around the block rather than through it, and then driving at a point past the block.

    Three states, from the block's frame: in front of the block (between block and goal), swing out
    sideways; behind but not yet in place, go to the approach point; in place, push.
    """
    agent, block = states[:, 0:2], states[:, 2:4]
    goal = torch.as_tensor(GOAL_XY, device=states.device, dtype=torch.float32)
    n = goal - block
    n = n / (n.norm(dim=1, keepdim=True) + 1e-6)
    behind = block - n * SWING["behind"]
    push = block + n * SWING["push"]
    perp = torch.stack([-n[:, 1], n[:, 0]], dim=1)
    side = torch.sign(((agent - block) * perp).sum(1, keepdim=True))
    side = torch.where(side == 0, torch.ones_like(side), side)
    waypoint = block + perp * side * SWING["clear"] - n * SWING["behind"] * 0.3
    in_front = ((agent - block) * n).sum(1, keepdim=True) > 0
    reached = (agent - behind).norm(dim=1, keepdim=True) < SWING["tol"]
    target = torch.where(reached, push, torch.where(in_front, waypoint, behind))
    delta = target - agent
    scale = torch.clamp(SWING["max_step"] / (delta.norm(dim=1, keepdim=True) + 1e-6), max=1.0)
    return torch.clamp(agent + delta * scale, 20.0, 492.0)


def dist_to_goal(states: torch.Tensor) -> torch.Tensor:
    goal = torch.as_tensor(GOAL_XY, device=states.device, dtype=torch.float32)
    return (states[..., 2:4] - goal).norm(dim=-1)


def rollout_stats(out: dict, sim_s: float, batch: int, steps: int) -> dict:
    d0, dT = dist_to_goal(out["states"][:, 0]), dist_to_goal(out["states"][:, -1])
    # Tracking error: how far the agent ends up from the position the policy commanded. gym_pusht's
    # action IS a target position, so this says whether the policy is in control of the agent at all —
    # a metric that does not depend on the controller being any good at the task.
    track = (out["states"][:, 1:, 0:2] - out["actions"]).norm(dim=-1).mean()
    return {"sim_s": sim_s, "env_steps_per_s": batch * steps / sim_s,
            "us_per_env_step": sim_s / (batch * steps) * 1e6,
            "block_dist_to_goal_start": float(d0.mean()), "block_dist_to_goal_end": float(dT.mean()),
            "block_dist_median_change": float((dT - d0).median()),
            "improved_fraction": float((dT < d0).float().mean()),
            "tracking_error_px": float(track)}


def show_unsupported(size: int) -> tuple[str, str]:
    """Call ``sample`` on the first renderer-only env in the registry and return what it raised."""
    for name, cls in sorted(se.registry.REGISTRY.items()):
        if hasattr(cls, "step"):
            continue
        try:
            env = se.make(name, num_envs=4, size=None, device="cuda")
            env.sample(steps=2, batch=4)
            return name, "NO ERROR RAISED"
        except NotImplementedError as e:
            return name, str(e)
        except Exception:                      # an env that cannot even be constructed here proves
            continue                           # nothing about sample(); try the next one
    return "", "every registered env implements step()"


def bytes_report(out: dict, frame_shape: tuple[int, int, int]) -> dict:
    B, T, D = out["states"].shape
    state_b = B * T * D * 4
    action_b = 0 if out.get("actions") is None else out["actions"].numel() * 4
    style_b = out["style"].numel() * 4
    frame_b = B * T * int(np.prod(frame_shape))
    return {"trajectories": B, "steps_per_trajectory": T, "state_dim": D,
            "state_bytes": state_b, "action_bytes": action_b, "style_bytes": style_b,
            "stored_bytes": state_b + action_b + style_b, "frame_bytes": frame_b,
            "frame_shape": list(frame_shape),
            "ratio": frame_b / (state_b + action_b + style_b)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-cpu", action="store_true", help="skip the CPU fallback baseline")
    ap.add_argument("--agree-episodes", type=int, default=16)
    ap.add_argument("--scale", default="256,1024,4096,16384")
    ap.add_argument("--out", type=Path, default=Path("examples/results/rollout_with_policy.json"))
    ap.add_argument("--corpus", type=Path, default=Path("examples/results/pusht_rollout.npz"))
    args = ap.parse_args()

    steppable = sorted(n for n, c in se.registry.REGISTRY.items() if hasattr(c, "step"))
    print(f"registry: {len(se.registry.REGISTRY)} envs, {len(steppable)} of which implement step(): {steppable}")
    only_render, msg = show_unsupported(args.size)
    print(f"\n{only_render}.sample(steps=2, batch=4) raises NotImplementedError:\n  {msg}\n")
    results = {"registry_size": len(se.registry.REGISTRY), "steppable": steppable,
               "renderer_only_example": only_render, "not_implemented_message": msg,
               "batch": args.batch, "steps": args.steps}

    env = se.make("PushT", num_envs=args.batch, size=args.size, device="cuda", seed=args.seed)
    init = env.sample_states(args.batch, np.random.default_rng(args.seed))
    # Colours only. The style space also carries goal.position, goal.angle and the geometry scales, and
    # randomising those would change what the picture MEANS relative to the simulator, whose goal pose
    # is fixed at (256, 256, pi/4) — the frame would advertise a target the dynamics and the scripted
    # policy do not share. Appearance is free to vary; geometry is not, once a simulator is attached.
    style = env.sample_style(args.batch, paths=["agent.color", "block.color", "background.color"],
                             rng=np.random.default_rng(args.seed + 1))

    # ---------------------------------------------------------------- native GPU rollouts
    runs = {}
    for label, policy in [("random", se.RandomPolicy(seed=args.seed)),         # no bounds needed:
                          ("scripted", se.ScriptedPolicy(push_towards_goal))]:  # the env declares them
        env.sample(steps=2, batch=args.batch, policy=policy, style=style, init_states=init)  # warm up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = env.sample(steps=args.steps, batch=args.batch, policy=policy, style=style,
                         init_states=init, seed=args.seed)
        torch.cuda.synchronize()
        runs[label] = rollout_stats(out, time.perf_counter() - t0, args.batch, args.steps)
        runs[label]["keys"] = {k: (list(v.shape), str(v.dtype), str(v.device))
                               for k, v in out.items() if torch.is_tensor(v)}
        print(f"sample(steps={args.steps}, batch={args.batch}, policy={type(policy).__name__}) "
              f"native GPU")
        for k, v in out.items():
            print(f"  {k:<8s} {tuple(v.shape) if torch.is_tensor(v) else v}"
                  f"{'' if not torch.is_tensor(v) else f'  {v.dtype}  {v.device}'}")
        r = runs[label]
        print(f"  {r['sim_s']:.3f} s   {r['env_steps_per_s']:,.0f} env-steps/s   "
              f"{r['us_per_env_step']:.2f} us per env-step")
        print(f"  command tracking error {r['tracking_error_px']:.1f} px; block distance to goal "
              f"{r['block_dist_to_goal_start']:.1f} -> {r['block_dist_to_goal_end']:.1f} px mean, "
              f"median change {r['block_dist_median_change']:+.1f} px, closer in "
              f"{r['improved_fraction'] * 100:.0f}%\n")
        if label == "scripted":
            last = out

    a, b = runs["random"], runs["scripted"]
    print("neither policy solves PushT. The scripted controller holds the block near where it found it "
          "and follows its\ncommands to "
          f"{b['tracking_error_px']:.0f} px; random actions scatter it "
          f"{a['block_dist_median_change']:+.0f} px in the median against "
          f"{b['block_dist_median_change']:+.0f}. That is a policy argument being load-bearing, "
          "measured;\nit is not a control result.")

    # ---------------------------------------------------------------- does the solver track pymunk?
    print(f"\nsolver agreement with the official CPU simulator "
          f"({args.agree_episodes} episodes x {args.steps} steps, identical actions and start states):")
    t0 = time.perf_counter()
    agree = check_step_matches_reference(env, steps=args.steps, episodes=args.agree_episodes,
                                         seed=args.seed)
    print(f"  block touched in {agree['block_touched']}/{agree['episodes']} episodes "
          f"(an untouched block agrees trivially and is excluded)")
    print(f"  agent position error   max {agree['agent_max_err']:.2e} px, "
          f"mean {agree['agent_mean_err']:.2e} px   <- kinematic: the integrator check")
    print(f"  block position error   median {agree['block_median_err']:.2f} px, "
          f"mean {agree['block_mean_err']:.1f} px, p90 {agree['block_p90_err']:.1f} px "
          f"after {args.steps} steps")
    print(f"                         median {agree['block_median_err_at_25']:.2f} px at 25 steps; "
          f"angle median {agree['angle_median_err']:.4f} rad")
    print(f"  measured in {time.perf_counter() - t0:.0f} s")
    assert agree["agent_max_err"] < AGENT_TOLERANCE_PX, (
        f"the kinematic agent diverged by {agree['agent_max_err']:.3e} px, over the "
        f"{AGENT_TOLERANCE_PX} px integrator tolerance. No contact acts on the agent, so this is an "
        f"integration bug, not a solver difference.")
    assert agree["block_median_err"] < MEDIAN_TOLERANCE_PX, (
        f"median block divergence {agree['block_median_err']:.2f} px exceeds the documented "
        f"{MEDIAN_TOLERANCE_PX} px. The mean is expected to be large (chaotic contact); the median "
        f"moving is a regression.")
    print(f"  within the documented tolerance (median < {MEDIAN_TOLERANCE_PX} px, "
          f"agent < {AGENT_TOLERANCE_PX} px)")
    results["agreement"] = agree
    results["tolerances"] = {"median_px": MEDIAN_TOLERANCE_PX, "agent_px": AGENT_TOLERANCE_PX}

    # ---------------------------------------------------------------- the CPU fallback, as a baseline
    if not args.skip_cpu:
        print(f"\nfallback path: {args.batch} official gym_pusht envs behind one batched step() ...",
              flush=True)
        t0 = time.perf_counter()
        cpu_env = ReferencePushT(num_envs=args.batch, size=args.size, device="cuda", seed=args.seed)
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        cpu_out = cpu_env.sample(steps=args.steps, batch=args.batch,
                                 policy=se.ScriptedPolicy(push_towards_goal), style=style,
                                 init_states=init, seed=args.seed)
        cpu = rollout_stats(cpu_out, time.perf_counter() - t0, args.batch, args.steps)
        cpu_env.close()
        print(f"  built in {build_s:.1f} s ({build_s / args.batch * 1e3:.1f} ms per env)")
        print(f"  {cpu['sim_s']:.1f} s   {cpu['env_steps_per_s']:,.0f} env-steps/s   "
              f"{cpu['us_per_env_step']:.0f} us per env-step")
        print(f"  same policy, same starts: block {cpu['block_dist_to_goal_start']:.1f} -> "
              f"{cpu['block_dist_to_goal_end']:.1f} px mean "
              f"(GPU: {b['block_dist_to_goal_end']:.1f}), tracking {cpu['tracking_error_px']:.1f} px "
              f"(GPU: {b['tracking_error_px']:.1f})")
        print(f"  native GPU is {cpu['sim_s'] / b['sim_s']:.1f}x faster at batch {args.batch}")
        results["cpu_fallback"] = {**cpu, "build_s": build_s,
                                   "speedup_native_vs_cpu": cpu["sim_s"] / b["sim_s"]}

    # ------------------------------------------------- how the native step scales, and why it matters
    # The CPU path costs one pymunk step per environment, so it is linear in B and its per-env-step cost
    # is a constant. The GPU step is ~2,000 tiny kernels captured into one graph, so its cost is nearly
    # batch-independent until the arithmetic finally matters — which means the speedup is not a number,
    # it is a function of how many environments you asked for.
    def time_step(env_, reps: int) -> float:
        act = torch.full((env_.B, 2), 256.0, device="cuda")
        for _ in range(3):
            env_.step(act)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            env_.step(act)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    scale = []
    for B in [int(x) for x in args.scale.split(",") if x]:
        e = se.make("PushT", num_envs=B, size=args.size, device="cuda")
        e.set_state(e.sample_states(B))
        dt = time_step(e, 50)
        # The same arithmetic without the graph, to show what the graph is actually buying. The step is
        # 10 substeps x 10 solver iterations of small ops; uncaptured, the cost is the launches.
        eager = se.make("PushT", num_envs=B, size=args.size, device="cuda")
        eager.use_step_graph = False
        eager.set_state(eager.sample_states(B))
        dt_eager = time_step(eager, 5)
        scale.append({"batch": B, "step_ms": dt * 1e3, "env_steps_per_s": B / dt,
                      "us_per_env_step": dt / B * 1e6, "eager_step_ms": dt_eager * 1e3,
                      "graph_speedup": dt_eager / dt})
        del e, eager
        torch.cuda.empty_cache()
    print("\nnative step cost against batch size:")
    print("| batch | ms / step | env-steps/s | µs per env-step | uncaptured ms | graph speedup |")
    print("|---|---|---|---|---|---|")
    for r in scale:
        print(f"| {r['batch']:,} | {r['step_ms']:.2f} | {r['env_steps_per_s']:,.0f} | "
              f"{r['us_per_env_step']:.3f} | {r['eager_step_ms']:.1f} | {r['graph_speedup']:.1f}x |")
    results["scale"] = scale

    # ---------------------------------------------------------------- storage
    rep = bytes_report(last, (args.size, args.size, 3))
    print(f"\nstorage for {rep['trajectories']} x {rep['steps_per_trajectory']} steps:")
    print(f"  states {rep['state_bytes']:,} B + actions {rep['action_bytes']:,} B + "
          f"style {rep['style_bytes']:,} B = {rep['stored_bytes'] / 1e3:.0f} kB")
    print(f"  the frames they replace: {rep['frame_bytes'] / 1e6:.0f} MB at "
          f"{args.size}x{args.size}x3  ->  {rep['ratio']:.0f}x")
    for s in (224, 512):
        n = rep["trajectories"] * rep["steps_per_trajectory"] * s * s * 3
        print(f"  at {s}x{s}: {n / 1e6:.0f} MB -> {n / rep['stored_bytes']:.0f}x "
              f"(the states do not change; the resolution is chosen at render time)")
    results["storage"] = rep

    # ------------------------------------------------- frames now vs frames regenerated from states
    t0 = time.perf_counter()
    live = env.sample(steps=args.steps, batch=args.batch, policy=se.ScriptedPolicy(push_towards_goal),
                      style=style, init_states=init, seed=args.seed, render=True)
    torch.cuda.synchronize()
    with_render_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    regen = se.render_states(env, live["states"], size=args.size,
                             style=live["style"].repeat_interleave(args.steps + 1, 0), chunk=8192)
    torch.cuda.synchronize()
    regen_s = time.perf_counter() - t0
    identical = torch.equal(live["frames"], regen)
    n_frames = live["frames"].shape[0] * live["frames"].shape[1]
    print(f"\nregeneration: {n_frames:,} frames re-rendered from the stored states in "
          f"{regen_s * 1e3:.0f} ms ({n_frames / regen_s:,.0f} frames/s)")
    print(f"  identical to the frames collected during the rollout: {identical}")
    print(f"  the same rollout with render=True took {with_render_s:.2f} s against "
          f"{b['sim_s']:.2f} s without frames")
    assert identical, "re-rendering the stored states did not reproduce the collected frames"
    results["regeneration"] = {"frames": int(n_frames), "regen_s": regen_s,
                               "frames_per_s": n_frames / regen_s, "identical": identical,
                               "rollout_with_render_s": with_render_s}
    results["runs"] = runs

    # ---------------------------------------------------------------- the corpus
    ds = se.TrajectoryDataset(env, last["states"].cpu().numpy(), style=last["style"].cpu().numpy(),
                              window=8, stride=4, size=args.size)
    print(f"\n{ds}")
    print(f"  compression {ds.compression['ratio']:.0f}x on the stored steps")
    args.corpus.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.corpus, states=last["states"].cpu().numpy(),
                        actions=last["actions"].cpu().numpy(), style=last["style"].cpu().numpy())
    print(f"  wrote {args.corpus} ({args.corpus.stat().st_size / 1e3:.0f} kB) "
          f"— feed it to hf_roundtrip.py --corpus")
    results["corpus"] = {"path": str(args.corpus), "bytes": args.corpus.stat().st_size,
                         "dataset": ds.describe(), "compression": ds.compression}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
