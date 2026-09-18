"""Stateful ``step`` vs functional ``transition`` / ``unroll``.

``step`` writes ``env.state``. ``transition`` and ``unroll`` do not. ``unroll`` takes known
open-loop actions with time first — it is not a policy rollout.

    python examples/functional_dynamics.py
    python examples/functional_dynamics.py --env CartPole --horizon 4
    python examples/functional_dynamics.py --env CraftaxClassic   # renderer-only refusal
"""
from __future__ import annotations

import argparse
import sys


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"{requested} was requested but torch.cuda.is_available() is False. Pass --device cpu.")
    return requested


def _actions(env, n, seed: int):
    import torch
    spec = env.action_spec
    g = torch.Generator().manual_seed(seed)
    if spec.low is not None and spec.high is not None:
        lo = torch.as_tensor(spec.low, dtype=torch.float32)
        hi = torch.as_tensor(spec.high, dtype=torch.float32)
        u = torch.rand(n, spec.shape[0], generator=g)
        return lo + (hi - lo) * u
    return spec.sample(n, env.rng)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="step is stateful; transition/unroll are pure in state; no env.rollout(policy).")
    p.add_argument("--env", default="PushT")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--horizon", type=int, default=5, help="T for unroll; actions are (T, num_envs, …)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    try:
        import stable_envs as snv
        import torch
    except ImportError as e:
        print(f"core dependency missing: {e}", file=sys.stderr)
        return 1

    if args.env not in snv.list_envs():
        print(f"{args.env!r} is not registered. import_errors={snv.import_errors()}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    cls = getattr(snv, args.env)
    try:
        env = cls(num_envs=args.num_envs, device=device)
    except Exception as e:
        print(f"{args.env} refused to construct on {device}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"{args.env} has_dynamics={env.has_dynamics}  device={device}")
    state = env.reset(seed=args.seed)

    if not env.has_dynamics:
        try:
            env.step(torch.zeros(args.num_envs, 1))
        except NotImplementedError as e:
            print(f"renderer-only refusal:\n{e}")
        env.close()
        return 0

    held_before = env.get_state().clone()
    a0 = _actions(env, args.num_envs, args.seed + 1)
    nxt = env.transition(state, a0)
    if not torch.equal(env.get_state(), held_before):
        print("FAIL: transition mutated env.state", file=sys.stderr)
        env.close()
        return 1
    print("transition left env.state untouched")

    out = env.step(a0)
    print(f"step keys={list(out.keys())}  (reward/done omitted unless the cell has them)")
    if torch.equal(env.pack(nxt), env.pack(out["state"])):
        print("transition(state, a) == step(a)['state']  (bit-identical on this device)")
    else:
        err = (env.pack(nxt) - env.pack(out["state"])).abs().max().item()
        print(f"transition vs step max abs {err}  (not identical on this cell/device)")

    acts = torch.stack([_actions(env, args.num_envs, 100 + t) for t in range(args.horizon)])
    held = env.get_state().clone()
    traj = env.unroll(state, acts)
    if not torch.equal(env.get_state(), held):
        print("FAIL: unroll mutated env.state", file=sys.stderr)
        env.close()
        return 1
    print(f"unroll actions {tuple(acts.shape)} -> traj batch {list(traj.batch_size)}")

    cur = state
    ok = True
    for t in range(args.horizon):
        cur = env.transition(cur, acts[t])
        if not torch.equal(env.pack(traj[t]), env.pack(cur)):
            ok = False
            break
    print("unroll == repeated transition" if ok else "unroll diverged from repeated transition")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
