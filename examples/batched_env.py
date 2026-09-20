"""Batched construction, specs, a short last batch, and explicit close.

python examples/batched_env.py
python examples/batched_env.py --num-envs 8 --short 3 --device cpu
"""

from __future__ import annotations

import argparse
import sys


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"{requested} was requested but torch.cuda.is_available() is False. Pass --device cpu.")
    return requested


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="num_envs, specs, short batches, and the context-manager lifecycle.")
    p.add_argument("--env", default="PushT")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--short", type=int, default=3, help="n <= num_envs to render as a short batch")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    try:
        import stable_envs as snv
    except ImportError as e:
        print(f"core dependency missing: {e}", file=sys.stderr)
        return 1

    if args.env not in snv.list_envs():
        print(f"{args.env!r} is not registered. import_errors={snv.import_errors()}", file=sys.stderr)
        return 1
    if not (1 <= args.short <= args.num_envs):
        print("--short must be in 1..--num-envs", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    cls = getattr(snv, args.env)
    print(f"stable-envs {snv.__version__}  {args.env}  num_envs={args.num_envs}  device={device}")
    print(f"this interpreter: {len(snv.list_envs())} envs, families={snv.list_families()}")
    meta = snv.info(args.env)
    print(
        f"info family={meta['family']} domain={meta['domain']} tier={meta['fidelity_tier']} "
        f"has_dynamics={meta['has_dynamics']} swm_ids={meta['swm_ids']}"
    )

    try:
        env = cls(num_envs=args.num_envs, device=device)
    except Exception as e:
        print(f"{args.env} refused to construct on {device}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    with env:
        assert len(env) == args.num_envs
        state = env.reset(seed=args.seed)
        print(f"state_spec keys: {list(env.state_spec.keys())}")
        print(f"frame_spec: {env.frame_spec}")
        try:
            print(f"action_spec: {env.action_spec}")
        except NotImplementedError as e:
            print(f"action_spec: (no dynamics) {e}")
        print(f"variation_spec.shape: {env.variation_spec.shape}")

        packed = env.pack_render_state(state)
        print(f"pack_render_state(state) -> {tuple(packed.shape)}  names={list(env.render_state_names)}")
        roundtrip = env.unpack_render_state(packed)
        print(f"unpack keys {list(roundtrip.keys())}")

        short = env.get_state()[: args.short]
        frames = env.set_state(short).render()
        print(f"short batch n={args.short} -> frames {tuple(frames.shape)}")
        assert frames.shape[0] == args.short

    print(f"closed={env.closed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
