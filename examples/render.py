"""Render a public ``State`` through a concrete environment class.

Uses only the Step-9 ``snv`` surface: construct by class (or ``snv.make``), ``reset``, ``render``,
``close``. Optional extras are not required for PushT.

    python examples/render.py
    python examples/render.py --env PushT --num-envs 8 --device cpu
    python examples/render.py --out /tmp/pusht.png
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


def load_env_class(name: str):
    import stable_envs as snv
    if name in snv.list_envs():
        return getattr(snv, name)
    errors = snv.import_errors()
    extra = f" Import errors for missing families: {errors}" if errors else ""
    raise SystemExit(
        f"{name!r} is not registered in this interpreter. "
        f"snv.list_envs() has {len(snv.list_envs())} names.{extra}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Render a batched public State with snv.<Env>(num_envs=..., device=...).")
    p.add_argument("--env", default="PushT", help="registered class name (default: PushT)")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--device", default="auto", help="cpu, cuda, or auto (cuda if available)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional PNG path for environment 0")
    args = p.parse_args(argv)

    try:
        import stable_envs as snv
        import torch
    except ImportError as e:
        print(f"core dependency missing: {e}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    cls = load_env_class(args.env)
    print(f"stable-envs {snv.__version__}  device={device}  env={args.env}")

    try:
        env = cls(num_envs=args.num_envs, device=device)
    except Exception as e:
        print(f"{args.env} refused to construct on {device}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    try:
        state = env.reset(seed=args.seed)
        pixels = env.render(state)
        print(f"state batch {list(state.batch_size)} keys {list(state.keys())}")
        print(f"pixels {tuple(pixels.shape)} {pixels.dtype} {pixels.device}")
        if args.out:
            try:
                from PIL import Image
            except ImportError:
                print("--out needs Pillow; pip install pillow", file=sys.stderr)
                return 1
            Image.fromarray(pixels[0].detach().cpu().numpy()).save(args.out)
            print(f"wrote {args.out}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
