"""Construct an environment, reset it, and render the returned ``State``.

PushT requires the ``pusht`` extra.

    python examples/render.py
    python examples/render.py --env PushT --num-envs 8 --device cpu
    python examples/render.py --out /tmp/pusht.png
"""

from __future__ import annotations

import argparse
import sys

import torch

import stable_envs as snv


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"{requested} was requested but torch.cuda.is_available() is False. Pass --device cpu.")
    return requested


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render a batched State with snv.<Env>(num_envs=..., device=...).")
    p.add_argument("--env", default="PushT", help="registered class name (default: PushT)")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--device", default="auto", help="cpu, cuda, or auto (cuda if available)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional PNG path for environment 0")
    args = p.parse_args(argv)

    if args.env not in snv.list_envs():
        print(f"{args.env!r} is not registered. import_errors={snv.import_errors()}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    print(f"stable-envs {snv.__version__}  device={device}  env={args.env}")

    with snv.make(args.env, num_envs=args.num_envs, device=device) as env:
        state = env.reset(seed=args.seed)
        pixels = env.render(state)
        print(f"state batch {list(state.batch_size)} keys {list(state.keys())}")
        print(f"pixels {tuple(pixels.shape)} {pixels.dtype} {pixels.device}")

    if args.out:
        from PIL import Image

        Image.fromarray(pixels[0].cpu().numpy()).save(args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
