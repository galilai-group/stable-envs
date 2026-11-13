"""Write PNG sheets for registered envs: reset frames, a short rollout, and sampled variation.

Per env, under ``--out/<env>/``:

  ``reset.png``       one frame per lane after ``reset``
  ``rollout.png``     lane 0 over ``--steps`` random actions, for envs with dynamics
  ``variation.png``   lane 0's state under one sampled appearance per lane, for envs whose
                      ``render`` takes ``variation=``

    python examples/gallery.py --out media
    python examples/gallery.py --envs CartPole,TwoRoom --num-envs 4 --steps 8 --device cpu
"""

from __future__ import annotations

import argparse
import inspect
import math
import struct
import zlib
from pathlib import Path

import numpy as np
import torch

import stable_envs as snv


def tile(frames: np.ndarray, cols: int, pad: int = 2) -> np.ndarray:
    n, h, w, _ = frames.shape
    rows = math.ceil(n / cols)
    sheet = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 235, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[pad + r * (h + pad) : pad + r * (h + pad) + h, pad + c * (w + pad) : pad + c * (w + pad) + w] = f
    return sheet


def write_png(path: Path, img: np.ndarray) -> None:
    """Write an ``(H, W, 3)`` uint8 image as PNG with the standard library only."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    h, w, _ = img.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in np.ascontiguousarray(img, np.uint8))
    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    )


def takes_variation(env) -> bool:
    params = inspect.signature(env.render).parameters.values()
    return any(p.name == "variation" or p.kind is p.VAR_KEYWORD for p in params)


def env_sheets(name: str, out: Path, n: int, steps: int, device: str, seed: int) -> list[str]:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    cols = min(n, 8)
    written = []
    with snv.make(name, num_envs=n, device=device) as env:
        state = env.reset(seed=seed)
        write_png(d / "reset.png", tile(env.render(state).cpu().numpy(), cols))
        written.append("reset.png")

        if env.has_dynamics:
            frames = []
            for _ in range(steps):
                result = env.step(env.action_spec.sample(n, env.rng))
                frames.append(result["frames"][0, -1].cpu().numpy())
            write_png(d / "rollout.png", tile(np.stack(frames), min(steps, 8)))
            written.append("rollout.png")

        if takes_variation(env):
            appearance = [p for p in env.variation_names if p not in env.dynamics_variation_leaves]
            rows = torch.as_tensor(env.sample_variation(n, paths=appearance, rng=env.rng), device=env.device)
            shared = state[torch.zeros(n, dtype=torch.long)]
            write_png(d / "variation.png", tile(env.render(shared, variation=rows).cpu().numpy(), cols))
            written.append("variation.png")
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PNG sheets of reset frames, rollouts and Variation per env.")
    p.add_argument("--out", type=Path, default=Path("media"))
    p.add_argument("--envs", default="", help="comma-separated names; default is every registered env")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    names = [s for s in args.envs.split(",") if s] or snv.list_envs()
    for name in names:
        written = env_sheets(name, args.out, args.num_envs, args.steps, args.device, args.seed)
        print(f"{name}: {', '.join(written)} in {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
