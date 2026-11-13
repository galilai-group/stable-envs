"""Write the README environment GIFs to docs/assets/media/envs/ with the public stable-envs API.

Each cell has two clips of the same seeded rollout: ``<Class>.gif`` in the default appearance and ``<Class>_var.gif``
with sampled appearance leaves. Classes without appearance leaves show a second class or another seed instead.
``--readme`` prints the HTML figure for the README. The Fetch classes need the ``robotics`` extra, which needs its own
environment, so pass their names to a second run there.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import stable_envs as snv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "media" / "envs"
URL = "https://raw.githubusercontent.com/galilai-group/stable-envs/main/docs/assets/media/envs"

STEPS = 40
HOLD = 8
SIDE = 120
FRAME_MS = 100
WIDTH = 120
DONE = ("terminated", "truncated", "done")
#: Steps per GIF frame for classes whose episodes are long.
STRIDE = {"PyFlytRocketLanding": 7, "MSHumanLocomotion": 3}
#: Classes that look better with zero actions than with random ones: the rocket falls straight to the pad.
ZERO_ACTIONS = {"PyFlytRocketLanding"}

#: Leaves that change only the picture: colors, materials, light, camera and weather.
APPEARANCE = re.compile(r"color|material\.|light\.|tint|camera\.|weather\.rain|cinematic")

#: README rows. A cell is a class name, or ``(top, bottom)`` to show a second class below.
ROWS: list[list[str | tuple[str, str]]] = [
    ["DMCBallInCupCatch", "DMCCartpoleSwingup", "DMCCheetahRun", "DMCFingerSpin", "DMCHopperHop"],
    ["DMCPendulumSwingup", "DMCQuadrupedWalk", "DMCReacher", "DMCWalkerWalk", "DMCHumanoidWalk"],
    ["DMCAcrobotBalance", "DMCManipulatorBringBall", "DMCPointMassEasy", "DMCSwimmerSwimmer6"],
    ["CartPole", "MountainCar", "Acrobot", "Pendulum"],
    ["OGBCube", "OGBScene", "PushT", "TwoRoom", "SimplePointMaze"],
    ["FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace", ("CraftaxClassicPixels", "CraftaxPixels")],
    ["PointMazeMediumNavigate", "AntMazeMediumNavigate", "HumanoidMazeMediumNavigate", "Piecewise"],
    ["PyFlytRocketLanding", "RoadsterLoop", "MSHumanLocomotion", "MiniWorldMovingBlockWorld"],
]


def _actions(env, seed: int, stride: int, zero: bool = False) -> torch.Tensor:
    """``(1, STEPS * stride, *A)`` uniform random actions, each held for ``HOLD * stride`` steps."""
    spec = env.action_spec
    rng = np.random.default_rng(seed)
    n = -(-STEPS // HOLD)
    shape = (n, *spec.shape)
    if not spec.dtype.is_floating_point:
        count = spec.discrete_n if spec.discrete_n is not None else int(np.max(spec.high)) + 1
        draws = torch.as_tensor(rng.integers(0, int(count), size=shape), dtype=spec.dtype)
    else:
        lo = np.broadcast_to(-1.0 if spec.low is None else spec.low, spec.shape).astype(np.float64)
        hi = np.broadcast_to(1.0 if spec.high is None else spec.high, spec.shape).astype(np.float64)
        lo, hi = np.where(np.isfinite(lo), lo, -1.0), np.where(np.isfinite(hi), hi, 1.0)
        draws = torch.as_tensor(rng.uniform(lo, hi, size=shape), dtype=spec.dtype)
    draws = draws.repeat_interleave(HOLD * stride, dim=0)[: STEPS * stride].unsqueeze(0)
    return torch.zeros_like(draws) if zero else draws


def _make(name: str, device: str | None):
    kwargs = {"num_envs": 1}
    if device is not None:
        kwargs["device"] = device
    if snv.info(name)["native_size"] is None:
        kwargs["size"] = SIDE
    return snv.make(name, **kwargs)


def _rollout(env, seed: int, actions: torch.Tensor | None) -> np.ndarray:
    """``(STEPS + 1, H, W, 3)`` frames from ``reset(seed)`` through ``actions``; an episode that ends resets with
    the next seed. Each call steps one fixed-length chunk, so that JAX and CUDA graph classes compile one shape."""
    state = env.reset(seed=seed)
    frames = [env.render(state)[0].cpu().numpy()]
    if actions is None or not env.has_dynamics:
        return np.stack(frames * (STEPS + 1))
    stride = actions.shape[1] // STEPS
    for t in range(STEPS):
        chunk = actions[:, t * stride : (t + 1) * stride].to(env.device)
        out = env.step(chunk[:, 0] if stride == 1 else chunk, states=None)
        frames.append(out["frames"][0, -1].cpu().numpy())
        if any(bool(out[key].any()) for key in DONE if key in out.keys()):
            seed += 1
            frames[-1] = env.render(env.reset(seed=seed))[0].cpu().numpy()
    return np.stack(frames)


def _appearance(env) -> list[str]:
    dynamics = set(env.dynamics_variation_leaves)
    return [p for p in env.variation_names if p not in dynamics and APPEARANCE.search(p)]


def clips(name: str, bottom: str | None, device: str | None, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Default and varied clips of ``name``, and a note on what the bottom clip varies."""
    env = _make(name, device)
    try:
        stride = STRIDE.get(name, 1)
        actions = _actions(env, seed, stride, name in ZERO_ACTIONS) if env.has_dynamics else None
        top = _rollout(env, seed, actions)
        if bottom is not None:
            other = _make(bottom, device)
            try:
                return top, _rollout(other, seed, _actions(other, seed, stride)), f"class {bottom}"
            finally:
                other.close()
        paths = _appearance(env)
        if not paths:
            # No appearance leaves: show the episode of another seed, the one with the most varied frames.
            runs = {k: _rollout(env, k, actions) for k in range(seed + 1, seed + 7)}
            k = max(runs, key=lambda k: float(runs[k].std(axis=(1, 2, 3)).mean()))
            return top, runs[k], f"no appearance leaves: seed {k}"
        best, best_gap = None, -1.0
        for k in range(6):
            rows = env.sample_variation(
                1, paths=paths, rng=np.random.default_rng((seed, zlib.crc32(name.encode()), k))
            )
            env.set_variation(rows)
            frames = _rollout(env, seed, actions)
            gap = float(np.abs(frames.astype(np.float32) - top.astype(np.float32)).mean())
            if frames.mean() < 0.95 * top.mean():
                gap /= 4  # a dark draw hides the scene
            if gap > best_gap:
                best, best_gap = frames, gap
            if gap > 25.0:
                break
        return top, best, f"{len(paths)} leaves, mean pixel gap {best_gap:.1f}"
    finally:
        env.close()


def _resize(frames: np.ndarray) -> list[Image.Image]:
    h, w = frames.shape[1:3]
    scale = SIDE / max(h, w)
    size = (max(2, round(w * scale)), max(2, round(h * scale)))
    return [Image.fromarray(f).resize(size, Image.Resampling.LANCZOS) for f in frames]


def write_gif(frames: np.ndarray, path: Path) -> int:
    images = _resize(frames)
    sheet = Image.new("RGB", (images[0].width, images[0].height * 8))
    for i, k in enumerate(np.linspace(0, len(images) - 1, 8).astype(int)):
        sheet.paste(images[k], (0, i * images[0].height))
    palette = sheet.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    quant = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    path.parent.mkdir(parents=True, exist_ok=True)
    quant[0].save(path, save_all=True, append_images=quant[1:], duration=FRAME_MS, loop=0, optimize=True)
    return path.stat().st_size


def _files(cell: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(cell, tuple):
        return f"{cell[0]}.gif", f"{cell[1]}.gif"
    return f"{cell}.gif", f"{cell}_var.gif"


def readme() -> str:
    lines = ['<div align="center">', ""]
    for row in ROWS:
        lines += ["<table>", "<tr>"]
        for cell in row:
            top, bottom = _files(cell)
            lines.append(
                f'<td align="center"><img src="{URL}/{top}" width="{WIDTH}"/><br>'
                f'<img src="{URL}/{bottom}" width="{WIDTH}"/></td>'
            )
        lines += ["</tr>", "</table>", ""]
    lines += [
        "<em>Top row: default appearance &nbsp;·&nbsp; Bottom row: visual factor of variation</em>",
        "",
        "</div>",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="top classes to render (default: every cell)")
    parser.add_argument("--device", default=None, help="device of every class (default: the class default)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--readme", action="store_true", help="print the README figure and exit")
    args = parser.parse_args(argv)
    if args.readme:
        print(readme())
        return 0
    cells = [cell for row in ROWS for cell in row]
    if args.names:
        cells = [c for c in cells if (c[0] if isinstance(c, tuple) else c) in args.names]
    failed = []
    for cell in cells:
        name, bottom = cell if isinstance(cell, tuple) else (cell, None)
        top_file, bottom_file = _files(cell)
        try:
            top, varied, note = clips(name, bottom, args.device, args.seed)
            sizes = write_gif(top, args.out / top_file), write_gif(varied, args.out / bottom_file)
            print(f"ok {name}: {sizes[0] // 1024} KB + {sizes[1] // 1024} KB ({note})", flush=True)
        except Exception:  # noqa: BLE001 — report every cell, then fail
            failed.append(name)
            print(f"FAILED {name}\n{traceback.format_exc()}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
