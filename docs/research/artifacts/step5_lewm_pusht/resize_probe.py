"""Which downsampling filter produced the stored corpus?

The parity probe turned up an asymmetry that has nothing to do with this repo: the frames stored in
``pusht_expert_train.h5`` are *blurrier* than the frames the live ``swm/PushT-v1`` env produces for the
same state. Measured on ``interior_mask`` — the fraction of pixels whose 3x3 neighbourhood is one flat
colour — the live env gives 0.867 and the stored corpus 0.757. A wider non-flat band around every edge
is the signature of an anti-aliased (support-scaled) downsample, whereas ``cv2.resize(..., INTER_LINEAR)``
from 512 to 224 is a sparse bilinear *sample* and leaves edges crisp.

This matters for the renderer swap, and only for one of its three surfaces. The live observation is
compared against the live env, so a filter mismatch there would show up in the parity probe (it did not:
interior agreement is exactly 1.0). But the *dataset-sourced* frames — the planner's first observation
and, more importantly, the goal image its cost is measured against — are read off disk in the official
run and re-rendered in the swapped run. If the corpus was written with a different filter than the one
used to re-render, the swap changes image statistics on top of changing the rasteriser, and a success-rate
difference could not be attributed to either.

So: render one 512 frame per state through the *reference* env, push it through every plausible
downsample, and see which one reproduces the stored bytes.

    python resize_probe.py --n 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_probe import load_rows  # noqa: E402


def downsamples(img512: np.ndarray, size: int) -> dict[str, np.ndarray]:
    """Every filter that plausibly produced a 224 px PushT corpus from the 512 px canvas."""
    import cv2
    from PIL import Image

    out = {
        "cv2_linear": cv2.resize(img512, (size, size), interpolation=cv2.INTER_LINEAR),
        "cv2_area": cv2.resize(img512, (size, size), interpolation=cv2.INTER_AREA),
        "cv2_nearest": cv2.resize(img512, (size, size), interpolation=cv2.INTER_NEAREST),
        "cv2_cubic": cv2.resize(img512, (size, size), interpolation=cv2.INTER_CUBIC),
    }
    pil = Image.fromarray(img512)
    for name, flt in (
        ("pil_bilinear", Image.BILINEAR),
        ("pil_bicubic", Image.BICUBIC),
        ("pil_nearest", Image.NEAREST),
        ("pil_box", Image.BOX),
        ("pil_lanczos", Image.LANCZOS),
    ):
        out[name] = np.array(pil.resize((size, size), flt))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="results/resize_probe.json")
    a = ap.parse_args()

    from stable_envs.testing.conformance import interior_mask, pixel_agreement

    states, stored = load_rows(a.n, a.seed)

    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    # resolution=512 makes the env's own cv2.resize a no-op, so this is the raw pygame canvas.
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=512)
    env.reset(seed=0)
    u = env.unwrapped
    canvases, poses = [], []
    for st in states:
        u._set_state(st)
        canvases.append(u.render().copy())
        poses.append(u._get_obs())
    canvases = np.stack(canvases)
    poses = np.stack(poses)

    res: dict = {"n": a.n, "size": a.size, "stored_interior_frac": float(interior_mask(stored).mean())}
    cands = {k: np.stack([downsamples(c, a.size)[k] for c in canvases]) for k in downsamples(canvases[0], a.size)}
    for k, v in cands.items():
        m = pixel_agreement(v, stored)
        res[k] = m
        print(
            f"[resize] {k:14s} vs stored: exact={m['exact_frac']:.4f} "
            f"interior_exact={m['interior_exact_frac']:.4f} mean_abs={m['mean_abs']:.4f} "
            f"own_interior_frac={float(interior_mask(v).mean()):.4f}",
            flush=True,
        )

    # And the same filters applied to OUR 512 raster, which is what the swap would use.
    from swap_renderer import _Renderer, _variation_from_env

    var = _variation_from_env(u)
    ours512 = _Renderer(size=512, mode="direct")(poses[:, :5], var)
    for k in cands:
        v = np.stack([downsamples(c, a.size)[k] for c in ours512])
        m = pixel_agreement(v, stored)
        res[f"ours_{k}"] = m
        print(
            f"[resize] ours_{k:9s} vs stored: exact={m['exact_frac']:.4f} "
            f"interior_exact={m['interior_exact_frac']:.4f} mean_abs={m['mean_abs']:.4f}",
            flush=True,
        )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[resize] wrote {a.out}")


if __name__ == "__main__":
    main()
