"""Where did the blur in the stored corpus come from?

``resize_probe.py`` ruled out the obvious answer: *no* single-stage 512 -> 224 filter reproduces the
stored frames, and every candidate comes out sharper than the corpus (flat-pixel fraction ~0.86 against
the corpus's ~0.76). A frame cannot gain blur from a filter that only ever removes detail once, so the
corpus must have been through a *round trip* — rendered small and enlarged to 224.

That is exactly what ``World(image_shape=(224,224))`` does when the env underneath renders at some other
``resolution``: ``AddPixelsWrapper`` PIL-resizes whatever it gets up to 224.

Sweep the intermediate resolution and report which one reproduces the corpus.

    python corpus_origin_probe.py --n 24
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_probe import load_rows  # noqa: E402


def two_stage(img512: np.ndarray, mid: int, size: int, up: str) -> np.ndarray:
    """The swm pipeline: env cv2-resizes the 512 canvas to ``mid``, the wrapper PIL-resizes up to 224."""
    import cv2
    from PIL import Image

    small = cv2.resize(img512, (mid, mid))
    flt = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "nearest": Image.NEAREST}[up]
    return np.array(Image.fromarray(small).resize((size, size), flt))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--mids", type=int, nargs="+", default=[64, 84, 96, 112, 128, 160, 192, 224])
    ap.add_argument("--out", default="results/corpus_origin.json")
    a = ap.parse_args()

    from stable_envs.testing.conformance import interior_mask, pixel_agreement

    states, stored = load_rows(a.n, a.seed)

    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=512)
    env.reset(seed=0)
    u = env.unwrapped
    canvases = []
    for st in states:
        u._set_state(st)
        canvases.append(u.render().copy())
    canvases = np.stack(canvases)

    res: dict = {"n": a.n, "stored_interior_frac": float(interior_mask(stored).mean())}
    print(f"[origin] stored flat-pixel fraction = {res['stored_interior_frac']:.4f}", flush=True)
    best = (None, -1.0)
    for up in ("bilinear", "bicubic", "nearest"):
        for mid in a.mids:
            v = np.stack([two_stage(c, mid, a.size, up) for c in canvases])
            m = pixel_agreement(v, stored)
            m["own_interior_frac"] = float(interior_mask(v).mean())
            res[f"{up}_{mid}"] = m
            print(
                f"[origin] mid={mid:4d} up={up:9s} exact={m['exact_frac']:.4f} "
                f"mean_abs={m['mean_abs']:.4f} flat={m['own_interior_frac']:.4f}",
                flush=True,
            )
            if m["exact_frac"] > best[1]:
                best = (f"{up}_{mid}", m["exact_frac"])
    res["best"] = {"key": best[0], "exact_frac": best[1]}
    print(f"[origin] best = {best[0]} at exact={best[1]:.4f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[origin] wrote {a.out}")


if __name__ == "__main__":
    main()
