"""Side-by-side frames and difference maps, so "imperceptible" is a claim a reader can check.

Every number in ``probe_budget_frontier.py`` is an average over a frame, and an average hides where
the error is. This writes, for one state:

    official | shipped render | candidate render | JPEG q95 of official

on the top row and, underneath, each one's absolute difference from the official frame on a shared
scale, with the scale printed. The JPEG column is the point: it is what a stored dataset hands the
consumer, and if the candidate's difference map is not obviously worse than the JPEG's then the
comparison the whole argument rests on is visible rather than asserted.

The difference maps are amplified by a stated factor because at 1x an error of 6/255 is invisible,
which is the finding and also the reason a raw map proves nothing.
"""
from __future__ import annotations

import argparse
import io
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from probe_budget_frontier import config_list, env_module, jpeg_roundtrip, patched, upsample


def label(img: np.ndarray, text: str, pad: int = 18) -> np.ndarray:
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    out = np.zeros((h + pad, w, 3), np.uint8)
    out[pad:] = img
    im = Image.fromarray(out)
    ImageDraw.Draw(im).text((3, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--config", default="r128")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--amplify", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import stable_envs as se
    from PIL import Image
    cls = se.REGISTRY[a.env]
    mod = env_module(a.env)
    family = "RoboCasa" if a.env.startswith("RoboCasa") else "HumanoidBench"

    ref = cls.reference_env(a.size)
    probe = se.make(a.env, batch=1, size=a.size, device="cuda")
    states = probe.sample_states(a.state + 1)
    official = cls.reference_render(ref, states)[a.state:a.state + 1]
    del probe
    mod._CTX_CACHE.clear()

    cols = [("official (mujoco.Renderer)", official[0])]
    for name in ("shipped", a.config):
        r, mode, ov, lod = config_list(family)[name]
        orig = patched(mod, ov, lod)
        try:
            env = se.make(a.env, batch=1, size=a.size, device="cuda")
            low = env.render(states[a.state:a.state + 1], size=r)
            cols.append((f"{name} (render {r}px, {mode})",
                         upsample(low, a.size, mode).cpu().numpy()[0]))
            del env, low
        finally:
            mod.MW.create_render_context = orig
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()
    jo, _ = jpeg_roundtrip(official, 95)
    cols.append(("JPEG q95 of official (the stored dataset)", jo[0]))

    top, bot = [], []
    for name, img in cols:
        d = np.abs(img.astype(np.int16) - official[0].astype(np.int16)).max(-1)
        top.append(label(img, name))
        bot.append(label(np.repeat(np.clip(d * a.amplify, 0, 255).astype(np.uint8)[..., None], 3, 2),
                         f"|diff| x{a.amplify}   mean {d.mean():.2f}/255   max {d.max()}"))
    grid = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
    out = a.out or (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/"
                    f"sbs_{a.env}_{a.config}.png")
    Image.fromarray(grid).save(out)
    print("wrote", out, grid.shape, flush=True)


if __name__ == "__main__":
    main()
