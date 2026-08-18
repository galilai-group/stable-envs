"""``(cheap render at r px, OFFICIAL 256 px frame)`` pairs for one Meta-World task, split by EPISODE.

The split is the reason any number downstream means anything, and on Meta-World the unit of a split
is the EPISODE, not the state. Meta-World writes the episode's placement into ``model.body_pos`` and
``model.site_pos`` at reset, so two states of one episode share their whole static scene -- the same
door in the same place, the same goal marker, the same plate. Splitting states would put the same
scene on both sides and measure a lookup table. So:

* **train** -- episodes from ``_seeded_reset`` seeds 20_000+,
* **heldout** -- episodes from seeds 30_000+, disjoint,
* **decl** -- the env's own ``sample_states`` at seeds 0..3, which draws from the SHIPPED pool
  (seeds 10_000..10_063) and is therefore disjoint from both. This is stronger than the RoboCasa
  study's ``decl``, which overlapped its training pool by construction.

Within an episode the states are the env's own generative process: 0-25 random actions out of the
reset, and a random yaw on every free joint (``common.episode_states``, a copy of ``_state_pool``
with the seed base made an argument).

Several resolutions are written in one pass. The official half is rendered once and HARD-LINKED into
every resolution's directory, so a frontier over r costs one MuJoCo pass and not four.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--split", required=True, choices=("train", "heldout", "decl"))
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--per-episode", type=int, default=4)
    ap.add_argument("--res", default="128,112")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--style", action="store_true")
    ap.add_argument("--frac-default", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--noblend", action="store_true",
                    help="cheap render WITHOUT the translucent second layer pass -- the lever the "
                         "two window tasks need (probe_window.py), which halves their frame and "
                         "paints the pane the way mujoco_warp's own shader does")
    a = ap.parse_args()

    import stable_envs as se
    cls = se.REGISTRY[a.env]
    sizes = [int(r) for r in a.res.split(",") if r]
    if a.noblend:
        C.mw().MetaWorldBase._blend = lambda self, c, rgb, plate, n, alpha: rgb

    # ---- states ------------------------------------------------------------------------------
    t0 = time.perf_counter()
    if a.split == "decl":
        allst = []
        for k in range(4):
            p = se.make(a.env, batch=1, size=a.size, device="cuda", seed=k)
            allst.append(p.sample_states(64))
            del p
        states = np.concatenate(allst).astype(np.float32)
        epi = np.repeat(np.arange(4), 64)                 # the draw, not the episode
        C.clear_caches()
    else:
        base = C.SEED_TRAIN if a.split == "train" else C.SEED_HELD
        states = C.episode_states(cls.TASK, a.episodes, base, a.per_episode,
                                  rng_seed=a.seed + (0 if a.split == "train" else 77))
        epi = C.episode_index(a.episodes, a.per_episode)
    print(f"{a.env} {a.split}: {len(states)} states from {len(np.unique(epi))} episodes in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    # ---- style -------------------------------------------------------------------------------
    probe = se.make(a.env, batch=1, size=a.size, device="cuda", seed=a.seed)
    style = None
    if a.style:
        rng = np.random.default_rng(a.seed + 7)
        style = np.asarray(probe.sample_style(len(states), rng=rng), np.float32)
        dflt = np.asarray(probe.default_style(), np.float32)
        pin = rng.random(len(states)) < a.frac_default      # a declaration is measured at the default
        style[pin] = dflt
    leaves = [p for p, _, _ in probe.variation_space.flat_layout()]

    # ---- official frames ---------------------------------------------------------------------
    t0 = time.perf_counter()
    hr = C.official_frames(cls, states, a.size, style, probe)
    print(f"official: {len(hr)} frames in {time.perf_counter() - t0:.1f}s "
          f"({len(hr) / (time.perf_counter() - t0):.1f} f/s)", flush=True)
    del probe
    C.clear_caches()

    # ---- cheap renders -----------------------------------------------------------------------
    first = None
    for r in sizes:
        tag = f"r{r}" + ("_noblend" if a.noblend else "") + ("_sty" if a.style else "")
        outdir = os.path.join(C.DATA, a.env, f"{a.split}_{tag}")
        os.makedirs(outdir, exist_ok=True)
        env = se.make(a.env, batch=1, size=a.size, device="cuda", seed=a.seed)
        lr = np.empty((len(states), r, r, 3), np.uint8)
        t0 = time.perf_counter()
        for i in range(0, len(states), a.chunk):
            j = min(i + a.chunk, len(states))
            sty = None if style is None else style[i:j]
            lr[i:j] = env.render(states[i:j], size=r, style=sty).cpu().numpy()
        print(f"cheap r{r}: {len(lr)} in {time.perf_counter() - t0:.1f}s", flush=True)
        del env
        C.clear_caches()

        np.save(os.path.join(outdir, "lr_0.npy"), lr)
        np.save(os.path.join(outdir, "states_0.npy"), states)
        np.save(os.path.join(outdir, "epi_0.npy"), epi)
        if style is not None:
            np.save(os.path.join(outdir, "style_0.npy"), style)
        hrp = os.path.join(outdir, "hr_0.npy")
        if first is None:
            np.save(hrp, hr)
            first = hrp
        else:                                   # one official pass, hard-linked into every res dir
            if os.path.exists(hrp):
                os.unlink(hrp)
            os.link(first, hrp)
        meta = {"env": a.env, "task": cls.TASK, "split": a.split, "n": len(states),
                "render_size": r, "size": a.size, "episodes": int(len(np.unique(epi))),
                "per_episode": a.per_episode, "style": a.style, "noblend": a.noblend,
                "style_leaves": leaves if a.style else None,
                "frac_default": a.frac_default if a.style else None, "seed": a.seed,
                "split_rule": ("episodes from _seeded_reset seeds "
                               f"{C.SEED_TRAIN}+ (train) / {C.SEED_HELD}+ (heldout), disjoint from "
                               "each other and from the shipped pool's 10_000..10_063; decl is the "
                               "env's own sample_states over the shipped pool")}
        with open(os.path.join(outdir, "meta_0.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print("wrote", outdir, flush=True)


if __name__ == "__main__":
    main()
