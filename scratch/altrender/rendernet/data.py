"""Loading the generated pairs, and the two derived masks every honest evaluation of them needs.

A corrector on a fixed camera in a fixed scene has an easy way to score well that has nothing to do
with correcting anything: memorise the frame. Most of a RoboCasa agentview is a counter, a
backsplash and a cabinet that do not move for any state, so a network that learned "paint marble at
these pixels" and ignored its input would take a large bite out of the mean error and would be
useless. The two masks here are what separates the two hypotheses:

* ``dynamic_mask`` -- pixels whose OFFICIAL value varies across the evaluation states. Error here can
  only be earned by reading the input.
* its complement, the static pixels, where a lookup table scores full marks.

Reporting the mean over the whole frame and the mean over the dynamic pixels side by side is the
whole difference between "the corrector works" and "the corrector memorised the room".
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = "/mnt/vast/home/randall.balestriero/data/rendernet"


def split_dir(env: str, split: str, tag: str) -> str:
    return os.path.join(ROOT, env, f"{split}_{tag}")


def geom_table(env: str):
    """``(ngeom+1, F)`` descriptor table written beside the env's data by ``gen_aux.py``."""
    p = os.path.join(ROOT, env, "geom_features.npy")
    return np.load(p) if os.path.exists(p) else None


def load_split(env: str, split: str, tag: str, limit: int | None = None):
    """``(lr, hr, states, style|None, meta, depth|None, segid|None)`` over shards, in shard order.

    The aux planes are returned as the last two elements rather than folded into a dict so that every
    existing caller breaks loudly on the shape rather than quietly ignoring them.
    """
    d = split_dir(env, split, tag)
    shards = sorted(glob.glob(os.path.join(d, "lr_*.npy")),
                    key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    if not shards:
        raise FileNotFoundError(f"no shards under {d}")
    def cat(kind):
        return np.concatenate([np.load(p.replace("/lr_", f"/{kind}_")) for p in shards])
    lr, hr, st = cat("lr"), cat("hr"), cat("states")
    sty = None
    if os.path.exists(shards[0].replace("/lr_", "/style_")):
        sty = cat("style")
    depth = seg = None
    if os.path.exists(shards[0].replace("/lr_", "/depth_")):
        depth, seg = cat("depth"), cat("segid").astype(np.int32)
    meta = json.load(open(shards[0].replace("/lr_", "/meta_").replace(".npy", ".json")))
    meta["n_total"] = len(lr)
    meta["shards"] = len(shards)
    if limit:
        lr, hr, st = lr[:limit], hr[:limit], st[:limit]
        sty = None if sty is None else sty[:limit]
        depth = None if depth is None else depth[:limit]
        seg = None if seg is None else seg[:limit]
    meta["has_aux"] = depth is not None
    return lr, hr, st, sty, meta, depth, seg


def dynamic_mask(hr: np.ndarray, frac: float = 0.25) -> np.ndarray:
    """``(H, W)`` bool: the ``frac`` of pixels whose official value moves most across the states.

    Standard deviation over the state axis, maxed over colour. ``frac`` is a quantile rather than a
    threshold so the mask has the same size on every cell and the numbers taken through it are
    comparable across cells that have very different amounts of moving content.
    """
    x = hr.astype(np.float32)
    sd = x.std(0).max(-1)
    thr = np.quantile(sd, 1.0 - frac)
    return sd >= thr


def to_gpu_uint8(a: np.ndarray, device="cuda"):
    import torch
    return torch.as_tensor(a, device=device)
