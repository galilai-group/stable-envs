"""Shared harness for the nvdiffrast probes: reference frames, scoring, and one-process discipline.

Two MuJoCo envs alive in one process can return byte-identical frames from cameras that provably
differ, so every probe here builds ONE scene per process and is launched once per scene.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")


def interior_mask(frames: np.ndarray) -> np.ndarray:
    """The conformance suite's definition: a pixel whose 3x3 neighbourhood in the REFERENCE is flat."""
    import scipy.ndimage as ndi
    out = []
    for f in frames:
        g = f.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
                + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
        out.append(ndi.minimum_filter(flat, size=3))
    return np.stack(out)


def score(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """``a`` ours, ``b`` reference. Optional extra ``mask`` (N,H,W) restricts every number to a region."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    if mask is not None:
        sel = mask.astype(bool)
        if sel.sum() == 0:
            return {"n": 0}
        dm = d[sel]
        res = {"n": int(sel.sum()), "mean_abs": float(dm.mean()),
               "exact_frac": float((dm == 0).mean()), "within8_frac": float((dm <= 8).mean()),
               "max_abs": int(dm.max())}
        im = interior_mask(b)
        both = im & sel[:, :-1, :-1]
        if both.sum():
            di = d[:, :-1, :-1][both]
            res.update(interior_frac=float(both.mean()), interior_exact=float((di == 0).mean()),
                       interior_mean=float(di.mean()))
        else:
            res.update(interior_frac=0.0, interior_exact=None, interior_mean=None)
        return res
    res = {"n": int(d.size), "mean_abs": float(d.mean()), "exact_frac": float((d == 0).mean()),
           "within8_frac": float((d <= 8).mean()), "max_abs": int(d.max())}
    im = interior_mask(b)
    di = d[:, :-1, :-1][im]
    if di.size:
        res.update(interior_frac=float(im.mean()), interior_exact=float((di == 0).mean()),
                   interior_mean=float(di.mean()))
    else:
        res.update(interior_frac=0.0, interior_exact=None, interior_mean=None)
    return res


def emit(tag: str, obj) -> None:
    print(f"@@{tag} " + json.dumps(obj, default=float), flush=True)


def save(name: str, obj) -> None:
    d = "/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff"
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
