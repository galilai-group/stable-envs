"""The torch fallback must agree with the Triton kernels, bit for bit.

`procgen.py` says the torch expression is "the readable definition the kernels are checked against",
and chaser adds a THIRD kernel (`_fill_kernel`) whose torch twin had never been run: triton imports
fine on this machine, so every parity number so far came from the fused path and the fallback was
dead code on a machine that has triton.

That is exactly the shape of bug this repo has hit before -- the `.contiguous()` note in `render()`
records a per-environment style read that was wrong and looked perfectly fine because the default
never loaded it. So: force `HAVE_TRITON = False`, render the same states, and diff against both the
Triton path and the official renderer.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(HERE, "chaser_torch_path.json")


def main():
    from stable_envs.envs import procgen as P
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {"triton_available": bool(P.HAVE_TRITON), "device": dev}
    if not P.HAVE_TRITON:
        raise SystemExit("triton is absent here, so there is nothing to compare the fallback against")

    env = P.ProcgenChaser(batch=1, device=dev)
    states = env.sample_states(64)
    st = torch.as_tensor(states, device=dev)

    fused = env.render(st).cpu().numpy().copy()
    P.HAVE_TRITON = False                       # the switch every draw method branches on
    try:
        plain = env.render(st).cpu().numpy().copy()
    finally:
        P.HAVE_TRITON = True

    d = np.abs(fused.astype(np.int32) - plain.astype(np.int32))
    rep["fused_vs_torch"] = {"frames": int(len(states)), "max_abs_diff": int(d.max()),
                             "pixels_exact_frac": float((d.max(-1) == 0).mean())}
    print("torch fallback vs triton:", rep["fused_vs_torch"], flush=True)

    # and against the reference, so a shared bug in both paths cannot pass
    ref = P.ProcgenChaser.reference_render(P.ProcgenChaser.reference_env(), states)
    dr = np.abs(plain.astype(np.int32) - ref.astype(np.int32))
    rep["torch_vs_official"] = {"max_abs_diff": int(dr.max()),
                                "pixels_exact_frac": float((dr.max(-1) == 0).mean())}
    print("torch fallback vs official:", rep["torch_vs_official"], flush=True)

    # the same for a per-environment style, which is the path the `.contiguous()` note is about
    sp = env.variation_space
    sty = env.sample_variation(batch=64) if hasattr(env, "sample_variation") else None
    if sty is not None:
        fu = env.render(st, style=sty).cpu().numpy().copy()
        P.HAVE_TRITON = False
        try:
            pl = env.render(st, style=sty).cpu().numpy().copy()
        finally:
            P.HAVE_TRITON = True
        ds = np.abs(fu.astype(np.int32) - pl.astype(np.int32))
        rep["per_env_style_fused_vs_torch"] = {"max_abs_diff": int(ds.max())}
        print("per-env style, torch vs triton:", rep["per_env_style_fused_vs_torch"], flush=True)
    else:
        rep["per_env_style_fused_vs_torch"] = {"skipped": "no sample_variation on this env"}
    _ = sp

    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
