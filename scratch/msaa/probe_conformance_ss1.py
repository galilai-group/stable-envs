"""Everything else the GPU suite asserts about a frame, with the env's supersample set to 1.

The parity survey answers one question -- would these cells still agree with the official renderer --
and a recommendation to drop supersampling has to answer the rest of them too. A frame is not only
required to match MuJoCo; it is required to be a pure function of ``(state, style, size)``,
independent of batch size, of chunking, of the order the batch arrives in and of the form the states
arrive as, and to stay consistent when the resolution changes. All of those are properties of the ray
grid, and the supersample factor IS the ray grid.

Nothing is edited to ask this. ``render`` reads ``self.ss`` when no supersample is passed, and ``ss``
is an instance attribute set from ``default_supersample`` at construction, so ``env.ss = 1`` puts the
env in exactly the state a changed default would -- through the public path, with the class untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import common as K  # noqa: E402

import numpy as np
import torch

from stable_envs.testing import conformance as C  # noqa: E402

CHECKS = ("batch_size_invariance", "chunked_render_matches", "order_independence",
          "render_is_pure", "resolution_consistency", "input_form_invariance")


def run_checks(env, states, style=None) -> dict:
    out = {}
    out["batch_size_invariance"] = C.check_batch_size_invariance(env, states[:64], style)
    out["chunked_render_matches"] = C.check_chunked_render_matches(env, states[:64], style)
    out["order_independence"] = C.check_order_independence(env, states[:64], style)
    out["render_is_pure"] = C.check_render_is_pure(env, states[:64], style)
    out["resolution_consistency"] = C.check_resolution_consistency(env, states[:16], style)
    out["input_form_invariance"] = C.check_input_form_invariance(env, states[:16], style)
    out["determinism"] = C.check_determinism(env)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="")
    ap.add_argument("--out", default=os.path.join(K.LOGS, "conformance_ss1.json"))
    a = ap.parse_args()
    cells = a.cells.split(",") if a.cells else [
        "DMCAcrobotSwingup", "DMCBallInCupCatch", "DMCCartpoleBalance", "DMCCartpoleSwingup",
        "DMCFingerSpin", "DMCFingerTurnEasy", "DMCFingerTurnHard", "DMCPendulumSwingup",
        "DMCPointMassEasy", "DMCReacherEasy", "DMCReacherHard",
        "MujocoReacher", "MujocoPusher", "MujocoInvertedPendulum", "MujocoInvertedDoublePendulum"]
    res = {"host": os.uname().nodename, "cells": {}}
    for cell in cells:
        env = K.make(cell, batch=16, size=int(K.cls_of(cell).default_size))
        states = np.asarray(env.sample_states(64, rng=np.random.default_rng(0)), np.float32)
        row = {}
        for ss in (2, 1):
            env.ss = ss
            t0 = time.perf_counter()
            try:
                row[f"ss{ss}"] = run_checks(env, states)
                row[f"ss{ss}"]["seconds"] = time.perf_counter() - t0
            except Exception as e:                     # a crash is a result, not a lost cell
                row[f"ss{ss}"] = {"error": repr(e)}
        res["cells"][cell] = row
        # The comparison that matters: does any check report a WORSE number at ss=1 than at ss=2.
        diffs = []
        for k in CHECKS:
            x, y = row["ss2"].get(k, {}), row["ss1"].get(k, {})
            if "skipped" in x or "skipped" in y:
                continue
            for field in ("max_abs", "exact_frac", "mean_abs"):
                if field in x and field in y and x[field] != y[field]:
                    diffs.append(f"{k}.{field}: ss2={x[field]} ss1={y[field]}")
        res["cells"][cell]["differences"] = diffs
        print(f"{cell}: " + ("identical at ss=1 and ss=2" if not diffs else "; ".join(diffs)),
              flush=True)
        del env
        torch.cuda.empty_cache()
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
