"""Attribute the ``check_state_roundtrip`` failure the closure sweep found on OGBCube and OGBScene.

``stable_envs/testing/conformance.py::check_state_roundtrip`` asserts that ``set_state(get_state())``
is the identity **to float32 exactly** -- ``tests/gpu/test_conformance.py`` spells the bound as
``max_abs_err == 0.0`` -- and both cells return ~1.2e-07 on a 3090. Nobody had run it on these two:
W5's Gate-A list covers determinism, batch, batch-size, order and state_dims_matter, and the tip
conformance run (job 6411148) was in the ``se7-rocket`` interpreter, where OGBench is not installed
and every OGB cell SKIPS.

This probe does not fix anything and touches no library file. It reports, per state dimension and by
name: the max absolute round-trip error, whether the error survives a second round trip (a one-shot
projection onto a constraint surface converges; a genuine loss of information does not), and the
relative size of the error against the value, so the report can say whether this is a quaternion
being renormalised, a float64->float32 narrowing, or a dimension that is genuinely not stored.

    python scratch/step7/closure/probe_ogb_state_roundtrip.py --out DIR/ogb_state_roundtrip.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="OGBCube,OGBScene")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    import torch
    import stable_envs as se

    rec = {"host": platform.node(), "device": a.device, "torch": torch.__version__,
           "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "git_sha": os.environ.get("STABLE_ENVS_GIT_COMMIT"),
           "bound": "conformance.check_state_roundtrip: max_abs_err == 0.0 exactly "
                    "(tests/gpu/test_conformance.py::test_state_roundtrip)",
           "cells": {}}
    for name in a.cells.split(","):
        if name not in se.REGISTRY:
            rec["cells"][name] = {"missing": "not registered in this interpreter"}
            continue
        env = se.make(name, device=a.device)
        st = env.sample_states(env.B)
        st_np = np.asarray(st.cpu().numpy() if hasattr(st, "cpu") else st, np.float64)
        env.set_state(st)
        back1 = env.get_state().cpu().numpy().astype(np.float64)
        env.set_state(env.get_state())
        back2 = env.get_state().cpu().numpy().astype(np.float64)

        e1 = np.abs(back1 - st_np)
        e2 = np.abs(back2 - back1)
        names = list(env.state_names or [f"dim{i}" for i in range(st_np.shape[1])])
        per_dim = []
        for i, nm in enumerate(names):
            m1, m2 = float(e1[:, i].max()), float(e2[:, i].max())
            v = float(np.abs(st_np[:, i]).max())
            per_dim.append({"i": i, "name": nm, "max_abs_err_first": m1,
                            "max_abs_err_second": m2, "max_abs_value": v,
                            "relative": (m1 / v) if v else None,
                            "float32_ulp_at_that_scale": float(np.spacing(np.float32(v))) if v else None})
        bad = [d for d in per_dim if d["max_abs_err_first"] > 0]
        rec["cells"][name] = {
            "state_dim": int(st_np.shape[1]), "batch": int(st_np.shape[0]),
            "max_abs_err_first_roundtrip": float(e1.max()),
            "max_abs_err_second_roundtrip": float(e2.max()),
            "idempotent_after_one_roundtrip": bool(e2.max() == 0.0),
            "n_dims_with_error": len(bad),
            "dims_with_error": bad,
            "per_dim": per_dim}
        print(f"[{name}] first {e1.max():.6g}  second {e2.max():.6g}  "
              f"idempotent={e2.max() == 0.0}  dims={[d['name'] for d in bad]}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
