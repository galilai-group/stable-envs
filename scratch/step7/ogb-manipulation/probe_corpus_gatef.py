"""Freeze the Step-4-schema corpora for the two cells, and run Gate F's arm D against swm.

**Corpus.** Reuses `examples/fidelity_step4.py::corpus_for` rather than reimplementing selection:
the pool comes from the env's own `sample_states`, and the 64 kept rows are the typical state, the
per-dimension extremes widest-dimension-first, and a farthest-point spread. Writing a second
selector would be a second notion of "extreme" that the frozen 49 were not chosen under. The pool is
saved beside the selection, which `corpus_for` does not do, because the coordinator asked for it and
because a selection whose pool is gone cannot be re-derived.

**Gate F, arm D.** swm's observation for this family is a 28-D / 40-D VECTOR, not pixels: the cells
register with `ob_type='states'`. So the thing stable-envs certifies is `render()`, and the gate's
"only render() is swapped" clause is the whole of it here -- there is no pixel observation to
replace. The arm therefore drives swm's OWN env through its OWN dynamics, reads the render-relevant
state off `data` at every step, renders those states with our cell, and pixel-compares against swm's
own `render()` on the same states. Arms that differ in simulator visit different episodes; this one
holds everything but the renderer fixed, which is the only comparison that means anything.
"""
from __future__ import annotations

import argparse, json, os, platform, socket, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--states-dir", default=None)
    ap.add_argument("--cells", nargs="+", default=["OGBCube", "OGBScene"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--steps", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    states_dir = Path(a.states_dir) if a.states_dir else (
        ROOT / "examples/results/step7/ogb-manipulation/states")

    sys.path.insert(0, str(ROOT))
    import torch
    import stable_envs as se
    from stable_envs.testing.conformance import interior_mask, ssim
    sys.path.insert(0, str(ROOT / "examples"))
    import fidelity_step4 as F

    rec = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
           "platform": platform.platform(), "mujoco_gl": os.environ.get("MUJOCO_GL"),
           "cells": {}}

    for cname in a.cells:
        cls = se.REGISTRY[cname]
        cell = {}
        env = se.make(cname, batch=a.n, device="cuda")

        # ---- corpus, in the Step-4 schema -------------------------------------------------
        pool = np.asarray(env.sample_states(a.pool, rng=np.random.default_rng(F.STATE_SEED)),
                          np.float32)
        np.savez_compressed(states_dir.parent / f"pool_{cname}.npz", pool=pool)
        states, meta = F.corpus_for(env, cname, states_dir, a.n, a.pool)
        cell["corpus"] = {k: v for k, v in meta.items() if k != "indices"}
        cell["corpus"]["n_indices"] = len(meta.get("indices", []))
        cell["corpus"]["pool_saved_to"] = str(
            (states_dir.parent / f"pool_{cname}.npz").relative_to(ROOT))
        print(f"{cname} corpus: kept {meta['kept']} of pool {meta['pool']}, "
              f"state_dim {meta['state_dim']}, sha {meta['sha256'][:12]}", flush=True)

        # ---- Gate F, arm D --------------------------------------------------------------
        import gymnasium, mujoco
        import stable_worldmodel  # noqa: F401
        from stable_envs.envs.ogb_manip import SPEC, _bundle
        b = _bundle(cls.CELL)
        sw = gymnasium.make(SPEC[cls.CELL]["swm_id"])
        ob, info = sw.reset(seed=11)
        sw.action_space.seed(11)
        u = sw.unwrapped
        gf = {"swm_id": SPEC[cls.CELL]["swm_id"],
              "observation_is_pixels": bool(np.asarray(ob).ndim >= 3),
              "observation_shape": list(np.asarray(ob).shape),
              "observation_dtype": str(np.asarray(ob).dtype),
              "reset_info_keys": sorted(info.keys()),
              "certified_artifact": ("render(): swm registers this family with ob_type='states', so "
                                     "the observation is a VECTOR and there is no pixel observation "
                                     "to swap. The frame stable-envs certifies is what render() "
                                     "returns."),
              "steps": int(a.steps)}
        got = []
        for _ in range(a.steps):
            sw.step(sw.action_space.sample())
            m, d = u.model, u.data
            bits = [float(x) for x in list(getattr(u, "_cur_button_states", []))[:b.nbuttons]]
            got.append(np.concatenate([d.qpos.copy(), d.mocap_pos.ravel().copy(),
                                       d.mocap_quat.ravel().copy(), np.asarray(bits, np.float64)]))
        st = np.stack(got).astype(np.float64)
        ours = env.render(st).cpu().numpy()
        official = cls.reference_render(cls.reference_env(), st)
        dmax = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
        im = interior_mask(official)
        gf.update({
            "n_states_from_swm_rollout": int(len(st)),
            "state_dim": int(env.state_dim),
            "shape_ours": list(ours.shape), "shape_official": list(official.shape),
            "dtype_ours": str(ours.dtype), "dtype_official": str(official.dtype),
            "shape_and_dtype_match": bool(ours.shape == official.shape
                                          and ours.dtype == official.dtype),
            "mean_abs": float(np.abs(ours.astype(np.int16) - official.astype(np.int16)).mean()),
            "within8_frac": float((dmax <= 8).mean()),
            "ssim": float(ssim(ours, official)),
            "all_exact_frac": float((dmax == 0).mean()),
            "interior_exact_frac": float((dmax[:, :-1, :-1][im] == 0).mean()),
        })
        sw.close()
        cell["gate_f_arm_d"] = gf
        print(f"{cname} GATE F arm D: obs_is_pixels={gf['observation_is_pixels']} "
              f"shape/dtype match={gf['shape_and_dtype_match']} over {a.steps} swm-driven states: "
              f"mean_abs {gf['mean_abs']:.4f} within8 {gf['within8_frac']:.4f} "
              f"ssim {gf['ssim']:.4f}", flush=True)
        rec["cells"][cname] = cell

    (out / "corpus_gatef.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
