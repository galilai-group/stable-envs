"""M3 — measure the OGBCube/OGBScene cells' visual fidelity, and exercise Gate A.

Three INDEPENDENT draws of 64 states each, per cell, against swm's own EGL frames at the registered
resolution. Three draws rather than one because the declared number has to survive a state draw:
`CubeEnv.reset` resamples its default variations from an RNG the gym seed does not control, so two
64-state pools are genuinely different scenes and a threshold set from a single pool is a tripwire.
The declaration takes the WORST of the three.

Gate A checks are run here too, because they need the real render path on a GPU:
determinism, batch invariance, order invariance, and state sufficiency.
"""
from __future__ import annotations

import argparse, json, os, platform, socket, sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=["OGBCube", "OGBScene"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    import stable_envs as se
    from stable_envs.testing.conformance import interior_mask, ssim

    rec = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
           "platform": platform.platform(), "mujoco_gl": os.environ.get("MUJOCO_GL"),
           "n": a.n, "draws": a.draws, "cells": {}}

    for cname in a.cells:
        cls = se.REGISTRY[cname]
        cell = {"name": cname, "size": int(cls.default_size), "draws": [], "gate_a": {}}
        env = se.make(cname, batch=a.n, device="cuda")
        ref = cls.reference_env()
        cell["state_dim"] = int(env.state_dim)
        cell["state_names_head"] = list(env.state_names)[:6]

        for k in range(a.draws):
            st = env.sample_states(a.n, np.random.default_rng(1000 + k))
            ours = env.render(st).cpu().numpy()
            off = cls.reference_render(ref, st)
            d = np.abs(ours.astype(np.int16) - off.astype(np.int16))
            dmax = d.max(-1)
            im = interior_mask(off)
            row = {"draw": k,
                   "mean_abs": float(d.mean()),
                   "within8_frac": float((dmax <= 8).mean()),
                   "ssim": float(ssim(ours, off)),
                   "all_exact_frac": float((dmax == 0).mean()),
                   "interior_frac": float(im.mean()),
                   "interior_exact_frac": float((dmax[:, :-1, :-1][im] == 0).mean()),
                   "max_abs": int(dmax.max())}
            cell["draws"].append(row)
            print(f"{cname} draw {k}: mean_abs {row['mean_abs']:.4f}  within8 {row['within8_frac']:.4f}"
                  f"  ssim {row['ssim']:.4f}  all_exact {row['all_exact_frac']:.4f}"
                  f"  interior_exact {row['interior_exact_frac']:.4f}", flush=True)
            if k == 0:
                np.savez_compressed(out / f"{cname}_frames.npz", ours=ours[:4], official=off[:4],
                                    states=st[:4])

        worst = {"mean_abs": max(r["mean_abs"] for r in cell["draws"]),
                 "within8_frac": min(r["within8_frac"] for r in cell["draws"]),
                 "ssim": min(r["ssim"] for r in cell["draws"])}
        cell["worst_of_draws"] = worst
        print(f"{cname} WORST-OF-{a.draws}: mean_abs {worst['mean_abs']:.6f}  "
              f"within8 {worst['within8_frac']:.6f}  ssim {worst['ssim']:.6f}", flush=True)
        print(f"{cname} PASTE INTO _MEASURED: \"{cname[3:]}\": "
              f"({worst['mean_abs']:.6f}, {worst['within8_frac']:.6f}, {worst['ssim']:.6f}),",
              flush=True)

        # ---- Gate A ------------------------------------------------------------------
        st = env.sample_states(8, np.random.default_rng(7))
        g = {}
        f1 = env.render(st).cpu().numpy()
        f2 = env.render(st).cpu().numpy()
        g["determinism_identical"] = bool(np.array_equal(f1, f2))
        # batch invariance: one at a time must equal the batch
        one = np.stack([env.render(st[i:i + 1]).cpu().numpy()[0] for i in range(len(st))])
        g["batch_invariance_identical"] = bool(np.array_equal(one, f1))
        g["batch_invariance_max_abs"] = int(np.abs(one.astype(np.int16) - f1.astype(np.int16)).max())
        # order invariance: a permutation of the batch must permute the frames
        perm = np.random.default_rng(3).permutation(len(st))
        fp = env.render(st[perm]).cpu().numpy()
        g["order_invariance_identical"] = bool(np.array_equal(fp, f1[perm]))
        g["order_invariance_max_abs"] = int(
            np.abs(fp.astype(np.int16) - f1[perm].astype(np.int16)).max())
        # state sufficiency: same state from a different construction must give the same frame
        env2 = se.make(cname, batch=len(st), device="cuda")
        f3 = env2.render(st).cpu().numpy()
        g["state_sufficiency_identical"] = bool(np.array_equal(f3, f1))
        g["state_sufficiency_max_abs"] = int(np.abs(f3.astype(np.int16) - f1.astype(np.int16)).max())
        # Every state dimension must matter, or it should not be in the state.
        #
        # The perturbation is PER DIMENSION KIND, and the first version of this probe was wrong
        # about one of them. Adding 0.05 to `mocap_quat[0]` -- the w component of a quaternion that
        # is (1,0,0,0) -- gives (1.05,0,0,0), which normalises back to exactly (1,0,0,0): the same
        # rotation, so zero pixels move and the dimension reads as inert when it is not. A quaternion
        # is perturbed as a ROTATION (a small rotation about the axis that component belongs to) and
        # a discrete bit is FLIPPED rather than nudged, since 0 + 0.05 rounds back to 0.
        names = list(env.state_names)
        moved = []
        for j in range(env.state_dim):
            s2 = st.copy()
            nm = names[j]
            if nm.startswith("mocap_quat["):
                k = int(nm.split("[")[1].rstrip("]")) % 4
                ang = 0.3
                q = s2[:, j - k: j - k + 4].copy()
                axis = np.zeros(3); axis[(k - 1) % 3] = 1.0
                dq = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * axis)])
                w0, v0 = q[:, :1], q[:, 1:]
                w1, v1 = dq[0], dq[1:]
                s2[:, j - k] = w0[:, 0] * w1 - v0 @ v1
                s2[:, j - k + 1: j - k + 4] = w0 * v1 + w1 * v0 + np.cross(v0, np.broadcast_to(v1, v0.shape))
            elif nm.startswith("button_state_"):
                s2[:, j] = 1.0 - s2[:, j]
            else:
                s2[:, j] += 0.05
            fj = env.render(s2).cpu().numpy()
            moved.append(int(np.abs(fj.astype(np.int16) - f1.astype(np.int16)).max()))
        g["per_dim_max_abs"] = moved
        g["perturbation"] = ("+0.05 for a scalar; a 0.3 rad ROTATION for a quaternion component "
                             "(adding to w and renormalising is a no-op); a BIT FLIP for a discrete "
                             "state bit (0 + 0.05 rounds back to 0)")
        g["dims_that_move_no_pixel"] = [int(j) for j, v in enumerate(moved) if v == 0]
        g["dim_names_that_move_no_pixel"] = [env.state_names[j] for j in g["dims_that_move_no_pixel"]]
        cell["gate_a"] = g
        print(f"{cname} GATE A: det {g['determinism_identical']}  batch {g['batch_invariance_identical']}"
              f"  order {g['order_invariance_identical']}  suff {g['state_sufficiency_identical']}"
              f"  inert dims {len(g['dims_that_move_no_pixel'])}/{env.state_dim}", flush=True)
        rec["cells"][cname] = cell

    (out / "m3_fidelity.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
