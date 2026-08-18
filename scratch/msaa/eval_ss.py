"""A trained corrector, scored the way the parity test scores a renderer, along the shrink frontier.

The single number this study needs is not "did the mean go down". It is the TRADE, and the trade has
a knob: how large a residual is allowed to survive quantisation. Soft-shrinking the residual by ``t``
half-levels makes every pixel where the corrector had less than ``t``/255 to say quantise back to the
input's own byte, so:

* ``t = 0``   -- the corrector's full opinion. Best tail, worst exactness, and on a parity cell it
                 destroys interior-exactness outright.
* ``t`` large -- the corrector only speaks where it is confident, every other pixel is bit-identical
                 to the ss=1 render, and interior-exactness comes back by construction.

Sweeping ``t`` traces the whole frontier from "a learned approximation" to "the ss=1 render with a few
edge pixels repaired", and the row a cell would actually ship is somewhere on it. That is the table a
decision can be made from, which is why this reports every metric at every ``t`` rather than picking
one.

Both evaluation sets are reported and they answer different questions. ``heldout`` is 2,048 states the
fit never saw -- the generalisation number. ``decl`` is the parity test's own 64-state draw at
``default_rng(20260811)`` -- the exact frames the assertion is computed on, so a number on it is
directly comparable to the floor the cell has to clear.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import common as K  # noqa: E402

import numpy as np
import torch

import nets_ss  # noqa: E402
import train_ss as T  # noqa: E402


def evaluate(ck_path: str, shrinks, splits=("heldout", "decl")) -> dict:
    ck = torch.load(ck_path, map_location="cuda", weights_only=False)
    cell, size, net = ck["cell"], ck["size"], ck["net"]
    model = nets_ss.build(net).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(ck["state_dict"])
    model = model.eval()
    out = {"ckpt": os.path.basename(ck_path), "cell": cell, "size": size, "net": net,
           "args": ck["args"], "best_epoch": ck["best_epoch"],
           "params": nets_ss.n_params(model),
           "gflops_per_frame": nets_ss.gflops_per_frame(net, size), "splits": {}}
    for split in splits:
        d = T.load(cell, size, split)
        x = torch.as_tensor(d["ss1"], device="cuda")
        y = torch.as_tensor(d["official"], device="cuda")
        ss2 = torch.as_tensor(d["ss2"], device="cuda")
        im = torch.as_tensor(T.interior_mask_cached(cell, size, split, d["official"]), device="cuda")
        rows = {"ss1_do_nothing": T.metrics(x, y, im), "ss2_shipped": T.metrics(ss2, y, im)}
        for t in shrinks:
            rows[f"corrected_shrink{t:g}"] = T.metrics(T.predict(model, x, t), y, im)
        # Against our own ss=2 rather than the official frame: how much of the MSAA itself was
        # learned, separated from how far ss=2 is from MuJoCo in the first place.
        rows["corrected_vs_ss2_shrink0"] = T.metrics(T.predict(model, x, 0.0), ss2, None)
        out["splits"][split] = rows
        print(f"\n=== {os.path.basename(ck_path)} / {split} ({len(x)} frames) ===", flush=True)
        for k, m in rows.items():
            ie = m.get("interior_exact_frac")
            print(f"  {k:26s} mean {m['mean_abs']:7.4f}  exact {m['exact_frac']:.4f}  "
                  f"w8 {m['within8_frac']:.4f}  max {m['max_abs']:3d}  p99.9 {m['p999']:5.0f}  "
                  f"p99.99 {m['p9999']:5.0f}  int-exact "
                  f"{('n/a' if ie is None else f'{ie:.8f}')}  "
                  f"bad {m.get('interior_bad_px', 0)}/{m.get('interior_px', 0)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(K.CKPT, "*.pt"))
    ap.add_argument("--shrinks", default="0,0.5,1,2,4,8,16")
    ap.add_argument("--out", default=os.path.join(K.LOGS, "eval.json"))
    a = ap.parse_args()
    shrinks = [float(s) for s in a.shrinks.split(",")]
    res = []
    for p in sorted(glob.glob(a.glob)):
        try:
            res.append(evaluate(p, shrinks))
        except Exception as e:
            print(f"{os.path.basename(p)}: {e!r}", flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print("\nwrote", a.out, flush=True)


if __name__ == "__main__":
    main()
