"""Does the OFFICIAL renderer draw the same styled scene the cheap one does, on HumanoidBench?

The whole HumanoidBench variation experiment rests on one thing: ``gen_pairs._hb_reference_style``
writes the env's own camera and light into the reference ``MjModel``, and if it writes them even
slightly differently from the way ``humanoidbench.render`` writes them into MJWarp's per-world
arrays, then every pair in the split is a frame of one scene against a frame of another. That failure
is not subtle in the numbers -- README.md §8 records it as mean 41.6 and SSIM 0.22 on RoboCasa -- but
it IS silent in the pictures, and it would be reported as "the corrector improves a lot", because a
corrector fitted on a mismatched pair learns the mismatch.

So this measures the SHIPPED 256 px render against the official frame on the styled split and on the
default one, and prints them side by side. If the styled row is close to the default row, the two
renderers were given the same scene; if it is several times worse, they were not.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

import data as D
from eval_corrector import full_metrics, shipped_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--tags", default="r112_tex1,r112_tex1_sty")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = []
    for tag in a.tags.split(","):
        lr, hr, states, sty, meta, _, _ = D.load_split(a.env, a.split, tag, a.limit)
        ship = shipped_frames(a.env, states, style=sty)
        dyn = D.dynamic_mask(hr)
        m = full_metrics(ship, hr, dyn)
        m.update(tag=tag, n=len(hr), styled=bool(meta.get("style")),
                 style_leaves=meta.get("style_leaves"))
        rows.append(m)
        print(f"{tag:20s} styled={m['styled']}  shipped-vs-official  mean {m['mean_abs']:7.3f}  "
              f"w8 {m['within8_frac']:.3f}  ssim {m['ssim']:.4f}  "
              f"iexact {m['interior_exact_frac']:.3f}  max {m['max_abs']}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1, default=float)
        print("wrote", a.out)
    if len(rows) == 2:
        r = rows[1]["mean_abs"] / max(rows[0]["mean_abs"], 1e-9)
        print(f"\nstyled / default shipped error = {r:.2f}x  "
              f"({'consistent' if r < 2.5 else 'THE TWO RENDERERS WERE GIVEN DIFFERENT SCENES'})")


if __name__ == "__main__":
    main()
