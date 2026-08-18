"""What does the STORED baseline already cost, in the same four metrics the tier declares?

Every throughput target in this campaign is "beat reading the env's own frames back off disk". Those
frames are JPEG-q95 in Lance -- not the official pixels, and not even our own pixels. So before any
approximation is judged, the incumbent has to be measured on the same ruler:

  ``ours``            the env as shipped, against MuJoCo's own raster. This reproduces the declaration.
  ``jpeg(official)``  the OFFICIAL frame through libjpeg q95 and back, against itself. THE BUDGET: an
                      approximation that costs less than this is, for a consumer reading stored
                      frames, not a degradation at all.
  ``jpeg(ours)``      our frame through the same codec, against our own frame -- what storage costs on
                      the pixels we would actually be storing.
  ``jpeg(ours) vs official`` what a consumer of the stored dataset is really handed.

Also dumps a side-by-side and an amplified difference map, because the point of the budget is that a
human should not be able to tell, and an SSIM cannot say that on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_common as RP  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/reproj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--quality", default="95")
    ap.add_argument("--tag", default="jpeg")
    a = ap.parse_args()

    import stable_envs as se

    out = {}
    for name in a.envs.split(","):
        env = se.make(name, batch=8, size=a.size, device="cuda")
        cls = type(env)
        st = env.sample_states(a.n)
        ours = env.render(st).cpu().numpy()
        ref = cls.reference_env(a.size)
        official = cls.reference_render(ref, st)
        row = {"declared": dict(cls.visual_fidelity),
               "ours_vs_official": RP.metrics(ours, official) | RP.interior_metrics(ours, official)}
        for q in [int(x) for x in a.quality.split(",")]:
            jo = RP.jpeg_roundtrip(official, q)
            ju = RP.jpeg_roundtrip(ours, q)
            row[f"q{q}"] = {
                "jpeg_official_vs_official": RP.metrics(jo, official)
                | RP.interior_metrics(jo, official),
                "jpeg_ours_vs_ours": RP.metrics(ju, ours) | RP.interior_metrics(ju, ours),
                "jpeg_ours_vs_official": RP.metrics(ju, official),
            }
        out[name] = row
        print(f"{name}: " + json.dumps(row), flush=True)
        _dump(os.path.join(OUT, f"{a.tag}_{name}.png"), official, RP.jpeg_roundtrip(official, 95),
              ours)
        del env
    with open(os.path.join(OUT, f"{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("JSON " + json.dumps(out), flush=True)


def _dump(path, official, jpeg_official, ours) -> None:
    """official | jpeg-q95 of it | ours | 8x |jpeg - official| | 8x |ours - official|."""
    from PIL import Image
    amp = lambda d: np.stack([np.clip(d.max(-1) * 8, 0, 255).astype(np.uint8)] * 3, -1)
    i = 0
    d1 = amp(np.abs(jpeg_official[i].astype(np.int32) - official[i].astype(np.int32)))
    d2 = amp(np.abs(ours[i].astype(np.int32) - official[i].astype(np.int32)))
    Image.fromarray(np.concatenate([official[i], jpeg_official[i], ours[i], d1, d2], 1)).save(path)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
