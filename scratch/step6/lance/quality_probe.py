"""How much of the stored baseline's cost is the q95 -> q90 quality difference?

Step 2's ``cores`` metric and the ``decode`` column of every Step-3/Step-6 sweep are one core of
**q90** decode. ``examples/bench_lance.py`` and ``stable_worldmodel.data.formats.lance`` both write
**q95**. The two are not interchangeable, and the direction matters: q95 is the quality a real corpus
is written at, so a comparison against the q90 bar flatters the stored path.

This measures the gap directly, on one core, in one process, on the same frames: each env's dumped
q95 blobs are decoded once, re-encoded at q90 and q75, and all three are then timed through the
IDENTICAL batched path ``bench_lance._decode`` uses (a list of ``torch.frombuffer`` into one
``decode_jpeg``), 256 at a time.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
import bench_lance as BL  # noqa: E402


def timeit(blobs, chunk, reps):
    n = len(blobs)
    starts = [(i * chunk) % max(n - chunk, 1) for i in range(reps)]
    best = None
    for s in starts:
        sub = blobs[s:s + chunk]
        t0 = time.perf_counter()
        f = BL._decode(sub)
        dt = (time.perf_counter() - t0) / len(sub)
        best = dt if best is None else min(best, dt)
        del f
    return 1 / best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--qualities", default="95,90,75")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from PIL import Image

    torch.set_num_threads(1)
    qs = [int(q) for q in a.qualities.split(",")]
    res = {"what": "one-core batched decode fps as a function of JPEG quality, same frames",
           "chunk": a.chunk, "reps": a.reps, "torch_threads": torch.get_num_threads(),
           "node": platform.node(), "cpu": subprocess.run(
               ["bash", "-c", "lscpu | grep 'Model name' | head -1"], capture_output=True,
               text=True).stdout.split(":", 1)[-1].strip(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "loadavg": os.getloadavg(), "envs": {}}
    for f in sorted(Path(a.frames_dir).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        q95 = [bytes(b) for b in z["blobs"]]
        frames = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in q95]
        entry = {"h": meta["h"], "w": meta["w"], "n": len(q95)}
        for q in qs:
            if q == meta["jpeg_quality"]:
                blobs = q95
            else:
                blobs = []
                for fr in frames:
                    buf = io.BytesIO()
                    Image.fromarray(fr).save(buf, format="JPEG", quality=q)
                    blobs.append(buf.getvalue())
            entry[f"q{q}"] = {"fps": timeit(blobs, a.chunk, a.reps),
                              "bytes_per_frame": sum(len(b) for b in blobs) / len(blobs)}
        base = entry[f"q{meta['jpeg_quality']}"]["fps"]
        for q in qs:
            entry[f"q{q}"]["rel_to_q95"] = entry[f"q{q}"]["fps"] / base
        res["envs"][meta["env"]] = entry
        print(f"[{meta['env']}] " + "  ".join(
            f"q{q} {entry[f'q{q}']['fps']:,.0f} f/s ({entry[f'q{q}']['bytes_per_frame']:,.0f} B)"
            for q in qs), flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
    import statistics as st
    for q in qs:
        v = [e[f"q{q}"]["rel_to_q95"] for e in res["envs"].values()]
        print(f"q{q}/q95 decode-fps ratio: median {st.median(v):.3f} range {min(v):.3f}-{max(v):.3f}")
    Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
