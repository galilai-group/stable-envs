"""Draw the same States with stable-worldmodel (swm) and with stable-envs, and write the frame MSE per env.

    python -m benchmarks.render_mse --env PushT TwoRoom --device cpu cuda --out render_mse.csv

The States come from seeded resets and random actions of stable-envs. swm draws each State after
``SwmSource.set_state``. Ours draws them with ``env.render`` at the frame size of swm's default renderer, with
the default appearance, camera and task of the class. The MSE is per frame on the 0-255 scale.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.speed import ROOT, _gl_env, _prepare_states, host_info, rows_path

COLUMNS = [
    "env",
    "device",
    "frames",
    "mse_mean",
    "mse_median",
    "mse_max",
    "size",
    "matched",
    "source",
    "status",
    "host",
    "gpu",
    "stable_envs_sha",
    "source_versions",
]


def measure(name: str, device: str, states_path: str) -> dict:
    """The frame MSE of one env on one device; swm draws first, before any stable-envs env exists here."""
    import torch

    from benchmarks.swm_source import SwmSource, source_id, versions

    states = torch.load(states_path, weights_only=False)
    src = SwmSource(name)
    src.reset(seed=0)
    theirs = []
    for row in torch.load(rows_path(states_path), weights_only=False):
        src.set_state(row)
        theirs.append(src.render()[..., :3])
    src.close()
    theirs = np.stack(theirs)
    size = int(theirs.shape[1])
    import stable_envs as snv

    with snv.make(name, num_envs=len(theirs), size=size, device=device) as env:
        ours = env.render(states.to(device)).cpu().numpy()
    per_frame = ((ours.astype(np.float64) - theirs.astype(np.float64)) ** 2).mean(axis=(1, 2, 3))
    return {
        "frames": len(per_frame),
        "mse_mean": f"{per_frame.mean():.4g}",
        "mse_median": f"{np.median(per_frame):.4g}",
        "mse_max": f"{per_frame.max():.4g}",
        "size": size,
        "matched": f"size {size}; default appearance, camera and task on both sides; swm kwargs {src.kwargs}",
        "source": source_id(name),
        "status": "ok",
        "source_versions": versions(),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", nargs="+", required=True)
    p.add_argument("--device", nargs="+", default=["cpu"], choices=["cpu", "cuda"])
    p.add_argument("--n", type=int, default=256, help="States per env")
    p.add_argument("--work", default=None, help="directory for the stored States (default: a temporary one)")
    p.add_argument("--out", default="render_mse.csv")
    a = p.parse_args(argv)
    work = Path(a.work or tempfile.mkdtemp(prefix="snv-mse-"))
    work.mkdir(parents=True, exist_ok=True)
    host = host_info(sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        w.writeheader()
        for name in a.env:
            states = work / f"{name}-states-{a.n}.pt"
            if not states.exists():
                _prepare_states(name, states, a.n)
            for device in a.device:
                row = _child(name, device, states, work)
                row.update(env=name, device=device, **host)
                print(row, flush=True)
                w.writerow(row)
                f.flush()


def _child(name, device, states, work) -> dict:
    """Run :func:`measure` in a fresh process with the OpenGL settings of ``device``."""
    import json
    import subprocess

    out = work / f"{name}-{device}-mse.json"
    code = (
        "import json, sys; from benchmarks.render_mse import measure\n"
        "try: r = measure(sys.argv[1], sys.argv[2], sys.argv[3])\n"
        "except Exception as e: r = {'status': f'error: {type(e).__name__}: {e}'[:300]}\n"
        "open(sys.argv[4], 'w').write(json.dumps(r))"
    )
    env = {**os.environ, **_gl_env(device == "cuda"), "PYTHONPATH": str(ROOT), "SDL_VIDEODRIVER": "dummy"}
    subprocess.run([sys.executable, "-c", code, name, device, str(states), str(out)], env=env, cwd=ROOT)
    return json.loads(out.read_text()) if out.exists() else {"status": "error: no result"}


if __name__ == "__main__":
    main()
