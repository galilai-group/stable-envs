"""Paired, alternating baseline/head throughput for the MJWarp and nvdiffrast hot paths (plan 8, 21).

WHY ALTERNATING, and why the same card. A "before" block followed by an "after" block measures the
node's mood as much as the code -- a thermal ramp, a second tenant, a clock step lands entirely on
one leg. So the legs interleave (baseline, head, baseline, head, ...) in separate interpreters, and
the statistic is the median over rounds of each leg. Same venv, same host, same GPU, same process
shape, same states; the only difference is which tree is on ``PYTHONPATH``. A baseline measured on a
3090 and a head measured on an L40S is not a before/after.

WHAT IS TIMED is the public caller-visible path -- ``env.render()`` with the state already set --
because that is what a consumer pays and where a structural change would show. Not a kernel.

WHY B=1 AND B=4096. The two ends answer different questions and this workstream can plausibly have
touched both. At B=1 the frame is dominated by per-call Python and cache lookups, which is exactly
what changed: a bounded LRU does a ``move_to_end`` where a plain dict did a lookup. At the large end
the frame is the kernel and the ratio should be 1.000 to within noise; anything else means something
structural moved. 4096 does not fit for every cell at every resolution (nvdiffrast does not raise
when it runs out -- it returns an illegal memory access and kills the interpreter), so the large leg
STEPS DOWN 4096 -> 1024 -> 256 -> 64 until a width runs, and records which width both legs used.

THE BAR (plan 8): a repeatable regression above roughly 10 % on a hot path blocks merge. ``ratio`` is
head/baseline wall time, and the per-leg spread beside it says whether the machine could resolve a
difference that size at all.

NOT RUN ANYWHERE YET: every backend here is CUDA-only and the local Mac has no CUDA. This is a
section of the coordinator's GPU oracle job (``gpu_oracle.sbatch``).

Usage (inside the job)::

    python scratch/step9/render-mjwarp-nvdiff/bench.py --cells DMCCheetahRun --tag vis311
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE_SHA = "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]

#: One cell per backend path, because the property being measured is the backend's and not the
#: cell's. ``(cell, corpus-or-None, which module, which venv it needs)``.
CELLS: list[tuple[str, str | None, str, str]] = [
    ("DMCCheetahRun", "examples/results/step4/states/DMCCheetahRun.npz",
     "mjwarp (BatchRenderer-shaped context, 84 px)", "se-vis311"),
    ("OGBCube", "examples/results/step7/ogb-manipulation/states/OGBCube.npz",
     "mjwarp, the largest prepared model (193k triangles)", "se7-ogb"),
    ("FetchReach", "examples/results/step4/states/FetchReach.npz",
     "mjwarp + gl_rounding, the parity Fetch path", "se-s3b-fetch311"),
    ("MujocoHalfCheetah", None,
     "nvdiff, BatchRenderer OWNING its Scene, 240 px, mipmapped", "se7-rocket"),
    ("PFRocketLanding", None,
     "nvdiff, Scene.from_triangles handed IN (the caller-owned-scene path)", "se7-rocket"),
    ("RoboCasaCounterToSinkCorrected", None,
     "mjwarp + lightspec + rendernet (the captured corrector graph)", "se-vis311"),
]

#: Large-end widths, tried in order until one runs on both legs.
LARGE_WIDTHS = (4096, 1024, 256, 64)

LEG = r'''
import json, sys, time
import numpy as np, torch
import stable_envs as se

name, batch, warmup, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
corpus = sys.argv[5] or None
cls = se.REGISTRY[name]
env = cls(batch=batch, device="cuda")
if corpus:
    s = np.load(corpus)["states"].astype(np.float32)
    s = np.resize(s, (batch, s.shape[1]))
else:
    s = np.asarray(env.sample_states(batch, np.random.default_rng(0)), np.float32)
env.set_state(s)
for _ in range(warmup):
    env.render(); torch.cuda.synchronize()
ts = []
for _ in range(reps):
    t0 = time.perf_counter(); env.render(); torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
peak = int(torch.cuda.max_memory_allocated())
try:
    env.close()
except Exception:
    pass
print("@@JSON@@" + json.dumps({"ms": ts, "size": int(env.size), "peak_bytes": peak,
                               "device": torch.cuda.get_device_name(0)}))
'''


def _worktree(sha: str) -> tuple[Path, bool]:
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True).stdout
    for block in listing.split("\n\n"):
        if "step9-mjwarp-baseline" in block:
            return Path(block.splitlines()[0].split(" ", 1)[1]), False
    path = Path(tempfile.gettempdir()) / f"step9-mjwarp-baseline-{sha[:12]}"
    if path.exists():
        return path, False
    subprocess.run(["git", "worktree", "add", "--detach", str(path), sha], cwd=str(ROOT),
                   check=True, capture_output=True)
    return path, True


def _leg(tree: Path, python: str, name: str, batch: int, warmup: int, reps: int,
         corpus: str | None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "egl")
    r = subprocess.run([python, "-c", LEG, name, str(batch), str(warmup), str(reps),
                        str(tree / corpus) if corpus else ""],
                       capture_output=True, text=True, cwd=str(tree), env=env)
    hit = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    if not hit:
        return None, f"{r.stdout[-600:]}\n{r.stderr[-600:]}"
    return json.loads(hit[-1][len("@@JSON@@"):]), None


def _paired(baseline: Path, python: str, name: str, batch: int, rounds: int, reps: int,
            warmup: int, corpus: str | None):
    legs: dict[str, list[float]] = {"baseline": [], "head": []}
    meta: dict = {}
    for _ in range(rounds):
        for label, tree in (("baseline", baseline), ("head", ROOT)):
            got, err = _leg(tree, python, name, batch, warmup, reps, corpus)
            if got is None:
                return None, err
            legs[label].append(statistics.median(got["ms"]))
            meta = got
    return (legs, meta), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--cells", default=None, help="comma separated subset of CELLS")
    ap.add_argument("--rounds", type=int, default=5, help="alternating baseline/head rounds")
    ap.add_argument("--reps", type=int, default=5, help="timed renders per leg")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/render-mjwarp-nvdiff"))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    want = set(a.cells.split(",")) if a.cells else None
    cells = [c for c in CELLS if want is None or c[0] in want]
    baseline, created = _worktree(a.baseline_sha)
    try:
        rows = []
        for name, corpus, why, venv in cells:
            for label, widths in (("B=1", (1,)), ("B=large", LARGE_WIDTHS)):
                row = {"config": name, "backend": why, "venv_expected": venv, "leg": label}
                got = err = None
                for w in widths:
                    got, err = _paired(baseline, a.python, name, w, a.rounds, a.reps, a.warmup,
                                       corpus)
                    if got is not None:
                        row["batch"] = w
                        break
                    row.setdefault("stepped_down_from", []).append(w)
                if got is None:
                    row.update({"status": "skipped",
                                "reason": (err or "").strip().splitlines()[-1][:200] if err else "?"})
                    rows.append(row)
                    print(f"{name:34s} {label:8s} SKIPPED: {row['reason'][:90]}")
                    continue
                legs, meta = got
                b = statistics.median(legs["baseline"])
                h = statistics.median(legs["head"])
                spread = (max(legs["baseline"]) - min(legs["baseline"])) / b if b else 0.0
                row.update({"status": "ok", "baseline_ms": round(b, 4), "head_ms": round(h, 4),
                            "ratio_head_over_baseline": round(h / b, 4),
                            "baseline_leg_spread": round(spread, 4),
                            "fps_baseline": round(row["batch"] / (b / 1e3), 1),
                            "fps_head": round(row["batch"] / (h / 1e3), 1),
                            "size": meta.get("size"), "card": meta.get("device"),
                            "peak_bytes": meta.get("peak_bytes"),
                            "baseline_rounds_ms": [round(x, 4) for x in legs["baseline"]],
                            "head_rounds_ms": [round(x, 4) for x in legs["head"]]})
                rows.append(row)
                print(f"{name:34s} {label:8s} B={row['batch']:<5d} baseline {b:9.3f} ms  "
                      f"head {h:9.3f} ms  ratio {h / b:5.3f}  (spread {spread:5.1%})")
        worst = max((r["ratio_head_over_baseline"] for r in rows if r["status"] == "ok"),
                    default=0.0)
        record = {"baseline_sha": a.baseline_sha, "host": os.uname().nodename,
                  "slurm_job": os.environ.get("SLURM_JOB_ID"), "python": a.python,
                  "rounds": a.rounds, "reps_per_leg": a.reps, "rows": rows, "worst_ratio": worst,
                  "regression_bar": 1.10, "blocks_merge": bool(worst > 1.10),
                  "head_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                             capture_output=True, text=True).stdout.strip()}
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        fn = f"timings_{a.tag or 'gpu'}.json"
        (out / fn).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out / fn}\nworst ratio {worst:.3f} against a 1.10 bar: "
              f"{'REGRESSION' if worst > 1.10 else 'within bar'}")
        return 0
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
