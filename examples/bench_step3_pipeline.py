"""Step 3 product C: can the input pipeline actually feed the renderer at large batch?

Kept in a separate file, and its numbers in a separate table, because mixing it into the renderer
microbenchmark is the specific error Step 2 section 6.1 forbids. The renderer sweep times
``states already resident -> frames``. This times the whole road to that point, and the question is
the one that only matters once rendering is cheap:

    stored states -> sampler/DataLoader -> collate -> host-to-device transfer -> renderer -> frames

Four arms, which are the four idioms ``stable_envs/data.py`` documents, run against the same states so
the difference is the plumbing and nothing else:

    no_loader          slice the tensor. The ceiling; every other arm is reported as a fraction of it.
    default_collate    a plain per-item ``DataLoader``. Costs ~3.4 us of python per row, invisible at
                       B=64 and 14.9 ms at B=4096 on the H200 this repo measured.
    batch_sampler      ``BatchSampler`` + ``batch_size=None``: the dataset is handed a LIST of indices
                       and returns the assembled batch, with torch's own collate.
    collate_batched    ``stable_envs.collate_batched``, the identity on an already-assembled batch.

The H200 figures those docstrings quote, PushT at B=1024, states only: 2.774 ms no loader, 2.886
BatchSampler, 2.907 collate_batched, 5.637 per-item. This script re-measures them on the project's own
hardware and adds the render and transfer terms beside them.

    python examples/bench_step3_pipeline.py --device cuda --out examples/results/step3/pipeline/3090.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_gpu_vs_cpu as B
from bench_gpu_vs_cpu import set_device
from bench_step3 import provenance

import stable_envs as se
from stable_envs.data import collate_batched

ROOT = Path(__file__).resolve().parents[1]


class BatchedStates(torch.utils.data.Dataset):
    """States with BOTH access shapes, so one object can serve every arm.

    ``__getitems__`` is what makes the ``BatchSampler`` and ``collate_batched`` arms meaningful:
    torch's fetcher calls it with the whole index list and skips per-item work entirely. The per-item
    arm therefore uses a plain ``TensorDataset`` instead -- giving this class to a per-item loader
    would silently measure the fast path and report it as the slow one.
    """

    def __init__(self, states: torch.Tensor):
        self.states = states

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i]

    def __getitems__(self, idx):
        return self.states[torch.as_tensor(idx)]


def _sync():
    B._sync()


def time_arm(make_iter, env, device: str, reps: int) -> dict:
    """One arm, one batch size: total, and the loader / transfer / render split inside it.

    The split is timed per batch rather than by subtracting whole-epoch totals, because the terms
    overlap on CUDA and a subtraction would hand the asynchrony to whichever term was measured last.
    """
    tot, load, xfer, rend, n = [], [], [], [], 0
    for _ in range(reps):
        it = make_iter()
        _sync()
        t_load = t_xfer = t_rend = 0.0
        t_all0 = time.perf_counter()
        t0 = time.perf_counter()
        for batch in it:
            st = batch[0] if isinstance(batch, (list, tuple)) else batch
            t_load += time.perf_counter() - t0
            t1 = time.perf_counter()
            st = st.to(device, non_blocking=True)
            _sync()
            t_xfer += time.perf_counter() - t1
            t2 = time.perf_counter()
            env.render(st)
            _sync()
            t_rend += time.perf_counter() - t2
            n += len(st)
            t0 = time.perf_counter()
        _sync()
        tot.append(time.perf_counter() - t_all0)
        load.append(t_load)
        xfer.append(t_xfer)
        rend.append(t_rend)
    med = statistics.median(tot)
    frames = n / len(tot)
    return {
        "seconds_total": med,
        "fps": frames / med,
        "frames": frames,
        "loader_s": statistics.median(load),
        "transfer_s": statistics.median(xfer),
        "render_s": statistics.median(rend),
        "loader_frac": statistics.median(load) / med,
        "transfer_frac": statistics.median(xfer) / med,
        "render_frac": statistics.median(rend) / med,
    }


def run_env(name: str, device: str, batches: list[int], n_states: int, reps: int) -> dict:
    rec: dict = {"env": name, "device": device}
    try:
        env = se.make(name, num_envs=max(batches), device=device)
        pool = np.asarray(env.sample_states(min(n_states, 4096)), np.float32)
        states = torch.as_tensor(np.resize(pool, (n_states, pool.shape[1])))
        rec.update(
            {
                "size": int(env.size),
                "supersample": int(env.ss),
                "n_states": n_states,
                "state_dim": int(states.shape[1]),
            }
        )
    except Exception as e:
        rec["blocked"] = {"reason": f"{type(e).__name__}: {e}"[:300], "kind": "blocked_environment"}
        return rec

    ds_batched = BatchedStates(states)
    ds_peritem = TensorDataset(states)
    rec["arms"] = {}
    for b in batches:
        cells = {}
        arms = {
            "no_loader": lambda b=b: (states[i : i + b] for i in range(0, len(states), b)),
            "default_collate": lambda b=b: DataLoader(ds_peritem, batch_size=b, num_workers=0),
            "batch_sampler": lambda b=b: DataLoader(
                ds_batched,
                batch_size=None,
                num_workers=0,
                sampler=BatchSampler(SequentialSampler(range(len(states))), batch_size=b, drop_last=False),
            ),
            "collate_batched": lambda b=b: DataLoader(
                ds_batched, batch_size=b, num_workers=0, collate_fn=collate_batched
            ),
        }
        for arm, mk in arms.items():
            try:
                cells[arm] = time_arm(mk, env, device, reps)
            except Exception as e:
                cells[arm] = {"error": f"{type(e).__name__}: {e}"[:250]}
        ceiling = (cells.get("no_loader") or {}).get("fps")
        for arm, c in cells.items():
            if c.get("fps") and ceiling:
                c["frac_of_ceiling"] = c["fps"] / ceiling
        rec["arms"][str(b)] = cells
        best = (cells.get("no_loader") or {}).get("fps") or 0
        print(
            f"    B={b:5d}  ceiling {best:10,.0f} f/s   "
            + "  ".join(
                f"{a}={(cells[a].get('frac_of_ceiling') or float('nan')):.2f}"
                for a in ("default_collate", "batch_sampler", "collate_batched")
            ),
            flush=True,
        )
    del env
    if device == "cuda":
        torch.cuda.empty_cache()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--envs", default="PushT,DMCReacherHard,DMCHumanoidStand")
    ap.add_argument("--batches", default="256,1024,4096")
    ap.add_argument("--n-states", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    set_device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but no CUDA is available")
    batches = [int(x) for x in a.batches.split(",")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "what": "Step 3 product C: end-to-end input pipeline, NOT a renderer certification number",
        "provenance": provenance(a.device, torch.get_num_threads()),
        "batches": batches,
        "n_states": a.n_states,
        "reps": a.reps,
        "envs": [],
    }
    for n in [x.strip() for x in a.envs.split(",") if x.strip()]:
        print(f"  {n}:", flush=True)
        res["envs"].append(run_env(n, a.device, batches, a.n_states, a.reps))
        out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
