"""Step 3: measure the CURRENT stable-envs renderer on the project's own hardware.

This is a driver, not a second benchmark framework. Every timing primitive comes from the two scripts
that already exist -- ``bench_gpu_vs_cpu.py`` for the render loop, the official-renderer baseline, the
resolved supersample, ``cpu_frac`` and the memory high-water mark, and ``bench_vs_decode.py`` for the
one-core JPEG-q90 bar. What is new here is only what Step 3 actually needs:

  * **a CPU sweep at all.** The repo had never timed its own renderer on a CPU. Every ``cpu`` column in
    every committed result file is the OFFICIAL renderer, which is why Step 1 recorded
    ``se_cpu_fps = needs_measurement`` on all 34 matched rows and Step 2 had to anchor its CPU ladder to
    a distribution derived from decode rather than to a measured one.
  * **the frozen Step 2 batch sweep**, including B=1 -- not a certification point, but the only way to
    compute ``BATCH_SCALES`` and the only place launch overhead is visible.
  * **a thread budget.** A CPU number taken with unrestricted machine-wide parallelism is a node
    measurement; the certified one is single-threaded, and the all-core run is reported beside it as a
    scaling result.
  * **one shard per interpreter**, because no single interpreter can import every family -- dm_control
    needs mujoco>=3.12 and gymnasium-robotics needs <=3.11, which Step 1 established by measurement.
    Shards merge deterministically downstream.

The universe is the **34 matched swm rows** of ``scratch/swm_matrix.json``, projected onto the 37
stable-envs envs that back them -- not the 196-env registry. Envs an interpreter cannot reach are
recorded as blockers with the importer's own message. A blocker is NEVER a performance tier: "we could
not build this here" and "this renderer is slow" are different findings and Step 2 forbids conflating
them.

    python examples/bench_step3.py --device cpu  --threads 1 --out examples/results/step3/cpu/m5_1t.json
    python examples/bench_step3.py --device cuda --out examples/results/step3/gpu/3090.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_gpu_vs_cpu import cpu_throughput, gpu_throughput, make_env, parse_ss, set_device
from bench_vs_decode import decode_baseline

import stable_envs as se

ROOT = Path(__file__).resolve().parents[1]

#: The frozen Step 2 sweeps. Read from scratch/certification.json rather than retyped, so a batch list
#: cannot drift from the policy that requires it -- section 5.2 (CPU) and 6.2 (GPU) of certification.md.
SPEC = json.loads((ROOT / "scratch" / "certification.json").read_text())


def batches_for(device: str, optional: bool) -> list[int]:
    key = "gpu" if device == "cuda" else "cpu"
    b = list(SPEC[key]["required_batches"])
    if optional:
        b += list(SPEC[key].get("optional_batches") or [])
    return sorted(set(b))


def matched_universe() -> dict[str, list[str]]:
    """``stable-envs env -> the swm ids that consume its frames``, from the Step 1 matrix.

    Projection is recorded rather than implied. PushT backs two swm ids (``PushT-v1`` and
    ``PushT-Discrete-v1``) and is timed ONCE; the merge step must be able to say that both rows came
    from one measurement instead of presenting two independent-looking numbers.
    """
    m = json.loads((ROOT / "scratch" / "swm_matrix.json").read_text())
    out: dict[str, list[str]] = {}
    for r in m["rows"]:
        for e in r["stable_envs"]["envs"]:
            out.setdefault(e, []).append(r["swm_id"])
    return {k: sorted(v) for k, v in sorted(out.items())}


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def provenance(device: str, threads: int) -> dict:
    """Everything a later reader needs to interpret or reject a row, captured once per shard."""
    cpu_name = platform.processor()
    if platform.system() == "Linux":
        cpu_name = _run(["bash", "-c", "lscpu | grep 'Model name' | head -1"]).split(":", 1)[-1].strip() or cpu_name
    elif platform.system() == "Darwin":
        cpu_name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu_name
    # WHICH TREE PRODUCED THIS SHARD. Two failures, both of which have already happened in Step 6.
    #
    # The measured copy on the cluster is an rsync without .git, so ``git rev-parse`` there returns
    # nothing and the shard recorded ``git_commit: null`` -- the same hole ``fidelity_step4`` grew
    # the STABLE_ENVS_GIT_COMMIT / GIT_COMMIT fallback for, reused verbatim here.
    #
    # And a before/after sweep that switches PYTHONPATH between two checkouts is only a before/after
    # if the import followed it: a script that inserts its own root ahead of PYTHONPATH makes both
    # legs measure one tree, and the shard cannot tell. ``stable_envs.__file__`` is where the import
    # actually resolved, so the artifact states which code it timed instead of the invocation being
    # taken on trust.
    commit, source = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]) or None, "git"
    dirty = bool(_run(["git", "-C", str(ROOT), "status", "--porcelain"])) if commit else None
    if not commit and os.environ.get("STABLE_ENVS_GIT_COMMIT", "").strip():
        commit, source = os.environ["STABLE_ENVS_GIT_COMMIT"].strip(), "env:STABLE_ENVS_GIT_COMMIT"
    elif not commit and (ROOT / "GIT_COMMIT").is_file():
        commit, source = (ROOT / "GIT_COMMIT").read_text().strip() or None, "file:GIT_COMMIT"
    p = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "git_commit_source": source if commit else None,
        "git_dirty": dirty,
        "stable_envs_file": se.__file__,
        "tree_root": str(Path(se.__file__).resolve().parents[1]),
        "host": platform.node(),
        "arch": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_name,
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": _physical_cores(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "torch_threads": threads,
        "loadavg_start": os.getloadavg()[0],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "certification_spec_version": SPEC.get("schema_version"),
    }
    if device == "cuda" and torch.cuda.is_available():
        p["device_name"] = torch.cuda.get_device_name(0)
        p["cuda"] = torch.version.cuda
        p["driver"] = _run(["bash", "-c", "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"])
        p["device_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    elif device == "cuda":
        p["device_name"] = "UNAVAILABLE: torch.cuda.is_available() is False"
    return p


def _physical_cores() -> int | None:
    if platform.system() == "Darwin":
        v = _run(["sysctl", "-n", "hw.physicalcpu"])
        return int(v) if v.isdigit() else None
    v = _run(["bash", "-c", "lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l"])
    return int(v) if v.isdigit() else None


def measure_env(
    name: str,
    swm_ids: list[str],
    device: str,
    batches: list[int],
    reps: int,
    max_out_gb: float,
    max_cell_s: float,
    ref_frames: int,
    decode_frames: int,
    ss_tok: str,
    scratch: Path,
    states_in: Path | None = None,
    states_out: Path | None = None,
) -> dict:
    """One env: the batch sweep, the official-renderer baseline, and the decode bar.

    Ascending batch order is deliberate and load-bearing on the CPU. A 500x500 classic-control frame at
    B=1024 is minutes of work single-threaded, and a sweep that discovers that by running it has spent
    the budget before it reaches the next env. Each cell's cost is therefore PROJECTED from the fps
    already measured at a smaller batch and skipped with that projection recorded, so a missing cell
    says why it is missing instead of looking like a failure.
    """
    rec: dict = {"env": name, "projected_to": swm_ids, "device": device}
    cls = se.registry.REGISTRY.get(name)
    if cls is None:
        rec["blocked"] = {
            "stage": "registry",
            "reason": f"{name} is not in stable_envs.registry.REGISTRY in this interpreter",
            "kind": "blocked_environment",
        }
        return rec

    ss = parse_ss(ss_tok)
    try:
        probe = make_env(name, 1, None, ss)
        # ``sample_states`` is not always cheap and is not always AVAILABLE: Fetch builds its pool by
        # resetting the official env and stepping it, so an interpreter that holds the renderer but not
        # the reference cannot draw one at all. A saved corpus lets the two halves of a split
        # measurement run in different interpreters on the same node over byte-identical states, which
        # is the only way their ratio means anything.
        cached = (states_in / f"{name}.npy") if states_in else None
        if cached and cached.is_file():
            pool = np.asarray(np.load(cached), np.float32)
            rec["states_source"] = str(cached)
        else:
            pool = np.asarray(probe.sample_states(64), np.float32)
            rec["states_source"] = "sample_states"
        if states_out:
            states_out.mkdir(parents=True, exist_ok=True)
            np.save(states_out / f"{name}.npy", pool)
            rec["states_written"] = str(states_out / f"{name}.npy")
        frame0 = probe.render(torch.as_tensor(pool[:1], device=device))
        hw = (int(frame0.shape[1]), int(frame0.shape[2]))
        rec.update(
            {
                "size": int(probe.size),
                "supersample": int(probe.ss),
                "shape": f"{hw[0]}x{hw[1]}",
                "backend": getattr(probe, "backend", None),
                "renderer_class": type(probe).__name__,
                "reference_id": getattr(cls, "reference_id", None),
                "fidelity_tier": getattr(cls, "fidelity_tier", None),
                "state_dim": int(getattr(cls, "state_dim", 0) or 0),
            }
        )
        del probe
    except Exception as e:
        rec["blocked"] = {
            "stage": "construct",
            "reason": f"{type(e).__name__}: {e}"[:400],
            "kind": "blocked_environment",
            "traceback": traceback.format_exc()[-1200:],
        }
        return rec

    # --- the renderer sweep ------------------------------------------------------------------
    cells, fps_seen = [], None
    for b in batches:
        if fps_seen and max_cell_s > 0:
            projected = b / fps_seen
            if projected > max_cell_s:
                cells.append(
                    {
                        "batch": b,
                        "skipped": f"projected {projected:,.0f} s per repetition at "
                        f"the {fps_seen:,.0f} frames/s already measured, over the "
                        f"{max_cell_s:.0f} s per-cell budget",
                        "kind": "not_measured_here",
                    }
                )
                continue
        cell = gpu_throughput(name, b, None, ss, reps, max_out_gb, pool, hw)
        cells.append(cell)
        if cell.get("fps"):
            fps_seen = cell["fps"]
    rec["cells"] = cells

    # --- the two same-host baselines ---------------------------------------------------------
    # Both are CPU numbers by construction and both are taken in THIS process on THIS host, which is
    # what makes the ratio meaningful. Step 2 forbids pairing a renderer on one machine with a
    # reference on another.
    try:
        rec["official_cpu"] = cpu_throughput(cls, name, int(rec["size"]), ref_frames, pool)
    except Exception as e:
        rec["official_cpu"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    if rec.get("official_cpu") is None:
        rec["official_cpu"] = {
            "absent": "the class declares no reference_env / reference_id, so no "
            "official renderer exists to form a ratio against",
            "kind": "no_reference",
        }
    try:
        f = None
        for c in cells:
            if c.get("fps"):
                e2 = make_env(name, 1, None, ss)
                f = np.asarray(e2.render(torch.as_tensor(pool[:1], device=device))[0].cpu(), np.uint8)
                del e2
                break
        rec["decode"] = decode_baseline(f, 90, decode_frames, scratch) if f is not None else None
    except Exception as e:
        rec["decode"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--threads", type=int, default=1, help="torch intra-op threads; 0 = leave alone")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", default="auto", help="'auto' = the matched swm universe, else comma list")
    ap.add_argument("--skip", default="", help="comma list to leave out")
    ap.add_argument("--optional-batches", action="store_true")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--max-out-gb", type=float, default=8.0)
    ap.add_argument("--max-cell-s", type=float, default=20.0)
    ap.add_argument("--ref-frames", type=int, default=60)
    ap.add_argument("--decode-frames", type=int, default=40)
    ap.add_argument("--ss", default="native")
    ap.add_argument("--resume", action="store_true", help="keep envs already present in --out")
    ap.add_argument(
        "--states-in", default=None, help="directory of <env>.npy state pools to render instead of sampling"
    )
    ap.add_argument(
        "--states-out", default=None, help="directory to save each env's state pool into, for a split-interpreter run"
    )
    a = ap.parse_args()

    set_device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(v, str(a.threads))
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda but torch.cuda.is_available() is False; refusing to write a "
            "shard that would look like a GPU measurement"
        )

    universe = matched_universe()
    names = sorted(universe) if a.envs == "auto" else [n.strip() for n in a.envs.split(",") if n.strip()]
    skip = {n.strip() for n in a.skip.split(",") if n.strip()}
    names = [n for n in names if n not in skip]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict = {}
    if a.resume and out.is_file():
        try:
            done = {r["env"]: r for r in json.loads(out.read_text()).get("envs", [])}
        except Exception:
            done = {}

    batches = batches_for(a.device, a.optional_batches)
    res = {
        "what": "Step 3 raw renderer benchmark shard",
        "provenance": provenance(a.device, torch.get_num_threads()),
        "batches": batches,
        "ss_request": a.ss,
        "reps": a.reps,
        "universe": "the 34 matched swm rows of scratch/swm_matrix.json, projected onto the "
        "stable-envs envs that back them",
        "states_in": a.states_in,
        "states_out": a.states_out,
        "envs": [],
    }
    scratch = Path(os.environ.get("TMPDIR", "/tmp"))
    for i, n in enumerate(names, 1):
        if n in done:
            print(f"[{i}/{len(names)}] {n}: kept from a previous run", flush=True)
            res["envs"].append(done[n])
            continue
        t0 = time.perf_counter()
        try:
            r = measure_env(
                n,
                universe.get(n, []),
                a.device,
                batches,
                a.reps,
                a.max_out_gb,
                a.max_cell_s,
                a.ref_frames,
                a.decode_frames,
                a.ss,
                scratch,
                Path(a.states_in) if a.states_in else None,
                Path(a.states_out) if a.states_out else None,
            )
        except Exception as e:  # one env must never end the sweep
            r = {
                "env": n,
                "projected_to": universe.get(n, []),
                "device": a.device,
                "blocked": {
                    "stage": "measure",
                    "reason": f"{type(e).__name__}: {e}"[:400],
                    "kind": "blocked_environment",
                    "traceback": traceback.format_exc()[-1200:],
                },
            }
        r["seconds"] = round(time.perf_counter() - t0, 2)
        res["envs"].append(r)
        best = max((c.get("fps") or 0 for c in r.get("cells", [])), default=0)
        print(
            f"[{i}/{len(names)}] {n}: "
            + (
                f"blocked ({r['blocked']['reason'][:70]})"
                if "blocked" in r
                else f"best {best:,.0f} f/s  ref "
                f"{(r.get('official_cpu') or {}).get('set_state_render', {}).get('fps', float('nan')):,.0f} f/s"
            )
            + f"  [{r['seconds']}s]",
            flush=True,
        )
        res["provenance"]["loadavg_end"] = os.getloadavg()[0]
        out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")  # resumable after every env
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    ok = sum(1 for r in res["envs"] if r.get("cells") and any(c.get("fps") for c in r["cells"]))
    print(f"\nwrote {out}: {ok}/{len(res['envs'])} envs measured, {len(res['envs']) - ok} blocked", flush=True)


if __name__ == "__main__":
    main()
