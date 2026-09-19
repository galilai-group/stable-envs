"""GPU renderer vs the official CPU renderer, per env in the registry.

The repo's claim is that regenerating a frame from its state is cheaper than reading a stored frame,
which is only worth asserting with the official renderer's own cost on the same states. So this script
measures the same transformation twice — ``state -> uint8 frame`` — once through the GPU path and once
through the reference env, and reports the ratio.

Two CPU timings are reported because the honest comparison is ambiguous by one step:

  ``set_state + render``   what the GPU path actually does: place the world at a stored state, then draw
                           it. This is the number that matters for re-rendering a dataset, and it is the
                           one used for the speedup column.
  ``render only``          the draw alone, with the reference env left wherever the previous call put it.
                           Reported separately so nobody can claim the speedup is an artifact of a slow
                           setter rather than a slow rasterizer.

The GPU column is a whole-batch render divided by the batch size, which is the point rather than a
concession: the kernel is bandwidth-bound on its own output, so per-frame cost only becomes meaningful
once the launch is saturated, and a batch of 1 is reported as well to show exactly how far off launch
overhead is.

NEITHER COLUMN IS UNCONDITIONALLY REPRODUCIBLE, and each row now carries the evidence for its own case
rather than inheriting a claim made here.

The CPU column is a single-threaded Python renderer competing with whatever else the node is running,
and the same measurement has produced 422 frames/s on an idle node and 3 frames/s with three other jobs
resident — a factor of 140 that has nothing to do with either renderer. So the load average is captured
around the sweep and written into the output, the speedup column is a ratio to read as an order of
magnitude rather than to three digits, and a CPU baseline taken on a busy node should be discarded, not
averaged.

The GPU column used to be described here as repeating across nodes to 0.4% — "one kernel on a device
this job owns". That is true of the fused rasterizer and FALSE of the ray-cast families, which was
found by two of this repo's own benchmarks disagreeing 3.6x on DMCBallInCupCatch at 84 px, batch 4096.
Measured: that render spends **196% of its wall time on CPU** — two cores saturated inside
``DMControlSuite.render``'s per-chunk Python loop, with the device waiting — so it is host-bound, and
the number it produces is a property of the node's processor and its load. The same env, same
supersample, measured 3,664 frames/s on an idle node and 271-978 on contended ones: a 13x spread on a
column that claimed 0.4%. Every GPU row therefore records ``cpu_frac`` and the load average alongside
the throughput. Read a row with ``cpu_frac`` near 1 or above as a host measurement that does not
travel; a row near 0 is the device and does.

Nothing here is env-specific. Every env in ``stable_envs.registry.REGISTRY`` is benchmarked, and any env with a
reference renderer gets the CPU comparison; the rest get the GPU columns only. Resolutions are per env:
``native`` resolves to the env's ``default_size``, and an explicitly requested size an env cannot produce
(MiniGrid needs a whole number of tiles) is recorded with the env's own message rather than silently
dropped. Every measurement is wrapped, so an env that cannot run leaves a row saying why instead of
ending the sweep.

    python examples/bench_gpu_vs_cpu.py --out examples/results
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import stable_envs as se


def _err(e: BaseException) -> dict:
    return {"error": f"{type(e).__name__}: {e}"}


def make_env(name: str, batch: int, size: int, ss: int | None):
    """Build the env, letting it choose its own supersample unless one is explicitly REQUESTED.

    ``ss=None`` means "whatever this class declares", and that is not the same thing as 1. Forty-four
    of the registered classes — every ``DMControlSuite`` domain, every OGBench and D4RL maze, four
    MuJoCo locomotion envs — set ``default_supersample = 2``, because their reference is MSAA'd and a
    hard silhouette is the wrong answer at a mover's edge. ``se.make(name)`` gives a user ss=2 for
    those; passing ``supersample=1`` gives a DIFFERENT PICTURE, four times cheaper.

    That substitution was live in this file, and it is why this script and ``bench_storage.py``
    (which never passed a supersample) described different pictures for the same env. Measured in one
    process on an idle node, ``supersample=1`` against the class's own value: DMCBallInCupCatch at
    84px batch 4096 is 4,272 against 3,683 frames/s (1.16x), AntmazeLargeNavigate at 224px 70,598
    against 55,947 (1.26x), MujocoPusher at 240px 14,938 against 4,475 (3.3x). The ratio is not a
    constant because these renders are host-bound (see ``_median_and_cpu``) and ss=2 quadruples the
    per-chunk Python loop, so it grows with how contended the node is — on a busy one the same
    ball_in_cup A/B measured 1,149 against 356.
    """
    kw = {} if ss is None else {"supersample": int(ss)}
    return se.make(name, num_envs=batch, size=size, device=DEVICE, **kw)


#: The device every timed render in this file runs on. It was ``"cuda"`` as a literal in three places
#: until Step 3 needed the SAME measurement on a CPU: the repo had never timed its own renderer there,
#: so every ``cpu`` column in every committed result file is the OFFICIAL renderer and the stable-envs
#: CPU throughput was simply unknown. Making it a variable is the whole change -- the timing loop, the
#: warm-up, the supersample resolution and the output-budget arithmetic are identical on both devices,
#: and duplicating them into a second script is how two files come to describe different pictures.
DEVICE = "cuda"


def set_device(dev: str) -> None:
    global DEVICE
    DEVICE = str(dev)


def _sync() -> None:
    """Block until queued work has actually finished, on devices where that is a distinct step.

    On CUDA the launch is asynchronous and timing without this measures how fast Python can enqueue,
    which for this kernel is about two orders of magnitude off. On CPU the call would raise if CUDA is
    absent, and there is nothing to wait for anyway -- ``render`` has already returned.
    """
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def parse_ss(tok: str) -> int | None:
    """``--ss native`` -> None (the env's own ``default_supersample``); a number -> that number."""
    return None if str(tok).strip().lower() in ("native", "default", "env", "none") else int(tok)


#: Wall-clock budget for one timed cell. Repetitions exist to beat down noise, and noise is a fixed
#: absolute quantity here (launch jitter, a few hundred microseconds), so a cell that already takes tens
#: of milliseconds does not need 25 of them. Without this the sweep is dominated by its slowest env: at
#: 5,000 frames/s a batch of 65,536 is 13 s per repetition, i.e. five minutes for one table cell.
CELL_BUDGET_S = 1.0
MIN_REPS = 3

#: There is deliberately NO synthetic "is this node busy" gate here, and the reason is a measurement.
#: A fixed single-threaded arithmetic loop was tried as one and does not detect the condition at all:
#: it takes 0.076 s on a node at load 3 and 0.061 s on the same cluster at load 209 — *faster* on the
#: busy host, because the loop is register-resident and what actually varies is clock. Meanwhile the
#: official renderers on the busy host ran up to 11x slow (MiniGridKeyCorridor 37 frames/s against 425,
#: Pendulum 38 against 341), because they are memory-heavy rather than arithmetic-heavy. So the sweep
#: records the load average and says plainly that the CPU column does not travel, rather than shipping
#: an instrument that reads plausibly and measures the wrong thing.


def _median_and_cpu(fn, reps: int) -> tuple[float, float]:
    """``(median wall seconds, CPU seconds per wall second)`` over up to ``reps`` synced timings.

    Timed with an explicit synchronize on both sides: the launch is asynchronous, so timing without it
    measures how fast Python can enqueue work, which for this kernel is roughly two orders of magnitude
    off.

    The CPU fraction is the honest label on the number and it was added because this file's own claim
    about reproducibility turned out to be false for a whole family. ``time.process_time`` counts only
    seconds this process spent ON a CPU, so a render blocked on the device costs wall time and no CPU
    time: a fraction near 0 means the device is the limit and the row travels between nodes, and a
    fraction at or above 1 means the host is busy for the whole render and the row is a property of
    the node's CPU and of what else is running on it.

    Measured, DMCBallInCupCatch at 84 px batch 4096 reports **196%** — two cores saturated, the GPU
    waiting — and the same configuration measured 3,664 f/s on an idle node and 271-978 f/s on
    contended ones, a 13x spread on what the report was calling a GPU number.
    """
    for _ in range(2):
        fn()
    _sync()
    ts = []
    spent = 0.0
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for i in range(reps):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        dt = time.perf_counter() - t0
        ts.append(dt)
        spent += dt
        if i + 1 >= MIN_REPS and spent > CELL_BUDGET_S:
            break
    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    return statistics.median(ts), (cpu / wall if wall > 0 else float("nan"))


def _median(fn, reps: int) -> float:
    return _median_and_cpu(fn, reps)[0]


def states_for(pool: np.ndarray, batch: int) -> np.ndarray:
    """``batch`` states, tiling the pool if it is smaller.

    The pool is drawn once per env rather than per cell because ``sample_states`` is not free for every
    env — the tile-based ones step the official simulator to build it — and drawing it seven times per
    resolution would dominate the sweep. Repeating states cannot flatter the renderer: it evaluates every
    primitive per environment regardless, and the output bytes are the same either way.
    """
    if len(pool) >= batch:
        return pool[:batch]
    return np.ascontiguousarray(np.resize(pool, (batch, pool.shape[1])))


def gpu_throughput(
    name: str, batch: int, size: int, ss: int, reps: int, max_out_gb: float, pool: np.ndarray, hw: tuple[int, int]
) -> dict:
    """One full-batch render, as frames/s, µs/frame and output GB/s.

    ``hw`` is the frame shape measured from an actual render rather than assumed to be ``size x size``.
    Several envs are not square — highway-env draws 150x600, Craftax 112x144 — and allocating a square
    ``out=`` buffer for them fails with a reshape error that reads like a renderer bug.
    """
    fh, fw = hw
    out_gb = batch * fh * fw * 3 / 1e9
    if out_gb > max_out_gb:
        return {
            "batch": batch,
            "size": size,
            "skipped": f"output would be {out_gb:.1f} GB, over the {max_out_gb} GB budget",
        }
    env = st = out = None
    try:
        env = make_env(name, batch, size, ss)
        st = torch.as_tensor(states_for(pool, batch), device=DEVICE)
        out = torch.empty((batch, fh, fw, 3), dtype=torch.uint8, device=DEVICE)
        for _ in range(5):  # warm up: JIT, autotune, and any static-prefix bake
            env.render(st, out=out)
        _sync()
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        dt, cpu_frac = _median_and_cpu(lambda: env.render(st, out=out), reps)
        # The RESOLVED supersample, read off the object rather than off the flag. A row that does not
        # carry it cannot be compared with a row from another script, which is exactly how this table
        # and bench_storage.py came to describe different pictures for the same env and resolution.
        return {
            "batch": batch,
            "size": size,
            "supersample": int(env.ss),
            "seconds_per_batch": dt,
            "fps": batch / dt,
            "cpu_frac": cpu_frac,
            "loadavg": os.getloadavg()[0],
            "us_per_frame": dt / batch * 1e6,
            "gb_per_s": out_gb / dt,
            "peak_gb": _peak_gb(),
        }
    except torch.OutOfMemoryError:
        return {"batch": batch, "size": size, "skipped": "OOM"}
    except MemoryError as e:
        return {"batch": batch, "size": size, "skipped": f"host OOM: {e}"}
    except Exception as e:
        return {"batch": batch, "size": size, **_err(e)}
    finally:
        del env, st, out
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def _peak_gb() -> float:
    """Peak bytes for the timed cell: device allocator on CUDA, process RSS on the CPU.

    They are not the same quantity and the field does not pretend they are -- the row carries its own
    ``device``, so a reader can tell which one this is. The CPU number is a high-water mark for the
    whole process rather than for this cell, which is the honest limit of what ``resource`` can report.
    """
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes.
        return peak / 1e9 if platform.system() == "Darwin" else peak * 1024 / 1e9
    except Exception:
        return float("nan")


def gpu_breakdown(
    name: str, batch: int, size: int, ss: int, reps: int, max_out_gb: float, pool: np.ndarray, hw: tuple[int, int]
) -> dict:
    """Split ``render`` into building the ``Scene2D`` and rasterizing it.

    The two halves have different cost models and the split is not a detail: the rasterizer is a single
    GPU launch whose cost scales with output pixels, while ``scene()`` is a few dozen small torch ops
    issued from Python, whose cost is dominated by launch overhead and is therefore almost independent of
    the batch. Reporting only the total hides that the small-batch numbers are measuring Python.

    Envs that do not build a ``Scene2D`` — the tile-atlas family renders by gathering instead — have no
    such split, and say so rather than reporting zeros.
    """
    from stable_envs.render.raster2d import render as render_scene

    fh, fw = hw
    if batch * fh * fw * 3 / 1e9 > max_out_gb:
        return {"batch": batch, "size": size, "skipped": "over the output budget"}
    env = st = out = sc = None
    try:
        env = make_env(name, batch, size, ss)
        ss = int(env.ss)  # the resolved value, not the request; see make_env
        st = torch.as_tensor(states_for(pool, batch), device="cuda")
        out = torch.empty((batch, fh, fw, 3), dtype=torch.uint8, device="cuda")
        sc = env.scene(st)
        render_scene(sc, size, ss, backend=env.backend, out=out)  # bake the static prefix first
        t_scene = _median(lambda: env.scene(st), reps)
        t_ras = _median(lambda: render_scene(sc, size, ss, backend=env.backend, out=out), reps)
        t_all = _median(lambda: env.render(st, out=out), reps)
        return {
            "batch": batch,
            "size": size,
            "supersample": ss,
            "scene_ms": t_scene * 1e3,
            "raster_ms": t_ras * 1e3,
            "render_ms": t_all * 1e3,
            "raster_fps": batch / t_ras,
            "raster_gb_per_s": batch * fh * fw * 3 / 1e9 / t_ras,
        }
    except NotImplementedError:
        return {"batch": batch, "size": size, "skipped": "no Scene2D; this env overrides render()"}
    except torch.OutOfMemoryError:
        return {"batch": batch, "size": size, "skipped": "OOM"}
    except Exception as e:
        return {"batch": batch, "size": size, **_err(e)}
    finally:
        del env, st, out, sc
        torch.cuda.empty_cache()


def cpu_throughput(cls, name: str, size: int, frames: int, pool: np.ndarray) -> dict | None:
    """The official renderer's cost for the same ``state -> frame`` transformation.

    Frames are rendered one at a time because that is the only thing the reference offers — there is no
    batched entry point to give it a fairer shot at. Where the env exposes a per-state setter the two
    halves are timed apart; where it only offers a batch-level ``reference_render`` the combined number
    is all there is, and only that is reported.
    """
    if not hasattr(cls, "reference_env") or getattr(cls, "reference_id", None) is None:
        return None
    ref = None
    try:
        ref = cls.reference_env(size=size)
        states = states_for(pool, frames + 8)

        # The combined number goes through the env's own reference bridge, which is the only entry point
        # every env is guaranteed to have and is exactly the transformation the GPU path performs.
        cls.reference_render(ref, states[:8])  # warm up surfaces, caches, JITs
        t0 = time.perf_counter()
        cls.reference_render(ref, states[8:])
        dt_full = (time.perf_counter() - t0) / frames

        # The draw-alone split is optional: it needs a per-state setter and a render hook, which is a
        # gym_pusht-shaped interface, not a universal one. Where it is missing the combined number stands
        # on its own rather than being approximated.
        # ``getattr``, not ``ref.unwrapped``: a reference is not always a gymnasium env. dm_control
        # hands back an ``Environment`` and Jumanji a tuple, neither of which has ``unwrapped``, and
        # reaching for it threw out of this try -- discarding the combined number that had ALREADY been
        # measured and reporting the whole row as an error. The draw-alone split is an optional extra;
        # failing to take it must not cost the measurement it is an extra to.
        dt_draw = None
        u = getattr(ref, "unwrapped", None)
        if u is not None and hasattr(cls, "reference_set_state") and hasattr(u, "_render"):
            try:
                cls.reference_set_state(u, states[8])
                u._render(visualize=False)
                t0 = time.perf_counter()
                for _ in range(frames):
                    u._render(visualize=False)
                dt_draw = (time.perf_counter() - t0) / frames
            except Exception:
                dt_draw = None

        res = {
            "size": size,
            "frames": frames,
            "set_state_render": {
                "seconds_per_frame": dt_full,
                "fps": 1 / dt_full,
                "us_per_frame": dt_full * 1e6,
                "gb_per_s": size * size * 3 / 1e9 / dt_full,
            },
        }
        if dt_draw is not None:
            res["render_only"] = {
                "seconds_per_frame": dt_draw,
                "fps": 1 / dt_draw,
                "us_per_frame": dt_draw * 1e6,
                "gb_per_s": size * size * 3 / 1e9 / dt_draw,
            }
        return res
    except Exception as e:
        traceback.print_exc()
        return _err(e)
    finally:
        if ref is not None:
            try:
                ref.close()
            except Exception:
                pass


def _carryover_note(results: dict) -> str:
    """Name the envs whose rows predate this run, or say there are none.

    The file is a MERGE: a sweep can be split by ``--envs`` and a reference env can take the process
    down mid-run, so rows accumulate across runs. That is deliberate. What is not acceptable is the
    header then reporting one ``supersample`` as though it described every row -- a crashed sweep left
    this file claiming ``native`` while nine envs still held rows forced to 1, and nothing said so.
    A row measured by the current code records its resolved supersample; one that does not is older
    than that change, and is named here rather than averaged in silently.
    """
    stale = sorted(
        name
        for name, e in results.get("envs", {}).items()
        if (rows := [r for blk in e.get("sizes", {}).values() for r in blk.get("gpu", []) if "error" not in r])
        and not any(r.get("supersample") is not None for r in rows)
    )
    if not stale:
        return " Every row below was measured by the current code, which records the value it resolved."
    return (
        f" **{len(stale)} envs below carry rows from an earlier run that did not record a resolved "
        f"supersample, so the line above does not describe them: "
        f"{', '.join(f'`{n}`' for n in stale)}.** Re-run with `--envs` naming them to replace those "
        f"rows."
    )


def markdown(results: dict) -> str:
    """One table per (env, resolution), plus the storage arithmetic the throughput exists to justify."""
    load = results.get("loadavg")
    L = [
        "# GPU vs official CPU renderer",
        "",
        f"Measured on `{results['device']}` / `{results['cpu']}`, "
        f"torch {results['torch']}, {results['cpu_count']} cores visible.",
        f"Supersample: `{results['supersample']}` — with `native`, every env renders at its own "
        f"`default_supersample`, which is the picture `se.make(name)` gives and the one every parity "
        f"number in this repo is measured under. Each row below carries the RESOLVED value; forcing "
        f"a number instead makes 44 of the classes report a cheaper picture than they ship."
        + _carryover_note(results),
        f"Median of {results['reps']} full-batch renders (GPU) and {results['cpu_frames']} sequential frames (CPU).",
        "",
        (
            f"Host 1-minute load average {load[0]:.2f} at the start and {load[-1]:.2f} at the end, over "
            f"{results['cpu_count']} visible cores. "
            if load
            else ""
        )
        + "Each GPU row carries `cpu%`, the process CPU seconds per wall second of the timed loop. "
        "Near 0 the device is the limit and the row travels between nodes; at or above 100% the host "
        "is busy for the whole render and the row is a measurement of this node's processor under "
        "this node's load — the ray-cast families sit there (DMCBallInCupCatch measures 196%, and "
        "the same configuration has produced 3,664 frames/s idle and 271 contended). The CPU column "
        "does not reproduce under contention at all: the same sweep on a busier host measured "
        "MiniGridKeyCorridor's official renderer at 37 frames/s against 425, and elsewhere in this "
        "repo one official renderer has gone from 422 frames/s idle to 3 with three other jobs "
        "resident. Read every speedup as an order of magnitude, and discard a baseline taken on a "
        "loaded node rather than averaging it in.",
        "",
        f"Registry at run time ({len(results['envs'])} envs): {', '.join(sorted(results['envs']))}.",
        "",
    ]
    for env_name, env_res in results["envs"].items():
        L += [f"## {env_name}", ""]
        if "error" in env_res:
            L += [f"Could not be benchmarked: {env_res['error']}", ""]
            continue
        meta = env_res["describe"]
        names = ", ".join(meta["state_names"][:8]) + (" ..." if len(meta["state_names"]) > 8 else "")
        L += [f"State: {meta['state_dim']} floats `{names}`. Reference: `{meta['reference_id'] or 'none'}`.", ""]
        for size, block in env_res["sizes"].items():
            cpu = block.get("cpu")
            ssv = sorted({r["supersample"] for r in block["gpu"] if "supersample" in r})
            L += [f"### {size}x{size}" + (f" (supersample {', '.join(map(str, ssv))})" if ssv else ""), ""]
            if cpu and "error" not in cpu:
                base = cpu["set_state_render"]
                line = (
                    f"Official CPU renderer: **{base['fps']:,.0f} frames/s** "
                    f"({base['us_per_frame']:,.0f} us/frame, {base['gb_per_s']:.4f} GB/s) "
                    f"for set_state+render"
                )
                if "render_only" in cpu:
                    line += (
                        f"; {cpu['render_only']['fps']:,.0f} frames/s "
                        f"({cpu['render_only']['us_per_frame']:,.0f} us/frame) for the draw alone"
                    )
                L += [line + ".", ""]
            elif cpu:
                L += [f"Official CPU renderer unavailable: {cpu['error']}", ""]
            L += [
                "| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for row in block["gpu"]:
                if "fps" not in row:
                    why = row.get("skipped") or row.get("error")
                    L.append(f"| {row['batch']:,} | {why} | | | | | | |")
                    continue
                # Two decimals below 10x, none above. A batch-1 cell where the GPU path is BEHIND the
                # reference is the honest shape of a batched renderer called one frame at a time, and
                # `,.0f` rounded 0.88x up to "1x" — the one place in the table where the formatting was
                # deciding the result.
                r_ = row["fps"] / cpu["set_state_render"]["fps"] if cpu and "error" not in cpu else None
                sp = "-" if r_ is None else (f"{r_:.2f}x" if r_ < 10 else f"{r_:,.0f}x")
                cf = row.get("cpu_frac")
                L.append(
                    f"| {row['batch']:,} | {row['seconds_per_batch'] * 1e3:.3f} | "
                    f"{row['us_per_frame']:.4f} | {row['fps']:,.0f} | {row['gb_per_s']:.1f} | "
                    f"{row['peak_gb']:.2f} | "
                    f"{'—' if cf is None else format(100 * cf, '.0f') + '%'} | {sp} |"
                )
            L.append("")
            st_bytes = meta["state_dim"] * 4
            fh, fw = block.get("frame_shape", [int(size), int(size)])
            px_bytes = fh * fw * 3
            L += [
                f"Storage: {st_bytes} B of state against {px_bytes:,} B of raw frame "
                f"({fh}x{fw}) — **{px_bytes / st_bytes:,.0f}x** smaller before any compression.",
                "",
            ]
            rows = [r for r in block.get("breakdown", []) if "scene_ms" in r]
            if rows:
                L += [
                    "Where the time goes:",
                    "",
                    "| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |",
                    "|---|---|---|---|---|---|",
                ]
                for row in rows:
                    L.append(
                        f"| {row['batch']:,} | {row['scene_ms']:.3f} | {row['raster_ms']:.3f} | "
                        f"{row['render_ms']:.3f} | {row['raster_fps']:,.0f} | "
                        f"{row['raster_gb_per_s']:.1f} |"
                    )
                L.append("")
            elif block.get("breakdown"):
                why = {r.get("skipped") for r in block["breakdown"] if r.get("skipped")}
                if why:
                    L += [f"No scene/rasterize split: {'; '.join(sorted(why))}.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", default="native,96,224", help="'native' resolves to each env's default_size; numbers are absolute"
    )
    ap.add_argument("--batches", default="1,64,256,1024,4096,16384,65536")
    ap.add_argument(
        "--ss",
        default="native",
        help="'native' lets every env use its own default_supersample (44 classes declare "
        "2); a number forces that value on every env and measures a picture their "
        "defaults do not produce",
    )
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--cpu-frames", type=int, default=200)
    ap.add_argument(
        "--max-out-gb", type=float, default=10.0, help="skip any (batch, size) whose output would exceed this"
    )
    ap.add_argument("--pool", type=int, default=8192, help="states drawn once per env/size and tiled up to the batch")
    ap.add_argument("--envs", default="", help="comma-separated subset; default is the whole registry")
    ap.add_argument(
        "--skip", default="", help="comma-separated envs to leave out; for reference envs that abort the process"
    )
    ap.add_argument("--out", default="examples/results")
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    batches = [int(b) for b in a.batches.split(",")]
    ss = parse_ss(a.ss)

    try:
        cpu_name = (
            subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | head -1"], capture_output=True, text=True)
            .stdout.split(":", 1)[-1]
            .strip()
        )
    except Exception:
        cpu_name = platform.processor()

    results = {
        "device": torch.cuda.get_device_name(0),
        "cpu": cpu_name or "unknown",
        "cpu_count": len(os.sched_getaffinity(0)),
        "torch": torch.__version__,
        "supersample": a.ss,
        "reps": a.reps,
        "cpu_frames": a.cpu_frames,
        # The CPU baseline is only meaningful on a node this job effectively owns, so the
        # evidence for that is recorded rather than assumed. One entry at the start, one after
        # every env, so a sweep that started idle and ended contended is visible in the file.
        "loadavg": [os.getloadavg()[0]],
        "registry_size": len(se.registry.REGISTRY),
        "import_errors": se.import_errors(),
        "envs": {},
    }
    # Merge into any previous run, per env. The registry is large enough that sweeping it at several
    # resolutions does not fit one allocation, so the sweep has to be resumable and splittable by
    # --envs; replacing the file would make the last partial run delete every earlier measurement.
    prev = outdir / "bench_gpu_vs_cpu.json"
    if prev.exists():
        try:
            results["envs"] = json.loads(prev.read_text()).get("envs", {})
        except Exception as e:
            print(f"previous results unreadable ({type(e).__name__}); starting fresh", flush=True)

    def flush() -> None:
        """Write after every env.

        A reference env can take the process down in a way Python cannot catch — highway-env's pygame
        path aborts with SIGABRT — and a sweep that only writes at the end loses every measurement taken
        before the crash. Writing incrementally means a crash costs one env and the rerun is a --skip
        away.
        """
        results["loadavg"] = results["loadavg"][:1] + [os.getloadavg()[0]]
        (outdir / "bench_gpu_vs_cpu.json").write_text(json.dumps(results, indent=2))
        (outdir / "bench_gpu_vs_cpu.md").write_text(markdown(results))

    skip = {s for s in a.skip.split(",") if s}
    wanted = [n for n in sorted(se.registry.REGISTRY) if (not a.envs or n in a.envs.split(",")) and n not in skip]
    results["skipped_envs"] = sorted(skip)
    for name in wanted:
        cls = se.registry.REGISTRY[name]
        try:
            probe = se.make(name, num_envs=1, device="cuda")
            entry = {"describe": probe.describe(), "sizes": {}}
            native = probe.size
            del probe
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{name}] cannot instantiate: {e}", flush=True)
            results["envs"][name] = _err(e)
            flush()
            continue

        sizes = []
        for tok in a.sizes.split(","):
            s = native if tok == "native" else int(tok)
            if s not in sizes:
                sizes.append(s)
        for size in sizes:
            try:
                probe = se.make(name, num_envs=1, size=size, device="cuda")
                pool = probe.sample_states(min(max(batches), a.pool))
                f = probe.render(pool[:1]).cpu().numpy()
                hw = (int(f.shape[1]), int(f.shape[2]))
                del probe, f
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[{name} @ {size}px] cannot sample states: {e}", flush=True)
                entry["sizes"][str(size)] = {"cpu": _err(e), "gpu": [], "breakdown": []}
                continue
            print(f"[{name} @ {size}px] CPU reference...", flush=True)
            cpu = cpu_throughput(cls, name, size, a.cpu_frames, pool)
            gpu = []
            for B in batches:
                r = gpu_throughput(name, B, size, ss, a.reps, a.max_out_gb, pool, hw)
                got = format(r["fps"], ",.0f") + " frames/s" if "fps" in r else (r.get("skipped") or r.get("error"))
                print(f"[{name} @ {size}px] batch {B}: {got}", flush=True)
                gpu.append(r)
            breakdown = [
                gpu_breakdown(name, B, size, ss, a.reps, a.max_out_gb, pool, hw)
                for B in batches
                if B in (1, 256, 4096, 16384)
            ]
            entry["sizes"][str(size)] = {"cpu": cpu, "gpu": gpu, "breakdown": breakdown, "frame_shape": list(hw)}
            results["envs"][name] = entry
            flush()

    flush()
    print(markdown(results))


if __name__ == "__main__":
    main()
