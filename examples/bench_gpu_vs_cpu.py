"""GPU renderer against the reference CPU renderer, per registered env.

Both sides perform the same ``state -> uint8 frame`` transformation. The CPU side reports two timings:

  ``set_state + render``   place the reference at a stored state, then draw it. This is what the GPU
                           path does and is the number the speedup column uses.
  ``render only``          the draw alone, where the reference exposes a per-state setter.

The GPU column is a whole-batch render divided by the batch size; batch 1 is included to show launch
overhead.

Both columns depend on the host. The CPU renderer is single-threaded Python and slows sharply under
load, and some GPU families (the ray-cast ones) are host-bound inside their per-chunk Python loop. Each
GPU row therefore records ``cpu_frac`` (process CPU seconds per wall second; near 0 means the device is
the limit) and the load average is recorded around the sweep. Read speedups as orders of magnitude.

Every env in ``stable_envs.registry.REGISTRY`` is measured; envs with a reference renderer also get the
CPU column. ``native`` resolves to each env's ``default_size``. Every measurement is wrapped, so a
failing env records its error instead of ending the sweep.

    python examples/bench_gpu_vs_cpu.py --out results
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
    """Build the env; ``ss=None`` leaves the class's own ``default_supersample`` in place.

    ``ss=None`` must not become ``supersample=1``: classes that declare ``default_supersample = 2``
    would then render a different, cheaper picture than ``se.make(name)`` gives.
    """
    kw = {} if ss is None else {"supersample": int(ss)}
    return se.make(name, num_envs=batch, size=size, device=DEVICE, **kw)


#: The device every timed stable-envs render runs on.
DEVICE = "cuda"


def _sync() -> None:
    """Wait for queued device work; a no-op on CPU."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def parse_ss(tok: str) -> int | None:
    """``--ss native`` -> None (the env's own ``default_supersample``); a number -> that number."""
    return None if str(tok).strip().lower() in ("native", "default", "env", "none") else int(tok)


#: Wall-clock budget for one timed cell; repetitions stop once it is spent and ``MIN_REPS`` are taken.
CELL_BUDGET_S = 1.0
MIN_REPS = 3


def _median_and_cpu(fn, reps: int) -> tuple[float, float]:
    """``(median wall seconds, CPU seconds per wall second)`` over up to ``reps`` synchronized timings.

    A CPU fraction near 0 means the device is the limit; at or above 1 the host is busy for the whole
    render and the number depends on the machine and its load.
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

    The pool is sampled once per env and size because ``sample_states`` steps the reference simulator
    for some envs. Repeated states do not change render cost.
    """
    if len(pool) >= batch:
        return pool[:batch]
    return np.ascontiguousarray(np.resize(pool, (batch, pool.shape[1])))


def gpu_throughput(
    name: str, batch: int, size: int, ss: int, reps: int, max_out_gb: float, pool: np.ndarray, hw: tuple[int, int]
) -> dict:
    """One full-batch render, as frames/s, µs/frame and output GB/s.

    ``hw`` is the frame shape measured from a render; several envs are not square.
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
        # Record the resolved supersample, not the requested one.
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
    """Peak GB: device allocator peak on CUDA, otherwise the whole process's peak RSS."""
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

    ``scene()`` cost is mostly Python launch overhead and nearly batch-independent; rasterizing scales
    with output pixels. Envs that do not build a ``Scene2D`` report a skip.
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
    """The reference renderer's cost for the same ``state -> frame`` transformation, one frame at a time.

    Where the env exposes a per-state setter, the draw alone is also timed.
    """
    if not hasattr(cls, "reference_env") or getattr(cls, "reference_id", None) is None:
        return None
    ref = None
    try:
        ref = cls.reference_env(size=size)
        states = states_for(pool, frames + 8)

        cls.reference_render(ref, states[:8])  # warm up surfaces, caches, JITs
        t0 = time.perf_counter()
        cls.reference_render(ref, states[8:])
        dt_full = (time.perf_counter() - t0) / frames

        # Optional draw-alone timing. Not every reference is a gymnasium env with ``unwrapped``, and
        # failing here must not discard the timing above.
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
    """Name the merged envs whose GPU rows carry no resolved supersample.

    Results merge across runs, so the header's supersample setting may not describe every row.
    """
    stale = sorted(
        name
        for name, e in results.get("envs", {}).items()
        if (rows := [r for blk in e.get("sizes", {}).values() for r in blk.get("gpu", []) if "error" not in r])
        and not any(r.get("supersample") is not None for r in rows)
    )
    if not stale:
        return ""
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
        f"`default_supersample`, which is the picture `se.make(name)` gives. Each row below carries "
        f"the resolved value." + _carryover_note(results),
        f"Median of {results['reps']} full-batch renders (GPU) and {results['cpu_frames']} sequential frames (CPU).",
        "",
        (
            f"Host 1-minute load average {load[0]:.2f} at the start and {load[-1]:.2f} at the end, over "
            f"{results['cpu_count']} visible cores. "
            if load
            else ""
        )
        + "Each GPU row carries `cpu%`, the process CPU seconds per wall second of the timed loop. "
        "Near 0 the device is the limit; at or above 100% the host is busy for the whole render and "
        "the row depends on this machine's processor and load. The CPU column slows sharply under "
        "contention. Read every speedup as an order of magnitude, and discard a baseline taken on a "
        "loaded machine.",
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
                # Two decimals below 10x, none above, so a batch-1 ratio below 1x is not rounded up to "1x".
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
    ap.add_argument("--out", default="results")
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
        # Load average at the start, then after every env.
        "loadavg": [os.getloadavg()[0]],
        "registry_size": len(se.registry.REGISTRY),
        "import_errors": se.import_errors(),
        "envs": {},
    }
    # Merge into any previous run, per env, so a sweep can be split with --envs and resumed.
    prev = outdir / "bench_gpu_vs_cpu.json"
    if prev.exists():
        try:
            results["envs"] = json.loads(prev.read_text()).get("envs", {})
        except Exception as e:
            print(f"previous results unreadable ({type(e).__name__}); starting fresh", flush=True)

    def flush() -> None:
        """Write after every env, so a reference that aborts the process loses only that env."""
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
