"""RAYCAST-3, design-first: what does one ray-primitive intersection COST on one CPU core?

DESIGN ONLY. Nothing here is wired into ``stable_envs``; the kernel lives in
``scratch/step6/raycast3d/cast_kernel.cpp`` and is built into a scratch extension directory.

Ticket 1 measured the CPU blocker: ``raycast3d.intersect`` is 76-95 % of every multi-mover CPU
render and costs **145-240 ns per ray-primitive test** on one core, because ``cast_nearest_torch``
pays 60-100 flops of arithmetic as ~200 elementwise kernels over materialised ``(B,N)`` temporaries.
``step6/cpu-3d``'s design reproduces the same constant (220 ns/test) on a different scene shape and
computes the bar this has to clear: ``cores_cpu >= 1`` at 84 px on ``node2339`` is one-core
JPEG-q90 decode at 6,629 fps, i.e. **151 us per frame, 21.4 ns per pixel**.

This asks the go/no-go question for a fused CPU cast, and asks it in the order that can kill it
cheapest:

  1. **Is a compiled transcription bit-identical to ``cast_nearest_torch``?** If not, nothing else
     matters -- the frozen Step-4 digests were produced by that expression and a CPU path that
     disagrees is a different renderer, not a faster one. Checked first, on uint32 bit patterns.
  2. **What is the ns/test floor of each option?** the shipped torch loop; torch with the rays
     compacted and chunked (no compiler, the cheapest thing that could work); a scalar C++ kernel
     with the K-loop in registers; and numba if the interpreter has it.
  3. **Does that floor reach CPU_FAST for the raycast3d parity cells?** Answered by multiplying the
     measured ns/test by each cell's MEASURED ray-primitive test count per frame (ticket 1's phase
     notes, quoted in CELLS below) and comparing against that cell's decode bar.

The scene is synthetic but shaped like the real ones: the kind mix and primitive count of each cell,
rays through a real pinhole over the cell's measured window. A synthetic bundle is the right unit
here -- the question is the cost of the arithmetic per test, and ticket 1 already established that
the cost is flat in what the rays hit.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))          # $PYTHONPATH wins; see replay_step4.py


def _root_tag(p: str) -> str:
    return re.sub(r"^/oscar/scratch/([^/]+)", r"$SCR", p)


#: cell -> (K, kind mix, rays cast per frame summed over the passes, decode fps at its size).
#: EVERY number here is measured, not assumed: the counts come from ticket 1's phase notes
#: (examples/results/step6/raycast3d/profile_phases_cpu/*.json, field ``facts``) at B=64, and the
#: decode rates from Step 3's frozen certification rows on node2349.
CELLS = {
    # name:            K, n_capsule, n_sphere, ray-prim tests per frame, decode fps (its size)
    "DMCReacherHard":        dict(K=4, cap=2, sph=2, tests=80884, decode=7014.9, size=84),
    "DMCFingerSpin":         dict(K=7, cap=5, sph=2, tests=93401, decode=7125.3, size=84),
    "DMCBallInCupCatch":     dict(K=36, cap=33, sph=3, tests=279972, decode=7186.9, size=84),
    "AntmazeMediumExplore":  dict(K=13, cap=12, sph=1, tests=53248, decode=2390.5, size=224),
    "PointmazeLargeNavigate": dict(K=1, cap=0, sph=1, tests=4096, decode=2390.5, size=224),
}


#: -ffp-contract=off is load-bearing: it is what stops ``a*b+c`` becoming one fma with ONE rounding
#: where torch -- two kernels -- rounds twice. Same knob as Triton's ``enable_fp_fusion=False``, and
#: the reason the fused CUDA cast is admissible at all. ``-fno-fast-math`` keeps ``/`` and ``sqrtf``
#: IEEE.
CFLAGS = ["-O3", "-march=native", "-ffp-contract=off", "-fno-fast-math"]
if os.environ.get("MB_NO_MARCH"):
    # Built a second time without -march=native to separate "my transcription is wrong" from "this
    # ISA's codegen reassociates". On gcc 11.5 / Cascade Lake the capsule NORMAL came back one ulp
    # off the torch expression while the sphere was exact and the winner index was exact on every
    # cell; on Apple clang / arm64 the same source was exact everywhere. That is the shape of a
    # codegen difference, and this is the cheapest experiment that can say so.
    CFLAGS = ["-O3", "-ffp-contract=off", "-fno-fast-math"]


def build_cpp(verbose: bool):
    """Build the kernel. ``cpp_extension.load`` when ninja is there, a plain g++ line when it is not.

    The fallback is not a nicety: ``se-step6`` has no ninja, and "the compiled option could not be
    measured because a build tool was missing" would be the wrong answer to a go/no-go question.
    """
    import subprocess

    import torch
    from torch.utils import cpp_extension as C

    src = Path(__file__).with_name("cast_kernel.cpp")
    build_dir = (Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp"))
                 / ("raycast3d_microbench_nomarch" if os.environ.get("MB_NO_MARCH")
                    else "raycast3d_microbench"))
    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        import ninja                                                    # noqa: F401
        return C.load(name="raycast3d_microbench", sources=[str(src)], extra_cflags=CFLAGS,
                      build_directory=str(build_dir), verbose=verbose), "cpp_extension.load+ninja"
    except ImportError:
        pass
    import importlib.util
    import sysconfig
    so = build_dir / "raycast3d_microbench.so"
    torch_lib = Path(torch.__file__).parent / "lib"
    # torch 2.14's headers use `requires` clauses, so the standard is not a free choice; try the
    # newer one first and fall back, rather than hard-coding a guess that fails on one toolchain.
    errs = []
    # macOS resolves the interpreter's symbols at load time, not at link time; Linux needs nothing.
    plat = (["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else [])
    for std in ("-std=c++20", "-std=c++17"):
        cmd = ([os.environ.get("CXX", "g++"), "-shared", std, "-fPIC",
                "-DTORCH_EXTENSION_NAME=raycast3d_microbench",
                "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
               + CFLAGS
               + [f"-I{p}" for p in C.include_paths()]
               + [f"-I{sysconfig.get_paths()['include']}", str(src),
                  f"-L{torch_lib}", "-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_python",
                  f"-Wl,-rpath,{torch_lib}", "-o", str(so)] + plat)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            break
        errs.append(f"{std}: {r.stderr[-300:]}")
    else:
        raise RuntimeError("g++ failed:\n" + "\n".join(errs))
    spec = importlib.util.spec_from_file_location("raycast3d_microbench", str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, "g++ (no ninja)"


def make_scene(spec, N, seed=20260913):
    """Rays and primitives shaped like ``spec``. float32 throughout, as the renderer is."""
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    K = spec["K"]
    kinds = np.array([2] * spec["sph"] + [3] * spec["cap"], np.int32)[:K]
    r = rng.uniform(0.01, 0.06, K).astype(np.float32)
    hh = rng.uniform(0.02, 0.12, K).astype(np.float32)
    sizes2 = np.stack([r, hh], 1)                                    # (K,2) for the C++ kernel
    pos = rng.uniform(-0.25, 0.25, (K, 3)).astype(np.float32)
    q = rng.normal(size=(K, 4)).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Rm = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                  1).astype(np.float32)                              # (K,9)
    # a pinhole looking at the cluster, rays through the window the cell actually casts
    o = np.array([[0.0, -1.2, 0.55]], np.float32)
    tgt = rng.uniform(-0.3, 0.3, (N, 3)).astype(np.float32)
    d = tgt - o
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d.astype(np.float32)
    t = torch.from_numpy
    return dict(kinds=t(kinds), sizes2=t(sizes2), pos=t(pos), Rm=t(Rm), o=t(o), d=t(d),
                # the shapes cast_nearest_torch wants: (B,K,S), (B,K,3), (B,K,3,3), (B,N,3)
                sizes3=t(np.stack([r, hh, np.zeros_like(r)], 1)).view(1, K, 3),
                pos3=t(pos).view(1, K, 3), R3=t(Rm).view(1, K, 3, 3),
                o3=t(o).view(1, 1, 3).expand(1, N, 3), d3=t(d).view(1, N, 3),
                kind_list=[int(v) for v in kinds])


def bits(x):
    import numpy as np
    a = x.detach().cpu().numpy()
    return a.view(np.uint32) if a.dtype == np.float32 else a


def _normal_delta(got, want):
    """How far apart are the two normals, in ulps and in absolute value, and on how many components?"""
    import numpy as np
    g, w = got.numpy(), want.numpy()
    gb, wb = g.view(np.int32).astype(np.int64), w.view(np.int32).astype(np.int64)
    diff = gb != wb
    ulp = np.abs(gb - wb)[diff]
    ad = np.abs(g.astype(np.float64) - w.astype(np.float64))
    finite = np.isfinite(ad)
    return {
        "normal_components_differing": int(diff.sum()),
        "normal_components_total": int(diff.size),
        "normal_rays_differing": int(diff.any(-1).sum()),
        "normal_max_ulp": int(ulp.max()) if ulp.size else 0,
        "normal_ulp_histogram": {str(int(k)): int(v) for k, v in
                                 zip(*np.unique(ulp, return_counts=True))} if ulp.size else {},
        "normal_max_abs_diff": float(ad[finite].max()) if finite.any() else 0.0,
    }


def timeit(fn, repeat, warmup):
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    return statistics.median(s)


def torch_chunked(R3mod, sc, N, chunk):
    """The shipped loop, with the rays cut into cache-sized chunks. No compiler, no new maths."""
    import torch
    outs = []
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        outs.append(R3mod.cast_nearest_torch(
            sc["kind_list"], sc["sizes3"], sc["pos3"], sc["R3"],
            sc["o3"][:, a:b], sc["d3"][:, a:b]))
    return (torch.cat([o[0] for o in outs], 1), torch.cat([o[1] for o in outs], 1),
            torch.cat([o[2] for o in outs], 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=list(CELLS))
    ap.add_argument("--rays", type=int, default=7056, help="rays per measurement (one 84px frame)")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verbose-build", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    import stable_envs.render.raycast3d as R3

    rep = {"host": socket.gethostname(), "cpu": platform.processor() or platform.machine(),
           "threads": torch.get_num_threads(), "torch": torch.__version__,
           "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "stable_envs_from": _root_tag(R3.__file__), "rays_per_measurement": args.rays,
           "bar": {"what": "cores_cpu >= 1 needs the whole frame inside one-core JPEG-q90 decode",
                   "ns_per_px_at_84": 21.4, "source": "step6/cpu-3d design.md 4.1/4.3"},
           "backends": {}, "cells": {}}

    ext = err = None
    try:
        ext, how = build_cpp(args.verbose_build)
        rep["backends"]["cpp"] = {"built": True, "flags": " ".join(CFLAGS), "via": how}
    except Exception as exc:                                            # noqa: BLE001
        err = repr(exc)[:400]
        rep["backends"]["cpp"] = {"built": False, "error": err}
        print(f"cpp extension did NOT build: {err}", flush=True)
    try:
        import numba                                                    # noqa: F401
        rep["backends"]["numba"] = {"available": True, "version": numba.__version__}
    except Exception as exc:                                            # noqa: BLE001
        rep["backends"]["numba"] = {"available": False, "why": repr(exc)[:160]}

    for name in args.cells:
        spec = CELLS[name]
        sc = make_scene(spec, args.rays)
        K, N = spec["K"], args.rays
        row = {"spec": spec, "K": K, "N": N, "ray_primitive_tests_per_measurement": K * N}

        ref = R3.cast_nearest_torch(sc["kind_list"], sc["sizes3"], sc["pos3"], sc["R3"],
                                    sc["o3"], sc["d3"])
        row["torch_loop_ms"] = timeit(lambda: R3.cast_nearest_torch(
            sc["kind_list"], sc["sizes3"], sc["pos3"], sc["R3"], sc["o3"], sc["d3"]),
            args.repeat, args.warmup) * 1e3

        for chunk in (1024, 4096):
            row[f"torch_chunk{chunk}_ms"] = timeit(
                lambda c=chunk: torch_chunked(R3, sc, N, c), args.repeat, args.warmup) * 1e3
            got = torch_chunked(R3, sc, N, chunk)
            row[f"torch_chunk{chunk}_bit_identical"] = bool(
                np.array_equal(bits(got[0]), bits(ref[0]))
                and np.array_equal(bits(got[1]), bits(ref[1]))
                and np.array_equal(got[2].numpy(), ref[2].numpy()))

        if ext is not None:
            out = ext.cast_nearest_cpu(sc["kinds"], sc["sizes2"], sc["pos"], sc["Rm"],
                                       sc["o"], sc["d"])
            same_t = np.array_equal(bits(out[0]), bits(ref[0][0]))
            same_n = np.array_equal(bits(out[1]), bits(ref[1][0]))
            same_i = np.array_equal(out[2].numpy(), ref[2][0].numpy())
            dt = np.abs(out[0].numpy().astype(np.float64)
                        - ref[0][0].numpy().astype(np.float64))
            finite = np.isfinite(dt) & (out[0].numpy() < 1e29)
            row["cpp"] = {
                "bit_identical_t": bool(same_t), "bit_identical_normal": bool(same_n),
                "same_winner_index": bool(same_i),
                "t_ulp_mismatches": int((bits(out[0]) != bits(ref[0][0])).sum()),
                "t_max_abs_diff_on_hits": float(dt[finite].max()) if finite.any() else 0.0,
                # "not bit-exact" is not actionable; "3 of 21,168 components differ by one ulp" is.
                # A one-ulp normal is a transcription that rounds differently somewhere, which
                # pinning a toolchain can hide and cannot fix; a large delta is a wrong expression.
                **_normal_delta(out[1], ref[1][0]),
                "ms": timeit(lambda: ext.cast_nearest_cpu(
                    sc["kinds"], sc["sizes2"], sc["pos"], sc["Rm"], sc["o"], sc["d"]),
                    args.repeat, args.warmup) * 1e3,
            }

        tests = K * N
        for tag, ms in [("torch_loop", row["torch_loop_ms"]),
                        ("torch_chunk1024", row["torch_chunk1024_ms"]),
                        ("torch_chunk4096", row["torch_chunk4096_ms"])] + \
                       ([("cpp", row["cpp"]["ms"])] if "cpp" in row else []):
            ns = ms * 1e6 / tests
            frame_us = spec["tests"] * ns / 1e3
            fps = 1e6 / frame_us if frame_us else 0.0
            row.setdefault("ns_per_ray_primitive_test", {})[tag] = ns
            row.setdefault("cast_only_frame_us", {})[tag] = frame_us
            row.setdefault("cast_only_fps_ceiling", {})[tag] = fps
            budget_us = 1e6 / spec["decode"]
            row.setdefault("fraction_of_decode_budget", {})[tag] = frame_us / budget_us
            # The other half of the question. At this ns/test, how many ray-primitive tests does the
            # WHOLE frame budget buy?  Compare against the measured count: if the ratio is large the
            # kernel is not the blocker, the number of tests issued is.
            row.setdefault("tests_affordable_in_full_budget", {})[tag] = budget_us * 1e3 / ns
            row.setdefault("test_count_reduction_needed", {})[tag] = (
                spec["tests"] / (budget_us * 1e3 / ns))
        rep["cells"][name] = row
        print(f"{name:24s} K={K:<3d} "
              + "  ".join(f"{k}={v:7.1f}ns" for k, v in row["ns_per_ray_primitive_test"].items())
              + (f"  cpp_bitexact={row['cpp']['bit_identical_t'] and row['cpp']['bit_identical_normal']}"
                 if "cpp" in row else ""), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2))

    args.out.write_text(json.dumps(rep, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
