"""Step 6C — phase attribution for the ``raycast3d`` path, without editing the renderer.

``scratch/step6/profile_note.py`` is the shared harness: it times ``env.render(states)`` on the
public path and attributes it with ``torch.profiler``. An operator trace answers "which aten op",
which is the wrong unit for this workstream's question — every raycast3d scene runs the same ops
and they differ by a factor of 12 in wall time — so this adds the unit that does answer it: the
PHASES of the render, measured by wrapping the functions the renderer already has.

Nothing in ``stable_envs`` is modified. The wrappers are installed here, on this process's
imported module objects, and every one of them is ``f(*a, **k)`` plus a clock, so the attribution
is of the real call graph rather than of a re-implementation of it.

Two numbers are reported for every phase because on CUDA they mean different things:

  * ``host_ms`` — inclusive wall time with NO synchronisation. This is what the Python thread
    spends inside the call. A phase that is large here and small in ``dev_ms`` is host work:
    a per-env loop, a numpy gather, a pageable host-to-device copy.
  * ``dev_ms`` — the same phase measured in a second pass with ``torch.cuda.synchronize()`` at
    both boundaries, so the device queue is drained inside the phase that filled it.

On CPU the two coincide and only ``host_ms`` is reported.

Also recorded, because "why is this scene slower than its sibling" is a question about the scene:
the primitive count and kind histogram actually cast, the ray count per pass, the screen window
the mover cast was restricted to, and (CPU only) per-primitive-kind intersection time from
``raycast3d.intersect``.

    python scratch/step6/raycast3d/profile_phases.py --env DMCBallInCupCatch --device cuda \
        --batch 1 16 64 256 1024 4096 --out examples/results/step6/raycast3d/profile_phases/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import platform
import socket
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# $PYTHONPATH DECIDES WHICH stable_envs IS MEASURED. A before/after run puts the baseline copy on
# PYTHONPATH for one leg and this branch for the other, so this file's own repo root must NOT be
# inserted ahead of it -- doing that made both legs import the branch and report a 1.00x speed-up as
# a real one. Appended, so the helper imports still resolve when PYTHONPATH is unset.
sys.path.append(str(ROOT))


def _root_tag(p: str) -> str:
    """An absolute Oscar path with the scratch root collapsed to ``$SCR``.

    The raw path is the evidence -- it says whether a leg loaded the baseline or the branch -- but
    written in full it reads to ``tests/unit/test_repo_contracts.py`` as a repo-relative
    ``scratch/...`` citation that cannot resolve, which is a false failure in a real check. Same
    string, written the way the sbatch files write it.
    """
    return re.sub(r"^/oscar/scratch/([^/]+)", r"$SCR", p)


sys.path.insert(0, str(ROOT / "scratch" / "step6"))

# reuse the shared harness rather than re-deriving its provenance block
from profile_note import git_commit, versions            # noqa: E402


class Rec:
    """Inclusive wall time per label, with a stack so a child can be subtracted from its parent."""

    def __init__(self):
        self.incl = defaultdict(float)
        self.calls = defaultdict(int)
        self.child = defaultdict(float)
        self.facts = defaultdict(list)
        self.stack = []
        self.sync = None            # callable or None
        self.on = False

    def reset(self):
        self.incl.clear(); self.calls.clear(); self.child.clear(); self.facts.clear()
        self.stack.clear()

    def note(self, key, value):
        if self.on:
            self.facts[key].append(value)

    def wrap(self, label, fn):
        def inner(*a, **k):
            if not self.on:
                return fn(*a, **k)
            if self.sync is not None:
                self.sync()
            t0 = time.perf_counter()
            self.stack.append(label)
            try:
                return fn(*a, **k)
            finally:
                self.stack.pop()
                if self.sync is not None:
                    self.sync()
                dt = time.perf_counter() - t0
                self.incl[label] += dt
                self.calls[label] += 1
                if self.stack:
                    self.child[self.stack[-1]] += dt
        inner.__name__ = getattr(fn, "__name__", label)
        inner.__wrapped_label__ = label
        return inner

    def table(self):
        out = {}
        for lab, tot in self.incl.items():
            out[lab] = {"incl_ms": tot * 1e3, "self_ms": (tot - self.child[lab]) * 1e3,
                        "calls": self.calls[lab]}
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["incl_ms"]))


REC = Rec()

KINDNAME = {0: "plane", 2: "sphere", 3: "capsule", 5: "cylinder", 6: "box"}


def install(env, device: str):
    """Wrap the renderer's own functions. Returns nothing; every patch is on an imported module."""
    import torch
    import stable_envs.render.raycast3d as R3
    import stable_envs.envs.dmcontrol as DMC
    import stable_envs.envs.ogbench as OGB

    if device.startswith("cuda"):
        REC.sync = torch.cuda.synchronize

    # ---- raycast3d module level
    raw_cast = R3.cast_nearest
    raw_any = R3.any_blocking
    raw_isect = R3.intersect

    def cast_nearest(kinds, sizes, pos, R, o, d, backend="auto"):
        REC.note("cast_K", int(len(kinds)))
        REC.note("cast_kinds", [KINDNAME.get(int(k), int(k)) for k in kinds])
        REC.note("cast_rays", int(o.shape[0]) * int(o.shape[1]))
        REC.note("cast_N_per_env", int(o.shape[1]))
        return raw_cast(kinds, sizes, pos, R, o, d, backend)

    def any_blocking(kinds, sizes, pos, R, o, d, lim, skip=None, backend="auto"):
        REC.note("shadow_K", int(len(kinds)))
        REC.note("shadow_rays", int(o.shape[0]) * int(o.shape[1]))
        return raw_any(kinds, sizes, pos, R, o, d, lim, skip, backend)

    def intersect(kind, size, pos, R, o, d):
        if not REC.on:
            return raw_isect(kind, size, pos, R, o, d)
        # The ray-primitive TEST COUNT, which is the quantity RAYCAST-4 exists to reduce: one
        # intersect call tests d.shape[0] x d.shape[1] (row, ray) pairs against one primitive.
        REC.note("ray_primitive_tests", int(d.shape[0]) * int(d.shape[1]))
        if REC.sync is not None:
            REC.sync()
        t0 = time.perf_counter()
        try:
            return raw_isect(kind, size, pos, R, o, d)
        finally:
            if REC.sync is not None:
                REC.sync()
            dt = time.perf_counter() - t0
            lab = "isect:" + KINDNAME.get(int(kind), str(int(kind)))
            REC.incl[lab] += dt
            REC.calls[lab] += 1

    R3.cast_nearest = REC.wrap("cast_nearest", cast_nearest)
    R3.any_blocking = REC.wrap("any_blocking(shadow ray)", any_blocking)
    R3.intersect = intersect                                    # own accounting, not in the stack
    R3.cached_rays = REC.wrap("cached_rays", R3.cached_rays)
    R3.screen_span = REC.wrap("screen_span(+sync)", R3.screen_span)
    R3.shade = REC.wrap("shade", R3.shade)
    R3.composite = REC.wrap("composite", R3.composite)

    # ---- DM Control
    def patch(cls, name, lab):
        raw = cls.__dict__.get(name)
        if raw is None:
            return
        if isinstance(raw, staticmethod):
            setattr(cls, name, staticmethod(REC.wrap(lab, raw.__func__)))
        elif isinstance(raw, classmethod):
            setattr(cls, name, classmethod(REC.wrap(lab, raw.__func__)))
        elif callable(raw):
            setattr(cls, name, REC.wrap(lab, raw))

    for name, lab in (("_fk", "fk (per-env mujoco loop)"),
                      ("_render_group", "render_group"),
                      ("_static_t", "static_t"),
                      ("_per_chunk_constants", "per_chunk_constants(rays)"),
                      ("_mover_idx", "mover_idx(tile ladder)"),
                      ("_reflect", "reflect"),
                      ("_shadow_candidates", "shadow_candidates"),
                      ("_floor_rays", "floor_rays"),
                      ("_static_shadow_rays", "static_shadow_rays"),
                      ("_occluded", "occluded(mover shadow)"),
                      ("_cast", "cast(mover)"),
                      ("_scene", "scene(bake)")):
        patch(DMC.DMControlSuite, name, lab)
    DMC._shade = REC.wrap("shade", DMC._shade)

    # ---- OGBench
    for name, lab in (("_fk", "fk (per-env mujoco loop)"),
                      ("_scene", "scene(bake)"),
                      ("_agent_appearance", "agent_appearance")):
        patch(OGB.OGBenchMaze, name, lab)


def scene_facts(env) -> dict:
    """What the scene IS, so a speed difference can be attributed to geometry rather than guessed."""
    out = {"class": type(env).__name__}
    try:
        sc = env._scene(env.size)
    except Exception as exc:                      # noqa: BLE001
        out["scene_error"] = repr(exc)
        return out
    prims = getattr(sc, "prims", None) or getattr(sc, "prims_base", None)
    if prims is not None:
        kinds = [KINDNAME.get(int(k), int(k)) for k in prims.kind]
        out["mover_primitives"] = len(kinds)
        h = defaultdict(int)
        for k in kinds:
            h[k] += 1
        out["mover_kind_histogram"] = dict(sorted(h.items()))
    td = getattr(sc, "tendon", None)
    out["tendon_segments"] = int(td.nseg) if td is not None else 0
    out["movers"] = [len(x) for x in getattr(sc, "movers", ([], []))] \
        if hasattr(sc, "movers") else None
    out["shadow_light"] = int(getattr(sc, "shadow_light", -1)) \
        if hasattr(sc, "shadow_light") else None
    out["reflectance"] = float(getattr(sc, "reflectance", 0.0) or 0.0) \
        if hasattr(sc, "reflectance") else None
    out["size"] = int(env.size)
    out["supersample"] = int(getattr(env, "ss", 1))
    out["fk_kernel"] = "mj_forward (tendon)" if out["tendon_segments"] else "mj_kinematics"
    if type(env).__module__.endswith("ogbench"):
        out["fk_kernel"] = "closed-form GPU" if getattr(env, "LOCO", "") == "point" \
            else "mj_kinematics"
        out["agent_geoms"] = len(getattr(sc, "agent_ids", []))
    return out


def timed(fn, device: str, repeat: int, warmup: int) -> dict:
    import torch
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    s = []
    for _ in range(repeat):
        if cuda:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            s.append(a.elapsed_time(b))
        else:
            t0 = time.perf_counter(); fn(); s.append((time.perf_counter() - t0) * 1e3)
    return {"ms_median": statistics.median(s), "ms_min": min(s), "ms_max": max(s),
            "ms_stdev": statistics.pstdev(s) if len(s) > 1 else 0.0, "samples": len(s)}


def kernel_census(fn, device: str) -> dict:
    """Kernel launches and device time, from one profiled iteration."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU]
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        fn()
        if cuda:
            torch.cuda.synchronize()
    evs = prof.key_averages()
    attr = "self_device_time_total" if hasattr(evs[0], "self_device_time_total") \
        else "self_cuda_time_total"
    launches = sum(e.count for e in evs if e.key in ("cudaLaunchKernel", "cudaLaunchKernelExC"))
    dev = sum(getattr(e, attr, 0) or 0 for e in evs)
    memcpy = sum(getattr(e, attr, 0) or 0 for e in evs if "emcpy" in e.key or "opy" in e.key)
    top = sorted(evs, key=lambda e: -(getattr(e, attr, 0) or 0))[:12]
    return {"op_calls": sum(e.count for e in evs), "cuda_launches": int(launches),
            "device_us_total": dev, "device_us_memcpy": memcpy,
            "host_op_us_total": sum(e.self_cpu_time_total for e in evs),
            "top_device": [{"op": e.key[:90], "calls": e.count,
                            "device_us": getattr(e, attr, 0) or 0,
                            "cpu_us": e.self_cpu_time_total} for e in top
                           if (getattr(e, attr, 0) or 0) > 0][:10]}


def phase_pass(fn, repeat: int) -> dict:
    REC.reset(); REC.on = True
    try:
        for _ in range(repeat):
            fn()
    finally:
        REC.on = False
    t = REC.table()
    for lab in t:
        t[lab]["incl_ms"] /= repeat
        t[lab]["self_ms"] /= repeat
        t[lab]["calls"] /= repeat
    facts = {}
    for k, v in REC.facts.items():
        if k.endswith("kinds"):
            facts[k] = v[0] if v else None
        else:
            vals = [x for x in v if isinstance(x, (int, float))]
            if vals:
                facts[k] = {"per_render_calls": len(vals) / repeat,
                            "min": min(vals), "max": max(vals),
                            "sum_per_render": sum(vals) / repeat}
    return {"phases": t, "facts": facts}


def cprofile_top(fn, n: int = 30) -> list:
    """Python-level attribution of ONE render. cProfile only sees Python frames, which is the
    point: a phase that is large here and absent from the operator trace is host work."""
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)
    rows = []
    for (fname, lineno, func), (cc, nc, tt, ct, _) in st.stats.items():
        rows.append({"func": f"{Path(fname).name}:{lineno}({func})", "ncalls": nc,
                     "tottime_ms": tt * 1e3, "cumtime_ms": ct * 1e3})
    rows.sort(key=lambda r: -r["tottime_ms"])
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-sync-phases", action="store_true",
                    help="attribute host time only (no cuda sync inside wrappers)")
    ap.add_argument("--cprofile", type=int, default=0,
                    help="also run cProfile on one render at this batch size and record the top")
    ap.add_argument("--fk-counterfactual", action="store_true",
                    help="also time render with _fk memoised, to bound what removing it buys")
    ap.add_argument("--states-dir", type=Path,
                    default=ROOT / "examples" / "results" / "step4" / "states")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    import stable_envs as se

    if args.threads:
        torch.set_num_threads(args.threads)

    note = {"tool": "profile_phases", "workstream": "step6/raycast3d", "env": args.env,
            "device": args.device, "baseline_commit": git_commit(),
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu": platform.processor() or platform.machine(),
            "threads": torch.get_num_threads(), "versions": versions(),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "stable_envs_from": _root_tag(se.__file__),          # WHICH code this number describes
            "pythonpath": _root_tag(os.environ.get("PYTHONPATH", "")), "batches": {}}

    probe = se.make(args.env, batch=1, size=args.size, device=args.device)
    note["size"] = int(probe.size)
    note["state_dim"] = int(probe.state_dim)
    note["scene"] = scene_facts(probe)

    npz = args.states_dir.expanduser().resolve() / f"{args.env}.npz"
    if npz.is_file():
        pool = np.load(npz)["states"]
        note["states_source"] = str(npz.relative_to(ROOT)) if str(npz).startswith(str(ROOT)) \
            else str(npz)
    else:
        pool = probe.sample_states(64)
        note["states_source"] = "env.sample_states"

    install(probe, args.device)
    if args.no_sync_phases:
        REC.sync = None

    for B in args.batch:
        states = np.asarray(pool[np.arange(B) % len(pool)], np.float32)
        e = se.make(args.env, batch=B, size=args.size, device=args.device)
        e.render(states)                                            # bake the scene out of the loop
        rec = {"batch": B}
        rec["set_state_ms"] = timed(lambda: e.set_state(states), args.device, args.repeat,
                                    args.warmup)
        rec["render_ms"] = timed(lambda: e.render(states), args.device, args.repeat, args.warmup)
        rec["fps"] = 1000.0 * B / rec["render_ms"]["ms_median"]
        rec["kernels"] = kernel_census(lambda: e.render(states), args.device)
        rec.update(phase_pass(lambda: e.render(states), max(3, args.repeat // 2)))
        # how much of the wall clock the phase pass accounted for
        rec["phase_overhead_ms"] = sum(v["self_ms"] for v in rec["phases"].values())
        if args.fk_counterfactual:
            import stable_envs.envs.dmcontrol as DMC
            import stable_envs.envs.ogbench as OGB
            cls = type(e)
            base = cls._fk
            cache = {}

            def frozen(self, s, sc, _b=base, _c=cache):
                k = (id(sc), int(s.shape[0]))
                if k not in _c:
                    _c[k] = _b(self, s, sc)
                return _c[k]
            cls._fk = frozen
            try:
                e.render(states)
                rec["render_ms_fk_frozen"] = timed(lambda: e.render(states), args.device,
                                                   args.repeat, args.warmup)
                rec["fps_fk_frozen"] = 1000.0 * B / rec["render_ms_fk_frozen"]["ms_median"]
            finally:
                cls._fk = base
        if args.cprofile and B == args.cprofile:
            rec["cprofile"] = cprofile_top(lambda: e.render(states))
        note["batches"][str(B)] = rec
        line = (f"B={B:<6} render {rec['render_ms']['ms_median']:9.2f} ms "
                f"({rec['fps']:9.1f} fps)  ops {rec['kernels']['op_calls']:6d} "
                f"launches {rec['kernels']['cuda_launches']:5d} "
                f"dev {rec['kernels']['device_us_total'] / 1e3:8.2f} ms")
        if "fps_fk_frozen" in rec:
            line += f"  | fk-frozen {rec['fps_fk_frozen']:9.1f} fps"
        print(line, flush=True)
        for lab, v in list(rec["phases"].items())[:8]:
            print(f"      {lab:34s} incl {v['incl_ms']:9.2f} self {v['self_ms']:9.2f} "
                  f"calls {v['calls']:6.1f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    host = note["host"].split(".")[0]
    tag = f"{args.env}_{args.device.replace(':', '')}_{host}_phases"
    (args.out / f"{tag}.json").write_text(json.dumps(note, indent=2))
    print(f"\nwrote {args.out / (tag + '.json')}", flush=True)


if __name__ == "__main__":
    main()
