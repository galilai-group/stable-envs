"""Oracle proof for the Step-9 render-2d workstream: the same pixels, from two different trees.

WHAT IT PROVES. Plan §7 and §27 say the architecture may change and the behaviour may not. For the
eight modules this workstream owns -- ``render/{scene,raster2d,gfxdraw,gfxdraw_classic,
gfxdraw_fused,tilemap,navdot,prng}.py`` -- "the behaviour" is the frames, so this runs EVERY
registered cell that reaches one of them through the PUBLIC path (``cls(batch=n, device=...)``,
``set_state``, ``render()``) in two separate interpreters, one with the Step-8 baseline tree on
``PYTHONPATH`` and one with this branch, and compares SHA-256 over the raw uint8 frame bytes.
Nothing is compared against a stored digest: both sides are computed here, now, in the same venv on
the same host, so a difference can only be ours.

WHICH CELLS, AND HOW THEY ARE CHOSEN. Not by reading the imports, and not from a list written by
hand: every registered config is constructed and rendered with twenty entry points of the eight
modules wrapped in counters, and a cell is IN if it called something. The call set (``backends`` in
the record) is itself compared between the arms, so a cell that silently stops using a backend is a
difference rather than a pass -- and because the cell list is discovered rather than declared, the
GPU arm automatically covers whatever the Oscar venv can import that the Mac cannot (CarRacing,
LunarLander, the MiniGrid family, Sokoban, Overcooked, Griddly, PGX, MiniHack). The two arms'
discovered lists are compared as sets, so a cell that appears in one arm only is a difference too.

WHICH STATES. The frozen corpus wherever one exists -- ``examples/results/step4/states/*.npz`` for
PushT, the five classic-control cells, the four swm_nav cells, SimplePointMaze and CraftaxClassic,
``examples/results/step7/craftax-full/corpus/states.npz`` for CraftaxFull -- and, for the cells that
never had one (the six highway configs and every tile family), the env's own ``sample_states`` at a
fixed seed. The digest of the states themselves is recorded and compared, so "the same frames" is
never an artefact of the two arms having been handed different inputs.

THE NON-FUSED PATHS ARE CHECKED AS WELL. Five of these modules ship a fused CUDA kernel and a torch
expression it must equal, and a refactor that preserved only the kernel would pass a naive frame
comparison. On ``cpu`` the torch expression IS what runs, so the CPU arm covers it by construction;
on ``cuda`` each cell is additionally rendered through whichever of these its call set says it has:

    backend_torch     raster2d.render_torch instead of render_triton  (Scene2D cells)
    cull_off/cull_on  the binned and unbinned scene kernels           (Scene2D cells)
    gfx_per_primitive gfxdraw's per-primitive path instead of gfxdraw_classic's kernel
    tilemap_torch     tilemap.blit_torch / visibility_scan_torch instead of the kernels
    navdot_torch      envs/navigation.py's torch blend instead of navdot.blend_dots

Usage (CPU arm, locally)::

    cd /Users/samib/stable-envs-step9-render-2d
    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/render-2d/oracle.py

GPU arm (inside the SLURM job, see ``gpu_oracle.sbatch``)::

    python scratch/step9/render-2d/oracle.py --devices cpu,cuda
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE_SHA = "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]

#: config -> frozen corpus, relative to the tree. A config absent from here has never had one and is
#: rendered from its own ``sample_states`` at seed 0 instead, which both arms do identically.
CORPORA: dict[str, str] = {
    n: f"examples/results/step4/states/{n}.npz" for n in (
        "PushT", "SimplePointMaze",
        "CartPole", "Acrobot", "MountainCar", "MountainCarContinuous", "Pendulum",
        "TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")
}
#: CraftaxClassic's Step-4 corpus is 127 wide and the cell has been 149 wide since Step 7 widened it,
#: so the Step-7 corpus is the frozen one that still loads. Both arms are handed the same file.
CORPORA["CraftaxClassic"] = "examples/results/step7/craftax-classic-full/corpus/states.npz"
CORPORA["CraftaxFull"] = "examples/results/step7/craftax-full/corpus/states.npz"

#: Run inside ONE tree, printing a JSON record on stdout. A string rather than an import because the
#: two trees define the same module names and cannot both be imported into one interpreter.
PROBE = r'''
import hashlib, json, sys, traceback
import numpy as np, torch

# The render modules are imported and INSTRUMENTED BEFORE ``stable_envs`` is, and that ordering is
# load-bearing rather than tidy. Some consumers bind a backend function at module import time
# (``env.render`` does ``from .render.raster2d import render as render_scene``), and a wrapper
# installed afterwards would never be seen by them -- in one arm only, if the two trees happen to
# bind at different moments, which is exactly the false difference this is here to avoid. The render
# package imports nothing from the core, so importing it first is safe.
from stable_envs.render import (gfxdraw as GD, gfxdraw_classic as GC, gfxdraw_fused as GF,
                                navdot as NV, prng as PR, raster2d as R2, scene as SC,
                                tilemap as TM)

corpora = json.loads(sys.argv[1])
devices = sys.argv[2].split(",")
nstates = int(sys.argv[3])
only = json.loads(sys.argv[4]) or None

# ---- the instrumentation that decides which cells are in ---------------------------------------
#: ``(module, canonical tag, names)``. One tag may cover several attribute names, because a function
#: that gained a public spelling in this branch still has to count as the SAME backend entry point as
#: the private one the baseline had -- otherwise the rename itself would read as a behaviour change,
#: and a consumer that reaches for the old name would read as a cell that stopped using the backend.
HITS = set()
#:
#: ``raster2d.render`` and ``scene.SceneBuilder`` are deliberately NOT wrapped. Both are bound by
#: name at module import time in a consumer -- the Step-8 ``core/env.py`` does ``from
#: ..render.raster2d import render as render_scene`` at module scope, and four envs do ``from
#: ..render.scene import SceneBuilder`` -- and a wrapper installed after that import cannot be seen,
#: whereas Wave 0 moved the ``render`` import inside the method, where it can. Instrumenting them
#: would therefore report a difference between the two TREES' import style rather than between their
#: pictures. Nothing is lost: ``render`` only dispatches to ``render_torch``/``render_triton``, which
#: it reaches through its own module globals at call time, and a scene that was not built cannot be
#: rasterized.
WRAPPED = [(R2, "render_torch", ["render_torch"]),
           (R2, "render_triton", ["render_triton"]), (R2, "_static_base", ["_static_base"]),
           (R2, "row_groups", ["row_groups", "_row_groups"]),
           (GD, "polygon", ["polygon"]), (GD, "circle", ["circle"]), (GD, "aaline", ["aaline"]),
           (GD, "blend", ["blend"]),
           (GC, "render_classic", ["render_classic"]),
           (GC, "render_classic_torch", ["render_classic_torch"]),
           (GF, "render_surface", ["render_surface"]), (GF, "edge_table", ["edge_table"]),
           (GF, "shrink", ["shrink"]),
           (TM, "blit", ["blit"]), (TM, "blit_torch", ["blit_torch"]),
           (TM, "recolour", ["recolour"]), (TM, "visibility_scan", ["visibility_scan"]),
           (TM, "visibility_scan_torch", ["visibility_scan_torch"]),
           (NV, "blend_dots", ["blend_dots"]), (NV, "dot_window", ["dot_window"]),
           (PR, "threefry2x32", ["threefry2x32"]), (PR, "uniform", ["uniform"]),
           (PR, "random_bits", ["random_bits"])]
for mod, canon, names in WRAPPED:
    present = [n for n in names if getattr(mod, n, None) is not None]
    if not present:
        continue
    orig = getattr(mod, present[0])
    def make(orig=orig, tag="%s.%s" % (mod.__name__.rsplit(".", 1)[-1], canon)):
        def f(*a, **k):
            HITS.add(tag)
            return orig(*a, **k)
        return f
    wrapper = make()
    for n in present:                 # every spelling of the one function points at one wrapper
        assert getattr(mod, n) is orig, "%s.%s is not the same object as %s" % (mod, n, present[0])
        setattr(mod, n, wrapper)

import stable_envs as se


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def states_for(env, corpus, n):
    if corpus is None:
        return np.asarray(env.sample_states(n, np.random.default_rng(0)), np.float32)
    return np.load(corpus)["states"].astype(np.float32)[:n]


def frames(env, s):
    env.set_state(s)
    return env.render().detach().cpu().numpy()


# ---- the non-fused variants, as (label, applies?, apply, undo) ----------------------------------
def _applies(tag_prefixes, hits):
    return any(t.startswith(p) for t in hits for p in tag_prefixes)


def variants_for(hits, dev):
    """Which second renders this cell's call set makes meaningful. cuda only: on cpu they ARE it."""
    out = []
    if dev != "cuda":
        return out
    if _applies(("raster2d.",), hits):
        out += ["backend_torch", "cull_off", "cull_on"]
    if _applies(("gfxdraw_classic.",), hits):
        out += ["gfx_per_primitive"]
    if _applies(("tilemap.",), hits):
        out += ["tilemap_torch"]
    if _applies(("navdot.blend_dots",), hits):
        out += ["navdot_torch"]
    return out


def render_variant(cls, label, dev, s):
    """Build the cell again with one fast path disabled and render the same states through it."""
    kw = dict(batch=len(s), device=dev)
    saved = {}
    try:
        if label == "backend_torch":
            kw["backend"] = "torch"
        if label == "tilemap_torch":
            saved[(TM, "HAVE_TRITON")] = TM.HAVE_TRITON
            TM.HAVE_TRITON = False
        if label == "navdot_torch":
            saved[(NV, "HAVE_TRITON")] = NV.HAVE_TRITON
            NV.HAVE_TRITON = False
        if label in ("cull_off", "cull_on"):
            saved[(R2, "_CULL_MIN_PRIMS")] = R2._CULL_MIN_PRIMS
            R2._CULL_MIN_PRIMS = 1 << 30 if label == "cull_off" else 0
        env = cls(**kw)
        if label == "gfx_per_primitive":
            env.use_fused_render = False
        return h(frames(env, s))
    finally:
        for (mod, attr), v in saved.items():
            setattr(mod, attr, v)


# ---- the run ------------------------------------------------------------------------------------
names = sorted(se.REGISTRY)
out = {"_registry": names, "_cells": {}}
for name in names:
    if only is not None and name not in only:
        continue
    cls = se.REGISTRY[name]
    corpus = corpora.get(name)
    rec = {}
    for dev in devices:
        key = "device_%s" % dev
        if dev == "cuda" and not torch.cuda.is_available():
            rec[key] = {"status": "skipped", "reason": "no cuda device"}
            continue
        HITS.clear()
        try:
            env = cls(batch=2, device=dev)         # cheap probe: does it build, and what does it use
            s2 = states_for(env, corpus, 2)
            frames(env, s2)
        except Exception as e:
            rec[key] = {"status": "skipped", "reason": "%s: %s" % (type(e).__name__, str(e)[:160])}
            continue
        hits = sorted(HITS)
        if not hits:
            rec[key] = {"status": "not_a_2d_cell"}
            continue
        try:
            HITS.clear()
            env = cls(batch=nstates, device=dev)
            s = states_for(env, corpus, nstates)
            if len(s) != nstates:
                env = cls(batch=len(s), device=dev)
            f = frames(env, s)
            r = {"status": "ok", "states_sha256": h(s), "n_states": int(len(s)),
                 "size": int(env.size), "frames_shape": list(f.shape), "frames_sha256": h(f),
                 "frames_mean": round(float(f.mean()), 9), "backends": sorted(HITS)}
            # the frames must not depend on how the rows arrive
            r["frames_from_rows_sha256"] = h(
                env.render(torch.as_tensor(s, device=env.device)).detach().cpu().numpy())
            vs = {}
            for label in variants_for(hits, dev):
                try:
                    vs[label] = render_variant(cls, label, dev, s)
                except Exception as e:
                    vs[label] = "error: %s: %s" % (type(e).__name__, str(e)[:120])
            if vs:
                r["variants"] = vs
            rec[key] = r
            try:
                env.close()
            except Exception:
                pass
        except Exception as e:
            rec[key] = {"status": "error", "reason": "%s: %s" % (type(e).__name__, str(e)[:200]),
                        "traceback": traceback.format_exc()[-800:]}
    if any(v.get("status") != "not_a_2d_cell" for v in rec.values()):
        out["_cells"][name] = rec
print("@@JSON@@" + json.dumps(out))
'''


def _run_probe(tree: Path, python: str, devices: str, nstates: int, only) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "glfw")
    corpora = {k: str(tree / v) for k, v in CORPORA.items() if (tree / v).exists()}
    r = subprocess.run([python, "-c", PROBE, json.dumps(corpora), devices, str(nstates),
                        json.dumps(only)], capture_output=True, text=True, cwd=str(tree), env=env)
    hit = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    if not hit:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    return json.loads(hit[-1][len("@@JSON@@"):])


def _baseline_worktree(sha: str) -> tuple[Path, bool]:
    """A detached worktree of ``sha``, so the comparison is against the tree and not a memory of it."""
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True).stdout
    for block in listing.split("\n\n"):
        if "step9-render2d-baseline" in block:
            return Path(block.splitlines()[0].split(" ", 1)[1]), False
    path = Path(tempfile.gettempdir()) / f"step9-render2d-baseline-{sha[:12]}"
    if path.exists():
        return path, False
    subprocess.run(["git", "worktree", "add", "--detach", str(path), sha], cwd=str(ROOT),
                   check=True, capture_output=True)
    return path, True


#: The keys that must agree, and what a difference in each one would mean.
COMPARED = {
    "status": "whether the cell renders at all on this device",
    "reason": "why a cell was skipped, when it was",
    "states_sha256": "the states the two arms were handed",
    "n_states": "how many states were rendered",
    "size": "the registered default frame size",
    "frames_shape": "the frame shape",
    "frames_sha256": "THE FRAMES, byte for byte",
    "frames_from_rows_sha256": "the frames rendered from explicit rows",
    "backends": "which 2-D backend entry points the cell actually called",
    "variants": "the non-fused expression behind each fast path",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--devices", default="cpu", help="comma separated: cpu, cuda")
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--only", default=None, help="comma-separated configs, for a quick check")
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/render-2d"))
    ap.add_argument("--tag", default=None, help="suffix for the record file (e.g. the host)")
    ap.add_argument("--arms", default="baseline,head",
                    help="which TREE each arm runs: baseline|head for each. ``baseline,baseline`` "
                         "runs the SAME tree twice in the same process shape, which is how a cell "
                         "that is not reproducible across processes is told apart from one this "
                         "branch changed.")
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None

    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        arms = a.arms.split(",")
        if len(arms) != 2 or any(x not in ("baseline", "head") for x in arms):
            raise SystemExit("--arms takes exactly two of baseline|head")
        trees = {"baseline": baseline, "head": ROOT}
        before = _run_probe(trees[arms[0]], a.python, a.devices, a.n_states, only)
        after = _run_probe(trees[arms[1]], a.python, a.devices, a.n_states, only)

        diffs, cells = {}, {}
        # The DISCOVERED cell list is itself part of the comparison: a cell that reaches a 2-D
        # backend in one arm and not in the other is a difference, not a cell to quietly drop.
        if sorted(before["_registry"]) != sorted(after["_registry"]):
            diffs["_registry"] = {"only_in_baseline": sorted(set(before["_registry"])
                                                             - set(after["_registry"])),
                                  "only_in_head": sorted(set(after["_registry"])
                                                         - set(before["_registry"])),
                                  "means": "the two arms did not even register the same configs"}
        names = sorted(set(before["_cells"]) | set(after["_cells"]))
        for name in names:
            b_all, m_all = before["_cells"].get(name, {}), after["_cells"].get(name, {})
            if (name in before["_cells"]) != (name in after["_cells"]):
                diffs[name] = {"discovery": {"baseline": name in before["_cells"],
                                             "head": name in after["_cells"],
                                             "means": "one arm found a 2-D cell the other did not"}}
                continue
            for dev in a.devices.split(","):
                key = f"device_{dev}"
                b, m = b_all.get(key, {}), m_all.get(key, {})
                d = {k: {"baseline": b.get(k), "head": m.get(k), "means": why}
                     for k, why in COMPARED.items() if b.get(k) != m.get(k)}
                if d:
                    diffs[f"{name}/{dev}"] = d
                cells[f"{name}/{dev}"] = m.get("status", "missing")
        record = {
            "arms": arms,
            "baseline_sha": a.baseline_sha,
            "head_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                       capture_output=True, text=True).stdout.strip(),
            "python": a.python,
            "devices": a.devices.split(","),
            "n_states_requested": a.n_states,
            "host": os.uname().nodename,
            "cells": cells,
            "compared": sorted(COMPARED),
            "unexplained_differences": diffs,
            "identical": not diffs,
            "baseline": before,
            "head": after,
        }
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        fn = f"oracle{'_' + a.tag if a.tag else ''}.json"
        (out / fn).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        ran = sum(1 for v in cells.values() if v == "ok")
        print(json.dumps({"arms": arms, "baseline_sha": record["baseline_sha"],
                          "head_sha": record["head_sha"], "identical": record["identical"],
                          "cells_ok": ran, "cells": cells,
                          "unexplained_differences": diffs}, indent=1))
        print(f"wrote {out / fn}")
        print("ORACLE:", "PASS" if record["identical"] and ran else "FAIL")
        return 0 if (record["identical"] and ran) else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
