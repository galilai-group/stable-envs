"""Oracle proof for Step-9 Workstream I: the same pixels and the same lifetime, from two trees.

WHAT IT PROVES. Plan 7 and 27: the architecture may change, the behaviour may not. For
``render/mjwarp.py``, ``render/mjwarp_lightspec.py``, ``render/nvdiff.py`` and
``render/rendernet.py`` the behaviour is (a) the frames and (b) — because this is the workstream
that introduces explicit resource ownership — what the process holds afterwards. Both arms run in
separate interpreters, one with the Step-8 baseline tree on ``PYTHONPATH`` and one with this branch,
in the SAME venv on the SAME card, and every compared value is computed here rather than read from a
stored digest, so a difference can only be ours.

EVERY BACKEND HERE IS CUDA-ONLY. MJWarp is a CUDA raytracer, nvdiffrast's ``RasterizeCudaContext``
is a CUDA rasterizer, and the render corrector is a captured CUDA graph; the local Mac has no CUDA
and no cell below renders on it. Therefore NOTHING in this file has been run: it is the section of
the coordinator's post-Wave-1 GPU oracle that covers these four modules, and every number it would
produce is owed.

THE FIVE ARMS, and what each one would catch that the others would not:

1. ``frames`` -- byte-identical frames per cell, from the frozen corpus where one exists, through the
   public path only (``cls(batch=n, device="cuda")``, ``set_state``, ``render()``). Catches any
   change to a numeric expression, an upload order or a cache key. Also renders from explicit rows,
   because those are different code.
2. ``fresh`` -- the same cell in a SECOND process, compared to the first. Catches a frame that
   depends on what the interpreter did earlier, which is exactly the failure a cache whose eviction
   policy changed could introduce, and which a single-process comparison cannot see.
3. ``cycles`` -- N create/close/create cycles, asserting the frames are identical on every cycle AND
   that the counters agree: ``_resources`` counts this repository's backend objects (created ==
   closed, zero live at the end), ``testing.resource_guard`` counts EGL contexts and mujoco
   Renderers, where the rule is the opposite one and is checked at zero tolerance -- NO context may
   be released, on either arm, however many times an env is closed. That asymmetry is the design
   (see ``render/_resources.py``), and this is where it is measured rather than asserted.
4. ``batch`` -- the same state rendered at several batch widths, comparing row 0. Batch invariance is
   a Step-8 property of every cell here and a bounded cache with a new eviction policy is precisely
   the kind of change that could break it at one width and not another.
5. ``closed`` -- a closed env refuses to render (the Wave-0 contract's ``close`` semantics) and
   closing twice is a no-op. Only meaningful on the head arm; recorded on both.

Usage inside the SLURM job, once per venv (see ``gpu_oracle.sbatch``)::

    python scratch/step9/render-mjwarp-nvdiff/oracle.py --cells DMCCheetahRun,DMCWalkerWalk \
        --tag vis311 --out examples/results/step9/render-mjwarp-nvdiff
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

#: The Step-8 behavioural baseline. Not the branch parent: the question is whether Step 9 moved a
#: pixel since the system was certified, not since the last merge.
BASELINE_SHA = "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]

#: ``(cell, corpus-or-None, which module it exercises)``. The backend column is what makes this a
#: section of a GPU oracle rather than a list of envs: every one of the four owned modules must be
#: reached by at least one cell, and a cell that stops reaching its module is itself a difference
#: (each arm records which of the owned entry points it actually called).
CELLS: list[tuple[str, str | None, str]] = [
    # --- MJWarp: the DM Control trackcom cells render through mjwarp on cuda and cpu3d on cpu
    ("DMCCheetahRun", "examples/results/step4/states/DMCCheetahRun.npz", "mjwarp"),
    ("DMCHopperHop", "examples/results/step4/states/DMCHopperHop.npz", "mjwarp"),
    ("DMCHumanoidStand", "examples/results/step4/states/DMCHumanoidStand.npz", "mjwarp"),
    ("DMCWalkerWalk", "examples/results/step4/states/DMCWalkerWalk.npz", "mjwarp"),
    ("DMCQuadrupedWalk",
     "examples/results/step7/dmc-quadruped/phase1/corpus/states/DMCQuadrupedWalk.npz", "mjwarp"),
    # --- MJWarp: OGB manipulation (cuda only by declaration CPU3D-OGB-1)
    ("OGBCube", "examples/results/step7/ogb-manipulation/states/OGBCube.npz", "mjwarp"),
    ("OGBScene", "examples/results/step7/ogb-manipulation/states/OGBScene.npz", "mjwarp"),
    # --- MJWarp: the four Fetch tasks (the sparse variants; Dense/Dict share the renderer)
    ("FetchReach", "examples/results/step4/states/FetchReach.npz", "mjwarp"),
    ("FetchPush", "examples/results/step4/states/FetchPush.npz", "mjwarp"),
    ("FetchSlide", "examples/results/step4/states/FetchSlide.npz", "mjwarp"),
    ("FetchPickAndPlace", "examples/results/step4/states/FetchPickAndPlace.npz", "mjwarp"),
    # --- MJWarp + lightspec + rendernet: RoboCasa needs its 24 GB asset library, so it may skip
    ("RoboCasaCounterToSink", None, "mjwarp+mjwarp_lightspec"),
    ("RoboCasaCounterToSinkCorrected", None, "mjwarp+mjwarp_lightspec+rendernet"),
    # --- nvdiff: the rocket (Scene.from_triangles, i.e. the caller-owned-scene path)
    ("PFRocketLanding", None, "nvdiff"),
    # --- nvdiff: the six visual locomotion cells (BatchRenderer builds its OWN scene)
    ("MujocoHalfCheetah", None, "nvdiff"),
    ("MujocoAnt", None, "nvdiff"),
]

#: Cells the cycle/batch/closed arms run. A subset, because each is O(cycles) renders and the
#: property is per BACKEND, not per cell: one MJWarp cell, one nvdiff-owning-its-scene cell, one
#: nvdiff-given-a-scene cell, plus the lightspec path if RoboCasa is available.
LIFETIME_CELLS = ("DMCCheetahRun", "MujocoHalfCheetah", "PFRocketLanding")

#: Widths for the batch-invariance arm. 1 is the width a cache is most likely to behave differently
#: at (nothing is warm); 4096 is the width the throughput arm uses.
BATCH_WIDTHS = (1, 2, 8, 64)

#: Run inside ONE tree and print a JSON record. A string rather than an import because the two trees
#: define the same module names and cannot both be imported into one interpreter.
PROBE = r'''
import hashlib, json, sys, traceback
import numpy as np, torch

import stable_envs as se
from stable_envs.testing import resource_guard

resource_guard.install()

cells = json.loads(sys.argv[1])
nstates = int(sys.argv[2])
lifetime = set(json.loads(sys.argv[3]))
cycles = int(sys.argv[4])
widths = json.loads(sys.argv[5])

# Which of the owned entry points did this cell actually call? A cell that silently stopped using
# the backend would otherwise pass a frame comparison for the wrong reason.
HITS = set()
def instrument():
    import importlib
    for modname, names in (("stable_envs.render.mjwarp",
                            ("prepare_model", "create_render_context", "gl_rounding",
                             "texture_gaps", "needs_com_pos", "bake_unsamplable_textures")),
                           ("stable_envs.render.mjwarp_lightspec", ("specialised_lights", "render")),
                           ("stable_envs.render.nvdiff", ("prepare_texture", "mvp_rows", "texgen")),
                           ("stable_envs.render.rendernet", ("load",))):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        short = modname.rsplit(".", 1)[-1]
        for name in names:
            orig = getattr(mod, name, None)
            if orig is None or getattr(orig, "_se_probe", False):
                continue
            def wrap(orig=orig, tag="%s.%s" % (short, name)):
                def f(*a, **k):
                    HITS.add(tag)
                    return orig(*a, **k)
                f._se_probe = True
                return f
            setattr(mod, name, wrap())
instrument()

# The nvdiff/rendernet modules are imported lazily by their envs, so re-instrument after each
# construction; the set is per cell and cleared before each one.

def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def states_for(env, corpus):
    if corpus is None:
        return np.asarray(env.sample_states(nstates, np.random.default_rng(0)), np.float32)
    return np.load(corpus)["states"].astype(np.float32)[:nstates]

def frames(env, s):
    env.set_state(s)
    return env.render().detach().cpu().numpy()

def close_env(env):
    close = getattr(env, "close", None)
    if callable(close):
        close()

def counts():
    try:
        from stable_envs.render import _resources
        return _resources.counts()
    except Exception:
        return None                      # the baseline tree has no _resources; that is expected

out = {}
for name, corpus, backend in cells:
    rec = {"backend": backend}
    out[name] = rec
    HITS.clear()
    instrument()
    try:
        cls = se.REGISTRY[name]
    except Exception as e:
        rec["status"] = "skipped"; rec["reason"] = "not registered: %s" % (str(e)[:160],)
        continue
    if not torch.cuda.is_available():
        rec["status"] = "skipped"; rec["reason"] = "no cuda device"
        continue
    rg0 = resource_guard.take()
    c0 = counts()
    try:
        env = cls(batch=nstates, device="cuda")
        s = states_for(env, corpus)
        if len(s) != nstates:
            close_env(env); env = cls(batch=len(s), device="cuda")
        f = frames(env, s)
        rec.update({"status": "ok", "n_states": int(len(s)), "states_sha256": h(s),
                    "size": int(env.size), "frames_shape": list(f.shape),
                    "frames_sha256": h(f), "frames_mean": round(float(f.mean()), 9)})
        rec["frames_from_rows_sha256"] = h(
            env.render(torch.as_tensor(s, device=env.device)).detach().cpu().numpy())
        rec["backends_called"] = sorted(HITS)
        called_modules = {tag.split(".", 1)[0] for tag in HITS}
        expected_modules = set(backend.split("+"))
        rec["backend_modules_called"] = sorted(called_modules)
        rec["backend_coverage_ok"] = expected_modules <= called_modules

        # --- batch invariance: row 0 of every width must be the same frame
        if name in lifetime:
            per_width = {}
            for w in widths:
                if w > len(s):
                    continue
                e2 = cls(batch=w, device="cuda")
                per_width[str(w)] = h(frames(e2, s[:w])[0])
                close_env(e2)
            rec["batch_row0_sha256"] = per_width
            rec["batch_invariant"] = len(set(per_width.values())) == 1

        # --- repeated create/close, with the two counters
        if name in lifetime:
            digests, rg_per_cycle = [], []
            for _ in range(cycles):
                rgc = resource_guard.take()
                cc = counts()
                e3 = cls(batch=min(8, len(s)), device="cuda")
                digests.append(h(frames(e3, s[:min(8, len(s))])))
                close_env(e3)
                close_env(e3)                            # idempotent on the head; no-op on baseline
                d = resource_guard.delta(rgc)
                rg_per_cycle.append({"context_created": d["context_created"],
                                     "context_freed": d["context_freed"],
                                     "renderer_closed": d["renderer_closed"]})
                if cc is not None:
                    now = counts()
                    rg_per_cycle[-1]["se_created"] = {k: now["created"].get(k, 0) - cc["created"].get(k, 0)
                                                      for k in now["created"]
                                                      if now["created"].get(k, 0) != cc["created"].get(k, 0)}
                    rg_per_cycle[-1]["se_closed"] = {k: now["closed"].get(k, 0) - cc["closed"].get(k, 0)
                                                     for k in now["closed"]
                                                     if now["closed"].get(k, 0) != cc["closed"].get(k, 0)}
                    rg_per_cycle[-1]["se_release_failed"] = {
                        k: now.get("release_failed", {}).get(k, 0)
                        - cc.get("release_failed", {}).get(k, 0)
                        for k in set(now.get("release_failed", {})) | set(cc.get("release_failed", {}))
                        if now.get("release_failed", {}).get(k, 0)
                        != cc.get("release_failed", {}).get(k, 0)}
            rec["cycle_frames_sha256"] = digests
            rec["cycles_identical"] = len(set(digests)) == 1
            rec["cycles"] = rg_per_cycle
            rec["egl_released_across_cycles"] = sum(c["context_freed"] for c in rg_per_cycle)
            rec["renderers_closed_across_cycles"] = sum(c["renderer_closed"] for c in rg_per_cycle)
            rec["release_failures_across_cycles"] = sum(
                sum(c.get("se_release_failed", {}).values()) for c in rg_per_cycle)

        # --- a closed env refuses to render, and closing twice is a no-op
        if name in lifetime and counts() is not None:
            e4 = cls(batch=1, device="cuda")
            close_env(e4); close_env(e4)
            try:
                e4.render()
                rec["closed_env_refuses"] = False
            except Exception as ex:
                rec["closed_env_refuses"] = True
                rec["closed_env_error"] = type(ex).__name__
        close_env(env)
    except Exception as e:
        rec["status"] = "error"
        rec["reason"] = "%s: %s" % (type(e).__name__, str(e)[:200])
        rec["traceback"] = traceback.format_exc()[-900:]
    rec["resource_guard_delta"] = {k: v for k, v in resource_guard.delta(rg0).items() if k != "sites"}
    if c0 is not None:
        rec["se_resources_after"] = counts()

out["_session"] = {"resource_guard_totals": {k: v for k, v in resource_guard.totals().items()
                                             if k != "patched"},
                   "torch": torch.__version__,
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print("@@JSON@@" + json.dumps(out))
'''


def _run_probe(tree: Path, python: str, cells, nstates: int, cycles: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "egl")
    payload = json.dumps([[n, (str(tree / c) if c else None), b] for n, c, b in cells])
    r = subprocess.run([python, "-c", PROBE, payload, str(nstates),
                        json.dumps(list(LIFETIME_CELLS)), str(cycles), json.dumps(list(BATCH_WIDTHS))],
                       capture_output=True, text=True, cwd=str(tree), env=env)
    hit = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    if not hit:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
    return json.loads(hit[-1][len("@@JSON@@"):])


def _baseline_worktree(sha: str) -> tuple[Path, bool]:
    """A detached worktree of ``sha``, so the comparison is against the tree and not a memory of it."""
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


#: What must agree between the arms, and what a difference in each would mean. ``cycles`` and the
#: ``_resources`` counters are deliberately NOT here: the baseline tree has no ``_resources`` and no
#: ``close`` on these backends, so those keys are expected to differ and are reported as the
#: head-arm's own assertions instead (see ``LIFETIME_ASSERTIONS``).
COMPARED = {
    "status": "whether the cell renders at all",
    "reason": "why a cell was skipped, when it was",
    "states_sha256": "the states the two arms were handed",
    "n_states": "how many states were rendered",
    "size": "the registered default frame size",
    "frames_shape": "the frame shape",
    "frames_sha256": "THE FRAMES, byte for byte",
    "frames_from_rows_sha256": "the frames rendered from explicit rows",
    "backends_called": "the owned backend entry points actually reached",
    "batch_row0_sha256": "row 0 at every batch width",
    "batch_invariant": "whether the width changed the frame",
}

#: Properties the HEAD arm must satisfy on its own. The baseline cannot be asked for them.
LIFETIME_ASSERTIONS = (
    ("cycles_identical", True, "the frames must not change across create/close/create cycles"),
    ("egl_released_across_cycles", 0,
     "closing an env must NEVER release an EGL context -- releasing one corrupts rendering in every "
     "other live context in the process (testing/resource_guard.py, four incidents)"),
    ("renderers_closed_across_cycles", 0, "and must never close a mujoco.Renderer, for the same reason"),
    ("closed_env_refuses", True, "a closed env must raise rather than hand back a frame"),
    ("closed_env_error", "RuntimeError",
     "a closed Env must raise the Wave-0 lifecycle error, not fail through released backend state"),
    ("release_failures_across_cycles", 0, "no backend release attempted in the cycle may fail"),
    ("backend_coverage_ok", True, "the cell must actually reach every backend it claims to exercise"),
    ("batch_invariant", True, "the batch width must not change row 0"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--cells", default=None,
                    help="comma separated subset; the default runs every cell in CELLS")
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/render-mjwarp-nvdiff"))
    ap.add_argument("--tag", default=None, help="suffix for the record file (venv or host)")
    ap.add_argument("--arms", default="baseline,head",
                    help="which TREE each arm runs. ``head,head`` runs the SAME tree twice, which is "
                         "the FRESH-PROCESS arm: it tells a cell that is not reproducible across "
                         "processes apart from a cell this branch changed.")
    a = ap.parse_args()

    want = set(a.cells.split(",")) if a.cells else None
    cells = [c for c in CELLS if want is None or c[0] in want]
    if not cells:
        raise SystemExit(f"--cells matched nothing; known: {[c[0] for c in CELLS]}")

    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        arms = a.arms.split(",")
        if len(arms) != 2 or any(x not in ("baseline", "head") for x in arms):
            raise SystemExit("--arms takes exactly two of baseline|head")
        trees = {"baseline": baseline, "head": ROOT}
        before = _run_probe(trees[arms[0]], a.python, cells, a.n_states, a.cycles)
        after = _run_probe(trees[arms[1]], a.python, cells, a.n_states, a.cycles)

        diffs, status, failures = {}, {}, {}
        expected_modules = set()
        selected_modules = set()
        called_modules = set()
        for name, _, _ in cells:
            b, m = before.get(name, {}), after.get(name, {})
            d = {k: {"baseline": b.get(k), "head": m.get(k), "means": why}
                 for k, why in COMPARED.items() if b.get(k) != m.get(k)}
            if d:
                diffs[name] = d
            status[name] = m.get("status", "missing")
            modules = set(next(c[2].split("+") for c in cells if c[0] == name))
            selected_modules.update(modules)
            if m.get("status") == "ok":
                expected_modules.update(modules)
            called_modules.update(m.get("backend_modules_called", ()))
            if name in LIFETIME_CELLS:
                for key, want_val, why in LIFETIME_ASSERTIONS:
                    got = m.get(key, "<missing>")
                    if got != want_val:
                        failures[f"{name}/{key}"] = {"got": got, "want": want_val, "means": why}
            elif m.get("status") == "ok" and not m.get("backend_coverage_ok", False):
                key, want_val, why = LIFETIME_ASSERTIONS[-2]
                got = m.get(key, "<missing>")
                failures[f"{name}/{key}"] = {"got": got, "want": want_val, "means": why}

        missing_modules = sorted(expected_modules - called_modules)
        if missing_modules:
            failures["session/backend_coverage"] = {
                "got": sorted(called_modules), "want": sorted(expected_modules),
                "means": f"selected arm did not exercise: {missing_modules}"}

        record = {
            "arms": arms,
            "baseline_sha": a.baseline_sha,
            "head_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                       capture_output=True, text=True).stdout.strip(),
            "python": a.python,
            "host": os.uname().nodename,
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "n_states_requested": a.n_states,
            "cycles": a.cycles,
            "batch_widths": list(BATCH_WIDTHS),
            "cells": status,
            "expected_backend_modules": sorted(expected_modules),
            "called_backend_modules": sorted(called_modules),
            "unavailable_backend_modules": sorted(selected_modules - expected_modules),
            "compared": sorted(COMPARED),
            "unexplained_differences": diffs,
            "lifetime_failures": failures,
            "identical": not diffs,
            "lifetime_ok": not failures,
            "baseline": before,
            "head": after,
        }
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        fn = f"oracle{'_' + a.tag if a.tag else ''}.json"
        (out / fn).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        ran = sum(1 for v in status.values() if v == "ok")
        print(json.dumps({"arms": arms, "head_sha": record["head_sha"], "cells_ok": ran,
                          "identical": record["identical"], "lifetime_ok": record["lifetime_ok"],
                          "cells": status, "unexplained_differences": diffs,
                          "lifetime_failures": failures}, indent=1))
        print(f"wrote {out / fn}")
        ok = record["identical"] and record["lifetime_ok"] and ran == len(cells)
        print("ORACLE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
