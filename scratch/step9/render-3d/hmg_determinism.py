"""Is ``HumanoidmazeGiantNavigate`` on CUDA reproducible from one process to the next?

WHY THIS EXISTS. The Step-9 render-3d GPU oracle (job 6464319) found 58 of 59 cell/device pairs
byte-identical between the Step-8 baseline tree and this branch, and ONE that was not:
``HumanoidmazeGiantNavigate`` on **cuda** only -- its cpu arm and every other maze on both devices
matched. The patch contains no arithmetic change at all (the moved functions are the same
expressions, re-imported under their old names), so before anything is attributed to it, the prior
question has to be answered: **does that cell give the same frames twice in a row from the SAME
tree?**

This is the cell the module already documents as the one with the long lever arm -- see
``raycast3d._BOUND_SLACK_NOISE``: the camera is ~100 m from a 5 cm capsule, ``c = olx^2 + oly^2 -
r^2`` loses the radius entirely in float32 there, and ``disc > 0`` is decided on rounding noise.
Job 6348207 moved one pixel of 3,211,264 by 22/255 on this exact configuration for exactly that
reason, and left the other twenty configurations alone.

WHAT IT RUNS, all inside one SLURM job on one card:

  * three FRESH processes per tree (baseline, head), each rendering the frozen 64-state corpus and
    printing the frame digest -- so "baseline vs baseline" and "head vs head" are measured, not
    assumed;
  * within one process, the same render twice, to separate "not reproducible across processes" from
    "not reproducible at all";
  * the same render with ``FORCE_BACKEND="torch"`` and with ``PRIMITIVE_BOUNDS=False``, which says
    whether any disagreement lives in the fused Triton cast or in the shared torch expression;
  * a pixel-level diff between one baseline frame set and one head frame set: how many pixels, by
    how much, in which frames.

HOW TO READ THE RESULT.

  * baseline digests disagree among themselves -> the cell is not process-reproducible on this card
    and the oracle difference is NOT evidence about the patch. It is a pre-existing property to be
    recorded (and, if Sami wants, chased separately).
  * baseline digests agree, head digests agree, and the two differ -> the patch changed this cell
    and the merge is blocked until it is explained.

Usage (inside the job)::

    python scratch/step9/render-3d/hmg_determinism.py --devices cuda
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
CELL = "HumanoidmazeGiantNavigate"
CORPUS = "examples/results/step4/states/HumanoidmazeGiantNavigate.npz"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]

PROBE = r'''
import hashlib, json, sys
import numpy as np, torch
import stable_envs as se
from stable_envs.render import raycast3d as R3

cell, corpus, device, dump = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = np.load(corpus)["states"].astype(np.float32)
env = se.REGISTRY[cell](batch=len(s), device=device)
env.set_state(s)


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def frames():
    return env.render().detach().cpu().numpy()


out = {"shipped": h(frames()), "shipped_again": h(frames())}
for label, attr, value in (("force_torch", "FORCE_BACKEND", "torch"),
                           ("primitive_bounds_off", "PRIMITIVE_BOUNDS", False),
                           ("composite_window", "COMPOSITE_COMPACT_MERGE", False)):
    saved = getattr(R3, attr)
    setattr(R3, attr, value)
    try:
        out[label] = h(frames())
    finally:
        setattr(R3, attr, saved)
if dump:
    np.save(dump, frames())
print("@@JSON@@" + json.dumps(out))
'''


def _worktree(sha: str) -> tuple[Path, bool]:
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True).stdout
    for block in listing.split("\n\n"):
        if "step9-render3d-baseline" in block:
            return Path(block.splitlines()[0].split(" ", 1)[1]), False
    path = Path(tempfile.gettempdir()) / f"step9-render3d-baseline-{sha[:12]}"
    if path.exists():
        return path, False
    subprocess.run(["git", "worktree", "add", "--detach", str(path), sha], cwd=str(ROOT),
                   check=True, capture_output=True)
    return path, True


def _run(tree: Path, python: str, device: str, dump: str = "", cell: str = CELL,
         corpus: str = CORPUS) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "egl")
    r = subprocess.run([python, "-c", PROBE, cell, str(tree / corpus), device, dump],
                       capture_output=True, text=True, cwd=str(tree), env=env)
    hit = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")]
    if not hit:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return json.loads(hit[-1][len("@@JSON@@"):])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--devices", default="cuda")
    ap.add_argument("--cell", default=CELL, help="another cell, for smoke-testing this script")
    ap.add_argument("--corpus", default=CORPUS, help="its corpus, relative to the tree root")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/render-3d"))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    import numpy as np
    baseline, created = _worktree(BASELINE_SHA)
    try:
        record = {"cell": a.cell, "baseline_sha": BASELINE_SHA, "host": os.uname().nodename,
                  "head_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                             capture_output=True, text=True).stdout.strip(),
                  "devices": {}}
        tmp = Path(tempfile.mkdtemp())
        for device in a.devices.split(","):
            runs = {"baseline": [], "head": []}
            for i in range(a.repeats):
                for label, tree in (("baseline", baseline), ("head", ROOT)):
                    dump = str(tmp / f"{label}_{device}.npy") if i == 0 else ""
                    runs[label].append(_run(tree, a.python, device, dump, a.cell, a.corpus))
            dev = {"runs": runs}
            for label in ("baseline", "head"):
                digests = {r["shipped"] for r in runs[label]}
                dev[f"{label}_self_consistent"] = len(digests) == 1
                dev[f"{label}_digests"] = sorted(digests)
            dev["baseline_equals_head"] = (runs["baseline"][0]["shipped"]
                                           == runs["head"][0]["shipped"])
            b = np.load(tmp / f"baseline_{device}.npy").astype(np.int16)
            h = np.load(tmp / f"head_{device}.npy").astype(np.int16)
            diff = np.abs(b - h)
            dev["pixels_total"] = int(diff.size)
            dev["pixels_differing"] = int((diff > 0).sum())
            dev["max_abs_diff_levels"] = int(diff.max())
            dev["frames_touched"] = int((diff.reshape(diff.shape[0], -1).max(1) > 0).sum())
            dev["frames_total"] = int(diff.shape[0])
            record["devices"][device] = dev
            print(json.dumps({k: v for k, v in dev.items() if k != "runs"}, indent=1))
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        fn = f"hmg_determinism_{a.tag or 'local'}.json"
        (out / fn).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out / fn}")
        verdict = []
        for device, dev in record["devices"].items():
            if not dev["baseline_self_consistent"]:
                verdict.append(f"{device}: BASELINE IS NOT PROCESS-REPRODUCIBLE -- the oracle "
                               f"difference is not evidence about the patch")
            elif dev["baseline_equals_head"]:
                verdict.append(f"{device}: identical on this run")
            else:
                verdict.append(f"{device}: both trees self-consistent AND different -- the patch "
                               f"changed this cell")
        print("VERDICT: " + "; ".join(verdict))
        return 0
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
