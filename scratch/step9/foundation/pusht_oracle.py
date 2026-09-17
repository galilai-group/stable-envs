"""Oracle proof for the Wave-0 PushT migration: the same pixels, from two different trees.

WHAT IT PROVES. The Step-8 behavioral oracle (plan §7) says the architecture may change and the
behavior may not. For PushT that is five statements, and each one is checked here by running the
FROZEN STEP-4 CORPUS (``examples/results/step4/states/PushT.npz``, 64 states) through BOTH trees in
two separate interpreters and comparing the results byte for byte:

  1. the frames: ``render(states)`` at the registered default size, SHA-256 over the raw uint8 bytes;
  2. ``describe()``: the dict a consumer reads to decide whether these frames are usable;
  3. the style layout: ``style_names`` and the default (official) appearance;
  4. the registry row: name, tier, decoder, state_dim, reference_id, module;
  5. the packed layout itself: ``state_dim``/``state_names``, and ``set_state``/``get_state``
     round-tripping the corpus.

It also checks the two claims that are NEW in Step 9 and therefore have no baseline to compare
against: ``transition`` equals the stateful ``step`` to the bit on the same state, and ``unroll``
equals repeated ``transition``. Those run on every available device — cpu always, cuda when there is
one — because PushT is a CPU-capable cell and the CUDA path is a captured graph, which is precisely
the path a functional reimplementation could diverge from.

HOW THE BASELINE IS OBTAINED. A detached ``git worktree`` of the Step-8 baseline SHA, so the
comparison is against the tree as it was and not against a memory of it. The baseline interpreter is
the same venv, so any difference that shows up is OURS and not a library's.

Usage::

    MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/foundation/pusht_oracle.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE_SHA = "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc"
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]

#: Run inside ONE tree, printing a JSON record on stdout. Written as a string and executed with
#: ``PYTHONPATH`` pointed at the tree under test, because the two trees define the same module names
#: and cannot both be imported into one interpreter.
PROBE = r'''
import hashlib, json, sys
import numpy as np, torch

import stable_envs as se

corpus = np.load(sys.argv[1])["states"].astype(np.float32)
n = int(corpus.shape[0])
cls = se.REGISTRY["PushT"] if hasattr(se, "REGISTRY") else se.PushT
kw = dict(device="cpu")
try:
    env = cls(num_envs=n, **kw)
except TypeError:
    env = cls(batch=n, **kw)

env.set_state(corpus)
rows = env.get_state().detach().cpu().numpy()
frames = env.render().detach().cpu().numpy()
frames_from_rows = env.render(torch.as_tensor(corpus)).detach().cpu().numpy()

def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def rows_of(table):
    return [r for r in table if r["name"] == "PushT"]

out = {
    "state_dim": int(env.state_dim),
    "state_names": list(env.state_names),
    "size": int(env.size),
    "roundtrip_exact": bool(np.array_equal(rows, corpus)),
    "rows_sha256": h(rows),
    "frames_shape": list(frames.shape),
    "frames_sha256": h(frames),
    "frames_from_rows_sha256": h(frames_from_rows),
    "frames_mean": float(frames.mean()),
    "describe": json.loads(json.dumps(env.describe(), default=str)),
    "style_names": list(env.style_names),
    "style_dim": int(env.style_dim),
    "default_style_sha256": h(np.asarray(env.default_style(), np.float32)),
    "default_style": [float(x) for x in np.asarray(env.default_style(), np.float32).reshape(-1)],
    "registry_row": rows_of(se.registry_table())[0],
    "registry_table": se.registry_table(),
    "import_errors": (se.import_errors() if hasattr(se, "import_errors")
                      else se.envs.import_errors()),
    "sample_states_sha256": h(np.asarray(env.sample_states(16, np.random.default_rng(0)),
                                         np.float32)),
}
print("@@JSON@@" + json.dumps(out))
'''


def _run_probe(tree: Path, corpus: Path, python: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "glfw")
    r = subprocess.run([python, "-c", PROBE, str(corpus)], capture_output=True, text=True,
                       cwd=str(tree), env=env)
    if r.returncode != 0:
        raise SystemExit(f"probe failed in {tree}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")][-1]
    return json.loads(line[len("@@JSON@@"):])


def _baseline_worktree(sha: str) -> tuple[Path, bool]:
    """A detached worktree of ``sha``. Returns the path and whether this call created it."""
    existing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout
    for block in existing.split("\n\n"):
        if sha[:12] in block and "step9-oracle-baseline" in block:
            return Path(block.splitlines()[0].split(" ", 1)[1]), False
    path = Path(tempfile.gettempdir()) / f"step9-oracle-baseline-{sha[:12]}"
    if not path.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(path), sha], cwd=str(ROOT),
                       check=True, capture_output=True)
        return path, True
    return path, False


def _functional_checks(python: str, tree: Path) -> list[dict]:
    """``transition`` vs ``step`` and ``unroll`` vs repeated ``transition``, per device."""
    code = r'''
import json, sys
import numpy as np, torch
import stable_envs as snv

corpus = np.load(sys.argv[1])["states"].astype(np.float32)
n = int(corpus.shape[0])
out = []
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devices:
    env = snv.PushT(num_envs=n, device=dev)
    st = env.unpack(torch.as_tensor(corpus, device=env.device))
    g = torch.Generator().manual_seed(0)
    a = (60.0 + 392.0 * torch.rand(n, 2, generator=g)).to(env.device)

    functional = env.transition(st, a)
    env.set_state(corpus)
    stateful = env.step(a)["state"]
    same_rows = bool(torch.equal(env.pack(functional), env.pack(stateful)))
    same_vel = bool(torch.equal(functional["agent", "velocity"], stateful["agent", "velocity"]))

    acts = torch.stack([(60.0 + 392.0 * torch.rand(n, 2, generator=g)).to(env.device)
                        for _ in range(4)])
    traj = env.unroll(st, acts)
    cur = st
    unroll_ok = True
    for t in range(4):
        cur = env.transition(cur, acts[t])
        unroll_ok &= bool(torch.equal(env.pack(traj[t]), env.pack(cur)))
        unroll_ok &= bool(torch.equal(traj[t]["agent", "velocity"], cur["agent", "velocity"]))
    graphed = bool(getattr(env, "_step_graph", None) is not None)
    frames = env.render(st)
    frames_from_rows = env.render(env.pack(st))
    out.append({"device": dev, "transition_equals_step_bitwise": same_rows and same_vel,
                "unroll_equals_repeated_transition": unroll_ok,
                "stateful_path_was_a_cuda_graph": graphed,
                "render_state_equals_render_rows": bool(torch.equal(frames, frames_from_rows))})
    env.close()
print("@@JSON@@" + json.dumps(out))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    env.setdefault("MUJOCO_GL", "glfw")
    corpus = tree / "examples/results/step4/states/PushT.npz"
    r = subprocess.run([python, "-c", code, str(corpus)], capture_output=True, text=True,
                       cwd=str(tree), env=env)
    if r.returncode != 0:
        raise SystemExit(f"functional checks failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.splitlines()[-1][len("@@JSON@@"):])


#: Every key the two trees must agree on, and the sentence that says what a difference would mean.
COMPARED = {
    "state_dim": "the packed row's width",
    "state_names": "the packed row's column names",
    "size": "the registered default frame size",
    "roundtrip_exact": "set_state/get_state round-tripping the frozen corpus",
    "rows_sha256": "the packed rows the renderer was handed",
    "frames_shape": "the frame shape",
    "frames_sha256": "THE FRAMES, byte for byte",
    "frames_from_rows_sha256": "the frames rendered from explicit rows",
    "describe": "describe(), which is what a consumer reads",
    "style_names": "the style layout",
    "style_dim": "the style width",
    "default_style_sha256": "the official appearance",
    "registry_row": "the registry row for PushT",
    "sample_states_sha256": "the env's own reset distribution at a fixed seed",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--baseline-sha", default=BASELINE_SHA)
    ap.add_argument("--out", default=str(ROOT / "examples/results/step9/foundation"))
    a = ap.parse_args()

    baseline, created = _baseline_worktree(a.baseline_sha)
    try:
        corpus = ROOT / "examples/results/step4/states/PushT.npz"
        before = _run_probe(baseline, baseline / "examples/results/step4/states/PushT.npz", a.python)
        after = _run_probe(ROOT, corpus, a.python)
        checks = _functional_checks(a.python, ROOT)

        diffs = {k: {"baseline": before[k], "migrated": after[k], "means": why}
                 for k, why in COMPARED.items() if before[k] != after[k]}
        record = {
            "baseline_sha": a.baseline_sha,
            "head_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                       capture_output=True, text=True).stdout.strip(),
            "corpus": "examples/results/step4/states/PushT.npz",
            "corpus_states": 64,
            "python": a.python,
            "compared": sorted(COMPARED),
            "unexplained_differences": diffs,
            "identical": not diffs,
            "baseline": before,
            "migrated": after,
            "functional_checks": checks,
        }
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pusht_oracle.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        # The baseline registry table, recorded for tests/unit/test_foundation_registry.py. Same
        # interpreter, so the set of families that imported is comparable.
        (out / "registry_baseline.json").write_text(json.dumps(
            {"baseline_sha": a.baseline_sha, "python": a.python,
             "registry_table": before["registry_table"],
             "import_errors": before["import_errors"]}, indent=1, sort_keys=True) + "\n")

        print(json.dumps({k: record[k] for k in ("baseline_sha", "head_sha", "identical",
                                                 "unexplained_differences", "functional_checks")},
                         indent=1))
        ok = record["identical"] and all(c["transition_equals_step_bitwise"]
                                         and c["unroll_equals_repeated_transition"]
                                         and c["render_state_equals_render_rows"] for c in checks)
        print("ORACLE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        if created:
            subprocess.run(["git", "worktree", "remove", "--force", str(baseline)], cwd=str(ROOT),
                           capture_output=True)


if __name__ == "__main__":
    raise SystemExit(main())
