# Step 6 — common operating instructions for every workstream subagent

Read this fully before touching anything. The coordinator (not you) merges, pushes, and decides
what gets implemented. You return evidence and, when authorized, a patch on your own branch.

## Baseline

- Final completed Step-5 baseline SHA: `89fa5ca81341dc4df7186ec325cb691431a2ef17`
  (branch `sami/pre-step6-baseline`; the earlier `3e35ef9` is NOT the parent).
- Every Step 6 branch was created from that SHA. Verify with `git merge-base --is-ancestor`.

## Your worktree — the ONLY place you edit

Each workstream has a dedicated git worktree at `/Users/samib/stable-envs-step6-<ws>` on branch
`step6/<ws>`. Never edit `/Users/samib/stable-envs` (baseline checkout), never edit another
workstream's worktree, never check out another branch, never merge, never rebase, never push.

Git identity is already configured (name `sami-bg`, real email). Commit on your branch with plain
messages. NO `Co-Authored-By`, NO AI attribution trailers, NO "Generated with" lines.

Do not modify: anything under `examples/results/step4/` (read-only frozen oracle),
`examples/results/step5/`, frozen Step-2 thresholds, existing fidelity declarations/bounds
(unless your brief explicitly concerns a declaration's provenance — then report, do not edit
without coordinator approval). Do not start Step 7 (new renderer coverage) work.

Write every Step-6 artifact under `examples/results/step6/<workstream>/` inside your worktree
(e.g. `profile_before/`, `profile_after/`, `replay/`, `sweep/`, `findings.md`). The profile
harness `scratch/step6/profile_note.py` writes host-tagged JSON+MD notes; reuse it and extend it
rather than writing a parallel harness.

If you discover the real fix belongs in a shared file another workstream owns (e.g. raster2d
and gfxdraw both touching `stable_envs/render/scene.py`, or raycast3d and cpu-3d touching
`stable_envs/render/raycast3d.py`), STOP that edit, record the finding in your findings.md, and
report it — the coordinator assigns ownership.

## Local machine

Apple M5 Mac, no NVIDIA GPU, no Triton. Shared CPU interpreter for all worktrees:
`/Users/samib/se6-venv/bin/python` (torch 2.14 CPU, mujoco 3.12.0, dm_control 1.0.45, pygame,
pymunk, pytest). Run tests from your worktree root so the root `conftest.py` puts the worktree
on `sys.path`, e.g. `cd /Users/samib/stable-envs-step6-<ws> && /Users/samib/se6-venv/bin/python -m pytest tests/unit -q`.
The local machine is for editing, static inspection, CPU-only tests, and git. Local CPU timings
are NOT certification numbers. Do not "fix" CUDA/Linux-only code because it fails locally.
Do not `pip install` into `/Users/samib/stable-envs/.venv`.

## Oscar (Brown HPC) — all GPU work and all certification measurements

SSH: `ssh -o BatchMode=yes sboughan@sshcampus.ccv.brown.edu`. If it returns "connection refused"
the VPN is off: report BLOCKED to the coordinator; do not retry in a loop.
Non-interactive shells do not source aliases; send multi-line scripts via
`ssh ... 'bash -ls' < script.sh`. Filter the login banner (`grep -v '^#'`).

Scratch root: `SCR=/oscar/scratch/sboughan`. **HOME (`/oscar/home/sboughan`) is at 100% of its
quota. Never write anything to HOME**: set `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`,
`TORCH_HOME`, `WARP_CACHE_PATH`, `HF_HOME` under `$SCR` as the existing sbatch scripts in
`scratch/step6/*.sbatch` do. Copy their header/env block.

Your code on Oscar lives at `$SCR/se6/<ws>/` — an rsync copy of your worktree (already seeded).
Sync after every local edit, and pull results back, with:

```bash
# push code (from the Mac)
rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude '*.pyc' \
  /Users/samib/stable-envs-step6-<ws>/ sboughan@sshcampus.ccv.brown.edu:/oscar/scratch/sboughan/se6/<ws>/
# pull artifacts back (from the Mac)
rsync -az sboughan@sshcampus.ccv.brown.edu:/oscar/scratch/sboughan/se6/<ws>/examples/results/step6/ \
  /Users/samib/stable-envs-step6-<ws>/examples/results/step6/
```

Always run with `PYTHONPATH=$SCR/se6/<ws>` and `cd $SCR/se6/<ws>` so the copy you measure is your
branch, not the stale `$SCR/stable-envs` checkout (which is an old snapshot on `main` — do not use
it, do not modify it).

Interpreters (do NOT rebuild, reinstall into, or delete any of these; create a new venv under
`$SCR/venvs/se6-<ws>-<purpose>` if you truly need different pins, and say so in your report):

| venv | python | torch | mujoco | notable |
|---|---|---|---|---|
| `$SCR/venvs/se-step6` | 3.12 | 2.14.0+cu126 | 3.12.0 | dm_control 1.0.45, triton, pygame, ogbench, pymunk, pytest — default for raycast3d/gfxdraw/raster2d/tilemap |
| `$SCR/venvs/se-step5` | 3.12 | 2.14.0+cu126 | 3.12.0 | + mujoco_warp, warp, stable_worldmodel, triton — use when MJWarp or SWM is required at mujoco 3.12 |
| `$SCR/venvs/se-s3b-fetch311` | 3.12 | 2.14.0+cu126 | 3.11.0 | the Fetch stack (Step 3B pin) |
| `$SCR/venvs/se-tex311` | 3.12 | 2.14.0+cu126 | 3.11.0 | dm_control 1.0.44, no warp |
| `$SCR/venvs/se-vis311` | built by `scratch/step6/visual311.sbatch` | | 3.11.0 | mujoco-warp 3.11 |

SLURM rules (from the `oscar*` skills in `~/.claude/skills/` on Oscar — read them if in doubt):
- Never run heavy compute on the login node. CPU certification benchmarks run on
  `--partition=batch --qos=normal` (never login-node numbers). GPU work runs under SLURM.
- Partition/QOS must match: `3090-gcondo`↔`cs-3090-gcondo`, `gpu`↔`norm-gpu`, `batch`↔`normal`.
- **Capacity right now:** Sami's `simclr_cifar10` campaign occupies the entire condo QOS budget
  (cpu=240, gpu 30/32). Condo jobs will pend with `QOSGrpGRES`/`QOSMaxCpuPerUserLimit` for many
  hours. For Step 6 GPU jobs use `--partition=gpu --qos=norm-gpu --gres=gpu:1
  --constraint=geforce3090` (a 3090 keeps before/after on the same card as Step 3/Step 4 GPU
  numbers). `gpu`/`norm-gpu` allows ~2 concurrent jobs per user across ALL Step 6 agents, so:
  at most ONE GPU job in flight per agent, `--time` ≤ 03:00:00 unless the coordinator approves
  more, `--cpus-per-task` ≤ 4, `--mem` ≤ 48G. Queue a second only after the first finishes.
- **NEVER cancel any `simclr_cifar10` job. Never cancel a job you did not submit.**
- Job names must start with `se6-<ws>-`. Logs to `$SCR/logs/%x_%j.out`. Record every job ID.
- `COMPLETED` is not proof of success: check the log for the expected output files.
- Prefer sbatch over interactive sessions; do not hold an interactive GPU node idle.
- Check `squeue -u sboughan` before submitting; do not flood the queue.

## Methodology (mandatory)

1. reproduce baseline  2. profile the PUBLIC caller-visible path (`se.make(...)`, `set_state`,
`render(states)`) — never a private kernel in isolation  3. attribute wall time  4. identify the
largest STRUCTURAL bottleneck  5. explain the mechanism  6. propose the smallest structural
intervention  7. implement only if authorized  8. re-profile  9. run regression suite  10. replay
ALL affected frozen Step-4 configs  11. run the required sweep  12. micro-optimize last.

Structural problems first: Python loops over batch/env/object; serial work behind a batched
API; per-frame/per-object kernel launches; repeated static scene construction/upload;
recomputed grids/backgrounds; host↔device sync; `.item()/.cpu()/.cuda()` in hot loops; B×H×W
temporaries per primitive; per-env state unpacking; failure to pre-bake static geometry.

Never: weaken a threshold; move work outside the timed region; add hidden global caches that
break batch/order invariance or render purity; silently use GPU on a `device="cpu"` path; tune
pixels against downstream task score.

## Regression machinery (use it, do not reinvent it)

`stable_envs/testing/conformance.py`, `stable_envs/testing/gate.py`,
`tests/gpu/test_conformance.py`, `tests/gpu/test_reproducibility.py`, `tests/gpu/test_fast_paths.py`,
`tests/unit/test_declarations.py`, `tests/unit/test_repo_contracts.py`, plus path-specific tests
(e.g. `tests/gpu/test_gfxdraw_fused.py`). Frozen Step-4 corpus: `examples/results/step4/`
(states in `states/<Config>.npz`, per-config contract in `certification.json`, digests/metrics in
`conformance/*.json`; `examples/fidelity_step4.py` is the replay entry point — read it). Replay
writes ONLY under `examples/results/step6/<ws>/replay/`.

## Report format (your final message)

Classify the result as exactly one of `FINDING` / `PATCH_CANDIDATE` / `PATCH` / `BLOCKER` / `NO-OP`
and give: baseline SHA; your branch HEAD SHA; hardware/venv; affected configs (enumerated);
profile-before artifact paths; bottleneck mechanism (measured numbers, not impressions);
proposed/implemented intervention; tests run and results; Step-4 replay result; tiers
before/after; pixels changed (yes/no/which configs); semantic follow-up recommendation
(none / DINO tripwire / semantic smoke / paired Gate H); SLURM job IDs; open questions for the
coordinator. Keep the report self-contained: the coordinator does not see your tool calls.
Write the same content to `examples/results/step6/<ws>/findings.md` and commit it on your branch.

## Addendum — measured Oscar capacity and the same-card rule

- `norm-gpu` QOS per-user limits: **gpu=2, cpu=12, mem=192G** shared by ALL Step 6 agents. So:
  one GPU job per agent, `--cpus-per-task=4`, `--mem<=48G`, `--time<=03:00:00`. Pending is fine;
  never cancel another agent's job. The `gpu` partition has 11 mixed `geforce3090` nodes, so
  `--constraint=geforce3090` is schedulable; if your job has pended > 45 min, resubmit with
  `--constraint='geforce3090|l40s|a5000|a5500|a40|a6000'` and cancel only your own pending copy.
- A pristine copy of the baseline SHA is at `$SCR/se6/baseline/`. When a GPU number matters,
  measure BEFORE (`PYTHONPATH=$SCR/se6/baseline`) and AFTER (`PYTHONPATH=$SCR/se6/<ws>`) in the
  SAME SLURM job on the same node, and record the card. A before on a 3090 and an after on an
  L40S is not a before/after. Never modify `$SCR/se6/baseline/`.
- Put your sbatch scripts under `scratch/step6/<ws>/` in your worktree so they are versioned and
  rsynced with the code.
- CPU certification: `--partition=batch --qos=normal --cpus-per-task=4`, torch threads pinned to 1
  (`--threads 1` in profile_note.py, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`).

## Policy addendum (Sami, 2026-09-14) — CPU/GPU backends

```text
backend implementation may differ by device
renderer semantics may not
```

- CPU and GPU may use structurally different backends (span rasterization, analytic raycasting,
  static-background caching + dynamic compositing, ...) as long as they implement the SAME renderer
  semantics — camera, geometry, transforms, visibility, materials, lighting, appearance — and are
  certified against the SAME frozen correctness/fidelity contract. Do not weaken or fork the contract
  per device.
- Do not force MJWarp onto CPU for implementation symmetry. Prefer one shared CPU-native 3-D backend
  for the MJWarp-only scenes (Cheetah/Hopper/Walker/Humanoid, and Fetch if feasible). Do not create
  bespoke per-environment approximate CPU renderers just to hit the performance bar.
- Device selection must be honest: `device="cpu"` -> genuine CPU backend; `device="cuda"` -> GPU
  backend. Never silently fall back to CUDA for a CPU request (raise if no CPU backend exists).
- Minimize coupling so a later cleanup can extract a shared Scene3D-like representation without
  rewriting the environments; do not block Step 6 on that refactor.
- Fetch CPU: quantify feasibility against the frozen CPU_FAST bar at its mesh/fill complexity BEFORE
  any large implementation.

## Addendum (Sami, 2026-09-15): login node usage
- NEVER run CPU-heavy operations on an Oscar login node (no test suites, probes, renders, benchmarks,
  venv builds, pip/uv installs, or large rsync/tar/gzip passes there). Login nodes are for ssh,
  sbatch/sacct/squeue, small file reads/writes, and lightweight git/rsync of sources only.
- Anything that computes runs inside a SLURM job (`batch`/`normal` for CPU, `gpu`/`norm-gpu` for GPU),
  including "quick" pytest runs and one-off probes. If a check is small, still submit it as a short
  CPU job (`--cpus-per-task=1 --time=00:15:00`).

## Addendum (Sami, 2026-09-15): GPU partitions
- GPU jobs may be submitted to `--partition=3090-gcondo` (condo QOS; usually faster to start) or to
  `--partition=gpu --qos=norm-gpu`. Prefer `3090-gcondo` for the remaining Step-7 closure jobs; keep the
  node exclusions (gpu2108,gpu2113,gpu2110,gpu2116) and the one-GPU-job-per-workstream rule; never
  cancel or touch the simclr jobs.

## Addendum (Sami, 2026-09-16): agent model choice
- Spawn workstream and reviewer agents on Opus by default. Use a different model family for an
  independent reviewer only when the author was Opus (then Sonnet); avoid Fable subagents (they have hit
  the monthly spend limit mid-task). Budget note: about $1000 of credits accrued across stable-envs so far.
