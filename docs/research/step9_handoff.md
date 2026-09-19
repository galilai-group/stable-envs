# Step 9 — coordinator handoff (2026-09-17, written by the Fable coordinator for an Opus successor)

**Superseded as the living status document.** v0 draft closure is `docs/research/step9_closure.md`
(2026-09-19). This file is the 2026-09-17 stop; do not treat section 0 as current.

Read this, then `STEP_09_AGENT_BRIEFS/STEP_09_OSS_REFACTOR_AND_PRODUCTIZATION.md` (Sami's binding)
plan), `STEP_09_AGENT_BRIEFS/WAVE0_CONTRACT.md` (the frozen API contract every workstream follows),
`examples/results/step9/dashboard.yaml` (machine-readable state; edit only via yaml safe_load/dump
with `/Users/samib/se6-venv/bin/python`), and the review reports under
`examples/results/step9/reviews/`. Git and artifacts are the source of truth; prose may be stale.

## 1. Where things are

| item | location |
|---|---|
| Step-8 baseline (immutable) | `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc` = `milestone/step8-functional-completion` = tag `step8-closure` (both on origin) |
| Step-9 integration branch | `step9/integration`, worktree `/Users/samib/stable-envs-step9-integration` (this file's home). Tip at handoff: see `git log -1` (5741b62 + this commit) |
| pre-Step-9 baseline branch | `sami/pre-step9-baseline` = 21cb306 |
| Step-8 oracle snapshot | `examples/results/step9/oracle/{step8_support_matrix,step8_gate_f_matrix,step8_tiers}.json` + `SHA256SUMS` (immutable input; every difference is API_ONLY / INTENTIONAL_IMPROVEMENT / REGRESSION; closure needs REGRESSION = 0) |
| Sami's briefs (tracked since the foundation merge) | `STEP_09_AGENT_BRIEFS/` (also untracked originals in `/Users/samib/stable-envs/STEP_09_AGENT_BRIEFS/`) |
| shared agent rules | `STEP_09_AGENT_BRIEFS/AGENT_COMMON.md` (login-node rule, GPU partitions, model choice, no pushes) |
| primary checkout | `/Users/samib/stable-envs` is still on `sami/pre-step6-baseline`; Step-6/7/8 worktrees under `/Users/samib/stable-envs-step{6,7,8}-*` are closed history — leave them |
| coordinator memory (Claude Code) | `/Users/samib/.claude/projects/-Users-samib-stable-envs/memory/step6-coordination-state.md` |
| local venvs | `/Users/samib/se6-venv` (CPU, the unit-suite interpreter; has tensordict+loguru), `se8-vm-venv` (stable_worldmodel + dm_control 1.0.44 + mujoco 3.11 + matplotlib), `se8-fetch-venv` (gymnasium-robotics; live Fetch blocked by numpy 2.5.3 + mujoco 3.12), `se7-ogb-venv` (ogbench), `se7-craftax-venv`, `se7-rocket-venv`. uv venvs: install with `/opt/homebrew/bin/uv pip install --python <venv>/bin/python <pkg>` (no pip inside) |
| unit-suite command (baseline number moves as workstreams merge) | `MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"` — 1513+ passed after the B merge (see git log); the two deselected are pre-existing baseline failures (citation scanner; robocasa example help) |

## 0. STOPPING POINT (final, 2026-09-17) — start here

The Fable coordinator stopped at `step9/integration` = the commit carrying this section (parent
678e1d8). Merged: W0, A, H, B, G. The Claude Code session that launched the agents below ends with this
handoff, so **every in-flight agent is dead**; their work is whatever is committed on their branches
plus the uncommitted files noted (checkpoints to inspect, not trust).

| workstream | branch @ HEAD | commits ahead of integration | worktree state | what to do first |
|---|---|---|---|---|
| G 2-D render | step9/render-2d @ 8afa2f4 | merged | MERGED (review_g.md APPROVE); its GPU sbatch is a section of the post-Wave-1 job | — |
| C Fetch | step9/envs-fetch @ 0ef8a97 | 4 | clean; DELIVERED (findings + oracle.json committed: 12 cells, 41 keys identical incl. baked plates and taxonomy 33/12/12/15/51; frames CUDA-only, owed to the GPU oracle; tensordict installed into se8-fetch-venv) | launch a reviewer (Sonnet is fine: mechanical), merge integration into the branch, merge on APPROVE; Workstream I must keep `HIDDEN_GROUP`, `DEFAULT_GROUPS`, `gl_rounding`, `create_render_context` re-exported from render/mjwarp |
| D OGBench | step9/envs-ogbench @ 7f90a06 | 4 | clean; DELIVERED (findings + oracle: 25 maze cells byte-identical on cpu over the 11 frozen scene corpora, 42 keys identical incl. registry module; 58/58 Step-8 names resolve; OGBCube/OGBScene frames CUDA-only owed; suite 1427; ratchet 41 -> 14; package = ogbench/{maze,manipulation}.py with ogb_manip.py as a sys.modules alias; the leftover copy of the plate texel substitution in ogb_manip now calls _plate_texels (plate_probe.py identical); tensordict installed into se7-ogb-venv) | launch a reviewer (Sonnet fine), merge integration into the branch, merge on APPROVE. R2: when the ogb_manip alias goes (Wave 3), move `manipulation._MODULE` and `scratch/step4/universe.py::shard_of` together. R3: docs/api/envs.md ogbench target → M. |
| E Craftax + Rocket | step9/envs-craftax-rocket @ 4f7350d | 7 | clean; DELIVERED (findings + oracle: Craftax frames identical on all three frozen corpora at both sizes and equal to the frozen meta digests; Rocket host-side chain identical, frames CUDA-only owed; suite 1419; craftax_full.py is a sys.modules alias of private _craftax_full.py — 21 module-level names differ between the two modules, so no merge; tensordict+loguru installed into se7-craftax-venv and se7-rocket-venv) | launch a reviewer (Sonnet fine), merge integration into the branch, merge on APPROVE. Its R1: `scratch/step3/certify.py::NO_BACKEND_MODULE_CLASS` keys on module basenames and now misses `navigation` and `_craftax_full` (also broken by A) — coordinator-owned one-line fix or key on class name (assign to K). R3: three docs cite envs/craftax_full.py by path → M. |
| F MuJoCo/non-42 wrappers | step9/envs-mujoco @ 3c83422 | 5 | IN PROGRESS at the stop: commits cover the 17 families' imports, `num_envs=` on the nine families that declared `batch=`, public State + dynamics + close() for all, and `scratch/step9/envs-mujoco/oracle.py` with `oracle.json`/`oracle_maze2d.json` committed; UNCOMMITTED: `tests/unit/test_envs_mujoco.py` (untracked) and edits to highway.py, minihack.py; no findings.md yet | inspect `git diff` and the untracked test; commit as a WIP checkpoint if coherent; re-launch with the §7 brief + "inspect the branch first, finish findings.md, run the unit suite (ratchet must drop 14 -> 0 for highway 6 / maze2d 4 / mujoco_locomotion 4)"; then Sonnet review, merge on APPROVE |
| I MJWarp/nvdiff/_resources | step9/render-mjwarp-nvdiff @ 5aeee37 | 9 | clean; DELIVERED (findings + 28 lifetime tests; static read-through: eleven removed lines, no numeric/upload/cache-key change; NO frame rendered — every backend is CUDA-only, so the oracle is UNESTABLISHED, not satisfied) | Opus review of the static evidence + design; do NOT merge on local evidence alone: run its `scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch` (four venv arms, frame identity, fresh-process, create/close cycles, batch invariance, B=1/4096 timings) as part of the post-Wave-1 GPU job, then merge on APPROVE. Design: `render/_resources.py` (Closeable, BoundedCache LRU, EGL contexts never released — `testing/resource_guard.py` incidents); five classes Closeable (`mjwarp.BatchRenderer`, `nvdiff.Scene`, `nvdiff.BatchRenderer`, `rendernet._Runner`, `LoadedCorrector`); no `__del__`. Consumer change list (one line each, coordinator assigns after those families merge): fetch.py:453, ogb_manip.py:471 (now under ogbench/), robocasa.py:336, humanoidbench.py:520, metaworld.py:760 → release MJWarp contexts on cache clear; dmcontrol `_TC_CACHE` same; mujoco_locomotion.py:1640, rocket_landing.py:729 → `br.close()` on cache clear; rocket_landing.py:739 `owns_scene = True`; robocasa `_CORRECTOR_CACHE` → BoundedCache(4, on_evict=close). Rulings owed: whether `BoundedCache` replaces the eight remaining clear()-on-full caches in envs/* in Wave 2 (behaviour-preserving; recommend yes, as a K/O sweep after Wave 1). |

Then: finish reviews/merges of C, D, E, F, I; ONE GPU oracle over all 42 ids on the merged Wave-1
tip (assemble from the workstreams' sbatch sections: A, B, G, H, I each committed one under
`scratch/step9/<ws>/`); then Wave 2 (J, K, L, M, N), Wave 3 (O), final comparison and closure.

## 2. Branches / worktrees at handoff

| workstream | branch | worktree | HEAD | state |
|---|---|---|---|---|
| W0 foundation | step9/foundation | /Users/samib/stable-envs-step9-foundation | 67e5039 (+ later merges) | MERGED d07ffa8; review_foundation.md APPROVE |
| A classic control + navigation | step9/envs-classic-nav | /Users/samib/stable-envs-step9-envs-classic-nav | c688882 | MERGED 4c88668; review_a.md APPROVE |
| H raycast3d/cpu3d/primitives | step9/render-3d | /Users/samib/stable-envs-step9-render-3d | d231a8f | MERGED 38de3bd; review_h.md APPROVE |
| B DMControl | step9/envs-dmcontrol | /Users/samib/stable-envs-step9-envs-dmcontrol | 82ce271 | MERGED (review_b.md APPROVE); oracle identical cpu+cuda |
| G 2-D rendering | step9/render-2d | /Users/samib/stable-envs-step9-render-2d | cd97cd3 (6 ahead) | IN PROGRESS (agent) |
| C Fetch | step9/envs-fetch | /Users/samib/stable-envs-step9-envs-fetch | b3f3a81 | IN PROGRESS (agent, just launched) |
| D OGBench | step9/envs-ogbench | /Users/samib/stable-envs-step9-envs-ogbench | b3f3a81 | IN PROGRESS (agent, just launched) |
| E Craftax + Rocket | step9/envs-craftax-rocket | /Users/samib/stable-envs-step9-envs-craftax-rocket | b3f3a81 | IN PROGRESS (agent, just launched) |
| F MuJoCo/non-42 wrappers | step9/envs-mujoco | /Users/samib/stable-envs-step9-envs-mujoco | b3f3a81 | IN PROGRESS (agent, just launched) |
| I MJWarp/nvdiff/_resources | step9/render-mjwarp-nvdiff | /Users/samib/stable-envs-step9-render-mjwarp-nvdiff | b3f3a81 | IN PROGRESS (agent, just launched); reviewer must be Opus High |
| J interop, K testing, L packaging/CI, M docs/examples, N benchmarks, O cleanup | not created | — | — | Wave 2/3, not started |

A stray detached worktree `/private/var/folders/.../step9-render2d-baseline-5befc46...` is G's
oracle baseline checkout; it removes itself (or `git worktree prune` after G finishes).

The running agents were launched from a Claude Code session that will not survive this handoff.
When you take over, each in-flight workstream is whatever is COMMITTED on its branch plus
uncommitted files in its worktree (`git status`). Treat uncommitted work as a checkpoint to inspect,
not to trust; every brief told agents to commit early. Re-launch an agent per unfinished workstream
with the brief text below (§6) and "inspect the worktree first".

## 3. Oscar

- SSH `ssh -o BatchMode=yes sboughan@sshcampus.ccv.brown.edu` (auth is intermittently denied; retry
  every few minutes). `SCR=/oscar/scratch/sboughan`. No CPU-heavy work on login nodes (git/rsync/uv
  installs are fine).
- Partition is `gpu` with `--qos=norm-gpu` (the 3090 partition `3090-gcondo` needs
  `--constraint=geforce3090` and its group quota is usually held by Sami's simclr/lejepa arrays; never
  touch those jobs). Jobs so far ran on A40 / RTX A6000 / L40S cards in 2–30 minutes.
- Step-9 trees live as REAL GIT CLONES under `$SCR/se9/<name>` made from bundles, because the oracle
  scripts `git worktree add --detach 21cb306` to build the baseline arm:
  `git bundle create x.bundle 21cb306 <branch>` → rsync → `git clone -b <branch> x.bundle <name>`;
  update an existing clone with `git fetch <bundle> <branch> && git checkout -q FETCH_HEAD -B <branch>`
  (move untracked leftovers aside first). Existing clones: foundation, integration, classic-nav,
  render-3d, envs-dmcontrol. Never rsync with `--delete`; pull results into a staging dir.
- Oscar venvs: `$SCR/venvs/se-step6` (torch cu126, tensordict+loguru installed via
  `~/.local/bin/uv pip install --python <venv>/bin/python ...`; no warp/mujoco_warp), `se-vis311`
  (mujoco 3.11 + mujoco_warp 3.11: the DMC/MJWarp arm), `se7-ogb` (ogbench, mjwarp), `se7-rocket`
  (PyFlyt, nvdiffrast), `se-s3b-fetch311` (gymnasium-robotics), `se7-craftax-gpu`. Install
  tensordict+loguru into any of them before running Step-9 code.
- Committed artifacts must not contain absolute `/oscar/...` paths (sed to `$SCR`); the citation
  scanner test enforces it.
- Job ids so far: 6463991/6464120 (PushT oracle, foundation/merge SHA), 6464309/6464378 (A),
  6464319/6464500/6464601 (H), 6469088 (B, COMPLETED, being incorporated by B's agent).

## 3a. Oscar: how it works and how this project uses it

**What it is.** Brown's Oscar cluster (CCV). Login nodes are shared, for editing/git/rsync/light
installs only — Sami's rule: **no CPU-heavy work on login nodes** (no pytest, probes, renders). All
compute goes through SLURM to compute nodes. The stable-envs Claude skills for it live on Oscar at
`~/.claude/skills/oscar*` (read them for partition/queue details before inventing commands).

**Access.** `ssh -o BatchMode=yes sboughan@sshcampus.ccv.brown.edu` (key auth; needs Sami's VPN up —
"Permission denied (keyboard-interactive)" means the VPN/auth is down, not a fault of yours: retry
every few minutes, ask Sami if it lasts). Every command below runs over that ssh. Two harmless
lines print on every login (`.bashrc: module: command not found` + the consent banner); strip them
with `sed '1,/^#####*$/d' | grep -v 'bashrc: line 51'`.

**Storage.** `SCR=/oscar/scratch/sboughan` (large, purgeable scratch: all trees, venvs, caches,
logs live here). `$SCR/logs/` holds job stdout (`%x_%j.out`). `$SCR/venvs/` holds the uv venvs
(see §3). `$SCR/cache/{uv,pip,xdg,torch,warp,hf}` — export the matching env vars in every job so
nothing writes to the small `$HOME`. Never `rsync --delete` onto Oscar; pull results into a staging
dir on the Mac, then copy into the worktree with `/oscar/scratch/sboughan` rewritten to `$SCR`.

**Getting code there.** The oracle scripts need a real git repo (they `git worktree add --detach
21cb306`), so ship branches as bundles:
```
git bundle create x.bundle 21cb306 <branch>
rsync -az x.bundle sboughan@sshcampus.ccv.brown.edu:$SCR/x.bundle
ssh ... 'cd $SCR/se9 && git clone -q -b <branch> $SCR/x.bundle <name>'      # first time
ssh ... 'cd $SCR/se9/<name> && git fetch <bundle> <branch> && git checkout -q FETCH_HEAD -B <branch>'  # update
```
Move untracked leftovers aside before a checkout (`git status --porcelain | grep '^??'`).

**Venvs on Oscar** (uv-made; no pip inside — install with `~/.local/bin/uv pip install --python
<venv>/bin/python <pkgs>`; `import stable_envs` needs `tensordict` + `loguru` since Wave 0):
`se-step6` (torch 2.14 cu126, general; no warp), `se-vis311` (mujoco 3.11.0 + mujoco_warp 3.11.0:
the DMC/MJWarp certification arm), `se7-ogb` (ogbench + mjwarp), `se7-rocket` (PyFlyt 0.29.0 +
nvdiffrast), `se-s3b-fetch311` (gymnasium-robotics + mujoco 3.11: the only interpreter where Fetch
builds), `se7-craftax-gpu` (craftax + jax cuda), `se8-mjwarp313` (3.13 evaluation, not adopted).

**Submitting work.** Batch jobs only (an interactive `interact3090`/`salloc` alias exists but the
held allocation pattern from Step 7/8 is gone). Header that works:
```
#SBATCH --partition=gpu --qos=norm-gpu --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=01:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
```
`gpu` is the partition, `norm-gpu` the QOS (the earlier `3090-gcondo` partition needs
`--constraint=geforce3090` and its group quota is usually held by Sami's simclr/lejepa arrays —
never touch those jobs; never `scancel` anything you did not submit). Cards you get on `gpu`: A40,
RTX A6000, L40S — fine for identity/isolation diffs (both arms run in the same job on the same
card); the frozen Step-4/7 digests were made on RTX 3090s, so a *cross-card* digest comparison is
not a valid oracle, a same-job baseline-vs-branch one is. `sbatch script.sbatch` → job id; watch
with `sacct -j <id> -n -X -o State,Elapsed,NodeList` (poll no more than every 2–3 min) or
`squeue -u sboughan`; read `$SCR/logs/<jobname>_<id>.out`. Typical Step-9 evidence job: 2–30 min.

**Job-script habits that saved us.** `set -uo pipefail` (not `-e`: later sections must still run
and report); print a provenance block first (host, job id, `git rev-parse HEAD` of the tree, GPU
name/driver, python/torch/mujoco versions); write full pytest output to a file and echo only the
summary (MuJoCo's EGL teardown prints dozens of `Exception ignored in __del__` lines AFTER the
verdict — they are noise, not failures — and `| tail` would hide the summary); end with an
`ALLDONE` marker line; report exit code 5 (no tests collected) as a warning, never as a pass.
`MUJOCO_GL=egl` on compute nodes (`glfw` on the Mac). Compute nodes may lack internet: install
packages from the login node with uv beforehand.

**Reading results.** Every committed artifact must be scrubbed (`sed 's#/oscar/scratch/sboughan#$SCR#g'`);
`tests/unit/test_repo_contracts.py` fails on absolute cluster paths. Sizes of things: the Step-9
oracle over 38 cells ≈ 10 min; a family GPU test selection ≈ 10–15 min; the `tests/gpu` full run
≈ 30 min plus 400+ "errors" that are just missing optional packages in whichever venv you chose.

## 4. Rulings in force (all recorded in dashboard `decisions`)

1. Wave-0 contract (WAVE0_CONTRACT.md): `num_envs=`; TensorDict State with `pack`/`unpack` to the
   unchanged Step-8 packed float32 row; `step` stateful returning `StepResult` (TensorDict) via
   `Env._step_result`; `transition`/`unroll` pure; `close`/context manager; registry `make/spec/list_envs`
   with SWM aliases; loguru config; tensordict + loguru core deps; deprecated aliases (`StableEnv`,
   `REGISTRY`, `register`, `batch=`, `se.make`) kept and warning once until Wave 3.
2. `state_layout` entries may carry a 4th element = public dtype (float64 simulators); `Env._layout`
   and `_layout_dtypes` are INSTANCE methods; `Env._hidden_state()` (fields the row does not carry)
   and `Env._live_state()` (drawn fields with a more precise live value) are the two hooks — no cell
   overrides the `state` property.
3. `unpack` copies (a State never aliases the env buffer) — a defect the coordinator introduced and
   Workstream A found; regression test in tests/unit/test_foundation_state.py.
4. Testing harness detects renderer-only cells with `dynamics.has_dynamics`, never
   `hasattr(env, "step")` (repo-wide test); every Env now has `step`.
5. Registry-row test: semantic columns compared exactly; `module` compared by reachability (old
   import path resolves to the same class) until Wave 3.
6. A package may replace a module of the same name (dmcontrol/, ogbench/); obligation = a test that
   every Step-8 module-level name still resolves; a module→module move keeps a `sys.modules` alias
   shim (A's swm_nav.py pattern).
7. StepResult cost on trivially cheap `step`s (+12–28 µs) accepted; the 10 % bar is for render/dynamics
   hot paths; Wave-3 candidate: lazy StepResult.
8. Sami (latest): run all Wave-1 workstreams in parallel; NO per-workstream GPU jobs; keep the
   per-workstream CPU oracle; ONE GPU oracle over all 42 ids after the Wave-1 merges (before K
   restructures tests) and the final §20 comparison at closure. Reviews: Sonnet for mechanical
   env-family reviews, Opus for B (plan says so), H/I/J/K (renderer/resource/interop/testing).
9. Fetch `background.color` stays partial (no re-freeze); trackcom 3.11 provenance; colour-check
   applicability; both Craftax corpora — all Step-8 rulings carry into Step 9 unchanged.

## 5. Procedure per workstream (what has worked)

1. Agent delivers on its branch with findings.md (§17 ten items), CPU oracle JSON (frames byte-identical
   vs a detached 21cb306 worktree in the same venv), unit-suite counts, paired CPU timings.
2. Coordinator handles its "requests" (foundation changes are coordinator-owned: env.py, dynamics.py,
   specs.py, registry.py, _config.py, __init__.py, pyproject.toml) on step9/integration, then tells
   the agent to `git merge step9/integration` (never rebase) and re-run its oracle.
3. Independent reviewer (fresh agent) writes `examples/results/step9/reviews/review_<x>.md`, first
   line APPROVE / APPROVE-WITH-CHANGES / BLOCK; only APPROVE merges; author fixes; reviewer re-verifies
   the delta and appends a final verdict.
4. Coordinator: record the review file on integration, `git merge --no-ff <branch>`, run the unit
   suite, update the dashboard row (status, head_sha, review, merge_sha, oracle_diff, tests,
   performance_diff, api_diff), commit. Author `sami-bg <samibghanem@gmail.com>`, NO AI trailers
   (`git log --format=%B | grep -ci "co-authored"` must stay 0), never push.
5. Additive conflicts (mkdocs nav, `__init__` exports, dashboard) are the coordinator's.

## 6. Remaining plan

- Finish and merge B, G, C, D, E, F, I (reviews as above). Expect each to file coordinator requests;
  answer them centrally, keep the contract frozen.
- After all Wave-1 merges: ONE GPU oracle job over all 42 ids on the merged tip — assemble it from the
  workstreams' sbatch sections (H: scratch/step9/render-3d/gpu_oracle.sbatch; B:
  scratch/step9/envs-dmcontrol/gpu.sbatch; I will write scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch;
  A/G have theirs) plus `tests/gpu` full run; classify every difference; REGRESSION must be 0.
- Wave 2: J interop (Opus High; `stable_envs/interop.py` over `_interop/`, tests/integration/test_worldmodel.py,
  42-id resolution, swap_renderer compatibility, native `swm.World(env=snv.PushT(...))` path), K tests
  (Opus High; tests/{unit,integration,gpu,certification,fixtures} + markers; `stable_envs/testing` → tests;
  `rollout.py` move), L packaging/CI, M docs/examples, N benchmarks.
- Wave 3: O cleanup (scratch/, examples/results/, shims, deprecated aliases, `stable_envs/core`,
  `compat`, legacy top-level names), only after canonical tests/docs/fixtures exist.
- Final: Step-9 support matrix vs the Step-8 oracle (§20), paired performance (§21), OSS gate (§22),
  closure packet `docs/research/step9_closure.md` + `examples/results/step9/support_matrix.json` (§24),
  recommendation (§23). Milestone branch/tag only when Sami accepts; push only when told.

## 7. Agent brief template (what every Wave-1 agent was told)

Read AGENT_COMMON.md, WAVE0_CONTRACT.md, the plan sections for its workstream, the templates
(`stable_envs/envs/pusht.py` + `examples/results/step9/foundation/findings.md`;
`stable_envs/envs/classic_control.py` + `examples/results/step9/envs-classic-nav/findings.md`;
DMC package on step9/envs-dmcontrol). Own only its listed files; render backends re-export every
name; never edit foundation files (requests go in findings); never override `state`; commit early;
CPU oracle script copied from A/B (`scratch/step9/<ws>/oracle.py`, detached 21cb306 worktree, same
venv, byte-identical frames + describe/style/taxonomy/fidelity/registry/state_names/round-trip);
GPU frames "owed to the post-Wave-1 GPU oracle"; unit suite command above; findings per §17;
final message with commits, oracle result, test totals, requests.

## 8. Known pitfalls

- Agents die on the Anthropic monthly spend limit mid-task; committed progress survives, uncommitted
  files are checkpoints. Sami's budget: prefer Sonnet where the brief allows; he considers Fable too
  expensive for coordination from here on.
- H found one cuda-only difference on `HumanoidmazeGiantNavigate`; classified DOCUMENTED_LIMITATION
  (baseline tree not process-reproducible on gpu1503; jobs 6464319/6464500/6464601). Expect it to
  reappear in the final GPU oracle; it is not a regression.
- `tests/gpu` in `se-step6` errors on missing optional packages (warp, gymnasium_robotics, minigrid);
  use the family venvs listed above per backend.
- The two pre-existing unit failures (citation scanner; robocasa example `--help`) predate Step 9.
