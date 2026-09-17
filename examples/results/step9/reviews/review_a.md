# Step 9 Wave 1 — Workstream A (classic control + navigation) — independent review

**Reviewed SHA:** `60b515dace52445dd85b28cd9e704ccb89f01538` (`step9/envs-classic-nav`)
**Diff reviewed:** `git diff 3490156 60b515d` (foundation code from the two integration merges excluded)

APPROVE-WITH-CHANGES

## The §17 ten items

1. **Files moved/removed/created** — Confirmed. `swm_nav.py` and `simple_point_maze.py` are now
   14/15-line `sys.modules` aliases of the new `stable_envs/envs/navigation.py` (1,527 lines);
   `classic_control.py` edited in place; `tests/unit/test_envs_classic_nav.py` created (118 collected
   tests); nothing importable removed. Both alias files verified by direct read: they set
   `sys.modules[__name__] = _navigation`, so a single module object serves both old paths.
2. **API changes** — Confirmed additive except `step`'s return type (Wave-0 `StepResult` contract).
   `classic_control.py` diff shows `transition`/`unroll` added via `_sim_from_state`/`_state_td`,
   `_live_state`/`_hidden_state` hooks, derived `state_layout` via `__init_subclass__`, and the new
   `frame_shape` property fixing a real `observation_spec` bug for the 400x600 cells.
3. **Step-8 IDs affected** — Behaviorally none; oracle confirms all eight served SWM ids byte-identical
   except the declared `module` column for the five navigation ids.
4. **Targeted tests** — `tests/unit/test_envs_classic_nav.py` re-run standalone: **118 passed** (see
   Defect D1: findings.md's "88"/"108" counts and its "1355 passed" full-suite transcript are stale
   relative to this SHA).
5. **Applicable certification checks** — Step-4 replay done for all ten cells (oracle re-run, see
   below); GPU conformance/fast-paths/gfxdraw/navdot re-verified from committed logs. Gate A/GPU
   conformance require CUDA and were run on Oscar (job 6464378) — reproduced from committed artifacts,
   not re-run locally (no GPU on this Mac, per instructions).
6. **Default-frame comparison** — Re-ran `scratch/step9/envs-classic-nav/oracle.py --device cpu` into
   a fresh temp dir: `identical: true`, `unexplained_differences: {}`, all ten frame SHA-256 digests
   byte-equal to the committed `oracle.json` (e.g. CartPole `4598212091e176d9`, SimplePointMaze
   `e520afaf03cb2b8a`, ...). Both old import paths (`swm_nav`, `simple_point_maze`) confirmed resolving
   to the merged module.
7. **Variation/state comparison** — Covered by the same oracle run (`describe()`, style layout,
   variation taxonomy, packed layout, `sample_states` all in the compared-keys set and all identical).
8. **Performance comparison for hot paths** — `performance.json` re-read and cross-checked:
   CartPole render −0.50%, TwoRoom render −0.12%, CartPole step +46.59% (+12 µs abs), TwoRoom step
   +28.39% (+20 µs abs). Matches findings.md exactly. The previously-found +137% regression
   (`_Classic.state` routing) is confirmed gone — `classic_control.py` no longer overrides `state`;
   `step` builds `StepResult` straight from `_sim` via `_state_td`.
9. **Resource-lifetime comparison** — `_Classic._close()` present, drops `_step_graph`, idempotent via
   base `Env.close`. EGL resource-guard counts unchanged (16 created/0 released) in the local unit run,
   consistent with a pre-existing Step-8 observation.
10. **Unexplained oracle differences** — **Zero**, confirmed by independent re-run, not just by
    reading the committed JSON.

## Commands run and tail output

```
$ MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/envs-classic-nav/oracle.py \
    --device cpu --out <tmp> --name oracle_review.json
...
"identical": true,
"unexplained_differences": {},
"old_module_paths_resolve": {"swm_nav": true, "simple_point_maze": true},
"functional_ok": true
  CartPole                 frames 4598212091e176d9  states   64
  ...
  SimplePointMaze          frames e520afaf03cb2b8a  states   64
ORACLE: PASS
```
All ten digests match `examples/results/step9/envs-classic-nav/oracle.json` exactly.

```
$ MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit \
    -x --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
1367 passed, 195 skipped, 112 deselected, 26619 warnings in 56.67s
```
(Brief's expected count of 1367 — matched; note this differs from findings.md's own quoted "1355
passed" transcript, see D1.)

```
$ MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
    tests/unit/test_envs_classic_nav.py tests/unit/test_variation_maps_dmc_control.py tests/swm
270 passed, 4088 warnings in 17.67s
```

GPU artifacts (read, not re-executed — no local CUDA):
- `oracle_6464309.json`: `identical: true`, `unexplained_differences: {}`.
- `oracle_head_cuda.json`: `compare_device: "cuda"`, `identical: true`, `unexplained_differences: {}`.
- `pytest_gfxdraw_classic.txt`: `80 passed`. `pytest_navdot_fused.txt`: `24 passed`.
- `pytest_conformance.txt`: `204 passed, 100 skipped, 1239 deselected`.
- `pytest_fast_paths.txt`: `42 passed, 30 skipped, 351 deselected`.
- `pytest_reproducibility.txt`: `18 failed, 288 passed, 66 skipped` — all 18 failures verified to be
  `DMCAcrobotSwingup`, `DMCCartpoleBalance`, `DMCCartpoleSwingup`, `DMCPendulumSwingup`,
  `MujocoInvertedPendulum`, `MujocoInvertedDoublePendulum` (× 3 tests each) — none are this
  workstream's ten cells. Root cause independently confirmed: `stable_envs/testing/render_worker.py:72`
  and `stable_envs/testing/conformance.py:1074` still guard with `hasattr(env, "step")`, which Wave 0
  made universally true; `git diff 3490156 60b515d` touches neither file (0 lines), confirming the
  branch correctly reported rather than fixed a shared-file defect (R9) it doesn't own.
- `grep -rl '/oscar/scratch/sboughan|/oscar/home'` over `examples/results/step9/envs-classic-nav/gpu/`:
  no hits — all paths scrubbed to `$SCR`.

## Spot-checks

- `_SWMPort` bridge merge: `stable_envs/envs/navigation.py:518` defines `_SWMPort(Env)` with
  `reference_kwargs`/`reference_render_args` as per-cell hooks and `reference_set_state`/
  `reference_sim_state` as the two per-cell overrides — matches the described duplication removal.
  Bridge behavior re-verified live via the `se8-vm-venv` run above (`tests/swm`, all passed).
- `StyleTensor.for_batch` replacing the hand-rolled `_style_for`: confirmed at
  `stable_envs/envs/navigation.py:605` and `:1485`; no remaining hand-rolled `_style_for` in the file.
- Derived `state_layout`: `classic_control.py`'s `_Classic.__init_subclass__` builds `state_layout`
  from `sim_state_names`/`render_from_sim`, with `_check_state_layout_is_complete()` raising at class
  creation for an incomplete layout — matches the claimed compile-time check.
- `tests/unit/test_foundation_registry.py` diff: the ruling is implemented exactly as described —
  semantic columns (all but `module`) compared for exact equality; `module`, when it changed, is
  checked by reachability (`getattr(old_module, name) is R.REGISTRY[name]`).
- `test_foundation_registry.py`/oracle.json's declared expected difference: exactly the five
  navigation cells, `module` only, both directions checked.

## Defects

**D1 (non-blocking, request changes).** `examples/results/step9/envs-classic-nav/findings.md` and
`oracle.json` are stale relative to the reviewed SHA `60b515d`:
- `oracle.json`'s `head_sha` field is `0af53ab2d0794aeb2ff3e5ad1a89d41b9c5eacf2` (the merge commit),
  not `60b515d`. The two commits after it (`c9a3ddf` retiring `_Classic.state`, `60b515d` filing R9)
  touch `classic_control.py`. I independently re-ran the oracle at the actual `60b515d` worktree state
  and it still reproduces `identical: true` with the same ten digests, so the *result* is not in
  question — but the committed JSON's provenance field understates what was actually verified, and a
  future auditor diffing `oracle.json.head_sha` against `git log` would flag a false mismatch.
- findings.md line 187's "exact output" transcript (`1355 passed ... in 54.83s`) is claimed as "the
  post-merge head" but does not match a fresh run at `60b515d`, which produces **1367 passed**
  (`test_envs_classic_nav.py` alone collects 118 tests, not the 88/108 quoted in §1/§4/top-of-file).
  The extra tests (e.g. `test_step_returns_exactly_env_state`) were added in `c9a3ddf` per the code
  comment in `classic_control.py`'s `step()`, but findings.md's counts and pasted transcript were never
  refreshed to match. Section 11 separately claims "1357 local tests" for the same two commits, which
  is a third, still-incorrect number.

Recommend: regenerate `oracle.json` at `60b515d` (or explicitly note in findings.md that it is
carried forward from `0af53ab` and why that's sound, as is already done for the GPU evidence at
`b8cc599`), and refresh the test-count accounting and pasted transcript in findings.md to the actual
final-head numbers. Purely a documentation-accuracy issue — every number I could independently
reproduce matched or exceeded what the branch claims; nothing in the diff is behaviorally wrong.

No other defects found. R7 (`_live_state` hook) and R8 (`unpack` aliasing) are foundation fixes
correctly identified, escalated, and (per the merge history) resolved upstream rather than patched
in this workstream's own files. R9 is correctly identified as a foundation-owned defect and correctly
left unfixed here.
