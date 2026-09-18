APPROVE

# Step 9 — independent review of Workstream F (MuJoCo wrappers and non-42-id families)

**Reviewed** `step9/envs-mujoco` at **`6a52607b9be62953865376f3bb9213e212cbe1be`** (author delivery through
`88aa37f`, then the coordinator's merge of `step9/integration` `dc887b8`). F-attributable delta:
`git diff step9/integration...HEAD` (31 files, +23,693/−97). Step-8 behavioural baseline
`21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.
**Reviewer host** Apple M5 Mac `cis1b1zs2j3.ad.brown.edu`, local CPU only, no CUDA, no Oscar, no
GPU job. `MUJOCO_GL=glfw`, `-p no:cacheprovider`. Interpreters `/Users/samib/se6-venv` (unit suite +
20-cell arm, with `SE9_EXTRA_PYTHONPATH` the four uv-cache archives) and `/Users/samib/se8-fetch-venv`
(maze2d). Nothing was edited, stashed, reset, rebased or pushed except this review file. The oracle
harness created and removed its own detached `git worktree` of `21cb306`; it cleaned up.

## Verdict

**APPROVE.** The F-attributable delta is exactly the seventeen family modules, `tests/unit/test_envs_mujoco.py`,
the authorized `NUM_ENVS_HOLDOUTS` 14 → 0 edit, and this workstream's own `scratch/` and
`examples/results/step9/envs-mujoco/` paths. No foundation file is touched. Live `num_envs=` holdouts
are **0**. I re-ran a representative two-tree CPU oracle at `6a52607` (six se6-venv cells plus
`PointMazeOpen`) into a fresh temp dir: **identical: true, unexplained_differences: {},
functional_ok: true**, and every frame digest matches the committed table. Targeted tests: **121
passed, 16 skipped** in se6-venv (the 16 are the four PointMaze cells, which this interpreter cannot
construct) and **20 passed** for PointMaze in se8-fetch-venv. Coordinator independently ran the
prescribed unit suite at this tip: **1691 passed, 279 skipped, 112 deselected**. Render timings are
under the 1.10 bar. Step-path ratios exceed 1.10 only because the baseline steps are 75–90 µs; the
absolute overhead is 10–21 µs, inside ruling 7's accepted StepResult band, and `blocking_configs` is
empty. CUDA-only and missing-package gaps are labelled UNESTABLISHED, not satisfied. REGRESSION
count on everything a CPU can measure: **0**.

Two documentation nits below; neither blocks.

## F-attributable delta

`git diff --name-only step9/integration...HEAD` is 31 paths: the seventeen family modules, the two
test files, three scratch scripts, nine artifacts under `examples/results/step9/envs-mujoco/`.
`git diff --name-only step9/integration...HEAD -- stable_envs/env.py stable_envs/dynamics.py
stable_envs/specs.py stable_envs/registry.py stable_envs/_config.py stable_envs/__init__.py
pyproject.toml` is empty. The sole shared edit is `tests/unit/test_foundation_api.py`:
`NUM_ENVS_HOLDOUTS = 14` → `0` plus the comment that F's fourteen (highway 6 / maze2d 4 /
mujoco_locomotion 4) are gone. That is the authorized ratchet, not a foundation-API change.

The merge commit `6a52607` itself only brings integration's Craftax-module-conflict evidence
(`craftax_full.py` + E tests/findings). It does not retouch F family code. F's last applicable code
change is `c0ea6f1` (State snapshot); `oracle_dynamics.json` covers it.

## The ten §17 items — my own numbers

1. **Files moved/removed/created.** Nothing moved, nothing removed. Seventeen modules edited in
   place (`StableEnv` → `Env`, `register` from `registry`, `num_envs=`, State layout, `close` /
   `_close`, `frame_shape` on five, `_advance_into` + `transition`/`step` on MiniGrid/Sokoban).
   Created: `tests/unit/test_envs_mujoco.py`, `scratch/step9/envs-mujoco/{oracle.py,bench.py,gpu_oracle.sbatch}`,
   nine files under `examples/results/step9/envs-mujoco/`. Confirmed against the diff stat, not the
   prose.

2. **API changes.** Additive: `num_envs=` on every class that used to declare `batch=` (and `batch=`
   still forwarded to `Env.__init__`); public `State` via `state_layout` / `_hidden_state`;
   `step`/`transition`/`unroll` on the seven steppable cells; `close()` on all seventeen, `_close` in
   eight; `frame_shape` on highway, minihack, lunarlander, overcooked, griddly. No `state` property
   override in any of the seventeen files (grep: only `state_from_blob` classmethods in `procgen.py`).
   No `__del__` added; griddly's existing wrapper-`__del__` policy is unchanged and documented.

3. **Step-8 IDs affected.** None of the 42. I enumerated `stable_envs_cells` from
   `examples/results/step9/oracle/step8_support_matrix.json` (43 names) against the 24 F classes this
   host registers: intersection **empty**. This patch cannot move the support matrix, Gate-F matrix
   or tier table.

4. **Targeted tests.** Re-run by me:

   ```
   SE9_EXTRA_PYTHONPATH=<four uv archives>
   MUJOCO_GL=glfw se6-venv pytest tests/unit/test_envs_mujoco.py \
       tests/unit/test_foundation_api.py::test_the_number_of_classes_that_still_reject_num_envs_only_goes_down
   -> 121 passed, 16 skipped in 629.32s
      resources: 4 EGL contexts created, 0 released, 4 live (the four gymnasium-MuJoCo cells;
                 shared caches deliberately not released — findings §9)

   MUJOCO_GL=glfw se8-fetch-venv pytest tests/unit/test_envs_mujoco.py -k PointMaze
   -> 20 passed, 116 deselected in 1.47s
      resources: 4 EGL contexts created, 0 released, 4 live
   ```

   The 16 se6 skips are exactly the four PointMaze cells × the four construction tests (layout /
   render / refuse / close); the signature check still ran for them. Fetch-venv constructs all four.
   Together that is 24/24 constructible cells on this host. Coordinator suite (not re-run by me):
   **1691 passed, 279 skipped, 112 deselected**.

5. **Applicable certification checks.** No frozen Step-4 corpus exists for any of these seventeen
   families (confirmed: the matrix/cell intersection is empty and `examples/results/step4/states/`
   holds none of these names). `check_state_roundtrip` is inside the oracle (`roundtrip_exact` true
   on the committed 24 and on my re-run). GPU conformance / reproducibility / fast-paths are owed
   to the post-Wave-1 job (ruling 8); `scratch/step9/envs-mujoco/gpu_oracle.sbatch` is prepared and
   not submitted.

6. **Default-frame comparison — re-run by me at 6a52607, not read off the committed JSON.**

   ```
   se6-venv oracle.py --cells MiniGridEmpty,Sokoban,Highway,Parking,MujocoReacher,Crafter
   head_sha 6a52607  baseline 21cb306  identical true  unexplained {}  functional_ok true
     MiniGridEmpty   28908c0eb8dd9ae7  size 256
     Sokoban         d54522ac1d324b99  size 160
     Highway         543ebae65390532f  size 150
     Parking         5e9997d80cf4c7d7  size 300
     MujocoReacher   435d981d598f460d  size 240
     Crafter         8d7adb9a16cef759  size 64
   ORACLE: PASS

   se8-fetch-venv oracle.py --cells PointMazeOpen
     PointMazeOpen   24af67f92b2a84a2  size 224
   ORACLE: PASS
   ```

   Every digest above is the digest `findings.md`'s table claims. I also flattened the committed
   `oracle.json` / `oracle_maze2d.json` / `oracle_dynamics.json`: 20 + 4 + 7 cells, all
   `identical: true`, `unexplained_differences: {}`, and all 24 findings-table prefixes match.
   `oracle_dynamics.json` at `c0ea6f1` still carries the same MiniGrid/Sokoban frame hashes, so the
   State-snapshot change moved no pixel. The six-cell re-run at `6a52607` (after the later
   integration merge) moved none either.

7. **Variation/state comparison.** Covered by the same oracle: `describe()`, style layout/default
   appearance, taxonomy, `state_dim`/`state_names`, round-trip, `sample_states`, fidelity
   declarations, registry row including `module` (exact, not reachability — nothing moved). Zero
   compared keys moved in the committed records; my re-run reported `unexplained_differences: {}`.
   Live: `pack(state) == get_state()` is asserted in `test_the_declared_state_layout_tiles_the_packed_row`
   and passed on every constructible cell.

8. **Performance.** I did not re-bench (the render path is declarations + G's already-approved 2-D
   backends; the step path is the snapshot already covered by `oracle_dynamics.json` and the
   non-aliasing test). I did recompute every stored ratio from the per-round arrays:

   * `timings_render.json` worst **1.0273** (Crafter), `blocks_merge: false`.
   * `timings_maze2d.json` worst **1.0181** (PointMazeLarge), `blocks_merge: false`.
   * `timings_render_selftime.json` worst **1.0245**, same-tree floor.
   * `timings_step.json` (preserved pre-optimization checkpoint) worst **1.5338**, `blocks_merge: true`
     — the copy cost they then fixed.
   * `timings_step_merged.json` worst **1.2804** (MiniGridDoorKey, +21 µs), Sokoban **1.1065** (+10 µs).
     `accepted_stepresult_overhead_ms: 0.028`, `blocking_configs: []`, `blocks_merge: false`.

   Stored `ratio_head_over_baseline` is computed from unrounded medians, then `baseline_ms`/`head_ms`
   are rounded to 3 decimals; reconstructing the ratio from the rounded fields differs in the 4th
   digit (e.g. MiniGridEmpty 1.2042 stored vs 1.2133 from 0.091/0.075). That is rounding, not a
   cooked number. Worst-ratio fields match `max(rows.ratio)` exactly.

   Plan §8 / ruling 7: the 10 % bar is for render/dynamics hot paths; StepResult tax of +12–28 µs on
   cheap steps is accepted. Render is inside 1.10. The step ratios look large because the baseline
   is 75–90 µs; the absolute cost is 10–21 µs, inside the accepted band. `bench.py` blocks a step
   config only if it exceeds **both** 1.10 **and** 28 µs. That dual gate is documented in the record
   and matches the ruling; it is not a silent bar-weakening.

9. **Resource lifetime.** `close()` is idempotent and a closed env refuses to render — passed on all
   24 constructible cells. `_close` is in the eight families that hold per-instance tables; shared
   caches (atlases, plates, griddly wrappers, MJWarp/EGL) are deliberately retained. My se6 run's
   safety plugin: **4 EGL contexts created, 0 released, 4 live** — all from the four gymnasium-MuJoCo
   cells, none released, which is the standing never-release rule. Workstream I's consumer change
   list is correctly not applied here (R2: line numbers have moved; assign by symbol after merge).

10. **Unexplained oracle differences: zero** on CPU over the 24 committed cells and over the 7 I
    re-ran at `6a52607`. **UNESTABLISHED**, not satisfied, for `metaworld` / `robocasa` /
    `humanoidbench` (CUDA-only, mujoco_warp) and for the seven CPU-capable families whose optional
    package is absent from every venv on this host (`carracing`, `lunarlander`, `griddly`, `jumanji`,
    `overcooked`, `pgx`, `procgen`). That split is honest: I independently listed `snv.import_errors()`
    and got exactly those ten F families, with the CUDA-only three naming mujoco_warp and the other
    seven naming the missing package.

## Contract / ruling checks

**`num_envs=` ratchet = 0.** Live, not read off the constant:

```
holdouts = [n for n, c in REGISTRY.items()
            if "batch" in signature(c.__init__).parameters
            and "num_envs" not in signature(c.__init__).parameters]
-> []   # len 0
```

The ratchet test passed in my run. 24 owned classes are registered in se6-venv with extra
pythonpath; the ten missing families appear only as `import_errors`.

**Ruling 2 (no `state` override).** Grepped all seventeen owned files for `def state`: no hits
other than `procgen.state_from_blob`. MiniGrid's carried object uses `_hidden_state` (offset
`None`), which I exercised via `test_minigrids_carried_object_is_hidden_state_the_packed_row_does_not_carry`
(passed).

**Ruling 3 (`unpack` copies).** None of the seventeen override `unpack`. The non-aliasing assertion
in `test_transition_is_pure_and_bit_identical_to_the_stateful_step` passed for all seven steppable
cells: a returned State is not a view of later `step`s.

**`transition` / `step` / `unroll` purity.** MiniGrid and Sokoban both call one `_advance_into`
over caller-named buffers. `transition` clones the packed row (and MiniGrid's carry) and does not
write `env.get_state()` — asserted live. `unroll` equals four `transition`s and four `step`s —
passed. `step` returns exactly `["state"]` (no fabricated reward/`done`) — passed. Renderer-only
cells raise `NotImplementedError` naming the class from **both** `step` and `transition` — passed
on the 17 renderer-only constructible configs (20 se6 constructible minus 7 steppable, plus 4
PointMaze in fetch-venv).

**`frame_shape`.** `Env.observation_spec` (`env.py:1074`) defaults to `(size, size)`. I read the
five declarations against the render reshape they claim to match:

* highway: `screen().H, screen().W` (Parking is 300×600; my oracle rendered Parking at size 300).
* minihack: `GRID_H * size, GRID_W * size` = 21×79 tiles; render writes `(n, GRID_H*t, GRID_W*t, 3)`.
* lunarlander: `native = (VIEWPORT_H, VIEWPORT_W) = (400, 600)`; `render` raises for `size != 400`.
* overcooked / griddly: `(grid_h * T, grid_w * T)` with `T = _tile(size)`; both `render` docstrings
  say that reshape.

`test_render_takes_a_state_a_packed_row_or_nothing` compares `observation_spec.shape` to a real
frame and passed on every constructible cell, including highway and minihack. lunarlander /
overcooked / griddly remain statically declared until stage 3 of the sbatch has the packages; that
limitation is stated, not hidden.

**Close semantics.** Idempotent `close()` + closed env refuses render + context manager, passed on
all 24. No new `__del__`.

## GPU / optional-package gaps (ruling 8)

I did not submit the sbatch and did not ssh to Oscar. The script is a section, not a claim of
results: stage 0 inventories per venv with the `ImportError` text; stage 1 runs the two-tree oracle
on cpu **and** cuda over discovered cells; exit 5 is a warning; `ORACLE_FAIL` is tracked via
`PIPESTATUS[0]`. HumanoidBench is opted in with `STABLE_ENVS_HUMANOIDBENCH=1`. That is the right
shape for the coordinator's single post-Wave-1 GPU job. R3/R4 in the findings stand: if stage 0
cannot build the seven CPU-capable optional-package families on the cluster either, that is a
DOCUMENTED_LIMITATION for §20, not an F omission; the dashboard must not say "identical" for the
three CUDA-only families.

## Defects (documentation only, neither blocking)

**F-1.** `MiniGridTopDown.step`'s docstring still says the carried object "is not in the returned
state — see `sim_state()`". `_step_state` (the hot-path snapshot used by `step`) **does** put
`agent.carrying` in the returned TensorDict, and `_hidden_state` splices it into `env.state`. The
packed row still does not carry it (offset `None`), which is the claim the findings and the unit
test pin. Behaviour is correct; the docstring is stale. Fix on a docs pass, not this merge.

**F-2.** Findings §2 says "`num_envs=` on every class of the **nine** families whose own `__init__`
declared `batch=`" and then lists highway, maze2d, mujoco_locomotion **plus** humanoidbench,
metaworld, jumanji, lunarlander, griddly, overcooked, pgx, procgen. The live ratchet was fourteen
**classes** in the three families that actually register on a CPU host; the others needed the same
signature change for when they import. The code matches the ratchet, the word "nine" does not.
Harmless.

No other defects found. Coordinator requests R1 (ratchet 0) is done; R2–R5 are correctly not F's
to execute.

Nothing above changes the verdict: **APPROVE**.
