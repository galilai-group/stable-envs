# Step 9 — Wave 1, Workstream F: the MuJoCo wrappers and the non-42-id families

**Workstream:** `step9/envs-mujoco` · **Parent:** `step9/integration`, merged at `70c7055`
(which brought Workstream B's DMControl package and Workstream G's 2-D render consolidation) ·
**Step-8 baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Host:** Apple M5 Mac, CPU only — no CUDA device in this ticket, and by ruling 8 no GPU job either.
**Interpreters:** `/Users/samib/se6-venv` (torch 2.14 CPU, mujoco 3.12.0, tensordict, loguru) and
`/Users/samib/se8-fetch-venv` (+ gymnasium-robotics 1.4.2). `MUJOCO_GL=glfw`, `-p no:cacheprovider`.
Four optional reference packages come off the uv cache on `SE9_EXTRA_PYTHONPATH`; see §6a.

**Result: PATCH.** Seventeen families migrated onto the Wave-0 foundation. The **24 cells this host
can construct render byte-identical frames** through this branch and through a detached worktree of
`21cb306` in the same venv, and `describe()`, the style layout, the default appearance, the variation
taxonomy, the fidelity declarations, the packed layout, `set_state`/`get_state` round-tripping,
`sample_states` and the registry row (module included) are identical on every one of them.
**Unexplained oracle differences: zero.** The 24 digests are also bit-for-bit the ones this branch
produced *before* the integration merge, so **Workstream G's 2-D backend consolidation moved no pixel
of these families**. Ratchet `NUM_ENVS_HOLDOUTS` **41 → 27**.

**What is NOT established and must not be read as established:** ten of the seventeen families own
no cell this host can build, for two different reasons (§6b). Three of them — `metaworld`,
`robocasa`, `humanoidbench` — are **CUDA-only by construction** (mujoco_warp's batch raytracer, no
CPU backend), so their frames were never rendered on this branch at all: their oracle is
**UNESTABLISHED, not satisfied**. `scratch/step9/envs-mujoco/gpu_oracle.sbatch` is prepared, not
submitted, for the coordinator's single post-Wave-1 GPU job.

**The seventeen families:**

| family | cells this host builds | where |
|---|---|---|
| `mujoco_locomotion` | 4 — MujocoInvertedPendulum, MujocoInvertedDoublePendulum, MujocoReacher, MujocoPusher | se6-venv |
| `highway` | 6 — Highway, Merge, Roundabout, Racetrack, Parking, Intersection | se6-venv |
| `minigrid` | 7 — MiniGridEmpty, MiniGridDoorKey, MiniGridFourRooms, MiniGridLavaCrossing, MiniGridKeyCorridor, BabyAIGoToLocal, MiniGridPOVFetch | se6-venv + `minigrid` 3.1.0 |
| `crafter` | 1 — Crafter | se6-venv |
| `minihack` | 1 — MiniHack | se6-venv |
| `sokoban` | 1 — Sokoban | se6-venv + `gym_sokoban` |
| `maze2d` | 4 — PointMazeOpen, PointMazeUMaze, PointMazeMedium, PointMazeLarge | se8-fetch-venv |
| `metaworld`, `robocasa`, `humanoidbench` | 0 — **CUDA-only** (mujoco_warp) | owed to the GPU job |
| `carracing`, `lunarlander`, `griddly`, `jumanji`, `overcooked`, `pgx`, `procgen` | 0 — CPU-capable, optional package absent from every venv on this host | owed to the GPU job |

---

## 1. Files moved / created / removed

**Nothing moved and nothing was removed.** No file in this workstream is renamed, split into a
package or aliased through `sys.modules` — unlike B (`dmcontrol/`), D (`ogbench/`) and A
(`swm_nav.py`). Seventeen modules were edited in place, which is why the oracle can compare the
registry row's `module` column **exactly** rather than by reachability (ruling 5): no consumer of
these families has an import path to update.

| file | edit |
|---|---|
| `stable_envs/envs/mujoco_locomotion.py` | imports, `num_envs=` (4 cells), public State, `_hidden_state`, `_declare_state` folded |
| `stable_envs/envs/humanoidbench.py` | imports, `num_envs=`, public State, `_hidden_state` |
| `stable_envs/envs/metaworld.py` | imports, `num_envs=`, public State (two halves), `_hidden_state` |
| `stable_envs/envs/robocasa.py` | imports, `num_envs=`, public State |
| `stable_envs/envs/maze2d.py` | imports, `num_envs=` (4 cells), public State, `_hidden_state` |
| `stable_envs/envs/highway.py` | imports, `num_envs=` (6 cells), State layout, `_close`, **`frame_shape`** |
| `stable_envs/envs/minigrid.py` | imports, `num_envs=`, State layout, `_hidden_state`, `_advance_into`, `transition`, `step` → StepResult, `_close` |
| `stable_envs/envs/sokoban.py` | imports, State layout, `_hidden_state`, `_advance_into`, `transition`, `step` → StepResult, `_close` |
| `stable_envs/envs/crafter.py` | imports, State layout |
| `stable_envs/envs/minihack.py` | imports, State (trivial), **`frame_shape`** |
| `stable_envs/envs/carracing.py` | imports, State layout |
| `stable_envs/envs/lunarlander.py` | imports, `num_envs=`, State layout, `_close`, **`frame_shape`** |
| `stable_envs/envs/griddly.py` | imports, `num_envs=`, `_close`, **`frame_shape`** |
| `stable_envs/envs/overcooked.py` | imports, `num_envs=`, `_close`, **`frame_shape`** |
| `stable_envs/envs/jumanji.py` | imports, `num_envs=` |
| `stable_envs/envs/pgx.py` | imports, `num_envs=`, `_close` |
| `stable_envs/envs/procgen.py` | imports, `num_envs=`, State layout, `_close` |

Created: `tests/unit/test_envs_mujoco.py`, `scratch/step9/envs-mujoco/{oracle.py,bench.py,gpu_oracle.sbatch}`,
`examples/results/step9/envs-mujoco/{oracle.json,oracle_maze2d.json,timings_*.json,findings.md}`.
Edited outside the family files: `tests/unit/test_foundation_api.py` (the ratchet constant and its
comment — one number and a note, §2).

**No foundation file is touched.** `git diff --name-only step9/integration...HEAD` is exactly the
seventeen family modules, the two test files and this workstream's own `scratch/` and
`examples/results/` paths. `env.py`, `dynamics.py`, `specs.py`, `registry.py`, `_config.py`,
`__init__.py` and `pyproject.toml` are untouched by this branch (they carry W0/A/B/G changes that
arrived through the merge).

## 2. API changes

**Additive only. Every Step-8 spelling still works.**

1. **`num_envs=`** on every class of the nine families whose own `__init__` declared `batch=`:
   `highway` (6 cells), `maze2d` (4), `mujoco_locomotion` (4), plus `humanoidbench`, `metaworld`,
   `jumanji`, `lunarlander`, `griddly`, `overcooked`, `pgx`, `procgen`. `batch=` is still accepted
   and deprecated by `Env.__init__` — these classes previously forwarded positionally and never
   reached its deprecation branch, so they *rejected* the Step-9 constructor with `TypeError`.
2. **The public State** (contract §3). Where the packed row already had structure it is named;
   where it does not, nothing is invented (§10, "deliberate non-decisions"). No column moves, no
   field is dropped, `state_names` is unchanged, `pack(state) == get_state()` on all 24 cells.
3. **`step`/`transition`/`unroll`** on the seven steppable cells (six MiniGrid top-down configs and
   Sokoban): `step` now returns a `StepResult` TensorDict carrying `"state"` and nothing else, and
   `transition`/`unroll` are new and pure.
4. **`close()` / context manager** on all seventeen; `_close` where an instance actually holds
   something (8 families).
5. **`frame_shape`** on five families (§7): `highway`, `minihack`, `lunarlander`, `overcooked`,
   `griddly`. This is a *read* of the Wave-0 hook at `env.py:1074`, not a change to it.

**Two Step-8 behaviours deliberately preserved, not "improved":**

* `step` reports **no reward and no `done`** on any of the seven steppable cells. MiniGrid's
  reference reward is `1 - 0.9 * step_count / max_steps` and neither env holds a step counter, so a
  number under that name would be a different quantity wearing it. Contract §5 says omit rather than
  fabricate, and that is what these do.
* MiniGrid's **carried object stays out of the packed row** (offset `None`, answered by
  `_hidden_state`), exactly as PushT's agent velocity. A stored trajectory therefore does not
  determine its own continuation for that one field; it does determine every frame, which is the
  claim this repo makes. Putting it in the row would have moved a frozen layout.

## 3. Step-8 IDs affected

**None of the 42.** Checked, not assumed: the 42 rows of
`examples/results/step9/oracle/step8_support_matrix.json` name 43 `stable_envs_cells`, and the
intersection with this workstream's cells is **empty**. That is the "non-42" in this workstream's
name, and it has a consequence worth stating plainly for the §20 closure comparison: **this patch
cannot move the Step-8 support matrix, the Gate-F matrix or the tier table**, because no row of any
of them describes a cell in these seventeen families. The oracle here is therefore not a check
against a frozen matrix row (there is none) but a two-tree diff — §6.

Registry rows compared and identical for all 24 cells: `name`, `tier`, `decoder`, `state_dim`,
`reference_id` **and `module`** (exactly, not by reachability — nothing moved).

## 4. Targeted tests

`tests/unit/test_envs_mujoco.py` (282 lines, 9 test functions). It pins the claims that are **NEW in
Step 9** and so have no Step-8 baseline to diff against:

| test | what it pins |
|---|---|
| `test_this_workstream_owns_some_registered_classes` | a guard on the guard: every test below parametrises over the discovered class list and would pass vacuously on an empty one |
| `test_every_owned_class_takes_the_step9_constructor` | `num_envs=` reaches the class, read from the **signature** — no construction, so a family whose optional dep is missing is still enforced rather than skipped |
| `test_the_declared_state_layout_tiles_the_packed_row` | every declared field is a State key, the offsets tile `state_dim` exactly, `pack`/`unpack` round-trip |
| `test_render_takes_a_state_a_packed_row_or_nothing` | the three spellings are the same frame **and `observation_spec` describes it** — this is the test that found the `frame_shape` defect |
| `test_a_cell_without_a_simulator_refuses_step_and_transition_by_name` | contract §5: `NotImplementedError` naming the class, from **both** `step` and `transition`, while `reset`/`render` keep working |
| `test_close_is_idempotent_and_a_closed_env_refuses_to_render` | `close()` twice is fine; a closed env raises; the context manager closes |
| `test_transition_is_pure_and_bit_identical_to_the_stateful_step` | the env's buffers are untouched and the two results agree bitwise |
| `test_unroll_is_repeated_transition_and_repeated_step` | `unroll` equals four `transition`s and four `step`s |
| `test_minigrids_carried_object_is_hidden_state_the_packed_row_does_not_carry` | the one hidden field is hidden deliberately and reachable through `_hidden_state` |

**Its honest coverage.** `_owned_classes()` reads the **registry**, so a family whose module failed
to import contributes nothing — which means the signature check, too, reaches only the classes that
registered. On this host that is 24 cells, and the run is **72 passed / 57 skipped**: the skips are
the ten families with no local package, named individually in each skip message. Stage 3 of
`gpu_oracle.sbatch` runs this same file in every Oscar venv, which is where those skips become
results. I did **not** reach for an AST walk to enumerate classes in a module that does not import;
that is a second discovery mechanism for one property, and §16 says no.

Full local suite (command from the handoff §1, run in `se6-venv`):

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
```

<!--SUITE-->

## 5. Applicable certification checks

The Step-4 frozen corpus (`examples/results/step4/states/`) holds 49 corpora and **not one of them
belongs to a cell in these seventeen families** — every one is PushT, classic control, navigation,
DM Control, Fetch, OGBench or Craftax. There is consequently **no frozen replay to run here**, and
inventing a corpus would freeze something Step 4 deliberately did not. What applies instead:

* `check_state_roundtrip` — run as part of the oracle (`roundtrip_exact`, true on all 24);
* the **declaration** checks (`tests/unit/test_declarations.py`) and the repo contracts, which run
  in the suite above and pass;
* the fidelity declarations (`parity_interior_min`, `parity_all_exact_min`, `fidelity_tier`,
  `render_decoder`) — compared between trees by the oracle, identical on all 24;
* `tests/gpu/test_conformance.py` / `test_reproducibility.py` / `test_fast_paths.py` for these
  families: **not runnable here** (no CUDA device, and most of the packages are absent). Stage 4 of
  the sbatch selects them.

## 6. The oracle — default frames, variation and state comparison

`scratch/step9/envs-mujoco/oracle.py` → `examples/results/step9/envs-mujoco/oracle.json` (20 cells,
`se6-venv`) and `oracle_maze2d.json` (4 cells, `se8-fetch-venv`). Both records at head `a466756`,
baseline `21cb306`, `identical: true`, `unexplained_differences: {}`, `functional_ok: true`.

**Where the states come from, and why this differs from A's and D's scripts.** No cell here has a
frozen Step-4 corpus (§5). So the **baseline tree** samples `sample_states(16, default_rng(0))` for
every cell and writes them to a temp `.npz`, and **both arms render that array**: the frames are
compared over identical inputs, exactly as a frozen corpus would be. The reset distribution is then
compared *separately* by having each arm sample its own and hashing it (`sample_states_sha256`), so
a changed distribution shows up as a difference in that key rather than silently changing what was
rendered. That is strictly stronger than letting each tree sample for itself.

**The 21 keys compared, per cell:** `frames_sha256` (the frames, byte for byte, over the raw uint8
bytes), `frames_from_rows_sha256` (rendered from explicit packed rows instead of the held state),
`frames_shape`, `rows_sha256`, `state_dim`, `state_names`, `size`, `roundtrip_exact`, `describe`,
`style_names`, `style_dim`, `default_style_sha256`, `registry_row`, `sample_states_sha256`,
`variation_names` (the taxonomy leaf for leaf), `parity_interior_min`, `parity_all_exact_min`,
`fidelity_tier`, `render_decoder`, `device`, `states`.

| cell | size | frames sha256 (16 states) | cell | size | frames sha256 (16 states) |
|---|---|---|---|---|---|
| MujocoInvertedPendulum | 240 | `ecdcff25fadc2523` | MiniHack | 16/tile | `685af1bc78e912b3` |
| MujocoInvertedDoublePendulum | 240 | `6b0243df9ba2e92a` | MiniGridEmpty | 256 | `28908c0eb8dd9ae7` |
| MujocoReacher | 240 | `435d981d598f460d` | MiniGridDoorKey | 256 | `2daddc1446bf8782` |
| MujocoPusher | 240 | `b4abce61d20bcfda` | MiniGridFourRooms | 608 | `84fb54c6a48ab251` |
| Highway | 150 | `543ebae65390532f` | MiniGridLavaCrossing | 288 | `736aeff968152abb` |
| Merge | 150 | `92f9a8490e74821e` | MiniGridKeyCorridor | 320 | `2eb1a2b562b449c4` |
| Roundabout | 600 | `4998e0e83b7e230c` | BabyAIGoToLocal | 256 | `fd76b5ba1b2d0b26` |
| Racetrack | 600 | `d830e2322fb25141` | MiniGridPOVFetch | 224 | `6e42fe8227472646` |
| Parking | 300 | `5e9997d80cf4c7d7` | Sokoban | 160 | `d54522ac1d324b99` |
| Intersection | 600 | `4b1bdc078ac925eb` | PointMazeOpen | 224 | `24af67f92b2a84a2` |
| Crafter | 64 | `8d7adb9a16cef759` | PointMazeUMaze | 224 | `ca6ba836791bc7f6` |
| | | | PointMazeMedium | 224 | `f098efcc6a9eec81` |
| | | | PointMazeLarge | 224 | `c997148c2f1037f8` |

**The merge did not move a frame, and that was the reason for re-running.** Nine of these families
render through the 2-D backends Workstream G consolidated (`tilemap`, `raster2d`, `scene`, `gfxdraw`,
`navdot`, `prng`), so the merge could in principle have moved pixels. Diffing this record against
the pre-merge one committed at `3c83422` over `frames_sha256`, `frames_from_rows_sha256`,
`rows_sha256`, `default_style_sha256`, `sample_states_sha256`, `state_dim`, `size` and
`frames_shape` gives **0 keys moved across 24 cells**. Classification: not applicable — there is no
difference to classify. G's tile law and Wu edge law are byte-compatible with what these families
were already drawing.

**The Step-9 functional claims**, checked in the migrated tree only (there is nothing to compare
them against), on all 24 cells:

* `pack(state) == get_state()`; `render(State) == render(rows) == render()` — **24/24**;
* the 17 renderer-only cells raise `NotImplementedError` naming the class from **both** `step` and
  `transition`, and `has_dynamics` is False — **17/17**;
* the 7 steppable cells: `transition` leaves the env's buffers untouched; `transition` is bitwise
  equal to `step` on every State key **and** on the packed row; `unroll` equals repeated
  `transition` **and** four stateful `step`s; `step` returns exactly `["state"]` — **7/7**;
* `close()` is idempotent and a closed env refuses to render — **24/24**.

### 6a. The four uv-cache archives, and a reproducibility defect fixed

The `se6-venv` arm only ever ran because four unpacked wheels were on `SE9_EXTRA_PYTHONPATH`, and
**the record committed at `3c83422` did not say so**. A fresh shell reproduces nothing: the run dies
with `ModuleNotFoundError: No module named 'minigrid'` *inside the baseline tree*, because
`render/tilemap.official_colors()` reads `COLOR_TO_IDX`/`COLORS` off the package and `make_obj`
builds real MiniGrid objects with it; `Sokoban` then fails to register at all because
`gym_sokoban`'s seven 16×16 PNGs **are** its frame and importing it executes `gym` and
`pkg_resources`. None of the four is installed in any venv on this host, and `setuptools` 84 no
longer ships `pkg_resources`, so the chain is `minigrid` → `gym_sokoban` → `gym` → `pkg_resources`.

`oracle.py` now resolves the four uv-cache archives itself (`--print-extra-pythonpath`), fails with
the package's name if the cache has been rebuilt instead of dying ten minutes later with a
`KeyError`, and records the resolved paths in the JSON under `extra_pythonpath`. That last part
matters: the cache holds **both** `minigrid` 3.0.0 and 3.1.0, and the pin is 3.1.0
(`docs/research/minigrid.md`), so without the record a reader cannot tell which package produced the
digests. **Nothing was installed into the shared `se6-venv`.**

### 6b. What is not covered, separated by reason

Ten families own no cell this host can build. They are **not** one category, and the sbatch keeps
them apart:

**CUDA-only by construction — oracle UNESTABLISHED, not satisfied:**

| family | why |
|---|---|
| `metaworld` | ray-casts the Sawyer's meshes with `mujoco_warp`'s batch raytracer; no CPU backend exists |
| `robocasa` | the same, for its kitchens (and `robocasa` pins mujoco 3.3.1, so the interpreter that *builds* a kitchen is never the one that renders it) |
| `humanoidbench` | the same; additionally **not registered by default** — `mujoco_warp`'s batched render is not bit-reproducible inside a long conformance session, so 2–3 of ~660 tests fail every run by 1–2/255 on a different cell each time. `STABLE_ENVS_HUMANOIDBENCH=1` registers it; the sbatch exports it |

**CPU-capable, optional package absent from every venv on this host:** `carracing` and
`lunarlander` (Box2D: `box2d-py` for the states *and* `pygame` for the reference renderer),
`griddly` (`griddly==1.6.7`, the only release, 2023-03-15, no successor), `jumanji`
(`jumanji`+`jax`+`matplotlib` — it builds its atlas by *calling* the official matplotlib viewer),
`overcooked` (`jaxmarl`, whose `_render_tile` output *is* the atlas), `pgx`
(`pgx`+`resvg_py`+`Pillow`), `procgen` (`procgen`, which supplies both the packed assets and the
reference). These are here because that is the tree with the packages, not because they need a GPU.

Nothing was installed to close this; every one of these families says in its own `ImportError` that
there is **no degraded mode** — the reference assets *are* the frame.

## 7. `frame_shape`: a real defect, found once by a test and then four more times by reading

`Env.observation_spec` (`env.py:1074`) reads `getattr(self, "frame_shape", (size, size))` — i.e. it
assumes a **square** frame unless the env says otherwise. **Five of this workstream's families are
not square**, and an unqualified spec told every consumer to allocate the wrong buffer:

| family | true frame | what the spec said | found by |
|---|---|---|---|
| `highway` | `(sc.H, sc.W)`; Parking is 300×600 | `(300, 300)` | `test_render_takes_a_state_a_packed_row_or_nothing`, against a real render |
| `minihack` | `(21·size, 79·size)` = 336×1264 at the default tile 16 | `(16, 16)` — `size` is the **tile** size here | the same test |
| `lunarlander` | always `native` = `(400, 600)`; `render` *raises* for any other size | `(400, 400)` | **reading** (§7a) |
| `overcooked` | `(grid_h·T, grid_w·T)`, `T = size // grid_h`; no shipped layout is square | `(size, size)` | **reading** |
| `griddly` | `(grid_h·T, grid_w·T)`; the pinned levels are wider than tall | `(size, size)` | **reading** |

### 7a. Why the last three had to be found by reading, which is the generalisable lesson

The test that caught `highway` and `minihack` compares `observation_spec` against a frame it actually
renders. It **could not** catch the other three: it skips a family whose optional dependency is
absent, and Box2D, `jaxmarl` and `griddly` are absent from every venv on this host. So the three
families with *the same defect* were passed over in silence — and a skip reads like a clearance. I
went through the remaining families' render paths by hand instead, and all three are provable from
the source without running:

* `lunarlander`: `native = (VIEWPORT_H, VIEWPORT_W) = (400, 600)` is a class attribute, `render`'s
  own docstring says `(...,400,600,3)`, and it **raises** for `size != 400` because every primitive
  is hardcoded in pixels. There is no size at which it is square, so this is a constant, not an
  aspect ratio.
* `overcooked` and `griddly`: `render` reshapes to `(grid_h*tile, grid_w*tile, 3)`
  (overcooked.py:512, griddly.py:598) with `tile = _tile(size) = size // grid_h`. The new property
  reads the **same two class attributes** the reshape reads, so the spec cannot drift from the frame.

Cleared by the same reading and deliberately left alone: `carracing` (the *observation* is a fixed
96×96 `smoothscale` of the 1000×800 surface — square), `pgx` (`reshape(B, size, ss, size, ss, 3)`),
`jumanji` (`(B, size, size, 3)`), `procgen` (`(...,S,S,3)`), and `crafter`, `minigrid`, `sokoban`,
`maze2d`, `mujoco_locomotion` (all square, and all confirmed against a real render by the test).

Fixed **in the families that know their aspect ratio, not in the base class, which does not** — and
`frame_shape` is the Wave-0 hook, so no foundation file is touched. Stage 3 of the sbatch runs the
spec-vs-render test in the venvs that have the packages, which is the first place these three static
declarations become *measured*. Until then they are read, and I am labelling them that way.

## 8. Performance — paired, alternating CPU timings

`scratch/step9/envs-mujoco/bench.py` → `timings_render.json`, `timings_step.json`,
`timings_maze2d.json`. Legs **interleaved** (baseline, head, baseline, head, …) in separate
interpreters — a before-block followed by an after-block measures the machine's mood, not the code —
5 rounds × 5 timed calls per leg, single-threaded (`OMP_NUM_THREADS=1`), states sampled at a fixed
seed, at the registered frame sizes. `ratio` is head/baseline, so >1.10 is the plan §8 regression
bar, and the same-arm spread beside it says whether this machine could resolve a smaller number at
all.

**Two hot paths, because this patch touches two and they fail differently.** `render` is what every
one of these families primarily is, and the branch now carries G's consolidated 2-D backends
underneath nine of them. `step` is the **only place this workstream changed executable code rather
than declarations**: `_advance` became `_advance_into(buffers…)` in `minigrid.py` and `sokoban.py`
so that `transition` is pure by construction, and `step` now allocates a `StepResult`. Ruling 7
accepted +12–28 µs of StepResult on trivially cheap steps; these steps are not trivially cheap, but
an indirection through a caller-named buffer is exactly the change that can cost a copy per step, so
it was measured rather than assumed.

<!--TIMINGS-->

## 9. Resource lifetime

Every one of the seventeen families is now closeable, and `_close` is implemented in the **eight**
that hold something per instance: `minigrid`, `sokoban`, `overcooked`, `griddly`, `highway`,
`lunarlander`, `pgx`, `procgen`. What they release is the sampled state pool, the default-style
tensor and the per-palette/per-resolution device caches — all rebuildable.

**The shared caches are deliberately NOT released**, and each one has a reason that is not "we
forgot":

* `tilemap.MiniGridAtlas`, the Overcooked tile atlas, PGX's plates and colour tables, Sokoban's
  16×16 art: keyed on `(tile size, device)` or `(game, size)` and shared by **every** instance of
  every config. Releasing a cache another live env is drawing through is the same class of error as
  releasing a shared EGL context.
* `griddly`'s wrapper cache is load-bearing in a stronger sense: `GymWrapper.__del__` runs
  `game.release()`, which **segfaults at interpreter exit**, so the cache exists precisely to keep
  every wrapper alive. That is the module's standing policy, recorded here so a later cleanup does
  not "fix" it.
* the MJWarp render contexts held by `metaworld`, `robocasa`, `humanoidbench` and
  `mujoco_locomotion` are Workstream I's consumer change list, not mine. I did not touch them.

This is the rule `testing/conformance.py` already carries, and nothing here depends on `__del__`.

**Workstream I's consumer change list touches four of my files** and I have deliberately **not**
applied it (per the handoff it is assigned after those families merge): `robocasa.py:336`,
`humanoidbench.py:520`, `metaworld.py:760` → release MJWarp contexts on cache clear;
`mujoco_locomotion.py:1640` → `br.close()` on cache clear; `robocasa`'s `_CORRECTOR_CACHE` →
`BoundedCache(4, on_evict=close)`. Those line numbers have **moved** on this branch — see R2.

## 10. Deliberate non-decisions (plan §16, recorded so they are not re-proposed)

1. **No shared base class for the five MuJoCo wrapper families.** `mujoco_locomotion`,
   `humanoidbench`, `metaworld`, `robocasa` and `maze2d` all declare one key over a `qpos`-shaped
   packed row whose width comes from a compiled model. That is four lines each, and a
   `MujocoWrapperEnv` would exist to hold four lines and a docstring while coupling families with
   different reference pins (mujoco 3.3.0 / 3.3.1 / 3.11+). §16: two users is the bar for an
   abstraction, but *simplifying nothing* is a veto.
2. **No `state_layout` for `griddly`, `jumanji`, `minihack`, `overcooked`, `pgx`.** Their packed row
   is a flat vector of cell codes with no entity structure to name, so they take `Env._layout`'s
   trivial one-key fallback (`{"state": (state_dim,)}`) rather than a ceremonial `{"grid": …}` that
   says nothing the key `"state"` does not. Contract §3 explicitly allows this.
3. **`_advance_into` and not a functional rewrite.** `transition` could have been a second
   implementation of MiniGrid's seven-branch decision table over fresh tensors. It is instead the
   *same body* with the buffers named by the caller, so `transition == step` holds by construction
   rather than by resemblance — and `step` keeps its captured CUDA graph, which reads fixed
   addresses and therefore cannot be handed the caller's buffers.
4. **No reward or `done` fabricated** (§2).
5. **No `BoundedCache` adoption.** Workstream I's ruling on whether it replaces the eight remaining
   `clear()`-on-full caches in `envs/*` is owed for Wave 2; four of those eight are in my files and I
   left them alone.

## 11. Requests for the coordinator

**R1 — the ratchet, and a line-level conflict that is yours by assignment.** I set
`tests/unit/test_foundation_api.py::NUM_ENVS_HOLDOUTS` to **27**, the count I measured live on this
branch after the merge. All 27 remaining are OGBench (25 in `envs/ogbench`, 2 in `envs/ogb_manip`)
and belong to Workstream D, which set the same constant to **14** on its own branch. The two edits
are on the same line; per the handoff that conflict is yours. F's contribution is these **fourteen**
by name: `Highway`, `Merge`, `Roundabout`, `Racetrack`, `Parking`, `Intersection`;
`PointMazeOpen`, `PointMazeUMaze`, `PointMazeMedium`, `PointMazeLarge`;
`MujocoInvertedPendulum`, `MujocoInvertedDoublePendulum`, `MujocoReacher`, `MujocoPusher`.
The correct value once both branches are in is **0**.

**R2 — Workstream I's line numbers are stale against this branch.** I's consumer change list cites
`robocasa.py:336`, `humanoidbench.py:520`, `metaworld.py:760`, `mujoco_locomotion.py:1640`. All four
files gained lines in this workstream's commits, so those numbers no longer point at the cache-clear
sites. Ask I (or K/O in the Wave-2 sweep) to re-locate them **by symbol** rather than by line, and
please do not apply the list to my files as part of merging F: it is a separate, assigned change and
mixing it in would make the oracle re-run ambiguous about which patch moved what.

**R3 — the GPU section, and one thing in it that needs a decision rather than a run.**
`scratch/step9/envs-mujoco/gpu_oracle.sbatch` is prepared and **not submitted** (ruling 8). It is
written as a *section*, and it asks for slightly more than "the cuda frames": **stage 0** discovers,
per interpreter, which of these seventeen families import and prints each `ImportError`, because no
local evidence says whether Oscar can certify `griddly`, `jumanji`, `overcooked`, `pgx`, `procgen`,
`carracing` or `lunarlander` **at all**. If stage 0 reports that none of those seven can be built on
the cluster either, that is a **DOCUMENTED_LIMITATION for §20 closure**, not a Workstream-F
omission, and it needs a decision from Sami rather than from me. Note also that packages must be
installed from the **login node** (compute nodes may have no internet) and that `minigrid` is absent
from every Oscar venv per the Step-6 findings — without that install, stages 1 and 3 would cover
none of the seven MiniGrid cells there.

**R4 — please do not let F's dashboard `oracle_diff` read "identical" unqualified.** It is identical
on **24 cells across 7 of the 17 families**; three families (`metaworld`, `robocasa`,
`humanoidbench`) have never had a frame rendered on this branch, on any device. If the post-Wave-1
job cannot build them either, F's row should say UNESTABLISHED for those three rather than inherit
the CPU verdict.

**R5 — flagged only, not mine.** `scratch/step3/certify.py::NO_BACKEND_MODULE_CLASS` keys on module
basenames and is broken for `navigation` and `_craftax_full`. I did not touch it; it is
coordinator-owned / Workstream K. Recording that **no module in this workstream moved or was
renamed**, so F adds nothing to that list.

**No contract change is requested.** WAVE0_CONTRACT.md stands as frozen; every claim above is made
inside it, `frame_shape` and `_hidden_state`/`_live_state` are the sanctioned hooks, and no cell
here overrides the `state` property (ruling 2 — checked: `grep -n "def state"` over the seventeen
files finds only `state_from_blob` classmethods in `procgen.py`).

**On `42bbf74`, the checkpoint the coordinator committed from my predecessor: I agree with it and
would not revert any of it.** The test file is honest about its own coverage, and the two
`frame_shape` declarations are correct, minimal and in the right place (the family, not the base
class). My one amendment is §7a — the same defect existed in three more families, the test could not
see them, so I found them by reading and fixed them the same way.

## 12. The §17 ten-item report

| # | item | answer |
|---|---|---|
| 1 | files moved/removed/created | **nothing moved, nothing removed**; 17 modules edited in place, 1 test file + 3 scratch scripts + this record created (§1) |
| 2 | API changes | additive only: `num_envs=`, public State, `step`/`transition`/`unroll` on 7 cells, `close()` on 17, `frame_shape` on 5 (§2) |
| 3 | Step-8 IDs affected | **none of the 42** — the intersection with the support matrix's 43 cells is empty (§3) |
| 4 | targeted tests | `tests/unit/test_envs_mujoco.py`, 9 functions, 72 passed / 57 skipped; full suite in §4 |
| 5 | applicable certification checks | no frozen Step-4 corpus exists for any cell here; `check_state_roundtrip`, the declaration checks and the fidelity declarations apply and pass (§5) |
| 6 | default-frame comparison | **24 cells byte-identical** vs `21cb306` in the same venv, and identical to the pre-merge digests, so G's consolidation moved no pixel (§6) |
| 7 | variation/state comparison | `variation_names` leaf for leaf, `style_names`/`style_dim`/`default_style`, `state_dim`/`state_names`, `roundtrip_exact`, `sample_states` — identical on all 24 (§6) |
| 8 | performance for hot paths | paired alternating CPU timings on **two** paths, render and step — §8 |
| 9 | resource-lifetime comparison | 17 closeable, `_close` in 8; shared caches deliberately retained with a named reason each; I's change list not applied (§9) |
| 10 | unexplained oracle differences | **zero** on cpu over 24 cells. **UNESTABLISHED** for `metaworld`, `robocasa`, `humanoidbench`: CUDA-only, no frame rendered on any device (§6b) |

## 13. Semantic follow-up recommendation

**None.** No frame in any reachable cell of these seventeen families moved by a single byte, so
there is nothing for a DINO tripwire or a paired Gate H to be sensitive to. The one genuinely new
numeric surface is `observation_spec`'s shape for five families, and that is a buffer description
rather than a pixel.

## 14. Commits

<!--COMMITS-->

## 15. Provenance of the records

| artifact | produced by | at head |
|---|---|---|
| `oracle.json` (20 cells) | `oracle.py`, `se6-venv`, `SE9_EXTRA_PYTHONPATH` = the four uv archives | `a466756` |
| `oracle_maze2d.json` (4 cells) | `oracle.py --python se8-fetch-venv --cells PointMaze…` | `a466756` |
| `timings_{render,step,maze2d}.json` | `bench.py`, alternating, 5 rounds × 5 reps, `OMP_NUM_THREADS=1` | `33d51c5` |
| `gpu_oracle.sbatch` | written, **not submitted** (ruling 8) | `33d51c5` |

The oracle record names head `a466756` while the code tip is later. The difference is three
`frame_shape` properties on `lunarlander`, `overcooked` and `griddly` — **three families that do not
import in either oracle venv and appear in neither record's cell list** — plus `bench.py`, the
sbatch and this file. None of them can move a compared key. If you would rather have the record
carry the tip SHA regardless, say so and I will re-run it.
