# Step 7 — closure verification (`step7/closure`)

**Classification: PASS-WITH-KNOWN-DELTAS on everything Step 7 froze; FAIL on two cells it merged.**

Nothing frozen moved, and `OGBCube` / `OGBScene` fail a mandatory Gate-A check plus thirteen further
`tests/gpu` assertions that no Step-7 interpreter was able to run on them. Section 7 is the verdict.

Certification of `step7/integration` as a whole. This workstream changed no renderer code and no
test: it ran the existing machinery — `examples/fidelity_step4.py`, `scratch/step4/certify.py`,
`scratch/step7/rocket-landing/refreeze_locomotion.py`, `tests/gpu`, `tests/unit` — against the
frozen Step-4 corpus, against the Step-7 correctness re-freeze, and against the seven cells Step 7
added, on their own committed corpora.

| | |
|---|---|
| parent baseline | `76dc5116f432e61dd47fba8f9e879cf9212c8e46` (Step 6 closed) |
| **integration SHA verified here** | **`275970cc2d9e0d92eebf5297ee175b08944b7d04`** (`step7/integration`, the commit after `fd2b8ef`) |
| my branch | `step7/closure`, branched from that SHA |
| cluster copy | `$SCR/se7/closure`, rsync of the worktree with `.git` excluded and `.git_sha` written first |
| pristine baseline on the cluster | `$SCR/se7/baseline`, `.git_sha` = `76dc5116f432e61dd47fba8f9e879cf9212c8e46` (untouched) |
| SLURM jobs | GPU `6413321` (gpu2107, RTX 3090); CPU `6413322` (Gate A on `cpu`), `6413370` (`tests/unit`); `6413171` superseded, `6413170` cancelled before it measured anything |

## 0. What this sweep is allowed to conclude

Four questions, kept apart, because conflating them is how a coverage claim gets read as a
certification claim:

1. **Did anything that was frozen move?** The 49 frozen Step-4 render-distinct configurations
   replayed on cuda against their frozen digests, each shard in the interpreter it was frozen in.
2. **Did the Step-7 re-freeze hold?** The six Mujoco locomotion nvdiff cells are NOT among the 49 —
   they have no `examples/results/step4/states/<Config>.npz` and appear in no
   `conformance/<shard>.json` — so the artifact they are held to is
   `examples/results/step7/rocket-landing/refreeze/branch.json`, the correctness re-freeze Sami
   authorised.
3. **Do the seven new cells pass Gate A on their own committed corpora**, on every device they
   claim, with the refusals recorded as refusals?
4. **Do the suites still say what they said**, with every remaining failure named in advance?

None of these is a performance question. No tier was measured, re-measured or recomputed anywhere in
this sweep; the coverage matrix copies each tier and prints the file it came from.

## 1. Interpreters — chosen to reproduce, not for convenience

Each frozen shard replays in the interpreter it was frozen in
(`examples/results/step4/conformance/<shard>.json` -> `provenance.reference_packages`); a digest
reproduced in a different dependency set answers a different question. That is Step 6's rule and it
is kept verbatim in `scratch/step7/closure/_env.sh`.

| purpose | interpreter | why this one |
|---|---|---|
| `dmc`, `ogb`, `raster` shards | `$SCR/venvs/se-step3` | dm_control 1.0.45, gym_pusht 0.1.6, ogbench 1.2.1, mujoco + mujoco_warp 3.12.0 — a field-for-field match to those shards' frozen provenance |
| `swm` shard | `$SCR/venvs/se-step4-swm` | craftax 1.6.1 + jax 0.11.1 + stable_worldmodel, no mujoco at all |
| `fetch` shard | `$SCR/venvs/se-s3b-fetch311` | gymnasium_robotics 1.4.2, mujoco + mujoco_warp 3.11.0 |
| the four visual-tier DM Control cells, kept as a shard of their own | `$SCR/venvs/se-vis311` | mujoco 3.11.0 + dm_control 1.0.44 — the pair their `visual_fidelity` is DECLARED against. `certify.py` reads the reference version off the shard, so these rows cannot sit in the 3.12 document |
| `CraftaxClassic`, `CraftaxFull` | `$SCR/venvs/se7-craftax` | jax 0.11.1 with `jax_threefry_partitionable` at its default, which is what the night parity claim names |
| `DMCManipulatorBringBall`, `DMCQuadrupedWalk` | `$SCR/venvs/se-vis311` | the declared 3.11 pair again |
| `OGBCube`, `OGBScene` | `$SCR/venvs/se7-ogb` + the swm checkout on `PYTHONPATH` | ogbench 1.2.1 + mujoco_warp 3.12.0; the reference is swm's own `render()` |
| `PFRocketLanding`, the six locomotion nvdiff cells | `$SCR/venvs/se7-rocket` | nvdiffrast 0.4.0, PyFlyt 0.29.0, pybullet 3.2.7 |
| `tests/gpu`, broad | `$SCR/venvs/se6-verify` | installs nothing: a `--without-pip` venv whose single `.pth` chains se-step6, se-step3, se-step5 and the swm checkout. The only interpreter that reaches the MJWarp families and pytest in one process |

**No venv was created and none was modified.** Every interpreter above already existed.

### One operational defect found and fixed inside this sweep

The first CPU job (`6413171`) set `MUJOCO_GL=osmesa`, reasoning that a batch node has no GPU. That
is wrong on Oscar: `stable_envs.envs.dmcontrol` then fails to import with
`AttributeError: 'NoneType' object has no attribute 'glGetError'`, which **silently removes every DM
Control cell from the registry** — the run reported `NOT_REGISTERED_HERE` for cells that are
registered, and `tests/unit` collapsed into collection errors. `MUJOCO_GL=egl` is what Step 6's own
CPU verification jobs used on these same nodes. Recorded because the failure mode is the dangerous
one: a missing backend and an empty table look identical unless the job prints the import error,
which `gate_a_new_cells.py` now does.

The same reasoning moved the `device="cpu"` refusal check for OGBCube, OGBScene and PFRocketLanding
off the batch node and onto the card. On a batch node `mujoco_warp` and `nvdiffrast` do not import
at all, so a refusal and a missing module are indistinguishable; the question is whether the class
refuses on a machine where the CUDA backend **is** available, and it has to be asked there.

## 2. The corpora the seven new cells were judged on

Copied, never redrawn, by `scratch/step7/closure/stage_corpora.py`, which records the source file and
the sha256 of the array it copied:

| cell | committed corpus | shape | sha256 (first 16) |
|---|---|---|---|
| `CraftaxClassic` | `examples/results/step7/craftax-classic-full/corpus/states.npz` | (64, 149) | `e403a5648d92156e` |
| `CraftaxFull` | `examples/results/step7/craftax-full/corpus/states.npz` | (64, 544) | `b0c38605fc983eca` |
| `DMCManipulatorBringBall` | `examples/results/step7/dmc-manipulator/corpus/states/DMCManipulatorBringBall.npz` | (64, 15) | `ebfc481ad7f23962` |
| `DMCQuadrupedWalk` | `examples/results/step7/dmc-quadruped/phase1/corpus/states/DMCQuadrupedWalk.npz` | (64, 23) | `1f0b38b4dca47b92` |
| `OGBCube` | `examples/results/step7/ogb-manipulation/states/OGBCube.npz` | (64, 24) | `560c1398a8a49a3f` |
| `OGBScene` | `examples/results/step7/ogb-manipulation/states/OGBScene.npz` | (64, 30) | `7a0cdff1849ee221` |
| `PFRocketLanding` | `examples/results/step7/rocket-landing/reference/states.npz` (`state13`, first 64 of 200) | (64, 13) | `f4707e54f3007b2e` |

Two of these needed a copy rather than a symlink: the two Craftax corpora were written without the
`meta` key that `fidelity_step4.corpus_for` reads unconditionally — already recorded as a Step-8
freeze carry-over by W2's reviewer — and PFRocketLanding's reference corpus is a multi-array file
whose states live under `state13`. The copy adds the `meta` key and nothing else; the state bytes
are the committed ones and their sha256 is printed above so that can be checked.

## 3. The frozen Step-4 replay — 48 of 49, and the 49th's obligation met

`--partition=gpu --qos=norm-gpu --gres=gpu:1 --constraint=geforce3090`; job `6413321` (gpu2107) for
`dmc`, `ogb`, `raster`, `fetch` and the 3.11 `dmcvis` arm, job `6417172` (gpu2101) for `swm`. Each
shard in the interpreter it was frozen in. `compare49_step7.py` diffs every field of every
configuration against `examples/results/step4/conformance/<shard>.json`.

**Under each shard's own frozen interpreter: 43 identical, 5 expected-known, 0 unexplained, 0
regressions.** The five are exactly Step 6's known set, with the same mechanisms and no new member:

| config | what moved | classification |
|---|---|---|
| `DMCReacherEasy`, `DMCReacherHard`, `DMCFingerTurnEasy`, `DMCFingerTurnHard` | the DECLARATION only — `style_dim` 17 -> 18 and `style_digest`; `digest_ours` and `digest_ours_default_style` byte-identical | expected-known (reacher-correctness's visibility leaf, Step 6) |
| `AntmazeTeleportExplore` | `digest_ours` and `digest_ours_default_style`; `digest_official` unchanged | expected-known (documented bake nondeterminism; the digest does not reproduce at the BASELINE either). Gate A FAIL -> PASS |

`HumanoidmazeMediumNavigate` improves Gate A **FAIL -> PASS** again, on `check_style_is_state`, as in
Step 6.

### The comparison is committed twice, and the difference between the two files is the point

`compare49.json` includes the `dmcvis` arm — the four MJWarp trackcom cells replayed under the
**declared** mujoco 3.11.0 / dm_control 1.0.44 pair their `visual_fidelity` names — and there they
read 39 identical / 9 expected-known, because `DMCCheetahRun`, `DMCHopperHop`, `DMCWalkerWalk` and
`DMCHumanoidStand` move their default-appearance digests.
`compare49_frozen_interpreters_only.json` leaves that arm out and they do not move.

**That difference is the reference stack, not the renderer.** The frozen shard's provenance is
`device cuda` under mujoco 3.12 on gpu2105; replaying at the 3.11 pin renders different pixels for a
reason Step 6 already recorded ("3.11 CPU replay digests differ from `conformance/dmc.json` on 10/14
cells: not a regression signal"). Publishing the 3.11 number without the 3.12 control beside it would
let a stack difference read as a Step-7 pixel change, so both are committed and both are cited.

### The one NEW delta Step 7 introduces, with its mechanism

**`CraftaxClassic` cannot replay its frozen digest, and must not.** W1 took
`swm/CraftaxClassicPixels-v1` from partial to full coverage: the observation grew from the
`(112,144,3)` world window to the whole `(144,144,3)` SWM frame and the state from 127 slots to 149.
`examples/fidelity_step4.py` therefore raises on the frozen corpus, by design and with the right
message:

```text
ValueError: CraftaxClassic.render: CraftaxClassic expects state_dim 149, got 127
(input shape (64, 127), dtype float32). ... This used to be reshaped to (-1, 149), which regroups
the input into a different number of states that each straddle two of your rows, and renders
without complaint.
```

That raise is the check doing its job; the alternative — a silent reshape — is the failure mode the
message names. Operationally it also means the cell cannot sit inside a `--shard swm` run: the raise
aborts the other five configurations before they are measured, which is why the shard runs with
`--configs` and `CraftaxClassic` is asked separately.

The obligation the coordinator set instead is **world-row byte identity at light 1.0**, and
`scratch/step7/closure/craftax_world_row_replay.py` measures exactly that against the frozen artifact:

| | |
|---|---|
| frozen corpus | `examples/results/step4/states/CraftaxClassic.npz`, 127 slots, read-only |
| zero-extended to | 149 slots (empty inventory, awake, day-cycle key zero), `light_level` forced to 1.0 |
| size | 16, the frozen size |
| full frame now | `(64, 144, 144, 3)` |
| world rows taken | `(64, 112, 144, 3)` = the frozen `frames_shape` |
| frozen `digest_ours` | `1aeabbe8bc6a9a2ee1fdb8ffc27a737bd659630dc6e3a1b98f6bae002c894a4d` |
| **world-row digest now** | **`1aeabbe8bc6a9a2ee1fdb8ffc27a737bd659630dc6e3a1b98f6bae002c894a4d`** |
| frozen mean pixel / now | `93.84458414713542` / `93.84458414713542` |
| inventory strip rows | all zero, as a zero inventory must be |

Digested with `conformance.frames_digest`, the function the frozen artifact used — it hashes shape
and dtype before the bytes, so a plain `sha256` of the same pixels is a different number and would
have looked like a failure. **Reproduced on both architectures**: Apple M5 `arm64` and an RTX 3090 on
gpu2101 (job `6417172`).

## 3b. The six Mujoco locomotion nvdiff cells against the Step-7 re-freeze

These six are **not** among the 49: they have no `examples/results/step4/states/<Config>.npz` and
appear in no `conformance/<shard>.json`. The only Step-4 artifact naming them is
`bounds.downsample_exact_frac`. So the artifact they are held to is the correctness re-freeze Sami
authorised, `examples/results/step7/rocket-landing/refreeze/branch.json`.

**6 of 6 re-render bit-identical.** Every `frames.sha256` equal, on the same identity draw
(`states_sha256` equal too, so the comparison is of frames and not of a different draw):

| cell | frames sha256 (frozen == fresh) |
|---|---|
| `MujocoHalfCheetah` | `17cf24506ae10a70…` |
| `MujocoHopper` | `f6f2aef63d97a4ab…` |
| `MujocoWalker2d` | `d95e510e20776cb1…` |
| `MujocoAnt` | `b66388957cbff323…` |
| `MujocoHumanoid` | `30948a9b3d1a2307…` |
| `MujocoSwimmer` | `389858758e2b5a03…` |

And Gate A passes on all six, including the check the re-freeze existed to satisfy:
`check_batch_size_invariance` `max_abs_diff = 0` at batch sizes 1, 3, 7, 8, 64 and 256, alongside
determinism, batch invariance, order independence and the rest.

## 4. Gate A on the seven cells Step 7 added — both devices

Measured by `scratch/step7/closure/gate_a_new_cells.py` on the committed Step-7 corpora at each
cell's registered default size, through `examples/fidelity_step4.measure`, with each mandatory
check's bound spelled out per check rather than inferred from the evidence's shape.

| cell | cuda | cpu | mandatory | interpreter | failing |
|---|---|---|---|---|---|
| `CraftaxClassic` | **PASS** | **PASS** | 14/14 | `se7-craftax` | — |
| `CraftaxFull` | **PASS** | **PASS** | 14/14 | `se7-craftax` | — |
| `DMCManipulatorBringBall` | **PASS** | **PASS** | 14/14 | `se-vis311` (declared 3.11 pair) | — |
| `DMCQuadrupedWalk` | **PASS** | **PASS** | 14/14 | `se-vis311` (declared 3.11 pair) | — |
| `PFRocketLanding` | **PASS** | refuses by design | 14/14 | `se7-rocket` | — |
| `OGBCube` | **FAIL** | refuses by design | 14 run | `se7-ogb` | `check_state_roundtrip` |
| `OGBScene` | **FAIL** | refuses by design | 14 run | `se7-ogb` | `check_state_roundtrip` |

Device coverage is recorded as it is, not as a gap. Four of the seven have a CPU backend and all four
pass every mandatory check on it. Three raise on `cpu` rather than falling back to CUDA, which is the
Step-6 policy addendum's rule, and the refusal was asked **on the card** — on a batch node
`mujoco_warp` and `nvdiffrast` do not import at all, so a refusal and a missing module would have
been indistinguishable. Both refusals carry the message the cell promised:

```text
ValueError: OGBCube has no CPU backend (CPU3D-OGB-1): this scene is 193,113 visible triangles at a
median projected edge of ~0.7 px, and the certification CPU bar leaves under a naive rasteriser …

NotImplementedError: PFRocketLanding has no CPU backend: it needs a CPU textured-triangle
rasteriser, which this repo does not have (cpu3d/raycast3d cast MuJoCo primitives). …
```

The two FAILs are CLOSURE-3 below.

## 5. The suites

### `tests/gpu` — against Step 6's run of the same files in the same interpreter

"Nothing else failing" is a claim about a *difference*, so the comparator is
`examples/results/step6/verify/gpu/*.results.txt`: Step 6's integration verification ran these files
in `$SCR/venvs/se6-verify`, the same chained interpreter, and committed the filtered output.
`scratch/step7/closure/compare_gpu_suite.py` diffs them, error counts included — 437 and 529 of the
lines are the 23 environments that interpreter cannot construct at all, and a change there would mean
the interpreter moved rather than the code.

| file | Step 6 | closure | error set | new failures |
|---|---|---|---|---|
| `test_conformance.py` | 7 failed, 845 passed, 387 skipped, 437 errors | 10 failed, 902 passed, 422 skipped, **437 errors** | **unchanged** | **3** |
| `test_reproducibility.py` | 17 failed, 1275 passed, 218 skipped, 529 errors | 22 failed, 1358 passed, 247 skipped, **529 errors** | **unchanged** | **12** |
| `test_fast_paths.py` | 168 passed, 180 skipped | 2 failed, 247 passed, 198 skipped | unchanged (0) | **2** |

**Every one of Step 6's failures is carried unchanged and none is fixed**; the error sets are
identical id-for-id. The 17 new failures are:

| new failure | cells | expected in advance? |
|---|---|---|
| `test_resolution_downsample_agrees_within_the_measured_bound` | `CraftaxClassic`, `CraftaxFull`, `DMCQuadrupedWalk` | **yes** — the downsample-table entries for new cells, a Step-8 freeze item. CraftaxClassic reads 81.638% against a bound of 100.000% (the night band declared not applicable); `CraftaxFull` and `DMCQuadrupedWalk` have **no bound recorded at all** and the test says so |
| `test_state_roundtrip` | `OGBCube`, `OGBScene` | **no** — CLOSURE-3 |
| `test_visual_tier_matches_its_declared_fidelity` | `OGBCube` | **no** — CLOSURE-4 |
| `test_a_shared_style_broadcasts_to_any_batch`, `test_rendering_a_sub_batch_refuses_environment_0_s_style` | `OGBCube`, `OGBScene` | **no** — CLOSURE-5 |
| `test_render_is_pure_and_set_state_is_idempotent`, `test_dataset_paths_regenerate_identical_frames`, `test_the_reference_renderer_on_this_machine_is_deterministic` | `OGBCube`, `OGBScene` | **no** — CLOSURE-5 |
| `test_the_fast_path_is_bit_identical_to_the_path_it_replaces[raster2d-tile-culling-ss2-*-native]` | `OGBCube`, `OGBScene` | **no** — CLOSURE-5 |

**Fourteen of the seventeen are OGBCube and OGBScene.** The three named in advance are the
downsample-table entries. The Rocket colour-knob failure the brief also named in advance does NOT
appear here because `PFRocketLanding` is not registered in `se6-verify` (no nvdiffrast) and skips;
it is a known failure of the `se7-rocket` arm, recorded by W6 at the tip (job `6411148`).

`test_conformance.py` was re-run in **`se7-ogb`**, the interpreter W5 declared against, with
`$SCR/se7/swm-src` on `PYTHONPATH`: **the same ten failures, id for id** (10 failed, 948 passed, 452
skipped, 361 errors — the error count differs only because that interpreter can construct 76 more
tests). So none of the OGB findings is an artefact of the Step-6 `stable_worldmodel` checkout that
`se6-verify` chains.

### `tests/unit` — the certification host and the Mac, each against its own baseline

Every run is paired with the SAME suite in the SAME interpreter on the SAME machine against the
pristine `76dc5116` tree, because "only the known pre-existing failure" is a claim about a
*difference*.

| host | interpreter | baseline `76dc5116` | integration `275970cc` | new failures |
|---|---|---|---|---|
| Oscar x86-64 | `se-step6` | 2 failed, 807 passed | 3 failed, 1018 passed | **1**: `test_from_triangles_clip_row_is_independent_of_batch_size` |
| Oscar x86-64 | `se7-craftax` | 2 failed, 776 passed | 3 failed, 919 passed | **1**: the same |
| Oscar x86-64 | `se7-rocket` | 11 failed, 780 passed | 12 failed, 974 passed | **1**: the same |
| Apple M5 | `se6-venv` | 2 failed, 822 passed | 2 failed, 1035 passed | **0** |
| Apple M5 | `se7-craftax-venv` | 2 failed, 773 passed | 2 failed, 918 passed | **0** |
| Apple M5 | `se7-rocket-venv` | 11 failed, 767 passed | 11 failed, 913 passed | **0** |

The failure SETS are otherwise identical on both trees:

- `test_repo_contracts.py::test_every_cited_scratch_path_resolves` — the known pre-existing failure,
  **9** broken citations at the baseline and **10** at the integration HEAD (CLOSURE-2).
- `test_examples.py::test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]`
  — pre-existing in every interpreter on both trees: that example loads its helpers from a hard-coded
  absolute path under `/mnt/vast/…` which exists on nobody's machine here. Previous Step-6 and Step-7
  runs deselected `test_examples.py`, which is why it does not appear in their counts.
- the nine `test_gfxdraw_bounds` / `test_gfxdraw_select` failures in the `rocket` interpreters only —
  the same hard-coded path, present at `76dc5116` on both machines. Interpreter-specific, not a
  Step-7 effect.

 — the certification host and the Mac, each against its own baseline

Every run is paired with the SAME suite in the SAME interpreter on the SAME machine against the
pristine `76dc5116` tree, because "only the known pre-existing failure" is a claim about a
*difference*.

| host | interpreter | baseline `76dc5116` | integration `275970cc` | new failures |
|---|---|---|---|---|
| Oscar x86-64 | `se-step6` | 2 failed, 807 passed | 3 failed, 1018 passed | **1**: `test_from_triangles_clip_row_is_independent_of_batch_size` |
| Oscar x86-64 | `se7-craftax` | 2 failed, 776 passed | 3 failed, 919 passed | **1**: the same |
| Oscar x86-64 | `se7-rocket` | 11 failed, 780 passed | 12 failed, 974 passed | **1**: the same |
| Apple M5 | `se6-venv` | 2 failed, 822 passed | 2 failed, 1035 passed | **0** |
| Apple M5 | `se7-craftax-venv` | 2 failed, 773 passed | 2 failed, 918 passed | **0** |
| Apple M5 | `se7-rocket-venv` | 11 failed, 767 passed | 11 failed, 913 passed | **0** |

The failure SETS are otherwise identical on both trees:

- `test_repo_contracts.py::test_every_cited_scratch_path_resolves` — the known pre-existing failure,
  **9** broken citations at the baseline and **10** at the integration HEAD (see CLOSURE-2).
- `test_examples.py::test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]`
  — pre-existing in every interpreter on both trees; `examples/robocasa_visual/render_mjwarp.py`
  loads its helpers from a hard-coded absolute path,
  `/mnt/vast/home/randall.balestriero/code/stable-envs/stable_envs/render/mjwarp.py`, which exists on
  nobody's machine here. Previous Step-6 and Step-7 runs deselected this file, which is why it does
  not appear in their counts.
- the nine `test_gfxdraw_bounds` / `test_gfxdraw_select` failures in the `rocket` interpreters only —
  same hard-coded path, same pre-existing status on both trees, present at `76dc5116` on both
  machines. Interpreter-specific, not a Step-7 effect.
## 6. What this sweep found that no workstream had seen

### FINDING CLOSURE-1 — `nvdiff.Scene.clip` is batch-size dependent on x86-64 CPU at widths 1 and 2

`tests/unit/test_nvdiff_triangles.py::test_from_triangles_clip_row_is_independent_of_batch_size` —
W6's own test, added in Step 7 — **passes on an Apple M5 and fails on Oscar's x86-64 batch nodes**.
No Step-7 workstream saw it because the Step-7 `tests/unit` runs quoted in the dashboard were taken
on the Mac; this is the first time that suite ran on the certification host at the integration HEAD.

Characterised by `scratch/step7/closure/probe_nvdiff_cpu_batch.py`, job `6413606` on `node1939`
(x86-64, torch 2.14.0+cu126), at three thread counts:

| batch width | `mvp_rows` (W6's float64 product) | `clip` positions | `clip` normals | `clip` world positions |
|---|---|---|---|---|
| 1 | **identical** | **36 of 96 differ**, max 3.05e-05 | identical | identical |
| 2 | **identical** | **60 of 192 differ**, max 3.05e-05 | identical | identical |
| 7, 8, 31, 32, 63 | identical | identical | identical | identical |

- **W6's fix is intact.** `mvp_rows` — the float64 `proj @ view @ M` rounded once, which is the
  product W6 made batch-invariant — is bit-identical at every width on both architectures. The
  residual is one product downstream: `Scene.clip`'s float32 `mvp @ vertices`, whose CPU GEMM picks a
  different summation order for a very small batch, exactly as cuBLAS did for the product W6 fixed.
- **The magnitude is one float32 ulp** at the coordinate scale involved (3.05e-05 at 306 units), on
  clip-space positions only.
- **Not a threading artifact**: identical behaviour at 1, 2 and 4 torch/OMP threads.
- **Architecture-specific**: every width is bit-identical on arm64 (Apple M5, torch 2.14 CPU).

**No shipped renderer is affected, and no frozen or certified artifact depends on it.** Nothing in
the registry draws through `nvdiff` on `device="cpu"`: `PFRocketLanding` raises on cpu by design and
the six Mujoco locomotion nvdiff cells render through nvdiffrast on the card. The CPU path of
`Scene.clip` exists so that the transform can be unit-tested without a GPU, which is what found this.

It is nevertheless a **failing test in a mandatory suite on the certification host**, and the fix
would be inside `stable_envs/render/nvdiff.py` — a shared renderer file — or inside `tests/`, both of
which this workstream may not edit and both of which are frozen for Step 7. **Reported, not fixed.**

### FINDING CLOSURE-2 — the broken-citation set grew from 9 to 10 during Step 7

`tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves` is the known pre-existing
failure. Its **set** is not what it was: at `76dc5116` it names **9** unresolvable paths, at
`275970cc` it names **10**. The tenth is

    scratch/sboughan/se7/swm-src/stable_worldmodel
        cited at examples/results/step7/ogb-manipulation/oscar/provenance_6395175_readback.txt:18

— a `$SCR` path inside a committed Oscar provenance capture, which the scanner reads because it reads
every `.txt` under `examples/`. Measured both ways in the same interpreter on the same tree
(`se6-venv`, Apple M5): baseline 9, integration HEAD 10, same nine plus that one.

Not fixed here, deliberately: the file is W5's committed *evidence capture* of what a job printed, and
editing it would make the provenance record disagree with the job that produced it. The right fixes
are either the exemption W4's reviewer already proposed for committed `review_*.md` (extended to
`oscar/*.txt` provenance captures) or a scrub at capture time — both changes to `tests/` or to a
workstream's artifacts, and both **Step-8 items**.


### FINDING CLOSURE-3 — `OGBCube` and `OGBScene` fail `check_state_roundtrip`, a MANDATORY Gate-A check

`stable_envs/testing/conformance.py::check_state_roundtrip` asserts that `set_state(get_state())` is
the identity, and `tests/gpu/test_conformance.py::test_state_roundtrip` spells the bound as
`max_abs_err == 0.0` — exactly zero, not a tolerance. Both cells return ~1.1e-07:

| cell | `max_abs_err` | where |
|---|---|---|
| `OGBCube` | `1.1682e-07` (Gate A), `1.1380e-07` (`tests/gpu`) | cuda, RTX 3090, job `6413321` |
| `OGBScene` | `1.1185e-07` (Gate A), `1.1053e-07` (`tests/gpu`) | cuda, RTX 3090, job `6413321` |

The check draws its own states with `env.sample_states(env.B)`, so this is a property of the cells
and not of the corpus this sweep staged.

**Nobody had run it on these two.** W5's Gate-A list in the dashboard covers determinism, batch
invariance, batch-size invariance, order independence and `state_dims_matter`; `check_state_roundtrip`
is not in it. And the tip conformance run the dashboard cites (job `6411148`) executed `tests/gpu` in
the `se7-rocket` interpreter, where OGBench is not installed and every OGB cell **skips**. This sweep
is the first time the full mandatory set has been run against these cells.

The magnitude is about one float32 ulp near 1.0, which is consistent with a quantity being projected
onto a constraint surface (a quaternion renormalised on the way in) or narrowed; the attribution
probe `scratch/step7/closure/probe_ogb_state_roundtrip.py` reports it per dimension by name and says
whether a second round trip is the identity, which separates "projected once" from "information
lost". **Reported, not fixed:** the fix would be inside `stable_envs/envs/ogb_manip.py`.

### FINDING CLOSURE-4 — `OGBCube`'s declared `mean_abs` ceiling does not hold under the conformance draw

`tests/gpu/test_conformance.py::test_visual_tier_matches_its_declared_fidelity[OGBCube]` fails:

```text
AssertionError: mean |diff| 3.287/255 exceeds the declared ceiling 3.2 — the visual tier is a
measured claim, and this env is no longer meeting it:
{'mean_abs': 3.2868, 'exact_frac': 0.8718, 'within8_frac': 0.9166, 'max_abs': 205,
 'interior_frac': 0.4113, 'interior_exact_frac': 0.9589, 'interior_mean_abs': 1.4179,
 'ssim': 0.9233, 'n_states': 64}
```

Everything else the cell claims is comfortably met — `interior_exact_frac` 0.9589 against a declared
0.9580, `within8_frac` 0.9166 against a floor of 0.85, `ssim` 0.9233 against 0.85. Only `mean_abs`
exceeds, by **2.7%**.

The declared ceiling is `worst-of-three × 1.3` over W5's own draw (worst 2.4417 -> 3.2). The
conformance check draws its own states, so the margin has to cover a distribution the declaration was
not measured on, and here it does not. `OGBScene` passes the same test: its ceiling is 7.0 against a
declared 5.38, which is more headroom for the same reason.

The obvious caveat was checked and does not hold: the same `test_conformance.py` re-run in
**`se7-ogb`** — the interpreter W5 declared against, with `$SCR/se7/swm-src` on `PYTHONPATH` rather
than the Step-6 `stable_worldmodel` checkout `se6-verify` chains — gives **the same ten failures, id
for id**, this one included. The number is a property of the cell and the draw, not of the reference
source. **Reported, not fixed:** changing a declared bound is a coordinator decision.

### FINDING CLOSURE-5 — `OGBCube` and `OGBScene` appear never to have been through `tests/gpu`

Fourteen of the seventeen new `tests/gpu` failures are these two cells, across three files:

| failure | cells | mechanism, quoted |
|---|---|---|
| `test_state_roundtrip` | both | `set_state(get_state()) is not the identity: {'max_abs_err': 1.1380e-07}` (Cube), `1.1053e-07` (Scene) — CLOSURE-3 |
| `test_visual_tier_matches_its_declared_fidelity` | Cube | `mean |diff| 3.287/255 exceeds the declared ceiling 3.2` — CLOSURE-4 |
| `test_a_shared_style_broadcasts_to_any_batch` | both | `OGBCube: under the default appearance, rendering 3 of 8 rows differs from those rows of the full batch — a shared style must broadcast unchanged` |
| `test_rendering_a_sub_batch_refuses_environment_0_s_style` | Cube | the same sub-batch assertion |
| `test_render_is_pure_and_set_state_is_idempotent` | both | `ValueError: zero-size array to reduction operation maximum which has no identity` — a crash, not an assertion |
| `test_dataset_paths_regenerate_identical_frames` | both | mechanism not extracted from the raw log (see below) |
| `test_the_reference_renderer_on_this_machine_is_deterministic` | both | mechanism not extracted |
| `test_the_fast_path_is_bit_identical_to_the_path_it_replaces[raster2d-tile-culling-ss2-*-native]` | both | mechanism not extracted |

**The sub-batch assertion is the one that matters most**, because it is not a tolerance question:
rendering rows 0-2 alone gives different pixels from rows 0-2 of a batch of 8, under the default
appearance. Gate A's `check_batch_invariance` and `check_batch_size_invariance` both PASS on these
cells, so whatever this is, it is specific to the shared-style sub-batch path and is not caught by
the checks the workstream ran.

**Why nobody saw it.** W5's Gate-A list in the dashboard is determinism, batch invariance, batch-size
invariance, order independence and `state_dims_matter` — repo checks called directly, not the pytest
suite. And every committed `tests/gpu` run in Step 7 was taken in an interpreter where OGBench is
absent, so all OGB cells **skipped**: W6's tip run (job `6411148`) was `se7-rocket`. This sweep is
the first `tests/gpu` execution that constructs these two cells.

Reproduced identically in `se7-ogb`, the interpreter W5 declared against, with `$SCR/se7/swm-src` on
`PYTHONPATH`, so none of it is an artefact of the `se6-verify` chain.

**Reported, not fixed.** Every fix would be inside `stable_envs/envs/ogb_manip.py` or in `tests/`,
and this workstream may edit neither. Four mechanisms could not be lifted out of the raw pytest log
(`scratch/step7/closure/extract_failures.py` matches sections by exact test id and those four
sections did not yield a first `E` line it could attribute safely — a fuzzy match produced a
demonstrable misattribution and was reverted rather than kept). The job log is
`$SCR/logs/6413321_test_reproducibility.verify.results.txt`; extracting them is a ten-minute job for
whoever picks the fix up, and the count and the ids above do not depend on it.

## 7. Verdict

**PASS-WITH-KNOWN-DELTAS on everything Step 7 froze, and FAIL on two cells it merged.**

Nothing that was frozen moved. 43 of the 48 replayable frozen configurations are bit-identical under
their own interpreters, the other 5 are Step 6's known set unchanged, the 49th meets the world-row
obligation exactly, the six re-baselined locomotion cells reproduce their re-freeze digests
bit-for-bit, and every Step-6 `tests/gpu` failure and error is carried unchanged with none fixed and
none added outside the two cells named below.

Five of the seven cells Step 7 added pass Gate A on every device they support.

`OGBCube` and `OGBScene` do not. They fail a mandatory Gate-A check (`check_state_roundtrip`) and
thirteen further `tests/gpu` assertions including a sub-batch rendering invariant and a crash, and
`OGBCube` no longer meets its own declared `mean_abs` ceiling. None of this was visible to the
workstream that merged them, because no interpreter it ran the suite in could construct them. The
coverage claim for `swm/OGBCube-v0` and `swm/OGBScene-v0` — that a renderer exists and draws them —
stands; the certification claim does not.

Recommended disposition, for the coordinator:

1. `OGBCube` / `OGBScene` Gate A is **FAIL**, not PASS, until `check_state_roundtrip` and the
   sub-batch invariant are fixed in `stable_envs/envs/ogb_manip.py`. The coverage matrix and the
   Step-7 page record them that way.
2. `OGBCube`'s `mean_abs` ceiling is a declaration decision, not a bug fix: 3.2 was `worst × 1.3`
   over the cell's own pool and does not cover the conformance draw.
3. `nvdiff.Scene.clip`'s CPU batch-size dependence (CLOSURE-1) reaches no shipped renderer and is the
   cheapest of the three to leave for Step 8.
4. Add `tests/gpu` in an OGBench-bearing interpreter to whatever gate merges a cell, since the
   mechanism that hid all of this is "the suite was green on a family it never constructed" — which
   is the exact failure `se6-verify` was built for in Step 6.
