# Step 9 — Workstream H (raycast3d / cpu3d / primitives)

**Branch** `step9/render-3d` · **parent** `d07ffa8` (step9/integration after Wave 0) ·
**Step-8 behavioural baseline** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc` ·
**host** Apple M5 Mac (`cis1b1zs2j3`), `/Users/samib/se6-venv` (torch 2.14 CPU, mujoco 3.12.0,
dm_control 1.0.45) and `/Users/samib/se8-vm-venv` for the DM Control subset.

**Result: PATCH, with one open item.** Six commits. On CPU: zero pixels moved on 21 cells, worst
paired timing ratio 1.017 against a 1.10 bar. On GPU (job 6464319, RTX A6000, `gpu1503`): 58 of 59
cell/device pairs byte-identical, worst paired timing ratio 1.006, and **one difference —
`HumanoidmazeGiantNavigate` on cuda only** — which is not yet classified and is the subject of a
prepared follow-up job (§9a). Until that job runs, plan §17's "unexplained oracle differences: zero"
is **not** satisfied and this branch should not merge.

---

## 1. Files created / moved / removed

| file | what happened |
|---|---|
| `stable_envs/render/primitives.py` | **new.** The vocabulary both 3-D backends share: the mjtGeom kind codes and `PRIMITIVE_KINDS`, `GEOM_NAMES`, `INF`, the unrolled `dot3`/`mv3`/`mtv3`, the bounding-radius law in its two spellings (`bound_radius` numpy, `bound_radius_tensor` torch), and the `Primitives` scene description. Imports only numpy and torch. |
| `stable_envs/render/raycast3d.py` | Those definitions **moved out** and are re-imported under their old names; the five near-copies of the memo idiom became one `_Memo`; type aliases `Kinds`/`Cast` and annotations on the cast/shade API. |
| `stable_envs/render/cpu3d.py` | Consumes `primitives` instead of its own copies (kind codes, `GEOM_NAMES`, `_dot`, `INF`, and a throwaway `Primitives` built only to reach `bound_radius`); `MOVER_KINDS` is now derived from `PRIMITIVE_KINDS`; `WINDOW_EMPTY` names the third return of `_mover_window`; annotations and two missing docstrings. |
| `tests/unit/test_render_primitives.py` | **new**, 16 tests: see §4. |
| `scratch/step9/render-3d/{oracle.py,bench.py,gpu_oracle.sbatch}` | **new** evidence harness. |
| `examples/results/step9/render-3d/{oracle_cpu_local.json,timings_cpu_local.json,findings.md}` | **new** artefacts. |

Nothing was removed from any other file. No file outside `render/primitives.py`,
`render/raycast3d.py`, `render/cpu3d.py`, their tests and `scratch|examples/results/step9/render-3d/`
was touched.

**Consumer call sites changed: none.** Every symbol the five consumers (`envs/dmcontrol.py`,
`envs/ogbench.py`, `envs/ogb_manip.py`, `envs/maze2d.py`, `envs/mujoco_locomotion.py`) and the tests
reach for — including the private ones the tests use (`R3._dot`, `R3._mv`, `R3._mtv`,
`R3._bound_radius`, `R3._sphere`, `R3._rays_near`, `R3._unit_rays`, `R3._expand_tile`,
`R3._fit_tile`, `R3._upsample_t`, `R3._composite_window`, `R3._kind_tensor`, `R3._BOUNDED_KINDS`,
`R3._UNIT_D_TOL`, `R3._BOUND_SLACK_NOISE`, `C3.MOVER_KINDS`, `C3.PLANE`) — resolves to the same
object at the same name. The re-export is deliberate: a rename would have cost ~74 edits in files
this workstream does not own and bought nothing the import line does not already say.

## 2. API changes

Additive only.

* New module `stable_envs.render.primitives` with `PLANE/SPHERE/CAPSULE/ELLIPSOID/CYLINDER/BOX`,
  `PRIMITIVE_KINDS`, `GEOM_NAMES`, `INF`, `dot3`, `mv3`, `mtv3`, `bound_radius`,
  `bound_radius_tensor`, `Primitives`. It is internal (`render/` is not in the public namespace per
  the Wave-0 contract §1) and imports nothing from `envs/`, core or SWM — the parametrised
  dependency-direction test `test_a_renderer_knows_nothing_about_environments_or_stable_worldmodel`
  picked the new file up automatically and passes.
* `raycast3d.Kinds` / `raycast3d.Cast` type aliases; annotations on `intersect`, `cast_nearest`,
  `cast_nearest_torch`, `any_blocking`, `any_blocking_torch`, their unbounded twins, `_rays_near`,
  `shade`, `cached_rays`, `Camera.basis`, `Camera.rays`.
* `raycast3d._Memo` (private) and `cpu3d.WINDOW_EMPTY`.
* Removed: nothing. Renamed: nothing (the moved functions are `dot3`/`mv3`/`mtv3`/
  `bound_radius_tensor` in their new home and keep `_dot`/`_mv`/`_mtv`/`_bound_radius` in
  `raycast3d`).

## 3. Step-8 IDs affected

Every cell that reaches either module. The list was **measured, not read off the imports**: each
registered environment was built on CPU with the six backend entry points wrapped in counters, and
these are the ones that called something.

| ID | cell | tier | backend path |
|---|---|---|---|
| `dm_control/{acrobot-swingup, ball_in_cup-catch, cartpole-balance, cartpole-swingup, finger-spin, finger-turn_easy, finger-turn_hard, pendulum-swingup, reacher-easy, reacher-hard, point_mass-easy, manipulator-bring_ball}` | 12 DMC parity cells | parity | `raycast3d` static plate + analytic cast |
| `dm_control/{cheetah-run, hopper-hop, humanoid-stand, walker-walk, quadruped-walk}` | 5 trackcom cells | visual | `cpu3d.render` on CPU (movers via `raycast3d.cast_nearest`), MJWarp on CUDA |
| `gymnasium/{InvertedPendulum-v5, InvertedDoublePendulum-v5, Pusher-v5, Reacher-v5}` | 4 gym MuJoCo | parity | `raycast3d` |
| `ogbench/{pointmaze,antmaze,humanoidmaze}-*` | 17 maze cells | parity | `raycast3d.composite` (CUDA only here — `ogbench` is installed in neither local venv) |
| `PointMaze_{UMaze,Open,Medium,Large}-v3` | 4 maze2d | parity | `raycast3d.composite` (needs `gymnasium_robotics`, absent locally) |
| `swm/{OGBCube,OGBScene}-v0` | 2 OGB manipulation | visual | `raycast3d` on CUDA; **no CPU backend by declaration** (`CPU3D-OGB-1`) |

21 of these render on the Mac and are covered by the CPU arm below; the remaining 23 are the GPU
arm's job and are already enumerated in `oracle.py`'s `CELLS`, skipped with a recorded reason here.

## 4. Targeted tests

| suite | venv | result |
|---|---|---|
| `tests/unit -x --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"` | se6-venv | **1263 passed**, 195 skipped, 112 deselected (56 s) |
| same, at the branch parent `d07ffa8` | se6-venv | 1246 passed — the brief's baseline, reproduced |
| `test_variation_maps_dmc_control.py`, `test_raycast3d_*.py`, `test_cpu3d.py`, `test_render_primitives.py` | se8-vm-venv | **341 passed** |
| `tests/gpu -k "raycast or cpu3d or DMC or Pointmaze or Fetch"` | se-step6 on `gpu1503` | **no result** — the summary was truncated out of the log by my own sbatch; re-run queued (§9) |

The 1246 → 1263 delta is fully accounted for: **+16** new tests in
`tests/unit/test_render_primitives.py` and **+1** because
`test_foundation_imports.py::test_a_renderer_knows_nothing_about_environments_or_stable_worldmodel`
is parametrised over `render/*.py` and now includes `primitives.py`. A collected-ID diff against the
parent confirms that is the only new ID and that nothing disappeared.

The new tests pin exactly what the extraction is supposed to buy, i.e. each way the two backends
could still disagree: the kind codes are mjtGeom's own numbers and both backends read the same ones;
`R3._dot is primitives.dot3` by **identity** (a second copy of the expression would pass a value
comparison, which is the failure mode being removed); `mv3`/`mtv3` are the matrix product and its
transpose and are bit-identical across batch sizes (the property the unrolled spelling exists for);
the numpy and torch bounding radii agree for every kind including ELLIPSOID and actually contain a
sampled surface; every raycast3d memo is bounded; and both dispatchers refuse `mjGEOM_MESH` with the
same reason.

One trap is recorded rather than fixed: **`INF` is `1e30`, which `np.isfinite` calls finite**, so
`Primitives.radii`'s `isfinite` guard never fires for a plane and its docstring's "or None if any
primitive is unbounded" is not what protects the screen-space caller — `screen_span`'s
`torch.isfinite(ndc)` probe on the projected corners is. That is Step-8 behaviour and is preserved
unchanged; the test says so, so a later reader "fixing" the guard finds out in a test rather than in
a frame.

## 5. Certification checks

Nothing in this patch touches a threshold, a declaration, a tier or a frozen digest. The applicable
harnesses are `tests/gpu/test_conformance.py`, `test_reproducibility.py` and `test_fast_paths.py`
(which diffs the fused Triton cast against `cast_nearest_torch` and the compacted composite against
`_composite_window` **on the environments themselves**) — all three are in the GPU arm, §9.

## 6. Default-frame comparison (the oracle)

`scratch/step9/render-3d/oracle.py` → `examples/results/step9/render-3d/oracle_cpu_local.json`.

Both arms run in the same venv on the same host, one with a detached `git worktree` of `21cb306` on
`PYTHONPATH` and one with this branch, through the public path only (`cls(batch=n, device=...)`,
`set_state`, `render()`), over the **frozen corpora** — `examples/results/step4/states/*.npz` for
the DMC parity and trackcom cells, `examples/results/step7/**` for Manipulator and Quadruped, 64
states each — and over seeded `sample_states` for the five cells that never had a frozen corpus
(DMCPointMassEasy and the four gym MuJoCo wrappers).

**21 cells, 64 states each, zero differences** in any compared key: the frame SHA-256, the frames
rendered from explicit rows, the state digest each arm was handed, the frame shape, the registered
size, and the *set of backend entry points the cell actually called* (so a cell that silently
stopped using the backend would fail rather than pass). The 17 cells whose dependency is absent
locally were skipped with byte-identical reasons in both arms.

Because three module switches select between a fast expression and the readable one it must equal,
two cells (`DMCReacherHard`, `DMCBallInCupCatch`) were additionally rendered with `PRIMITIVE_BOUNDS`,
`COVER_COMPACTION` and `COMPOSITE_COMPACT_MERGE` off, in both trees. All four digests per cell are
the same one string: the refactor preserved the reference expressions too, not only the shipped
path.

## 7. Variation / state comparison

Untouched by construction: neither module knows what a variation is. `mat_rgb` and `size_scale`
reach `composite` as tensors and pass through expressions this patch did not modify, and the
oracle's per-cell run uses each env's registered default style, whose frames are identical. The
DM Control variation map suite passes (341 tests in se8-vm-venv with the raycast/cpu3d suites).

## 8. Performance

`scratch/step9/render-3d/bench.py` → `timings_cpu_local.json`. Five **alternating** baseline/head
rounds (a before-block followed by an after-block measures the machine's mood; interleaving does
not), five timed `render()` calls per leg, separate interpreter per leg, single-threaded, frozen
states, at the registered tier sizes.

| cell | px | batch | baseline ms | head ms | ratio | same-arm spread |
|---|---|---|---|---|---|---|
| DMCReacherHard | 84 | 64 | 168.74 | 171.53 | **1.017** | 5.6 % |
| DMCBallInCupCatch (36 primitives) | 84 | 64 | 333.91 | 333.36 | 0.998 | 4.8 % |
| DMCFingerTurnEasy | 84 | 64 | 208.97 | 210.88 | 1.009 | 2.8 % |
| DMCManipulatorBringBall | 224 | 16 | 280.78 | 273.68 | 0.975 | 1.2 % |
| MujocoPusher | 240 | 16 | 170.86 | 170.17 | 0.996 | 4.7 % |
| DMCWalkerWalk (cpu3d) | 84 | 64 | 241.93 | 241.94 | 1.000 | 4.5 % |
| DMCHumanoidStand (cpu3d) | 84 | 64 | 291.70 | 292.21 | 1.002 | 3.5 % |
| DMCQuadrupedWalk (cpu3d, ellipsoid) | 224 | 16 | 393.86 | 394.15 | 1.001 | 2.7 % |

Worst ratio **1.017**, inside every row's own same-arm spread and far inside the 1.10 bar. An
independent smoke run at 3 rounds x 3 reps gave the same picture (worst 1.021). **No regression.**
This is expected rather than lucky: no arithmetic, no allocation and no kernel launch changed — the
moved functions are the same objects called through the same names, and `_Memo.put` runs only on a
cache miss.

GPU timings are §9's job: the same script with `--set gpu` runs both legs on the same card inside
the one SLURM job, over the fused Triton cast and the tiled composite, which is where the CPU arm
has nothing to say.

## 9. GPU arm — job 6464319 (COMPLETED, 29:39, `gpu1503`, RTX A6000, driver 575.57.08)

`scratch/step9/render-3d/gpu_oracle.sbatch`, tree `$SCR/se9/render-3d` at `899e4a6`,
`baseline_present yes`, venv `$SCR/venvs/se-step6/bin/python` (torch 2.14.0+cu126, triton 3.8.0).
Artifacts (absolute `/oscar/...` paths scrubbed to `$SCR`): `oracle_gpu_6464319.json`,
`timings_gpu_6464319.json`, `provenance_gpu_6464319.txt`.

**Oracle, `--devices cpu,cuda`, 64 frozen states per cell: 59 of 76 cell/device pairs ran, 58
byte-identical, 1 different.**

* cpu: 32 of 38 ran, **all identical** — including the 17 OGBench maze IDs, which the Mac cannot
  build, and the five `cpu3d` trackcom cells.
* cuda: 27 of 38 ran, 26 identical. The two switch cells were also rendered on cuda with
  `PRIMITIVE_BOUNDS`, `COVER_COMPACTION` and `COMPOSITE_COMPACT_MERGE` off: every digest equals the
  shipped one, so the fused Triton cast, the torch fallback and the uncompacted composite all agree
  with the baseline.
* **The difference: `HumanoidmazeGiantNavigate` on cuda.** Its own cpu arm is identical, and every
  other maze (3 other humanoidmazes, 4 antmazes, 4 pointmazes) is identical on both devices. The
  frame mean moves from 172.771413583 to 172.771393653 — 1.99e-5 over 9,633,792 subpixels, a signed
  total of about 192 levels, i.e. of the order of three levels per frame over 64 frames. See §9a.

**Environment gaps in `se-step6` — these IDs are still uncovered on cuda and nothing in this job
says anything about them:**

| missing package | consequence |
|---|---|
| `warp` | the 5 DM Control trackcom cells skipped on **cuda** (that is the MJWarp path, Workstream I's, not this one). Their **cpu** arm — `cpu3d`, which is mine — ran and is identical. |
| `stable_worldmodel` | `swm/OGBCube-v0` and `swm/OGBScene-v0` skipped on cuda; they have no CPU backend by declaration, so **these two IDs are not covered at all**. |
| `gymnasium_robotics` | the four `PointMaze_*-v3` maze2d IDs skipped on both devices — **not covered at all**. |

**GPU paired timings** (same card, same job, 5 alternating rounds x 5 reps):

| cell | batch | baseline ms | head ms | ratio | same-arm spread |
|---|---|---|---|---|---|
| DMCReacherHard | 256 | 14.61 | 14.69 | 1.006 | 2.0 % |
| DMCBallInCupCatch | 256 | 40.10 | 39.97 | 0.997 | 0.7 % |
| DMCManipulatorBringBall | 64 | 11.08 | 11.07 | 0.999 | 1.8 % |
| PointmazeLargeNavigate | 256 | 5.58 | 5.59 | 1.001 | 3.7 % |
| AntmazeLargeExplore | 256 | 8.25 | 8.26 | 1.001 | 1.3 % |
| HumanoidmazeGiantNavigate | 64 | 6.02 | 5.99 | 0.994 | 7.6 % |

Worst **1.006** against the 1.10 bar; OGBCube and DMCWalkerWalk skipped for the missing packages
above. **No GPU regression.**

**GPU tests: no result, and that is my harness's fault.** The sbatch piped pytest into `tail -40`,
and MuJoCo's EGL teardown — dozens of `Exception ignored in __del__` lines printed *after* pytest's
verdict — pushed the summary out of the window. `TEST_RC=1` is therefore uninterpretable: it is
equally consistent with real failures and with a nonzero exit from teardown. The sbatch now writes
the whole output to `$OUT/gpu_tests_<tag>.log` and echoes only the summary, and the selection is
re-run in the follow-up job.

## 9a. The open item, and the job that will classify it (prepared, NOT submitted)

`scratch/step9/render-3d/hmg_determinism.py` + `hmg_determinism.sbatch`
(`--partition=gpu --qos=norm-gpu --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=00:45:00`).

**The prior question is not "what did the patch change" but "is this cell reproducible at all".**
The patch contains **no arithmetic change whatsoever** — `git diff 21cb306..HEAD` over both modules
is imports, type annotations, comments, the `_Memo` refactor and the moved-verbatim definitions;
every numeric expression is character-identical and the moved functions are the same objects under
the same names. And this is precisely the configuration `raycast3d._BOUND_SLACK_NOISE` already
documents as the one with the lever arm: the camera is ~100 m from a 5 cm capsule, `c = olx^2 +
oly^2 - r^2` loses the radius entirely in float32 there, `disc > 0` then decides hit-or-miss on
rounding noise, and job 6348207 moved one pixel of 3,211,264 by 22/255 on **this cell and no
other**.

So the follow-up runs, on one card in one job: three *fresh processes per tree* (baseline vs
baseline, head vs head, baseline vs head), a second render inside one process, the same render with
`FORCE_BACKEND="torch"` and with `PRIMITIVE_BOUNDS=False` (which says whether any disagreement lives
in the Triton kernel or in the shared torch expression), and a pixel-level diff (how many pixels, by
how many levels, in how many frames). It prints one `VERDICT` line:

* *baseline is not process-reproducible* → the oracle difference is not evidence about the patch;
  record it as a pre-existing property of the largest maze on this card.
* *both trees self-consistent and different* → the patch changed this cell, and the merge is blocked
  until it is explained.

The script was smoke-tested locally end to end on `DMCReacherHard`/cpu (both trees self-consistent,
identical, 0 differing pixels of 1,354,752), so only the cuda answer is missing. The same job
re-runs the GPU test selection with its summary preserved.

## 10. Resource lifetime

Neither module owns a process resource: no EGL context, no CUDA context, no file handle, no MuJoCo
renderer. What they do own is five module-level memos in `raycast3d` (camera basis, corner masks,
kind vector, ray bundle, tile offsets) and one per-instance `Primitives._radii_cache`. All five are
now the same `_Memo` type and every one carries a cap (32, 32, 64, 8, 16);
`test_every_raycast3d_cache_is_bounded` asserts it. `cpu3d.Scene` holds only torch tensors and is
released by refcount. Nothing here needs a `close()` and nothing depends on `__del__`, so plan §2's
resource rule is already satisfied and Workstream I's work does not extend to these files.

## 11. Unexplained oracle differences

**CPU: zero** (21 cells, 64 frozen states each, fast and reference expressions).
**GPU: one** — `HumanoidmazeGiantNavigate` on cuda, of 59 cell/device pairs — **currently
unclassified**. It is not yet attributable to the patch (the patch has no arithmetic in it) and it is
not yet excused either; §9a is the job that decides which. Plan §17 requires zero, so **this is an
open gate item, not a footnote**.

## 12. Deliberate non-consolidations (per plan §16, recorded so they are not re-proposed)

* **`cpu3d`'s two inline ray/plane tests** (primary rays in the local frame, shadow rays in world
  dot products) look like one law written twice, and merging them would change the bytes: they are
  not the same float program — different decomposition, different shapes, different rounding. The
  frames are certified against MJWarp's picture, so this is not a place to be tidy.
* **`cpu3d._mover_window` vs `raycast3d.screen_span`/`_fit_tile`** both derive a pixel window from
  bounding radii, but through different cameras (a `fovy` frustum against an `mjvGLCamera` frustum),
  different padding and different fallbacks. Two callers do not make one abstraction when the
  expression differs.
* **`cpu3d._normalize` vs the normalisations inside `shade`/`_capsule`** differ in their guard
  (`where(n > 0, n, 1)` against `clamp(min=1e-12)`). Unifying them would move values.
* **`static_depth`'s per-geom `torch.tensor` loop** could be one `cast_nearest` call, but its
  reduction is `minimum` rather than the ordered `t < best`, and it runs once per baked scene rather
  than per frame. Not worth a behavioural risk for a bake-time cost.

## 13. Semantic follow-up recommendation

**Deferred to the §9a verdict.** No pixel moved on 21 CPU cells or on 26 of 27 cuda cells, so there
is nothing for a DINO tripwire to be sensitive to there. If §9a returns "the patch changed this
cell", `HumanoidmazeGiantNavigate` needs a paired semantic check before merge; if it returns
"baseline is not process-reproducible", none is needed and the instability should be recorded
against the cell rather than against this branch.

## 14. Commits

```
7f8a7af  render: extract the 3-D primitive vocabulary both backends share
c0c98e0  raycast3d: one bounded memo instead of five near-copies
a0887c7  step9 render-3d: oracle, paired timing harness, GPU sbatch
6bf885a  render 3-D: types on the cast API, a named window sentinel, tests for the shared vocabulary
899e4a6  step9 render-3d: findings, CPU oracle record and paired timings
<this>   step9 render-3d: GPU arm results, and the probe for the one difference
```
