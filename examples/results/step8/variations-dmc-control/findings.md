# Step 8 Workstream A — VariationMaps: DMC + classic control — findings

**Verdict: `PATCH_CANDIDATE`** (with one `FINDING` for the coordinator, §6, and an upstream bundle
of twenty-one dead swm leaves, §3).

* **Baseline** `43ac95238f7472137619f3e69e7636e11dd088b1`
* **Branch** `step8/variations-dmc-control`
* **swm** `galilai-group/stable-worldmodel` @ `176d545`, editable in `/Users/samib/se8-vm-venv`
  (python 3.12, torch 2.14.0 CPU, mujoco 3.12.0, dm_control 1.0.45, gymnasium 1.3.0, `MUJOCO_GL=glfw`).
  `/Users/samib/se6-venv` was not modified.
* **Full write-up**: [`docs/research/step8_variations_dmc_control.md`](../../../../docs/research/step8_variations_dmc_control.md)
* **Machine-readable**: [`leaf_classification.json`](leaf_classification.json); raw probes in
  [`measure/`](measure/), probe sources in `scratch/step8/variations-dmc-control/`.

## 1. Coverage

17 swm ids, 18 stable-envs cells, **161 leaves, 0 undocumented**.

| | mapped | unsupported | dead_upstream | not_applicable | total |
|---|---:|---:|---:|---:|---:|
| **all 17 ids** | **46** | **94** | **21** | **0** | **161** |
| classic control (5 ids) | 15 | 24 | 0 | 0 | 39 |
| DM Control (12 ids) | 31 | 70 | 21 | 0 | 122 |

`not_applicable` is unused by design, per the coordinator's ruling: a physics-only leaf of a
render-from-state cell is `unsupported`-by-that-cell, because it asks for a dynamics surface the cell
does not have rather than asking nothing. The 94 split into 11 appearance leaves the cell has no knob
for, 13 appearance leaves that arrive through the physics (§4), and 70 physics-only leaves measured at
0 px — and each leaf's reason says which it is, and whether the cell in question simulates.

Per-id counts are in `leaf_classification.json` and asserted by
`test_classification_counts_match_the_declaration` against the **live** swm space.

## 2. Registered defaults — no pixel movement

* **Classic control:** the mapped swm default renders **byte-identical to swm's own frame** over 8
  states on all five (`test_classic_default_matches_the_official_swm_render`).
* **DM Control:** 8 of 13 cells — the mapped swm default **is** the cell's default style, 0 px.
* 5 cells reach a genuinely different swm-registered appearance through named leaves (the Step-7
  "Reacher pattern" ruling): `DMCBallInCupCatch`, `DMCCheetahRun`, `DMCFingerTurnHard`,
  `DMCManipulatorBringBall`, `DMCReacherHard`. The test asserts the exact leaf set.
* **Pixel-movement classification: NONE.** No renderer, no cell and no declaration was changed; every
  frame this branch can produce at a default style is the frame the baseline produced.

## 3. Upstream findings — 21 swm leaves are dead (for Workstream D's bundle)

Sixteen leave the compiled model unchanged: eleven are never read and five density writes are
overridden by explicit mass. Five more reach hidden target elements and change no pixels.
Model effects were measured by hashing every `MjModel` array at two leaf values through the live wrapper.

*Never read by swm's own `modify_mjcf_model`* (7 `*_locked` leaves, plus
`BallInCup.ball.color`, `Cartpole.agent.cart_shape`, `Pendulum.agent.mass_shape`,
`Manipulator.agent.upper_arm_length`).
*Written but ignored by MuJoCo* — the geom carries an explicit `mass`, which beats `density`:
`Acrobot.agent.{upper,lower}_arm_density`, `Cartpole.agent.pole_density`,
`Pendulum.agent.{mass,pole}_density`.
*Reaches the model, is never drawn* — geom/site group 3 or 4, outside `mjvOption`'s 0-2:
`Acrobot.target.{color,shape}`, `BallInCup.{rendering.render_target,target.color,target.shape}`.

None is a stable-envs defect; none should be implemented.

## 4. Two measurements that changed a classification

* **`trackcom` mass leaves move pixels.** On cheetah/walker/hopper/humanoid/quadruped the camera is
  `mode="trackcom"` and rides `subtree_com`, so a density change moves the camera: 2,115-137,533 px
  over 8 states with `qpos` and `geom_xpos` bit-identical. Every non-trackcom cell measures exactly
  0 px. These ten leaves are `unsupported`, not physics-with-no-pixel.
* **`BallInCup` gravity moves pixels** through `mjv_catenary`, which draws the cup-ball tendon from
  the model's gravity: `gravity.x` 178 px, `gravity.y` 110 px at a fixed `qpos`.

## 5. Two semantics differences, measured and documented, not removed

* **Classic control recolor.** swm recolours the rendered frame by exact uint8 match; we draw in the
  colour. 0 px at the registered default on all five; up to 0.59 % of the frame at a probe value, all
  of it on anti-aliased boundaries.
* **`light.intensity` dark end.** Mapped over [0.2 * base, 1.0]; swm's [0, 0.2 * base) is unreachable
  (base 0.7 -> gap [0, 0.140); base 0.6 -> gap [0, 0.120)).

## 6. FINDING — `floor.color` is an escalation candidate, and the recommendation is to ship the gap

swm sets the `grid` texture's `rgb1`/`rgb2` (6 scalars); our only floor knob is `material.grid` (3),
and `_tex_rgb` collapses the checker to its mean colour. **0 % of swm's range is reachable** — a
measure-zero subset — while **the registered default is exact and needs no leaf**, because swm's
`init_value` is `common/materials.xml`'s own grid texture. Closing the range needs a texture edit plus
a change to `_bundle`'s never-closed one-Renderer-per-(domain, size) policy, i.e. a renderer change,
which plan §4 says to stop and report. Recommend `DOCUMENTED_LIMITATION`. Coordinator decision.

## 6b. Class extension for Workstream C — `affine`

The coordinator routed Workstream C's request (`OGBMaze.floor.pixel_encoding`, a switch rendered with
inverted sense) here. Delivered as `VariationMap.affine`: `our_path -> (swm_path, factor, offset)`
with `ours = theirs * factor + offset`, so an inverted switch is `(-1.0, 1.0)` and a unit shift is
`(1.0, k)`. An `inverted` flag was not chosen because it answers one case and this answers the class
of them at the same cost.

* NOT a widening of `scaled`: `pairs()` still returns two-tuples, `affine_pairs()` is the uniform
  three-tuple view. Changing `scaled`'s arity would break any caller that unpacks it, silently.
* `factor == 0` is refused in `__post_init__` — it is not invertible, so the leaf could not round-trip.
* Backward compatibility is asserted, not asserted-about: `test_the_four_pre_step8_maps_are_untouched`
  checks `unmapped() == swm_only`, `mapped_swm_paths() == {t for t, _ in pairs().values()}` and
  `affine_pairs() == {o: (t, f, 0.0) for o, (t, f) in pairs().items()}` for all four pre-Step-8 maps.
* Exercised by `test_affine_is_invertible_and_additive` (including the inverted 0/1 switch) and
  `test_a_non_invertible_affine_is_refused`.

Workstreams B and C should build on it, and should add their maps as a module under
`stable_envs/compat/variation_maps/` plus one line in that package's `_MODULES` — not by editing
`worldmodel.py`.

## 7. Shared files touched

| file | change |
|---|---|
| `stable_envs/compat/worldmodel.py` | **shared.** Seven additive `VariationMap` fields (including `affine`, added at the coordinator's request for Workstream C) + four methods; `style_from_variation`/`variation_from_style`/`check_mapping` honour them; new `swm_variation_space()` (bug fix: `unwrapped.variation_space` is wrong for the five classic-control ids); ONE registration line. Backward compatibility of the four pre-Step-8 maps is asserted by test. |
| `stable_envs/compat/variation_maps/__init__.py` | **new, and shared going forward** — Workstreams B and C add a module plus one line in `_MODULES` here and never touch `worldmodel.py`. |
| `stable_envs/compat/variation_maps/dmc_control.py` | new, owned by this workstream |
| `tests/unit/test_variation_maps_dmc_control.py` | new, owned by this workstream |
| `docs/research/step8_variations_dmc_control.md` | new, owned by this workstream |
| `examples/results/step8/variations-dmc-control/**`, `scratch/step8/variations-dmc-control/**` | new, owned by this workstream |

`stable_envs/envs/dmcontrol.py` and `stable_envs/envs/classic_control.py` are **owned but unchanged**:
no cell needed a new leaf to make the registered default reachable.
Nothing under `stable_envs/render/`, `stable_envs/testing/`, `tests/gpu/` or any frozen corpus was
touched.

## 8. Tests

| suite | venv | result |
|---|---|---|
| `tests/unit/test_variation_maps_dmc_control.py` | `se8-vm-venv` (with swm) | **138 passed** |
| `tests/unit/test_variation_maps_dmc_control.py` | `se6-venv` (no swm) | 10 passed, 128 skipped |
| `tests/unit` | `se6-venv` | 1043 passed, 137 skipped, **2 failed — both pre-existing at the baseline** (`test_examples.py::...robocasa_visual/render_mjwarp.py`, `test_repo_contracts.py::test_every_cited_scratch_path_resolves`), verified by re-running them with this branch's changes stashed |

`tests/gpu` not run. This patch changes how callers select existing styles; it does not change
the renderer that consumes a state/style. The renderer-gate carry-forward scope and evidence are
recorded in §11; this is not a claim of a fresh GPU pass. No SLURM jobs were submitted for this patch.

## 9. Method note

`test_dead_upstream_leaves_really_are_dead` runs its probe in a **subprocess**: every swm variation
reset that changes a model-level leaf makes swm recompile and drop the previous `dm_control`
`Physics`, which frees an EGL context on GC, and `stable_envs.testing.resource_guard` correctly fails
a test that releases one. The isolation is what makes the measurement safe to run beside the suite.

## 10. Semantic follow-up

**None required.** No renderer semantics changed, no pixels moved at any default, no fidelity
declaration or frozen corpus was touched.

## 11. Review follow-up and the ten-part merge gate

This record applies to implementation commit `063f2168ebcc563cd7f433c99dc3d2140d25530c` plus the
review follow-up correcting prose/docstrings only. It does not authorize a release limitation,
alter any gate contract, or turn an inherited failure/unmeasured status into a pass.

Independent review on 2026-09-16 reproduced all **138 tests** using live SWM in `se8-vm-venv`
(`MUJOCO_GL=glfw PYTHONPATH=. .../bin/python -m pytest -q tests/unit/test_variation_maps_dmc_control.py`).
It also regenerated `leaf_classification.json` byte-for-byte and reproduced the generated documentation
tables byte-for-byte. Therefore the generator and generated taxonomy do not need a semantic change:
the corrections are to hand-written prose (21 dead leaves = 11 never-read + 5 ignored density +
5 hidden target) and the module docstring. Final counts remain 46 mapped, 94 unsupported, 21 dead,
0 not applicable across 161 distinct per-ID leaves.

### Renderer evidence applicability

`git ls-tree` at baseline `43ac95238f7472137619f3e69e7636e11dd088b1` and implementation
`063f2168ebcc563cd7f433c99dc3d2140d25530c` returns identical object IDs:

| Path | Identical Git tree object |
|---|---|
| `stable_envs/envs` | `b6ba355d0d997ed437db53cc7fcf8fd33993d949` |
| `stable_envs/render` | `6abd350f218d5f41f7ed9c4158c4b9fe43fa6055` |
| `stable_envs/testing` | `0a6299acdbd0124fb2426993c14b9049be20e5d3` |
| `examples/results/step4` | `15a0fc2f71259a70eab41d67158e913a67b55628` |
| `examples/results/step7` | `93858185db77b45907c3560b7c44ca5480bf4558` |

The accepted corpora and earlier results themselves remain authoritative. No new replay result is claimed.
The only changed runtime files are `compat/worldmodel.py` and the new `compat/variation_maps`
package. Neither the renderer/cell implementation nor the testing/GPU replay code imports the
changed mapping helpers (`rg` for `style_from_variation`, `variation_from_style`, and
`compat.worldmodel` in `stable_envs/testing` and `tests/gpu` finds no matches).
For the frozen replay's same stored states/styles, the execution surface, corpus, and declared
reference remain identical. Existing renderer Gate-A and frozen replay evidence therefore carries
forward **with its original device/version applicability and limitations**, rather than requiring
a new GPU run for this mapping-only patch. This source-identity argument does not establish new
SWM API parity or certify newly selected nondefault styles; those use targeted mapping checks and
the later integration pass. The coordinator must still run the plan's final full-universe pass.

| §18 item | Evidence / disposition |
|---|---|
| 1. Explicit scope | Five classic-control IDs and twelve DMC IDs; mapping/setup only. |
| 2. Before/after diagnosis | Missing maps become measured leaf classifications; corrected prose agrees with generated artifacts. Existing appearance/range gaps remain visible in §5–6. |
| 3. Targeted unit tests | Independently reproduced 138 passes; follow-up changes only prose/docstrings. |
| 4. Affected Gate A/F | Default reproduction, classic official pixel equality, isolation, round-trip, model-light and dead-leaf tests passed. Unchanged renderer invariants carry forward only within the scope above. Full 42-ID Gate F remains D's deliverable, not a claimed result here. |
| 5. Affected frozen-corpus replay | Inherited unchanged by verified source/corpus identity as above; no fresh replay was performed or claimed. |
| 6. Pixel movement | No change to existing state/style rendering. Five SWM registered defaults intentionally select pre-existing nondefault styles; their differences remain recorded in the artifact. |
| 7. Performance regression | Not applicable: renderer hot paths and performance contracts unchanged. No new throughput claim. |
| 8. Independent verdict | Initial review APPROVE-WITH-CHANGES; this follow-up requires explicit re-review and APPROVE before merge. |
| 9. Semantic follow-up | No added renderer information, so no downstream rerun required for this patch. No broader semantic claim added. |
| 10. Dashboard | Coordinator owns recording the follow-up SHA, final reviewer verdict, actual changed files and remaining decisions before merge. |

The floor range-gap decision in §6 remains open for release closure. Carrying forward unchanged
renderer evidence does not approve shipping that feasible-but-unimplemented variation, relax
fidelity, or authorize a re-freeze.
