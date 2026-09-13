# Step 6B — Reacher target visibility, and the metric that could not see it

**Classification: PATCH**

| | |
|---|---|
| baseline SHA | `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`; `git merge-base --is-ancestor` verified) |
| branch | `step6/reacher-correctness` |
| HEAD | `ec3ac960aba3f7301bbb9c7c9787eaf0937391a9` |
| the two required commits | `262a7c8` Problem 1 (visibility), `a60ae8f` Problem 2 (guard) |
| follow-ups on the same branch | `11ff2da`, `a9674df`, `465354e`, `0aa0d59` verification harness; `1d620f7` re-derive the guard's bound after it failed on GPU; `93eeb85`, `ec426e4`, `ec3ac96` robustness + recorded numbers |
| hardware | Oscar `--partition=gpu --qos=norm-gpu --constraint=geforce3090`, NVIDIA GeForce RTX 3090 (gpu2113 and gpu2101), driver 575.57.08; Apple M5 Mac for CPU/static work |
| venvs | `$SCR/venvs/se-step5` (torch 2.14.0+cu126, mujoco 3.12.0, mujoco_warp 3.12.0, warp 1.17.0 — the only one with MJWarp, has no pytest) for pixels/replay; `$SCR/venvs/se-step6` (+ pytest 9.1.1, no mujoco_warp) for the suites; `/Users/samib/se6-venv` locally with `MUJOCO_GL=glfw` (macOS has no EGL) |
| SLURM job IDs | **6344667** (baseline-vs-branch pixels on the four affected cells + first dmc replay), **6344904** (all-16-cell pixel diff, the three GPU suites, `tests/unit`, the guard's GPU spread, the re-replay) |

---

## What was wrong

Two things, which are the same thing seen from two sides.

**1. Visibility had no representation.** `stable_worldmodel`'s reacher hides its target by zeroing the
target material's alpha (`render_target = 0`); MuJoCo then declines to draw the geom, which Step 5
measured from the outside rather than assuming — moving the reference's target changes 0/255
(`examples/results/step5/reacher/parity_probe.json`, `reference_target_drawn: false`). stable-envs'
DM Control appearance space carried material colour, light diffuse and camera fovy and **no
visibility**, so the Step-5 harness said "do not draw the target" by writing 10 m into the two state
columns the episode draws into `geom_pos['target']`, pushing the geom outside the camera frustum
(`examples/downstream_step5/reacher/swap_renderer.py:OFFSCREEN`, `state_bridge.read_state`'s
`target_override`, `parity_probe.py:OFFSCREEN`, asserted by
`tests/unit/test_step5.py::test_the_offscreen_placement_is_outside_any_reacher_scene`). The state
then did not mean what it said, and the geometry was still built, transformed and cast against every
frame.

**2. The fidelity contract could not see the difference.** `conformance.interior_mask` keeps only
pixels whose 3×3 neighbourhood in the reference is one flat colour, so an object on a textured floor
contributes no interior pixels at all. Measured: the two Step-5 target placements differ by a whole
red disc — mean |d| 0.459 against 0.171 — and report `interior_exact_frac = 0.9998668941566535` and
`interior_px = 82641`, identical to the last digit. At the shipped 84 px `DMCReacher*` reports
`interior_frac = 0.000000`, so the strong assertion is not merely blunted there, it does not apply.

## Affected configurations

Four of the 49 frozen Step-4 configurations gain a declaration. **Zero change what they draw at the
default appearance.**

| config | change |
|---|---|
| `DMCReacherEasy` | `style_dim` 17 → 18, new leaf `visibility.target` (geom) |
| `DMCReacherHard` | `style_dim` 17 → 18, new leaf `visibility.target` (geom) |
| `DMCFingerTurnEasy` | `style_dim` 17 → 18, new leaf `visibility.target` (site) |
| `DMCFingerTurnHard` | `style_dim` 17 → 18, new leaf `visibility.target` (site) |

The other 7 parity DM Control cells (`DMCCartpoleSwingup`, `DMCCartpoleBalance`,
`DMCPendulumSwingup`, `DMCAcrobotSwingup`, `DMCPointMassEasy`, `DMCFingerSpin`, `DMCBallInCupCatch`)
and all 5 trackcom cells are untouched in pixels *and* declaration: their `initialize_episode` draws
nothing per episode, and `DMControlTrackCom.VISIBILITY_KNOB = False`. `examples/results/step4/` was
not modified.

## Diagnosis, before any code changed

`design.md` in this directory, written first, records where the workaround lives, how
`render_target = 0` maps onto the official renderer, what the appearance space carried, the proposed
representation and the files it touches. The probes are in `scratch/step6/reacher-correctness/`
(its README says what each one answers). Three measurements decided the design:

* **dm_control's reacher DRAWS its target.** `mat_rgba['target'] = [0.6, 0.3, 0.3, 1.0]` and
  `geom_rgba['target']` is the compiler default, so `_effective_alpha` resolves to 1.0; hiding it in
  the reference changes 19 px, max 215/255, at 84 px. **So the stable-envs default must keep drawing
  it.** Its reference is `dm_control/reacher-{easy,hard}`, `parity_interior_min = 1.0` is measured
  against that, and the frozen digests are taken at the default. This is the one place the brief's
  parenthetical is read differently and deliberately: *"faithful to SWM" is taken to mean the knob
  must reproduce `render_target = 0` exactly when asked, not that the stable-envs default flips.*
  Flipping the default would break the Step-4 parity contract and change the frozen digests.
* **Both hiding mechanisms work in MuJoCo's own renderer, and per-element is the finer one.**
  `probe_hide.py`: on reacher-hard, zeroing the element's own rgba alpha and zeroing its material's
  alpha both change 19 px / max 215 — the target geom is that material's only wearer. On
  finger-turn_hard the element knob changes 43 px and the material knob 58, because `target` is worn
  by both the `target` and `tip` sites.
* **The census of what to declare was already in the code.** `probe_slots.py`: `_episode_slots` finds
  a per-episode draw on 4 of 11 parity cells (reacher ×2, finger-turn ×2) and on swimmer, and on
  nothing else.

## The intervention

### Problem 1 — `stable_envs/envs/dmcontrol.py`, and nothing else in `stable_envs/`

`DMControlSuite` declares one `Discrete(2)` `visibility.<element>` leaf per element the **episode
draws**, derived from `_episode_slots` via a new `_DMCModel._hideable()`, with the default read off
the pristine model's own resolved alpha. `_apply_variation` zeroes that element's own
`*_rgba[:, 3]` when the knob is 0.

The renderer needed **no new code**, only a way in: `_effective_alpha` already resolves alpha the way
MuJoCo does, `_visible` already drops a zero-alpha element, `_movers` therefore never lists it, and
the three plate shots are taken by MuJoCo itself, which skips it too. `PLATE_PATHS` is already *every*
leaf, so `render()`'s existing group-by-style path bakes one scene per distinct visibility and
scatters rows back by index — batching, batch/order invariance, render purity and style-as-state hold
by construction and are measured below.

Binary rather than a continuous alpha: MuJoCo **blends** `0 < alpha < 1` and the analytic ray-caster
is opaque-only, so a continuous knob would advertise a range two thirds of which this renderer cannot
honour, and no check in the suite could catch it.

**A second, worse instance of the same class, found in review and fixed here.**
`DMControlSuite._static_t` casts the static depth lazily, per (scene, supersample), because the
supersample is a render-time argument — and it derived the STATIC primitive set at that moment from
the **live** shared `MjModel`, which carries whatever the last bake applied. So a scene baked with an
element hidden, whose static depth was first cast after some *other* appearance had been baked, saw
the element visible again, failed to find it in `sc.movers`, and promoted it to a **static** primitive
at whatever `geom_pos` the last `_fk` had written. Reproduced on CPU over the 64 frozen states
(`scratch/step6/reacher-correctness/probe_static_leak.py`): `render(hidden, ss=2)` alone against
`render(hidden) → render(other appearance) → render(hidden, ss=2)` differed by **18 px / max 215 / 3
rows on `DMCReacherHard`** and **88 px / max 140 / 28 rows on `DMCFingerTurnHard`**; after the fix,
**0/255** on both. The supersample change is the trigger and is why nothing caught it: `static_t` is
cached per (scene, ss), and at an env's default supersample it is always cast during that scene's own
first render, while that scene's appearance is still on the model — and every check in this repository
renders at the default. Fixed by recording `gstat`/`sstat` on the `_Scene` at bake time (`sc.statics`,
already computed there) and having `_static_t` read them. Regression test:
`tests/gpu/test_reproducibility.py::test_a_baked_scene_does_not_inherit_the_last_bake_s_visibility`,
parameterised over every env declaring a `visibility.*` leaf, which renders hidden → other bake →
hidden at a fresh supersample and asserts 0/255.

**One latent bug fixed in passing.** `_used_materials` read `_effective_alpha` off the *live* shared
`MjModel`, which every bake mutates. Harmless until a bake could legitimately leave an element at
alpha 0 — after which the *declared* variation space of the next env built from the class would
depend on what the previous env last drew, the failure `_pristine`'s docstring records one array
earlier. It now reads the pristine snapshot, through new optional `rgba`/`mat_rgba` arguments on the
module-local `_effective_alpha`.

The Step-5 harness now uses the knob: `target_mode="hidden"` is the default, `Renderer._variation` is
the single place it is expressed, `"offscreen"` still works so `examples/results/step5/reacher/` stays
reproducible, and `parity_probe.py` measures `as_placed`, `hidden` and `offscreen` on the same states
in one run.

### Problem 2 — `conformance.check_task_object_visibility`

Three plug-in points, all additive:

| where | what |
|---|---|
| `stable_envs/testing/conformance.py` | `check_task_object_visibility(env, states=None, n=16, seed=0)`, `PRESENCE_TOL = 8` |
| `tests/gpu/test_conformance.py` | `test_a_declared_task_object_is_drawn_iff_the_reference_draws_it`, `TASK_OBJECT_PRESENCE_MARGIN = 0.50` |
| `examples/fidelity_step4.py` | recorded in `family_specific` on the frozen corpus; `not_applicable` for every env with no `visibility.*` leaf |

For each declared object it measures:

* **`footprint_px`** — the exact pixel set where the declared value and the flipped one differ, i.e.
  the object's own pixels. Zero means the declaration is inert.
* **`per_row_max_abs_diff`** — a style alternating the knob down the batch must reproduce the two
  uniform renders bit for bit. `check_per_env_variation` structurally cannot ask this of a `Discrete`
  leaf (it skips anything without `.low`), and per-row visibility is the reason visibility is style
  rather than a constructor argument.
* **`presence_margin = 1 − mean|d|(declared) / mean|d|(flipped)`** over that footprint, against the
  **official** frame, **pooled over the whole state stack** — one number per object, not one per
  frame. One when the declared arm is exact, zero when the two are equally wrong, sharply negative
  when they are the wrong way round. `per_row_max_abs_diff` is the only per-row quantity here, and
  `rows_with_footprint` says how many frames contributed pixels to the pool.
  `presence_agree_{declared,flipped}` (the within-8 fractions) are reported as diagnostics.

**Bound `TASK_OBJECT_PRESENCE_MARGIN = 0.50`** — "the declared appearance must be at least twice as
close to the official frame as the flipped one" — set from the spread over five draws (four of 16
sampled states plus the frozen 64-state Step-4 corpus) per cell, on **both** machines:

| cell | margin (CPU) | margin (RTX 3090) | declared mean\|d\| | flipped mean\|d\| |
|---|---|---|---|---|
| `DMCReacherEasy` | 0.9646 – 0.9676 | 0.9548 – 0.9575 | 5.2 – 7.2 | 156 – 161 |
| `DMCReacherHard` | 0.8730 – 0.8806 | 0.8513 – 0.8604 | 17.0 – 20.3 | 132 – 144 |
| `DMCFingerTurnEasy` | 0.8263 – 0.8506 | 0.9445 – 0.9503 | 5.1 – 16.9 | 97 – 104 |
| `DMCFingerTurnHard` | 0.8199 – 0.8713 | 0.9090 – 0.9147 | 9.0 – 18.4 | 99 – 108 |

The failure this exists to catch inverts the ratio — ~140/255 in the numerator against ~15 in the
denominator, a margin around −8 — so the pass and fail populations are nowhere near each other.

**The first draft of this bound was wrong and the branch caught it.** `presence_margin` was initially
the *difference of two within-8 agreement fractions*, floored at 0.25 from CPU draws (worst 0.3966 on
`DMCReacherHard`). The frozen corpus then measured **0.2422 on the 3090** — the branch would have
failed its own new bound on the certification hardware by 0.008. The number moved, not the renderer:
a task object's footprint is nearly all silhouette (reacher-hard's target is ~15 px a frame), so the
fraction of it landing within 8/255 is a statement about two rasterizers' edge rules — 0.60 on CPU,
0.26 on the GPU, identical code. Commit `1d620f7` replaces it with the error ratio, which cannot move
that way, and re-derives the floor on both machines.

Nothing existing is redefined, removed or re-thresholded: `interior_mask`, `pixel_agreement`,
`INTERIOR_EXACT_MIN`, `ALL_EXACT_MIN`, `MEAN_ABS_MAX`, every `parity_interior_min`,
`scratch/certification.json` and every frozen Step-2 bound are byte-identical on this branch.

---

## Verification

### Pixels — the headline

Both trees rendered the frozen Step-4 corpora at the **default appearance**, on one 3090, in one
SLURM job (6344904), over **every** DM Control cell — the same-card rule applied to digests, because
a "before" on one node and an "after" on another is not a before/after even for a hash
(`examples/results/step6/reacher-correctness/gpu/pixels_all.txt`):

```
VERDICT: 0 env(s) moved between the two TREES;
         4 env(s) where the BASELINE tree already differs from the frozen Step-4 digest
```

All 16 cells read `baseline_vs_branch=identical`. The four that differ from the frozen Step-4 digest
— `DMCCheetahRun`, `DMCHopperHop`, `DMCHumanoidStand`, `DMCWalkerWalk`, every one a **trackcom /
MJWarp** cell — differ **on the baseline too**, so that drift predates this branch (see the shared
findings below). The 11 parity cells, reachers included, reproduce the frozen Step-4 digest exactly:
`DMCReacherEasy` `4d684da4…`, `DMCReacherHard` `04e3961c…`, and `mean_pixel` 80.1058127243953 /
79.02629780210695 matching the frozen oracle to every digit.

### The guard fires (synthetic) and passes (frozen corpus)

`tests/unit/test_task_object_visibility.py` — 7 tests, CPU, no GL and no dm_control. A fake env draws
a disc over a checkerboard and a fake official renderer decides independently whether *it* draws one:

| case | result |
|---|---|
| both draw it | `presence_margin > 0.9` — pass |
| neither draws it (the SWM configuration, declared 0) | `presence_margin > 0.9` — pass |
| **we draw it, the reference does not** (the Step-5 situation) | `presence_margin < 0` — **FIRES** |
| **the reference draws it, we do not** | `presence_margin < 0` — **FIRES** |
| the flipped arm is bit-exact (ratio undefined) | reported as `why_no_margin`, asserted as a failure — **FIRES** |
| a style collapse to row 0 | `per_row_max_abs_diff > 0` — **FIRES** |
| the premise itself | `interior_exact_frac` and `interior_px` are *unchanged* by the same appear/disappear event |

That last row is the justification for the check existing, asserted rather than claimed: on the very
frames where the guard fires, the interior metric does not move.

On the frozen Step-4 corpora (GPU, 64 states, in the replay):

| cell | footprint px | declared mean\|d\| | flipped mean\|d\| | margin | per-row |
|---|---|---|---|---|---|
| `DMCReacherEasy` | 9 891 | 7.05 | 157.26 | **0.9552** | 0 |
| `DMCReacherHard` | 966 | 18.72 | 134.02 | **0.8604** | 0 |
| `DMCFingerTurnEasy` | 11 774 | 5.65 | 101.69 | **0.9445** | 0 |
| `DMCFingerTurnHard` | 1 827 | 9.17 | 105.42 | **0.9131** | 0 |

All ≥ 0.50. The margin is one pooled number per object over the 64-state stack; `rows_with_footprint`
is 64/64 on every cell, so every frame contributed pixels to that pool, and `per_row_max_abs_diff` —
the per-row quantity — is 0 everywhere.

### Targeted tests (job 6344904, RTX 3090, `se-step6`)

| suite | selection | result |
|---|---|---|
| `tests/gpu/test_conformance.py` | `-k "Reacher or FingerTurn or required_envs or registry_is_not_empty or relaxed_parity"`, `SE_REQUIRE_ENVS` = the four cells | **72 passed, 26 skipped, 0 failed** — includes `test_render_parity_with_official_renderer`, `test_style_is_state`, `test_default_style_is_the_official_appearance`, `test_per_env_variation_is_per_env`, `test_variations_all_do_something`, `test_state_is_sufficient` and the new `test_a_declared_task_object_is_drawn_iff_the_reference_draws_it` on all four cells |
| `tests/gpu/test_reproducibility.py` | `-k "Reacher or FingerTurn"` | **90 passed, 25 skipped, 0 failed** — includes the fresh-process bit-identity, batch-size independence, sub-batch style refusal, dataset round-trip and HuggingFace round-trip cases |
| `tests/gpu/test_fast_paths.py` | `-k "Reacher or FingerTurn or unknown_geom or flat_polygon or declared_fast_path"` | **23 passed, 10 skipped, 0 failed** |
| `tests/unit` (on Oscar) | full | **507 passed, 8 skipped, 2 failed** |
| `tests/unit` (locally, `MUJOCO_GL=glfw`) | full | **509 passed, 6 skipped, 2 failed** |

Both `tests/unit` failures are **present at the baseline commit, unchanged by this patch**, verified
by `git stash`: `test_examples.py::test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]`
and `test_repo_contracts.py::test_every_cited_scratch_path_resolves`. The latter's unresolved set is
back to **exactly the three entries it has at the baseline** — the test names them itself, and this
page deliberately does not repeat them, because that scanner reads every `.md` and `.txt` under
`examples/` and a report quoting an unresolved path becomes one more citer of it. The one citation
this patch adds resolves.

That same scanner is why **no raw pytest log is committed**. The first version of this directory
carried the full `tee`d output of each suite, and those logs contain absolute interpreter paths from
the cluster whose `.../scratch/...` substrings the scanner matches: the unresolved set went from 3 to
18 entries, on files that are evidence rather than documentation. The logs are now filtered to their
OUTCOME lines (`gpu/*.results.txt`), and both sbatch scripts gained a `keep` helper that tees the
full log to the cluster's own log directory, outside the tree, and writes only those lines here.

`tests/unit/test_declarations.py`, `test_repo_contracts.py` (bar the pre-existing case),
`test_style.py`, `test_step4.py`, `test_step5.py` and the new `test_task_object_visibility.py` run
clean locally: 161 passed, 2 skipped.

CPU measurements supporting the same claims (`probe_checks.py`, `probe_perrow.py`, `MUJOCO_GL=glfw`,
`device="cpu"`, `DMCReacherHard` and `DMCFingerTurnHard`): `variations_do_something` no inert leaves,
`per_env_variation` no collapses, `style_is_state` 0, `default_style_is_official` 0,
`state_dims_matter` no dead dims, `order_independence` 0/0, `batch_size_invariance` 0,
`render_is_pure` 0/0/0/0, `colors_apply_exactly` clean; and, with the knob alternating down a batch
of 8, on-rows match the all-visible render 0/255, off-rows match the all-hidden render 0/255, 8
distinct frames, permutation 0, sub-batch 0, style mutated by render 0.0.

### Step-4 replay — the whole `dmc` shard, not only the four

`examples/fidelity_step4.py --shard dmc` against the frozen `examples/results/step4/conformance/dmc.json`;
output under `examples/results/step6/reacher-correctness/replay/`, comparator
`scratch/step6/reacher-correctness/compare_replay.py`. 14 configurations:

* **10 pixel-identical, declaration unchanged**: Acrobot, BallInCup, both Cartpoles, FingerSpin,
  Pendulum — plus the four affected cells, which are **pixel-identical with
  `declaration_changed=True`** (`style_dim` 17→18). `digest_ours` and `digest_ours_default_style`
  match the frozen oracle byte for byte on all ten.
* `digest_ours_corpus_style` and `style_digest` change on exactly the four affected cells and nowhere
  else, **by construction**: they are taken under `uniform_style`, which draws from the variation
  space, and the space now has one more leaf. That is not a pixel change and the comparator reports
  it on its own line so it cannot be mistaken for one.
* **4 trackcom cells differ from the frozen digest — and so does the baseline.** See below.
* No check that was `measured` in the frozen run is anything other than `measured` here; no check
  errored.

## Pixels changed?

**No, for every configuration, at the default appearance.**

* **Default configuration:** unchanged, byte for byte, on all 11 parity cells and all 5 trackcom
  cells, measured as baseline-vs-branch on one card in one job (0/16 moved) and as branch-vs-frozen
  Step-4 digests (11/11 parity cells identical).
* **Target-visible configuration:** this *is* the default. `visibility.target = 1` is the compiled
  model's own alpha, and the digests above are taken at it.
* **Target-hidden configuration (`visibility.target = 0`):** new as a *representation*, but **not a
  new picture**. Against the default it changes ~15 px/frame on reacher-hard (max 219/255), 966 px
  over the 64-state corpus. Against the Step-5 `offscreen` workaround it changes **nothing**:
  measured on 16 frozen states at 84 px *and* 224 px, on both reacher cells, `max |d| = 0`, 0 of
  112 896 and 0 of 802 816 pixels (`scratch/step6/reacher-correctness/probe_mode_equivalence.py`).
  That was worth measuring rather than assuming — `offscreen` still builds, transforms and casts
  against the geom, so it could in principle have leaked a floor reflection or a shadow ray, and on
  this scene it does not. **An earlier draft of this report claimed the two differ; that claim was
  wrong and the measurement above replaces it.**

## Semantic follow-up

**None required.** The evidence is the digest.

The Step-5 Reacher + DINO-WM result is a **live-surface** comparison: only `DMControlWrapper.render`
is swapped, and it renders the state the official env is in at that instant, at the env's default
appearance. Those frames are byte-identical here — same sha256, same card, same job, on the frozen
corpora of both reacher cells. The Step-5 harness additionally ran `target_mode="offscreen"`, and
that mode is unchanged, still the default in `eval_reacher.py`, and still selectable, so the exact
configuration those 600 paired episodes ran under reproduces bit for bit. **The Step-5 semantic
result does not need rerunning** — and the conclusion is stronger than "the default did not move":
`hidden` and `offscreen` are **bit-identical on this scene** (max |d| 0 at 84 and 224 px), so even
the arm re-run under the new representation would see the same pixels the recorded one saw.
`eval_reacher.py` now accepts `hidden` on both `--target-mode` and `--dataset-target-mode`, with
`offscreen` kept as the default for Step-5 reproducibility.

Recommendation: **no rerun, no tripwire** (a tripwire on a byte-identical default measures nothing).
`parity_probe.py` now measures `as_placed`, `hidden` and `offscreen` on the same states in one run,
so a Step-8 reader can re-confirm the equivalence against `stable_worldmodel`'s own env rather than
against the frozen corpus.

---

## Shared-file findings for the coordinator

1. **This branch does not edit `stable_envs/render/raycast3d.py` or `stable_envs/render/mjwarp.py`.**
   `git diff 89fa5ca --stat` touches neither. The only `stable_envs/` files changed are
   `envs/dmcontrol.py` and `testing/conformance.py`.
2. **MJWarp cannot carry per-row visibility without an upstream change — recorded, not worked
   around.** `stable_envs/render/mjwarp.py`'s own docstring (point 4) says MJWarp's shader never reads
   alpha; `_tc_context` hides zero-alpha elements via `MW.prepare_model`, which moves them out of the
   **enabled geom groups** at `create_render_context` time, after which the five kernels are
   CUDA-graph captured. Visibility is fixed for the life of a context. A per-world value needs either
   one captured context per distinct combination (the trade `CAMERA_KNOB = False` already refuses on
   this class) or an alpha-aware shader in `mujoco_warp`. `DMControlTrackCom.VISIBILITY_KNOB = False`
   carries that reason. The single cell this costs is `DMCSwimmerSwimmer6` (a `target` geom and a
   `target_light` per episode, visual tier). **For the cpu-3d / mjwarp owner, as information.**
3. **Four trackcom cells no longer reproduce their frozen Step-4 digests, and it is not this
   branch.** `DMCCheetahRun`, `DMCHopperHop`, `DMCHumanoidStand`, `DMCWalkerWalk` — every MJWarp cell
   in the shard — differ from the frozen `digest_ours`, and the **baseline tree differs identically**
   on the same card in the same job. `mean_pixel` moves by 0.16–0.61/255 and the distinct-colour count
   by ~1%, which reads as a shading-level drift rather than a geometry error. The frozen Step-4 dmc
   shard ran under a venv with `ogbench` and `gym_pusht` and no `stable_worldmodel` (job 6176998,
   gpu2105); this ran under `se-step5`. Same mujoco 3.12.0, mujoco_warp 3.12.0, warp 1.17.0, same
   driver, same GPU model. **Unexplained and outside this brief — it belongs to whoever owns the
   MJWarp path, and it means the trackcom rows of the frozen oracle currently cannot be reproduced
   from the pre-Step-6 baseline in `se-step5`.** Artifacts: `gpu/pixels_all.txt`,
   `gpu/pixels_all_{baseline,branch}.json`, `replay/compare.txt`.
4. **`certification.json` lists `backends: ["mjwarp", "raycast3d"]` for `DMCReacher*`, and that is an
   artefact.** `scratch/census/census.py::_backends` unions the `stable_envs.render.*` modules
   imported by the *module* that defines `render`; `dmcontrol.py` imports both because the trackcom
   tier uses MJWarp. No Reacher pixel goes through MJWarp — `DMCReacher*` is `DMControlSuite`, i.e.
   baked plates plus `raycast3d`. Worth correcting in Step 8 so a reader does not conclude Reacher has
   two backends to keep in agreement.
5. **`$SCR/venvs/se-step5` has no `pytest`** (AGENT_COMMON's table does not claim it does, but the
   obvious reading is that a venv used for GPU work can run the suite). It is the only venv with
   `mujoco_warp`, and `se-step6` — which has pytest — no longer has `mujoco_warp` even though the
   frozen Step-4 dmc run recorded `mujoco_warp 3.12.0`. Any agent needing both in one interpreter will
   hit this. My jobs are split across the two for that reason.

## Open questions for the coordinator

1. **`certify.py` has no judge for the new check.** An unknown check name lands as
   `no_declared_bound` (`scratch/step4/certify.py:296-310`) — honest, but it does not gate. The bound
   lives in `tests/gpu/test_conformance.py:TASK_OBJECT_PRESENCE_MARGIN`, following the
   `INTERIOR_EXACT_MIN` precedent that `certification.json` already cites by that path. Folding it
   into the frozen Step-2 spec and adding a judge is a Step-8 decision; I did not edit
   `scratch/certification.json`.
2. **Four Step-4 signatures move** (`describe_digest`, `style_dim` 17→18). The frames do not. If Step
   8 re-freezes signatures, these are the four rows and the reason is one variation leaf each.
3. **Should the knob cover elements the episode does *not* draw?** The current rule is "whatever
   `_episode_slots` measured", which is the smallest defensible derivation and covers the SWM
   `render_target` case exactly. A wider rule (one leaf per used *material*) would let a caller hide
   the floor or the arm, changes 11 cells' declarations instead of 4, and nothing has asked for it.
4. **`docs/research/step5_downstream_semantics.md` issue 1 says "A DM Control visibility knob is Step
   6 work".** That sentence is generated by `scratch/step5/report.py:46` and is now done. I edited
   neither, because the Step-5 page is a record of what Step 5 did; the generator's sentence probably
   wants updating when the Step-6 page is written.
5. **The brief's parenthetical on the default.** Recorded again because it is the one deliberate
   divergence: the stable-envs default stays target-*visible*, because its reference is dm_control and
   dm_control draws it. See the Diagnosis section for the measurement. Say so if the coordinator wants
   the opposite; it is a one-line change to `_hideable`'s `init_value` and it would move the frozen
   Reacher digests.
