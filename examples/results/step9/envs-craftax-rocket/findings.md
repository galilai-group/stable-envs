# Step 9 — Wave 1, Workstream E: Craftax (Classic + Full) + PFRocketLanding

**Workstream:** `step9/envs-craftax-rocket` · **Parent:** `step9/integration` `5741b62`, merged in
(no rebase) · **Step-8 baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Host:** Apple M5 Mac (`cis1b1zs2j3`), CPU only — no CUDA device in this ticket.
**Interpreters:** `/Users/samib/se6-venv` (torch 2.14 CPU, tensordict, loguru),
`/Users/samib/se7-craftax-venv` (+ `craftax` 1.6.1, `jax` CPU), `/Users/samib/se7-rocket-venv`
(+ PyFlyt, scipy), `/Users/samib/se8-vm-venv` (+ `stable_worldmodel`). `tensordict` 0.14.2 and
`loguru` 0.7.3 were installed into the two `se7-*` venvs (they had neither; Wave 0 did the same for
`se6-venv`/`se8-vm-venv`). `MUJOCO_GL=glfw`, `-p no:cacheprovider` throughout.

**Result: PATCH.** Three cells migrated onto the Wave-0 foundation — the repository's **first
renderer-only migrated cells**. Both Craftax cells render **byte-identical frames** through this
branch and through a detached worktree of `21cb306` in the same venv, on **all three frozen corpora
at both certified tile sizes each**, and the four digests also reproduce the **frozen Step-7
metadata** exactly, which is an oracle the baseline tree is not needed for. Rocket's frames are owed
to the post-Wave-1 GPU oracle, and everything the frame is a function of on the CPU side is compared
here and identical. **Unexplained oracle differences: zero.** Local unit suite **1419 passed**
(baseline 1385 + 34 new); craftax venv **179 passed / 1 skipped**; rocket venv **70 passed**; SWM venv
**35 passed**.

Cells: `CraftaxClassic`, `CraftaxFull`, `PFRocketLanding`.

---

## 1. Files moved / created / removed

| before | after |
|---|---|
| `stable_envs/envs/craftax_full.py` (1,531 lines) | `git mv` to `stable_envs/envs/_craftax_full.py`; `craftax_full.py` is now a 24-line `sys.modules` alias |
| `stable_envs/envs/craftax.py` | edited in place (`StableEnv` → `Env`, the Step-9 block) |
| `stable_envs/envs/rocket_landing.py` | edited in place (`StableEnv` → `Env`, `num_envs=`, the Step-9 block, `_close`) |
| `stable_envs/envs/_craftax_full_tex.py` | **untouched** (455 lines of baked texture tables; nothing in it is API) |

Created: `tests/unit/test_envs_craftax_rocket.py` (34 tests),
`scratch/step9/envs-craftax-rocket/oracle.py`, `scratch/step9/envs-craftax-rocket/perf.py`,
`examples/results/step9/envs-craftax-rocket/{oracle.json,oracle_rocket.json,performance.json,performance_5x11.json,findings.md}`.

Removed: **nothing importable.**

### THE MODULE DECISION, and the measurement behind it

The brief offered two shapes: merge `craftax_full.py` into `craftax.py` (plan §3 names one
`craftax.py` for this family), or keep a private `_craftax_full.py` **if the merge is not
behaviour-neutral**. It is not, and the reason is countable rather than aesthetic: **the two modules
share 21 module-level names and the shared names hold DIFFERENT values.**

| name | `craftax` | `_craftax_full` | | name | `craftax` | `_craftax_full` |
|---|---:|---:|---|---|---:|---:|
| `N_BLOCK` | 17 | 37 | | `STRIP_H` | 2 | 4 |
| `N_STRIP` | 16 | 41 | | `TILE_PX_AGENT` | 7 | 10 |
| `N_COUNT` | 10 | 100 | | `OBS_H`, `OBS_W` | 7, 9 | 9, 11 |
| `N_TILE` | 160 | 99 | | `REFERENCE_TILE_PX` | (7, 16) | (10, 16, 64) |

plus `N_FACING`, `N_KEY_HALVES`, `NIGHT_TINT`, `NIGHT_THRESHOLD`, `DAY_LENGTH`, `NO_PLAYER`,
`PLAYER_ASLEEP`, `TILE_PX_IMG`, `strip_geometry`, `night_vignette`, `_unpack` and the three
per-device caches. One module object cannot serve both spellings, so a merge would have to **rename
constants that tests and scratch scripts import by name** — `tests/unit/test_craftax_full.py` alone
reaches for 48 of them through `CF.<name>`. Renaming them is exactly the kind of non-neutral change
the brief conditioned the merge on, so the fallback was taken and is asserted by
`test_the_two_craftax_modules_could_not_have_been_merged`.

`stable_envs.envs.craftax_full` **is** `stable_envs.envs._craftax_full` — a `sys.modules` alias, one
module object, so a monkeypatch through either path is seen by both and the oracle checks the
identity. Auto-discovery skips `_`-prefixed modules and imports the alias, so `CraftaxFull` registers
exactly as before and a missing `craftax` is still recorded under the key `craftax_full`. Wave 3
decides whether the underscore stays.

**Duplication removed: none, deliberately.** The two Craftax renderers share `render/prng` and nothing
else that is not already shared; their atlases, strips, lighting chains and state layouts are
different programs over different games. The one thing that *looks* shared —
`craftax.night_and_sleep` — is already module-level and shape-parameterised so full Craftax's
`light_night_and_sleep` can sit beside it, and merging them further would be the shared-renderer
rewrite the brief forbids.

## 2. API changes

Additive. These three cells define no `step`, so nothing about their behaviour changed except the
constructor and the arrival of the public State:

```python
env  = snv.CraftaxClassic(num_envs=64, device="cpu")   # num_envs=, batch= still accepted (deprecated)
st   = env.state                                       # State: see the table below
px   = env.render(st) == env.render(env.pack(st)) == env.render()
env.close(); with snv.CraftaxFull(...) as env: ...
env.step(a) / env.transition(st, a) / env.unroll(st, A) / env.action_spec   # -> NotImplementedError
```

Public State keys:

| cell | State |
|---|---|
| `CraftaxClassic` | `{"world": {"blocks" (63), "mobs" (63), "light_level" (1), "rng_key" (4)}, "player": {"facing", "is_sleeping"}, "inventory": {<16 names> (1)}}` |
| `CraftaxFull` | `{"world": {"blocks", "items", "mobs", "projectiles", "light_map" (99 each), "light_level" (1), "rng_key" (4)}, "player": {"facing", "is_sleeping", "level"}, "inventory": {<41 slots> (1)}}` |
| `PFRocketLanding` | `{"rocket": {"angular_velocity" (3), "orientation" (4), "linear_velocity" (3), "position" (3)}}` |

The inventory is one key PER NAMED SLOT rather than one vector, because the slot names are the
env's own vocabulary and `state["inventory", "diamond"]` is what a user means; the quaternion is ONE
field of width 4 rather than four scalars, because it is one quantity with a norm constraint and
splitting it invites a caller to write three of its components.

**Not one packed column moved.** Every `state_layout` offset is the Step-4/Step-7 column it has always
been, in `state_names` order; the layouts tile the row with no gap and no overlap (asserted), so
`pack(unpack(rows)) == rows` bit for bit and every frozen digest and corpus is untouched.

**No field has offset `None`, and that is a claim rather than an omission.** Offset `None` means "real
dynamical state the packed row does not carry" — PushT's agent velocity, classic control's
velocities. A renderer-only cell has none by definition: there is no simulator to hold it. A field
appearing with offset `None` here would mean a hidden buffer had been introduced, which is the thing
that makes `transition` impure, so `test_the_packed_row_is_the_frozen_one_and_no_field_is_hidden`
pins the absence.

**`num_envs=`.** The two Craftax cells inherit `Env.__init__` and already took it. **`PFRocketLanding`
was one of the 41 Wave-0 holdouts** — it declared `batch=` itself and forwarded positionally, so it
never reached the deprecation branch and raised `TypeError` for the Step-9 spelling. It now takes
`num_envs=` in the slot `batch=` occupied (positional callers unaffected) and still accepts
`batch=`. **The Wave-0 ratchet
`test_foundation_api.py::test_the_number_of_classes_that_still_reject_num_envs_only_goes_down` is
unaffected on this host**, because it counts from signatures over the classes that register and
`PFRocketLanding` does not register without nvdiffrast; on a CUDA host the count should drop by one.
Flagged as request **R5** so the coordinator is not surprised by a ratchet that moves on the GPU host
and not here.

**`frame_shape` (new, both Craftax cells), and it is a genuine bug fix.** `Env.observation_spec`
assumes a square frame of `size` pixels, and for Craftax `size` is the **TILE** edge: the spec was
reporting `(16, 16, 3)` for a 144x144 picture — a buffer **81x too small** — and `(10, 10, 3)` for
full Craftax's `(130, 110)` one, which is the wrong shape as well as the wrong size. Fixed in the
family, exactly as Workstream A fixed classic control's 400x600 there rather than in the base class.
Craftax Classic's frame happens to be square (7 + 2 == 9 rows by 9 columns) and the arithmetic is
written out rather than relying on that coincidence.

**`PFRocketLanding._close` (new).** `_NVD_CACHE` is keyed by `(size, supersample, device)` and is
deliberately SHARED between envs — a `BatchRenderer` allocates a CUDA context and the scene's vertex
buffers, and two envs at the same resolution should not pay twice. So `_close` does **not** tear the
cache down: another live env may be rendering through the same entry and freeing it here would be a
use-after-free in that env rather than a leak in this one. What `close` does own is this env's refusal
to render afterwards, which `Env.close` installs. Written out as a no-op with the reason, rather than
omitted, so the next reader does not "fix" it.

Unchanged and asserted: `state_dim`, `state_names`, the packed row and its column order,
`sample_states`, `set_state`/`get_state`, `render(rows)`, the whole style API, `describe()`,
`variation_space`, every reference-bridge staticmethod, and every fidelity/applicability declaration.

## 3. Step-8 IDs affected

**Behaviorally: none.** Five of the 42 SWM ids are served by these three cells:

| SWM id | cell |
|---|---|
| `swm/CraftaxClassicPixels-v1` | `CraftaxClassic` |
| `swm/CraftaxClassicSymbolic-v1` | symbolic — refuses, unchanged |
| `swm/CraftaxPixels-v1` | `CraftaxFull` |
| `swm/CraftaxSymbolic-v1` | symbolic — refuses, unchanged |
| `swm/PFRocketLanding-v0` | `PFRocketLanding` |

The two pixel ids are byte-identical (§6). The two symbolic ids still resolve to the Wave-0 refusal
that says they are symbolic rather than "unknown env". `swm/PFRocketLanding-v0`'s frames are owed
(§6, §10).

## 4. Targeted tests

`tests/unit/test_envs_craftax_rocket.py`, **34 tests**, marker `unit`, parametrised over both Craftax
cells where the cell can be built and written against the CLASS for Rocket, which cannot be built on a
host without nvdiffrast.

| test | what it pins |
|---|---|
| `takes_the_step9_constructor` | `num_envs=`, `len`, the context manager, `close` idempotence, a closed env refusing `render` |
| `the_rocket_constructor_takes_num_envs_and_still_takes_batch` | from the SIGNATURE, so it is enforced on any machine: `num_envs` in the slot `batch` had, `batch=` still accepted, `device="cuda"` default |
| `state_keys_are_the_declared_ones_and_cover_the_packed_row` | the public vocabulary WRITTEN OUT rather than derived from `state_layout` (a test that reads the attribute the code reads asserts only that the attribute exists); `pack(unpack(rows)) == rows` |
| `the_packed_row_is_the_frozen_one_and_no_field_is_hidden` | no offset `None`; the layout tiles the row with no gap and no overlap; `_hidden_state`/`_live_state` are both empty |
| `state_names_of_agrees_with_the_frozen_column_names` | the public keys re-derive `state_names` exactly, in order |
| `a_state_the_caller_keeps_is_not_a_view_of_the_env` | foundation defect R8, pinned for these cells: every field here is float32, i.e. exactly the shape that was exposed before `unpack` got its `copy=True` |
| `the_rocket_state_is_one_rigid_body_grouped_by_what_it_is` | the four keys, their widths, their offsets, and that the offsets are the frozen obs layout |
| `render_takes_a_state_a_packed_row_or_nothing` | `render(State) == render(rows) == render()` |
| `the_observation_spec_is_the_frame_and_not_the_tile` | the `frame_shape` fix, per cell, against the real frame |
| `the_state_spec_leaves_match_the_layout` | every leaf's shape |
| `these_cells_have_no_dynamics_and_say_so_precisely` | **the brief's check, answered**: `step`, `transition`, `unroll` and `action_spec` all raise the contract's `NotImplementedError` with its exact wording |
| `the_rocket_refuses_a_cpu_device_by_design` | the CPU refusal on a CUDA host, and the earlier `ImportError` on a host with no nvdiffrast — both asserted, so the test means something on either machine |
| `the_rocket_is_registered_exactly_when_it_can_render` | registration is honest about the backend |
| `the_old_craftax_full_import_path_is_the_private_module` | one module object, and the ten names the existing tests reach for |
| `the_two_craftax_modules_could_not_have_been_merged` | the evidence for the fallback, kept beside the choice |
| `a_missing_craftax_is_still_recorded_under_the_old_module_key` | discovery still keys on `craftax_full` |
| `the_jit_wrapper_declaration_survives_on_both_craftax_cells` | `SWM_WRAPPER_GATE_F`: the jitted/un-jitted split, 0.997854 / 0.997225, max \|diff\| 1 |
| `the_resolution_consistency_applicability_declarations_survive` | Classic's `light >= 0.5` band scoping with 0.1634; Full's all-bands declaration and its `still_required` |
| `the_parity_claims_and_their_reference_versions_survive` | `parity_all_exact_min` 1.0, tier `parity`, decoder `renderer`, craftax 1.6.1 / jax 0.11.1, threefry in the note |
| `the_rocket_colour_exactness_declaration_survives_verbatim` | the five lit leaves by name, the 0.993567 ceiling, `{200: 198, 33: 32, 17: 16}`, `official_renderer_fails_the_same_check` |
| `the_rocket_tier_and_reference_versions_survive` | visual tier, PyFlyt 0.29.0 / pybullet 3.2.7, the three thresholds 10.0 / 0.91 / 0.80 and the three measurements |
| `the_rocket_flame_and_ignition_semantics_survive` | `FLAME_RGBA`, `IGNITION_RGBA`, `flame.ignited` at style column 0, the corpus style paths |
| `the_batch_invariant_float64_mvp_does_not_depend_on_the_batch` | `lagged_pose`/`camera_pose` in float64 give bit-identical answers for three different sub-batches |
| `both_frozen_craftax_classic_corpora_are_kept_and_both_replay` | Sami's ruling: 64x127 and 64x149 both present; the narrow one RAISES on a direct render and replays zero-extended |
| `the_complete_observation_is_149_dimensional_and_the_strip_is_in_it` | the 149-dim complete observation, its 16 inventory keys and the four PRNG halves |

Exact outputs, all from the worktree root, at HEAD:

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
1419 passed, 195 skipped, 112 deselected, 26619 warnings in 123.35s
```

Baseline for that command, measured at `step9/integration` `5741b62` before any edit on this branch:
**`1385 passed, 195 skipped, 112 deselected`** (160.31s). 1385 + 34 new = 1419, i.e. **no pre-existing
test changed its result**.

```
MUJOCO_GL=glfw /Users/samib/se7-craftax-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_step8_semantic_craftax.py tests/unit/test_step8_gate_record.py \
  tests/unit/test_craftax_atlas.py tests/unit/test_craftax_full.py \
  tests/unit/test_craftax_prng.py tests/unit/test_craftax_strip.py \
  tests/unit/test_envs_craftax_rocket.py
179 passed, 1 skipped in 21.91s          (145 passed, 1 skipped without my file, = the baseline count)

MUJOCO_GL=glfw /Users/samib/se7-rocket-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_rocket_landing.py tests/unit/test_envs_craftax_rocket.py \
  tests/unit/test_nvdiff_triangles.py
70 passed

MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/swm tests/unit/test_step8_gate_record.py
35 passed
```

The SWM run is included because `tests/swm/test_gate_f_matrix.py` imports
`stable_envs.envs.craftax_full` by name — the alias is exercised there rather than only asserted.

**No pre-existing test was edited.** Workstream A's registry-row ruling (their R2, accepted
Wave-1-wide) already made `test_foundation_registry.py` compare the `module` column by what it
guarantees, so `CraftaxFull` moving module does not fail it.

## 5. Applicable certification checks

The frozen-corpus replay for both Craftax cells on all three corpora at both certified sizes each
(§6), the Step-8 support-matrix rows (`test_foundation_registry.py`), and the Step-8 semantic record
(`test_step8_semantic_craftax.py`, which re-derives Workstream H's byte-identity claim from its
artifacts). Gate A, the GPU conformance suite, the tier measurements, and **everything about
`PFRocketLanding`'s pixels** need a CUDA host and are owed — §10.

## 6. The oracle — default frames, variation/state comparison

`scratch/step9/envs-craftax-rocket/oracle.py` pushes each corpus through BOTH trees in two separate
interpreters — a detached `git worktree` of `21cb306` and this branch, the same venv — and compares
the 27 keys listed in the script's `COMPARED` table: the frames (twice: from the held state and from
explicit packed rows), `describe()`, the style layout and default appearance, the variation taxonomy
leaf for leaf, the registry row, `state_dim`/`state_names`, the corpus round trip, `sample_states` at
a fixed seed, and every declaration plan §12 E names — `SWM_WRAPPER_GATE_F`,
`resolution_consistency_applicability`, `parity_note`, `reference_versions`,
`certification_state_distribution`, `visual_fidelity`, `color_exactness_applicability`.

**The corpora, all three of them** (Sami's Step-8 ruling: keep both of Classic's):

| corpus | shape | role |
|---|---|---|
| `examples/results/step4/states/CraftaxClassic.npz` | 64 x **127** | the historical WORLD-ROW oracle, under the zero-extension interpretation |
| `examples/results/step7/craftax-classic-full/corpus/states.npz` | 64 x **149** | the complete observation (16 day / 25 dusk / 23 night, 30 sleeping) |
| `examples/results/step7/craftax-full/corpus/states.npz` | 64 x **544** | full Craftax's five cost bands |

The 127-wide array is zero-extended into the 149-slot layout with `light_level` forced to 1.0,
verbatim from `scratch/step7/closure/craftax_world_row_replay.py`; and the oracle **checks that
rendering it directly RAISES** rather than silently reshaping, because that raise is the correct
Step-7 behaviour and a migration could have turned it into a wrong picture.

Per-corpus, per-size frame digests (SHA-256 over the raw uint8 bytes), **identical between the two
trees**:

| cell | corpus | 7 / 10 px | 16 px |
|---|---|---|---|
| `CraftaxClassic` | Step-4 127, zero-extended | `6a64b4fd360e765f…` | `319792d73f89906f…` |
| `CraftaxClassic` | Step-7 149 | `79aebe383560d3d2…` | `d06c468ce0eea2c8…` |
| `CraftaxFull` | Step-7 544 | `82d83319a48bd3c8…` | `650477921f68f055…` |

**And the four Step-7 digests are the FROZEN ones.** `corpus/states.meta.json` records
`79aebe383560d3d2…` at 7 px and `d06c468ce0eea2c8…` at 16 px for Classic, and
`82d83319a48bd3c8…` at 10 px and `650477921f68f055…` at 16 px for Full — reproduced here to every
digit. That is an oracle that does not depend on the baseline tree being checked out correctly: the
frames agree with a number frozen in Step 7 and committed since.

| compared | result over both Craftax cells |
|---|---|
| frames, both corpora, both certified sizes, from the held state and from explicit rows | **identical** |
| the narrow-corpus refusal (`ValueError: expects state_dim 149, got 127`) | identical, word for word |
| `describe()` | identical |
| `style_names`, `style_dim`, the official default appearance | identical |
| `variation_space.names()` — the taxonomy, leaf for leaf | identical |
| `parity_interior_min`, `parity_all_exact_min`, `fidelity_tier`, `render_decoder`, `parity_note` | identical |
| `SWM_WRAPPER_GATE_F`, `resolution_consistency_applicability`, `certification_state_distribution`, `reference_versions` | identical |
| registry row (name, tier, decoder, state_dim, reference_id) | identical |
| `state_dim` / `state_names`, `set_state`/`get_state` round-trip | identical, exact |
| `sample_states(64, seed 0)` | identical |

**Unexplained differences: zero** (`"unexplained_differences": {}`, `"identical": true` in
`oracle.json` and `oracle_rocket.json`).

**The two EXPECTED differences**, declared in the script rather than discovered in the diff:

1. `CraftaxFull`'s `registry_table()[i]["module"]` moved from `stable_envs.envs.craftax_full` to
   `stable_envs.envs._craftax_full` (§1). The script checks that the old import path still resolves to
   the same module object. Consequences are location-only: `spec("CraftaxFull")["family"]` and
   `list_envs(family=...)` now answer `_craftax_full`. See request **R1**.
2. `constructor_takes_num_envs` goes False → True on all three cells. That is contract §2 and it is
   the patch working; its *absence* would be the failure, so the oracle now **requires** it.

### PFRocketLanding — what was and was not proved here

Rocket renders only on `cuda` through nvdiffrast, refuses `device="cpu"` by design, and cannot even
reach a renderer on this host (`ImportError: nvdiffrast`), so **no frame was produced or compared**.
What IS fully determined on the CPU and was compared, over the frozen 200-state Step-7 reference
corpus in float64:

| compared | digest (migrated == baseline) |
|---|---|
| the frozen corpus itself | `state13`, 200 x 13 |
| `geom_poses` — the 13 geom positions and orientations | `aa9f38121803a165…` / … |
| `lagged_pose` — the pose the camera reads, one substep behind the body | identical |
| `camera_pose` — **the batch-invariant float64 MVP** | `59fc03c2ce6f7e9b…` |
| `material_table` — the TinyRenderer material table from the style row | `e4b648155709b5c5…` |
| `corpus_style_to_env` | identical |
| `build_scene_arrays` — vertices, normals, triangles, vertex→geom, ground UVs | `abe22462dd45fe8b…` / … |
| `checker_texture` — the nearest-sampled checker ground | `3ef5ee1a7ba81d42…` |
| `style_layout`, `style_dim` (16), the default style row | identical |
| `visual_fidelity`, `visual_note`, `color_exactness_applicability`, `DOCUMENTED_LIMITATIONS`, `REFERENCE_UNREACHABLE_SHARE`, `reference_id`, `default_size`/`default_supersample` | identical |
| `missing_backend()`, and that the class registers exactly when it can render | identical |

Every one of those is an input to the frame. That they are byte-identical is the CPU half of the
frame claim; the GPU half is **owed** and listed in the record's own `owed` block, not implied.

### The Step-9 claims (no baseline to compare against)

Checked in the migrated tree for both Craftax cells on `cpu`: `pack(unpack(rows))` is the identity,
`pack(env.state) == get_state()`, `render(State) == render(rows) == render()`,
`observation_spec` matches the real frame shape, a `State` the caller keeps does not change when the
env is re-stated, `has_dynamics` is False, a closed env refuses `render`, and `step`/`transition`/
`unroll` each raise the contract's refusal. `"functional_ok": true`.

## 7. Performance (plan §8)

`scratch/step9/envs-craftax-rocket/perf.py`, the foundation's method: ALTERNATING subprocesses between
the two trees (A B A B …), `torch.set_num_threads(1)`, `OMP/MKL_NUM_THREADS=1`, CPU, median of the
per-repeat medians. Craftax's render has **two paths** and the migration had to leave both alone, so
each is timed on a batch drawn entirely from its own band rather than on a mixed batch that would
average them and hide a regression in either:

* the **uint8 gather** — fully lit, awake, unit gain: one `index_select` from a uint8 atlas, no float
  buffer;
* the **float chain** — sleeping and at `light_level` 0.4: the pre-rounding float table plus the
  reference's per-pixel post-process, about 100x the gather on one core.

11 repeats x 31 iterations, 64 envs, each cell at its registered tile size
(`examples/results/step9/envs-craftax-rocket/performance.json`):

| path | baseline | migrated | delta |
|---|---|---|---|
| `CraftaxClassic.render`, uint8 gather, 16 px | 0.714 ms | 0.650 ms | **−8.97 %** |
| `CraftaxClassic.render`, float chain, 16 px | 69.08 ms | 72.67 ms | **+5.18 %** |
| `CraftaxFull.render`, uint8 gather, 10 px | 3.680 ms | 3.783 ms | **+2.79 %** |
| `CraftaxFull.render`, float chain, 10 px | 19.40 ms | 19.43 ms | **+0.15 %** |

**All four are inside this machine's run-to-run spread, and the spread is the finding.** The
per-repeat legs are recorded in the JSON: `CraftaxClassic`'s float chain spans **50.6–96.4 ms on the
BASELINE arm alone** (a 1.9x spread) and its uint8 gather 0.58–0.92 ms. A first, shorter run (5
repeats x 11 iterations, kept as `performance_5x11.json`) read **+22.18 %** on that same uint8 gather
and **−0.35 %** on the float chain — i.e. the 5x11 pass put a +22 % where the 11x31 pass puts a −9 %,
on code that was not touched. That is reported rather than quietly replaced: on an unpinned laptop a
sub-millisecond path needs more repeats than five before a double-digit percentage means anything,
and the 11x31 numbers are the record for that reason.

**Nothing in the render path was edited**, which is why no delta needs a mechanism: `state_layout` is
a class attribute built once at class creation, `frame_shape` is a property `render` never consults,
and `_close` is a no-op. The measurement exists to catch an accident, and it did not find one. There
is no `step` to time: these cells have none.

`PFRocketLanding` is absent from the table because it does not render on this host at all. Its timing
is **owed** to the GPU oracle and is recorded in the JSON's `owed` block.

## 8. Resource lifetime

`Env.close` is explicit and idempotent; a closed env refuses `render`/`reset`/`step` with a message
naming the call, and correctness never depends on `__del__`. The two Craftax cells hold no external
resource — their atlases are per-device tensors in a module-level cache that is never written through
(`test_craftax_strip.py::test_the_strip_grid_cache_is_not_written_through`) — so neither declares a
`_close`. `PFRocketLanding` declares one, and it deliberately does **nothing to the shared
`_NVD_CACHE`**, for the reason given in §2. The suite's resource guard reports the same counts as at
baseline: 16 EGL contexts created / 0 released over the unit run, unchanged and a pre-existing Step-8
observation.

## 9. The §17 ten-item report

1. **files moved/removed/created** — §1. One file moved and kept as an alias, two edited in place,
   five created, nothing removed.
2. **API changes** — §2. Additive; the constructor, the public State, one genuine bug fix
   (`frame_shape`), one documented no-op (`_close`).
3. **Step-8 IDs affected** — §3. Behaviorally none; the two pixel ids these cells serve are
   byte-identical, the two symbolic ids still refuse, Rocket's is owed.
4. **targeted tests** — §4. 34 new; 1419 local, 179/1 craftax venv, 70 rocket venv, 35 SWM venv.
5. **applicable certification checks** — §5. Frozen-corpus replay done for both Craftax cells on
   three corpora at two sizes each; Gate A, the GPU suites and all of Rocket's pixels owed (§10).
6. **default-frame comparison** — §6. Identical SHA-256 on all six (cell, corpus, size) combinations,
   384 frames, and four of them reproduce the frozen Step-7 metadata exactly.
7. **variation/state comparison** — §6. Style layout, default appearance, taxonomy, packed layout,
   round trip, reset distribution and every declaration identical.
8. **performance comparison for hot paths** — §7: −8.97 % / +5.18 % / +2.79 % / +0.15 %, all inside a
   measured run-to-run spread that is itself larger than any of them.
9. **resource-lifetime comparison** — §8: explicit `close`, the shared-cache decision written down,
   guard counts unchanged.
10. **unexplained oracle differences** — **zero**.

## 10. ESCALATIONS, owed GPU items, and requests

### Owed to the GPU host (nothing here is a result that was claimed and wrong; these are results that
were not produced)

| # | owed | what to run |
|---|---|---|
| **O1** | **`PFRocketLanding`'s frames.** The whole pixel claim for this cell. No frame was rendered or compared anywhere in this ticket. | the 200-state Step-7 reference corpus at 480 px through both trees on ONE card in ONE job, byte-compared; plus `tests/gpu` for this cell. The CPU inputs to those frames are already proved identical (§6). |
| **O2** | the Craftax `--device cuda` arm of the oracle | `scratch/step9/envs-craftax-rocket/oracle.py --device cuda` — one flag. Craftax's render is the same torch gather on both devices (no CUDA-only fused kernel, unlike `gfxdraw_classic`/`navdot`), so this is a confirmation rather than a different code path, but it has not been run. |
| **O3** | GPU timings for all three cells | `perf.py`'s method on the card; Rocket has no CPU number at all. |
| **O4** | Gate A, `tests/gpu/test_conformance.py`, `test_reproducibility.py` and `test_fast_paths.py` for `CraftaxClassic`, `CraftaxFull`, `PFRocketLanding` | the tier and resolution-consistency rows for these three live there |
| **O5** | the Wave-0 `num_envs` ratchet | `PFRocketLanding` does not register on this host, so the count did not move here; it should drop by one on a CUDA host (§2). |

No sbatch was written, per the brief.

### Requests

**R1 — `scratch/step3/certify.py::NO_BACKEND_MODULE_CLASS` keys on the module's last path segment,
and Wave 1 is moving modules.** The table is `{"craftax": "R", "craftax_full": "R", "minihack": "R",
"swm_nav": "R"}`, and its own comment says CraftaxFull sat at `unknown` for a whole step *purely
because this table keyed on the module NAME*. `craftax_full` is now `_craftax_full`, so the same
fall-through returns. **Workstream A already broke the `swm_nav` entry the same way** (`swm_nav.py`
became `navigation.py`), which makes this a Wave-1-wide issue rather than mine. I did **not** edit the
file — it is `scratch/step3` and the fix should land once. No unit test fails today
(`tests/unit/test_step3.py` drives `backend_class` from a frozen census, not from the live module), so
this is latent until someone reclassifies. **Ask:** add `_craftax_full` and `navigation` to that dict,
or key it on the registered class name instead of the module basename.

**R2 — the registry-row `module` comparison.** Already the Wave-1-wide ruling (Workstream A's R2,
accepted). Recorded here only to say this workstream relies on it: `CraftaxFull.module` moves.

**R3 — `docs/api/envs.md` and anything that cites `stable_envs/envs/craftax_full.py` by path.**
`docs/research/step8_upstream_findings.md`, `docs/research/step8_certification.md` and
`docs/research/step7_coverage.md` name the file. The module is still importable under both spellings,
so nothing is broken; the doc paths are Workstream M's to refresh.

**R4 — foundation R9 is CLOSED and this workstream is the reason it mattered.** Workstream A found
`render_worker.py:72` and `conformance.py:1074` still guarding with `hasattr(env, "step")`, which is
now always True, so renderer-only cells fell through to `env.sample` and raised. Both are fixed on
`step9/integration` (they call `dynamics.has_dynamics` at `5741b62`) and I verified it here. Flagging
it because **all three of my cells are renderer-only** — they are precisely the population that guard
broke, and the fix is untested on a GPU for them (O4).

**R5 — the `num_envs` ratchet moves on the CUDA host, not here.** §2. Not a defect; a number the
coordinator should expect to change when the GPU run happens.

**R6 (Wave 3, no action now) — the private module's underscore.** `_craftax_full.py` satisfies plan
§3's spirit (one public `craftax.py` for the family) at the cost of a name that discovery skips and
that `scratch/step3` does not recognise (R1). If Wave 3 would rather have `craftax_full.py` back as
the real module, that is a one-line revert of the alias and the oracle's one expected difference
disappears with it. Recorded so the choice is re-decidable rather than inherited.
