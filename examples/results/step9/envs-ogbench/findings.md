# Step 9 — Wave 1, Workstream D: the OGBench family

**Workstream:** `step9/envs-ogbench` · **Parent:** `step9/integration` `b3f3a81` (Wave 0 + A + B's
`_layout` instance-method ruling + Workstream H's raycast3d/cpu3d consolidation), no rebase ·
**Step-8 baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Host:** Apple M5 Mac (`cis1b1zs2j3`), CPU only — no CUDA device in this ticket.
**Interpreters:** `/Users/samib/se7-ogb-venv` (mujoco 3.12.0, ogbench, stable_worldmodel, gymnasium
1.3.0 — plus `tensordict` 0.14.2, installed for this work, see request R4) and
`/Users/samib/se6-venv` (the CPU-only suite). `MUJOCO_GL=glfw`, `-p no:cacheprovider`.

**Result: PATCH.** `stable_envs/envs/ogbench.py` (712 lines) and `stable_envs/envs/ogb_manip.py`
(1,582 lines) are now the PACKAGE `stable_envs/envs/ogbench/` — `maze.py`, `manipulation.py` and an
`__init__.py` — with `ogb_manip.py` kept as a `sys.modules` alias. All **27** cells migrated. The
eleven frozen Step-4 maze corpora replay to **byte-identical frames** across the 25 maze cells
through this branch and through a detached worktree of `21cb306` in the same venv, and `describe()`,
the style layout, the default appearance, the variation taxonomy, the fidelity declarations, the
parity floor and `parity_note`, `visual_fidelity` in full, `visual_note` (the button-texture bake
note included), `state_dim`/`state_names`, the corpus round trip, `sample_states`, the SWM id
aliases, the registry row **including `module`** and the CPU refusal **with its message** are all
identical. **Unexplained oracle differences: zero.** Local unit suite **1385 → 1427** (59 new tests,
no pre-existing test changed its result).

Cells — 25 PARITY mazes over 11 distinct scenes (`Pointmaze{Medium,Large,Giant,Teleport}{Navigate,
Stitch}`, `Antmaze{Medium,Large}{Navigate,Stitch,Explore}`, `AntmazeGiant{Navigate,Stitch}`,
`AntmazeTeleport{Navigate,Stitch,Explore}`, `Humanoidmaze{Medium,Large,Giant}{Navigate,Stitch}`) and
two VISUAL manipulation cells (`OGBCube` 4.3 / `OGBScene` 8.7 declared mean-abs ceilings), which are
CUDA-only by declaration and whose FRAMES are owed to the post-Wave-1 GPU oracle (§10).

---

## 1. Files moved / created / removed

| before | after |
|---|---|
| `stable_envs/envs/ogbench.py` (712) | `stable_envs/envs/ogbench/maze.py` (772), via `git mv` |
| `stable_envs/envs/ogb_manip.py` (1,582) | `stable_envs/envs/ogbench/manipulation.py` (1,667), via `git mv` |
| — | `stable_envs/envs/ogbench/__init__.py` (79), new |
| — | `stable_envs/envs/ogb_manip.py` (23), rewritten as a `sys.modules` alias |

Created besides: `tests/unit/test_envs_ogbench.py` (59 tests),
`scratch/step9/envs-ogbench/oracle.py` and `scratch/step9/envs-ogbench/plate_probe.py`,
`examples/results/step9/envs-ogbench/{oracle.json,plate_probe.json,findings.md}`.

**`stable_envs/envs/ogbench.py` is REMOVED rather than kept as an alias, and it has to be.** The new
home is a package with the SAME name, and a module and a package cannot coexist in one directory —
the package wins the import and the `.py` would be unreachable dead code. This is Workstream B's
`dmcontrol` precedent exactly, including its obligation: the oracle reads **17** module-level names
off the baseline source of `ogbench.py` and checks each one resolves off the package. All 17 do, and
so do all **25** generated cell classes. There are no declared-absent names.

**`stable_envs/envs/ogb_manip.py` KEEPS its name**, because nothing claims it, and it is a
`sys.modules` ALIAS rather than a set of re-exports — Workstream A's `swm_nav` pattern. That matters
here more than it did there: `tests/unit/test_ogb_manip.py` reads `inspect.getsource(ogb._context)`
and reaches for eleven private names, `tests/unit/test_variation_maps_newcells.py` imports
`OGBManipBase` and `MJW_CHUNK`, and there is exactly ONE module object, so every one of them resolves
and a monkeypatch through either path is seen by both. The oracle checks all **41** module-level
names of the Step-8 `ogb_manip.py` off that path; all 41 resolve.

**THE PACKAGE IS TWO MODULES AND NOT FIVE, deliberately.** Plan §3's target tree names exactly
`__init__.py`, `maze.py` and `manipulation.py` for this family, and §16 forbids an abstraction two
things do not need. `manipulation.py` is 1,667 lines and could be split into a model half and a
warp-context half; Workstream B split `dmcontrol.py` because it was 3,376 lines and genuinely
eighteen domains with two disjoint renderers, and this is two cells of ONE scene family whose
reference env, bundle, plate rig and render context are a single chain of construction. Splitting it
would buy file sizes and cost a re-export layer over a chain that is read top to bottom. The two
halves of the FAMILY are split, because those really are two scenes, two renderers, two fidelity
tiers and two device stories, and they share no base class, no cache and no code.

**The line count grew by 4.5 % (2,294 → 2,541) and none of it is duplication.** `+91` of Step-9
machinery and its comment block in the two constructors and `_close`, `+79` for the package
`__init__` (which is mostly the note on why the two import paths are different shapes), `+72` of
docstring in the two modules saying what Step 9 added, and **−61 of DELETED duplication** (below).

**The one deletion is a duplication, and it is the one this family should never have had.** Step 8
lifted this file's texel substitution into `stable_envs/envs/_plate_texels.py` so that `dmcontrol.py`
could run the same code — and left the original standing, so the module the helper was extracted
FROM was the one family not running it. `mj_quantise`, `_grid_texture`, `_floor_key` and
`_apply_floor` keep their names (they are Step-8 module-level names) and their behaviour; their
bodies are now `quantise`, `texel_slots`, `texel_key` and `substitute`. `mj_quantise` **IS**
`_plate_texels.quantise` — the same function object, asserted, so the truncation rule cannot drift.
`_plate_texels.py` itself was NOT touched (§6 measures that this changed no plate).

## 2. API changes

**Additive.** Nothing a Step-8 caller did stopped working; the packed row, the frames and every
declaration are unchanged.

```python
env = snv.PointmazeLargeNavigate(num_envs=64, device="cpu")   # num_envs=, batch= still accepted
st  = env.state             # State: {"root": {"position", "orientation"}, "joints"} (legged)
                            #        {"root": {"position"}}                          (point)
rows = env.pack(st)         # the frozen packed row, byte for byte
px  = env.render(st) == env.render(rows) == env.render()
env.step(a)                 # NotImplementedError: no batched simulator (below)
env.close(); with snv.OGBCube(num_envs=16, device="cuda") as env: ...
snv.OGBCube(num_envs=1, device="cpu")     # ValueError: ... (CPU3D-OGB-1) ... Use device='cuda'.
```

**The public State is the BODY's own structure, declared per body and not per cell.** `qpos` for a
point is two slide joints; for the legged bodies it is a free joint (3 position + 4 quaternion)
followed by 8 (ant) or 21 (humanoid) hinges. One vocabulary covers all three —
`root.position` / `root.orientation` / `joints` — and the point simply has neither of the last two,
which is what contract §3 means by no ceremony for a state that is one vector. The manipulation
cells' State is `{"qpos": (nq,), "mocap": {"position": (3,)}}` plus `{"buttons": (2,)}` on the
Scene: three keys because the three halves are three KINDS of quantity — the simulator's
configuration, a mocap body's pose (which is why it needed a name of its own at all), and two bits
that are neither and reach the frame only as colours.

**NOTHING IS HIDDEN, and that is a statement rather than an omission.** Every packed column belongs
to exactly one key, so no field has offset `None`, `_hidden_state` is never consulted, `pack` drops
nothing and `unpack` invents nothing — asserted on all three bodies. There is no `_live_state`
either: the packed row IS this family's live state, since `set_state` writes it and `render` reads
it, unlike classic control's float64 simulator. `state`, `_layout`, `_hidden_state` and
`_live_state` are all UNOVERRIDDEN in both modules.

**`state_layout` is assigned in `__init__`** in both halves, which is the foundation's R1 instance
method in use: the manipulation cells genuinely do not know their width until swm has compiled a
model (`nq`, `3 * nmocap`, the button count), and the maze cells' layout depends on `LOCO`, which the
generated subclass sets. `Env._layout` reads `self.state_layout`, so neither family overrides it.

**NO DYNAMICS, declared rather than omitted.** There is no batched simulator in this package for any
OGBench cell, so `has_dynamics` is False and `step`/`transition` raise `Env`'s precise refusal naming
the cell and the reason; `reset` and `render` work as usual. This is the Step-8 behaviour said out
loud, and `test_a_renderer_only_cell_refuses_step_and_transition_and_says_why` pins the message.

**`num_envs=`: 27 of the Wave-0 holdout list of 41 are now migrated.** The ratchet
(`test_foundation_api.py::test_the_number_of_classes_that_still_reject_num_envs_only_goes_down`)
measures **14** on this branch, down from 41: what remains is `highway` (6), `maze2d` (4) and
`mujoco_locomotion` (4), i.e. Workstream F's. The test asserts `<=`, so it passes and the ratchet
tightens for whoever goes next.

**`close()`** on a maze cell releases nothing — the baked scenes, the reference envs and their
`mujoco.Renderer`s are MODULE caches shared by every instance, and releasing one EGL context corrupts
every other live one in the process, which is the rule this family's history exists to record. On a
manipulation cell `_close` drops the per-instance index tables and the style-host cache and nothing
else, for the same reason (`_KEEPALIVE` and `_PlateRig` enforce the never-released invariant
directly). What `close()` buys is the contract's other half: a closed env REFUSES to render.

**`__module__` is PINNED on all 27 cells**, mazes to `stable_envs.envs.ogbench` and the two
manipulation cells to `stable_envs.envs.ogb_manip`. `registry.spec` derives an env's FAMILY from the
last component of `__module__`, so `type()`'s default would have renamed the families to `maze` and
`manipulation` and moved `spec(name)["family"]`, `list_envs(family=...)`, `list_families()`, the
registry row every Step-8 artifact recorded, and `scratch/step4/universe.py::shard_of`, which reads
the same field. Both pins are accurate as well as unchanged: each class IS reachable at the path it
names. **This workstream therefore does NOT hit the registry-row `module` comparison Workstream A's
R2 ruling covers** — `module` is compared like every other column and is identical.

Unchanged, and asserted: `state_dim`, `state_names`, the packed row and its column order,
`sample_states`, `set_state`/`get_state`, `render(rows)`, the whole style API, `PLATE_PATHS`, `TILE`,
the fitted shading constants, `default_size`, `default_supersample`, every reference-bridge
behaviour, `describe()`, `variation_space`, and every fidelity declaration.

## 3. Step-8 IDs affected

**Behaviorally: none.** Three of the 42 SWM ids are served by these cells and all three are
unchanged:

| SWM id | cells | result |
|---|---|---|
| `swm/OGBMaze-v0` | `PointmazeLargeNavigate`, `PointmazeLargeStitch` | frames byte-identical |
| `swm/OGBCube-v0` | `OGBCube` | declarations identical; frames owed to the GPU oracle |
| `swm/OGBScene-v0` | `OGBScene` | declarations identical; frames owed to the GPU oracle |

The oracle compares each cell's declared `swm_id` off the VariationMap tables (the source BOTH trees
have — the registry's own `SWM_IDS` table is a Wave-0 addition, so asking the baseline for it would
compare a thing against its own absence) and then checks the Step-9 registry's alias table agrees
with those declarations, cell for cell. `swm/OGBMaze-v0` still resolves to two cells and still
refuses to pick one for the caller.

## 4. Targeted tests

`tests/unit/test_envs_ogbench.py`, **59 tests**, marker `unit`, in two halves. The STRUCTURAL half
needs no MuJoCo and no `ogbench`; the BEHAVIOURAL half constructs cells and renders them on the CPU
ray-caster.

| test | what it pins |
|---|---|
| `the_import_path_is_the_package_and_it_carries_the_step8_names` | `stable_envs.envs.ogbench` is the package, 21 named Step-8 attributes resolve off it, and every one of the 25 cells is the REGISTERED class (the oracle checks all 17 + 41 names) |
| `the_manipulation_module_path_is_an_alias_and_not_a_set_of_re_exports` | `ogb_manip IS ogbench.manipulation` — the identity, because re-exports would satisfy a read and not a monkeypatch — plus the 24 names its tests reach for and that `getsource(_context)` still works |
| `the_module_caches_are_one_object_each_across_the_package` | **Workstream B's R3 lesson applied before it bites**: `_SCENE_CACHE`/`_ENV_CACHE` are the same dict through the package and through `maze`, checked by identity AND by a write, so a re-export written as a copy fails loudly instead of leaving `test_ogbench_scene_cache.py` isolating a cache nothing uses |
| `every_cell_reports_the_step8_module_so_the_registry_row_did_not_move` | both pins, both families, `spec(...)["family"]`, `list_envs(family=...)` — and that the pinned path is one the class is actually reachable at |
| `a_maze_cell_is_a_declaration_and_overrides_no_renderer_method` | no hierarchy grew: 25 cells over ONE base, each leaf's own namespace holding no callables |
| `the_maze_declarations_are_still_exactly_what_step_8_declared` (×25) | `parity_interior_min == 1.0` NOT relaxed on any cell, the tier, 224 px, `default_supersample == 2`, `TILE == 32`, `PLATE_PATHS`, the `parity_note` claim and `reference_id` |
| `the_fitted_shading_law_constants_did_not_move` | the four numbers the whole agent-pixel claim rests on, and the per-light split beside them |
| `the_manipulation_declarations_are_still_the_measured_visual_ones` (×2) | thirteen `visual_fidelity` keys, every ceiling above its measurement, the mujoco 3.12.0 / ogbench 1.2.1 provenance, `measured_with`, the registered size and the tier claim in the docstring |
| `the_scene_carries_the_button_texture_bake_note_and_the_cube_does_not` | the `btn_metal` bake, declared per geom, on the cell that has it and not on the one that does not |
| `cpu_is_refused_rather_than_served_from_the_gpu` (×2) | `CPU3D-OGB-1`, "no CPU backend", "Use device='cuda'", and that the message quotes THIS cell's triangle count — through the class and through `snv.make` |
| `the_plate_texel_helpers_are_the_shared_module_and_not_a_second_copy` | `mj_quantise is _plate_texels.quantise`, the three call sites calling the shared functions, and — the Step-8 `38150c3` fix — that `make_current()` precedes `mjr_uploadTexture` in the code that now carries it |
| `the_maze_state_layout_covers_the_packed_row_and_hides_nothing` (×3) | the layout is contiguous from 0, covers `state_dim` exactly, and no offset is `None` |
| `the_maze_variation_taxonomy_is_the_step8_one` | the seven leaves and four init values, including `rendering.decolor_floor`, whose SWM counterpart is UNSUPPORTED-PENDING |
| `the_manipulation_variation_taxonomy_is_the_two_mapped_leaves` | `cube.color` flat `(3,)` and `floor.color` `(2, 3)` — the Step-8 boundary shapes |
| `takes_the_step9_constructor_and_the_lifecycle` / `batch_is_still_accepted…` (×3 each) | `num_envs=`, `len`, idempotent `close`, a closed env refusing `render`, the context manager, `batch=` still meaning `num_envs`, and the two disagreeing being a `TypeError` |
| `the_public_state_is_the_bodys_own_structure_and_packs_back_to_the_frozen_row` (×3) | `pack(state) == get_state()`, `pack(unpack(rows)) == rows`, both hooks empty, and the keys naming the packed columns IN ORDER |
| `a_renderer_only_cell_refuses_step_and_transition_and_says_why` (×3) | `has_dynamics` False, both refusals naming the cell and the reason and saying what DOES work, and `reset`/`render` still working |
| `render_takes_a_state_a_packed_row_or_nothing` (×3) | `render(State) == render(rows) == render()`, the shape, the dtype, `observation_spec` |
| `the_dataset_siblings_of_one_scene_draw_the_same_frame` | the fact the plate cache is keyed on, and the fact that lets the oracle replay fourteen cells on a sibling's corpus |

Exact outputs, from the worktree root at HEAD:

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
1427 passed, 212 skipped, 112 deselected, 26619 warnings in 132.05s
```

Baseline for that command on this branch's parent (`step9/integration` `b3f3a81`), measured before
any edit: **`1385 passed, 195 skipped, 112 deselected`**. 1385 + 42 = 1427 and 195 + 17 = 212, i.e.
**no pre-existing test changed its result** (the 17 skips are the behavioural half and the
swm-dependent taxonomy test, which need `ogbench`).

```
MUJOCO_GL=glfw /Users/samib/se7-ogb-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_envs_ogbench.py tests/unit/test_variation_maps_newcells.py \
  tests/unit/test_ogb_manip.py tests/unit/test_ogbench_scene_cache.py
178 passed, 2 skipped, 39 warnings in 16.77s
```

The three PRE-EXISTING files alone are `119 passed, 2 skipped` — **and a detached worktree of
`b3f3a81` produces `119 passed, 2 skipped` for the identical command**, so the family's own Step-8
tests are unmoved by the migration. (The two skips are `cvxpy` and `matplotlib`, neither of which
this venv has and neither of which is this family's.)

**Not one pre-existing test file was edited.** Workstream A's R2 registry-row concession is not
needed here (the `module` column is identical, §2) and Workstream B's two source-reading edits have
no analogue: `test_ogb_manip.py` reads its source through the alias, which is the migrated file, and
`test_ogbench_scene_cache.py` reaches the cache through the package, which is the same dict object.

The two tests the brief deselects were run anyway. `test_every_cited_scratch_path_resolves` fails on
**14** unresolved citations — the same 14 Workstream B reported, all of them Oscar `sboughan` paths
and two `carracing`/`robocasa` output directories. None is an OGBench path and no file this patch
touches cites a scratch path that does not exist.

## 5. Applicable certification checks

The Step-4 frozen corpus replay for all 25 maze cells (§6), the plate-bake probe for the two
manipulation cells (§6), the Step-8 support-matrix rows (`test_foundation_registry.py`), and
`tests/unit/test_variation_maps_newcells.py` driven against `stable_worldmodel` — every Step-8
variation mapping still resolves, unchanged, 119/2 before and after. Gate A, the GPU conformance
suite, `check_visual_fidelity` and the mujoco_warp tier measurements need a CUDA host and are owed
(§10, R1).

## 6. The oracle — default frames, variation/state comparison

`scratch/step9/envs-ogbench/oracle.py` pushes all 27 cells through BOTH trees in two separate
interpreters — a detached `git worktree` of `21cb306` and this branch, the same venv — and compares
**42 keys** per cell:

| compared | result over all 27 cells |
|---|---|
| frames, SHA-256 over the raw uint8 bytes, at the registered default size, on `cpu` | **identical** (25 render; 2 are owed, below) |
| frames rendered from explicit packed rows | identical |
| `describe()` (tier, decoder, parity floor, style names, variations, reference versions) | identical |
| `style_names`, `style_dim`, the official default appearance | identical |
| the variation space in full: taxonomy, flat layout AND every leaf's init value | identical |
| `visual_fidelity` in full: measurements, thresholds, `draw_spread`, `measured_at` | identical |
| `visual_note` — including the Scene's `btn_metal` bake residual | identical |
| `parity_interior_min`, `parity_all_exact_min`, `parity_note`, `fidelity_tier`, `render_decoder` | identical |
| `reference_versions` / `measured_with`, `reference_id`, `default_size`, `default_supersample` | identical |
| registry row **in full, `module` included**; the SWM id each cell declares | identical |
| `state_dim` / `state_names`, `set_state`/`get_state` round trip on the corpus | identical, exact |
| `sample_states(16, seed 0)` | identical |
| the CPU refusal, message included (`OGBCube`, `OGBScene`) | identical |
| `TILE`, `PLATE_PATHS`, `(LOCO, MAZE)` / `CELL`, the class docstring | identical |

**Unexplained differences: zero** (`"unexplained_differences": {}`, `"identical": true` in
`examples/results/step9/envs-ogbench/oracle.json`), and **`"expected_differences": {}`** — unlike
Workstream B there is not even one declared exception, because every Step-8 module-level name of both
modules still resolves.

The record's `head_sha` is `f00ddcf`, the last commit before the one that carries it — the commit
that adds a record cannot be the one it measured. The oracle was re-run there after the findings
landed, and the ONLY key that changed against the record measured at `a1352ab` (the migration commit
itself) was `head_sha`: all eleven frame digests, `describe()`, the style layout, the registry rows,
the declarations and the functional checks are byte for byte what they were.

Per-scene default-frame digests (first 16 hex of the SHA-256 over 64 frames at 224 px, ss=2; the
dataset siblings of a scene share its digest, which is the fact the plate cache is keyed on):

| scene | frames | scene | frames |
|---|---|---|---|
| pointmaze-medium | `22185b713581b94a` | ant-teleport | `da777169b3c52a2f` |
| pointmaze-large | `b35a6631e7d20247` | humanoid-medium | `be9993beeca4641d` |
| pointmaze-giant | `577646f1835f113f` | humanoid-large | `b4dd4403238962d7` |
| pointmaze-teleport | `3cde1e67c10fb539` | humanoid-giant | `55534ade7b9bd328` |
| ant-medium | `3234e84199fd6e7b` | ant-large | `d648171bd3bf0bb3` |
| ant-giant | `368f7277bf4b3adc` | | |

**Fourteen cells have no frozen corpus of their own and are replayed on their scene sibling's**,
which is not a shortcut: `navigate`/`stitch`/`explore` select an offline DATASET and the task never
reaches the renderer, so a sibling's states are exactly the states of this cell's scene. There is
nothing else to replay, and `test_the_dataset_siblings_of_one_scene_draw_the_same_frame` asserts the
premise rather than assuming it.

**THE PLATE BAKE, measured where the cells themselves refuse to run.** The one behavioural change
outside the constructors is the texel deduplication (§1), and it is on the manipulation cells' plate
path — which is CUDA-only territory in principle but is drawn by `mujoco.Renderer` in practice. So
`scratch/step9/envs-ogbench/plate_probe.py` drives it on the CPU through the unbound methods, in both
trees, and `examples/results/step9/envs-ogbench/plate_probe.json` records the result:

| check | Cube | Scene |
|---|---|---|
| default plate, SHA-256 | `5058c1bff67b8eb2…` | `99bdf3792b1fdfca…` — **identical to the baseline's** |
| plate at an explicit default `floor.color` | identical to the `None` plate | identical |
| plate at a custom `floor.color` | differs (mean 55.47 → 140.10) | differs (80.65 → 146.79) |
| the default, re-baked AFTER the custom one | back to the original bytes | back to the original bytes |
| `_floor_key(default)` / `(None)` / `(default + 1e-4)` | `None`, `None`, `None` | same |
| `_floor_key(custom)` | `(229, 25, 51, 76, 178, 102)` — the bytes MuJoCo stores | same |

Every pre-existing key is identical across the two trees. The only two fields that differ are two
that the shared `texel_slots` ADDS and the copy it replaces never had (`texels` 262,144 and `covered`
261,121 — the 1,023 uncovered texels are MuJoCo's `markrgb` grid lines, which swm does not touch
either). So the deduplication is a measurement and not a reading of two sources: the plate, the cache
key and the re-bake-back-to-default all produce the same bytes they did.

The **Step-9 claims**, which have no baseline to compare against, are checked in the same script on
`cpu` for all 27 cells (`"functional_ok": true`): every cell accepts `num_envs=`; `pack`/`unpack`
round-trip the packed row bit for bit; `pack(env.state) == get_state()`; no field has offset `None`;
`has_dynamics` is False and both refusals raise `NotImplementedError`; `render(State)` equals
`render(rows)` equals `render()`; `reset(seed=0)` works; and a closed env refuses. The two
manipulation cells report their declared CPU refusal instead, which is the correct answer for them
and is compared against the baseline's own refusal.

## 7. Performance (plan §8)

**No paired timing run, and the reason is that this patch has no hot-path edit to measure.** Plan §8
asks for before/after on changes affecting rendering or dynamics. `render`, `_fk`, `_bake`, `_scene`,
`composite`, the mujoco_warp launch, the mover LUT, the per-world uploads and the chunk loop are
**byte-identical** to Step 8 (§1's diff removes 61 lines and adds none inside any of them). What Step
9 added is `__init__` work — a `state_layout` tuple — and methods a render never calls (`pack`,
`unpack`, `state`, `_close`). The three functions whose bodies DID change are `mj_quantise`,
`_grid_texture` and `_apply_floor`, and all three are on the plate-bake path, which runs once per
plate-cache MISS and never per render; `_floor_key` runs once per render and is the same arithmetic
in a shared function.

Unlike classic control and PushT, neither half of this family has a `step` to pay the StepResult cost
on: both refuse. GPU timings for the manipulation cells are owed with their frames (R1).

## 8. Resource lifetime

Unchanged, and that is the point. `_ENV_CACHE`, `_SCENE_CACHE`, `_CTX_CACHE`, `_PLATE_CACHE`,
`_PLATE_RIGS`, `_BUNDLE_CACHE`, `_REF_CACHE` and `_KEEPALIVE` are all MODULE caches and all of them
moved with their module rather than being rebuilt; `test_the_module_caches_are_one_object_each…`
asserts the re-exports share them. No renderer and no EGL context is closed anywhere in this family,
which is the rule the `_PlateRig`/`_KEEPALIVE` comments record and the reason
`OGBenchMaze.reference_env` says "never closed". `close()` is explicit, idempotent and never required
for correctness from `__del__`; a closed env refuses `render`/`reset`/`step` with a message naming the
call. The suite's resource guard reports the same counts as at baseline over the unit run —
**16 EGL contexts created / 0 released, 15 renderers created / 0 closed**, identical before and after.

## 9. The §17 ten-item report

1. **files moved/removed/created** — §1. Two files `git mv`d into a package, one rewritten as a
   23-line alias, one package `__init__` created, five artifacts/scripts/tests created, nothing
   importable removed.
2. **API changes** — §2. Additive: `num_envs=`, the entity-grouped State, `close()`. The packed row,
   the frames and every declaration are untouched.
3. **Step-8 IDs affected** — §3. Behaviorally none; the three ids these cells serve are unchanged,
   two of them pending their GPU frames.
4. **targeted tests** — §4. 59 new, 1427 passing locally, 178/2 in the OGBench venv, and the three
   pre-existing family test files at 119/2 before and after.
5. **applicable certification checks** — §5. Step-4 replay done for all 25 mazes and the plate probe
   for both manipulation cells; Gate A and the GPU suites owed (R1).
6. **default-frame comparison** — §6. Identical SHA-256 over 25 cells × 64 states, 11 distinct
   digests, one per scene.
7. **variation/state comparison** — §6. The taxonomy leaf for leaf WITH its init values, the style
   layout, the default appearance, the packed layout, the round trip and `sample_states`: identical.
8. **performance comparison for hot paths** — §7. No hot-path edit exists to measure; the render
   path is byte-identical and the changed functions are per-cache-miss.
9. **resource-lifetime comparison** — §8. Guard counts identical (16/0, 15/0); explicit `close`.
10. **unexplained oracle differences** — **zero**, with **zero** declared exceptions.

## 10. Owed to the GPU oracle

Recorded rather than claimed, and no sbatch was written (the brief forbids one):

| owed | why | what closes it |
|---|---|---|
| `OGBCube` / `OGBScene` FRAMES | no CPU backend by declaration (`CPU3D-OGB-1`); there is no honest way to render them on this host | `MUJOCO_GL=egl <cuda python> scratch/step9/envs-ogbench/oracle.py --device cuda --cells OGBCube,OGBScene` on a CUDA host with `mujoco_warp` — the script already runs both arms and needs no edit |
| the maze CUDA arm | the mazes render on `cpu` here; the `cuda` raycast3d path is unproved by this ticket | the same script with `--device cuda` and no `--cells` |
| `check_visual_fidelity` for the two visual cells | it grades the declaration this patch preserves, and it needs frames | the GPU conformance run |
| Gate A, `tests/gpu/*`, the tier measurements | CUDA-only | the coordinator's GPU pass |

The declarations, the CPU refusal, the variation space, the state layout, the plate bake and the
registry row of both manipulation cells ARE compared here (§6), so what is owed is the pixels and
nothing else.

## 11. Requests for the coordinator

| # | request | detail |
|---|---|---|
| **R1** | **run the GPU arm** | §10. One command, no edit; the script's `--device cuda` arm is the same code the CPU arm ran. |
| **R2** | **Wave 3 must move the family name and `shard_of` TOGETHER** | The two manipulation cells' `__module__` is pinned to `stable_envs.envs.ogb_manip`, which keeps the registry row identical today. When Wave 3 deletes that alias the family becomes `ogbench` (or `manipulation`, if the pin is simply dropped), and `scratch/step4/universe.py::shard_of` — which maps `"ogbench" -> "ogb"` and has no `ogb_manip` entry, so these two cells currently shard as `"raster"` — moves with it. That is one line in `manipulation._MODULE` plus one in `shard_of`, and it is a coordinator decision because both files are outside this workstream's ownership. Filed now so it is not discovered in Wave 3. |
| **R3** | **`docs/api/envs.md` line 26** | `::: stable_envs.envs.ogbench` still resolves (mkdocstrings renders a package), but it now documents only what the package `__init__` re-exports. Workstream M should decide whether to add `::: stable_envs.envs.ogbench.maze` and `::: stable_envs.envs.ogbench.manipulation` — the two module docstrings are where this family's measured facts live. Not edited here; `docs/` is not this workstream's. |
| **R4** | **`tensordict` 0.14.2 was installed into `/Users/samib/se7-ogb-venv`** | Wave 0 made it a core dependency and installed it into `se6-venv` and `se8-vm-venv`; the OGBench venv did not have it, so `import stable_envs` failed there outright. Nothing else in that venv was touched. Reported because it is an environment change, per the brief's rule about venvs. |
| **R5** | **the extraction that was never adopted — now closed, and worth a look elsewhere** | `envs/_plate_texels.py` was lifted OUT of `ogb_manip.py` in Step 8 "so that `dmcontrol.py` runs the same code rather than a second copy of it that is free to drift" — and `ogb_manip.py` kept its copy, so the one family the module was extracted from was the one family not running it. Closed here (§1, §6). `envs/fetch.py` NAMES the helper in a docstring (line 599, describing its own plate rig) and does not import it; whether that is a third copy waiting to be folded in, or a genuinely different operation, is for the Fetch workstream (C) to judge — it is their file and I did not touch it. |
| **R6** | **14 pre-existing unresolved scratch citations** | `test_every_cited_scratch_path_resolves` fails on the identical 14 Workstream B reported (Oscar `sboughan` paths, `carracing/logs/`, `robocasa/out/corpus`). None is this family's, and this patch adds none. Noted so two workstreams do not report it as new. |
