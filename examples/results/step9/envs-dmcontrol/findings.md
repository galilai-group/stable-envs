# Step 9 — Wave 1, Workstream B: the DM Control family

**Workstream:** `step9/envs-dmcontrol` · **Parent:** `step9/integration`, merged twice: `86a6ba4`
(A merged) and `b3f3a81` (the R1 instance-method `_layout`, and Workstream H's raycast3d/cpu3d
consolidation), no rebase ·
**Step-8 baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Host:** Apple M5 Mac (`cis1b1zs2j3`), CPU only — no CUDA device in this ticket.
**Interpreters:** `/Users/samib/se8-vm-venv` (dm_control 1.0.44 + mujoco 3.11 + stable_worldmodel +
tensordict) and `/Users/samib/se6-venv` (the CPU-only suite). `MUJOCO_GL=glfw`, `-p no:cacheprovider`.

**Result: PATCH.** `stable_envs/envs/dmcontrol.py` (3,376 lines) is now the PACKAGE
`stable_envs/envs/dmcontrol/` — one small base, one module per renderer, three private helper
modules moved verbatim, and one module per domain. All **eighteen** cells migrated. Every cell's
corpus renders to **byte-identical frames** through this branch and through a detached worktree of
`21cb306` in the same venv, and `describe()`, the style layout, the default appearance, the
variation taxonomy, the fidelity declarations, `visual_fidelity`, the 3.11 `reference_versions`
provenance, the parity floors and their prose, Manipulator's state exclusions and their contract, the
shape knobs, the registry row **including `module`**, `state_dim`/`state_names`, the
`set_state`/`get_state` round trip, `sample_states` and the CPU refusal with its message are all
identical. **Unexplained oracle differences: zero.** T3 `floor.color` bake-vs-recompile is still
IDENTICAL. Local unit suite **1368 → 1496** (128 new, no pre-existing test changed its result).

Cells — twelve PARITY (fixed model camera, baked plates + analytic ray-cast): `DMCAcrobotSwingup`,
`DMCBallInCupCatch`, `DMCCartpoleBalance`, `DMCCartpoleSwingup`, `DMCFingerSpin`,
`DMCFingerTurnEasy`, `DMCFingerTurnHard`, `DMCManipulatorBringBall`, `DMCPendulumSwingup`,
`DMCPointMassEasy`, `DMCReacherEasy`, `DMCReacherHard`; six VISUAL (`mode="trackcom"` camera,
whole scene to MJWarp/cpu3d every frame): `DMCCheetahRun`, `DMCHopperHop`, `DMCHumanoidStand`,
`DMCQuadrupedWalk`, `DMCSwimmerSwimmer6`, `DMCWalkerWalk`.

---

## 1. Files moved / created / removed

| before | after |
|---|---|
| `stable_envs/envs/dmcontrol.py` (3,376 lines) | `stable_envs/envs/dmcontrol/`, 22 modules, 3,721 lines |

| module | lines | what it is |
|---|---:|---|
| `base.py` | 548 | `_DMCModel`: model plumbing, the widened state, the appearance knobs, the visited distribution, the bridge to dm_control's own renderer. Properties of the DOMAIN. |
| `suite.py` | 1,139 | the PARITY renderer (`DMControlSuite`), moved verbatim |
| `trackcom.py` | 412 | the VISUAL renderer (`DMControlTrackCom`, `_TCtx`, `_tc_context`, `_tc_frame`), moved verbatim but for the CPU-refusal lookup (§2) |
| `_model.py` | 362 | what a compiled `MjModel` says; the constants; the caches; `_episode_slots` |
| `_tendon.py` | 298 | `mjv_catenary` and the capsule chain it becomes — ball_in_cup's string |
| `_shading.py` | 139 | the lit-surface law both tiers evaluate, and `_Lights` (where MuJoCo's headlight exists) |
| `__init__.py` | 146 | the family docstring, the re-exports, and the four collected tables |
| `_cells.py` | 142 | `parity_cell`/`trackcom_cell` — how a cell is DECLARED — and the provenance strings several share |
| `quadruped.py` | 140 | SWM's model (`_swm_quadruped_env`, `_SWMQuadrupedModel`), `_QUADRUPED_AT`, `QUADRUPED_NOTE`, its `visual_fidelity` |
| `swimmer.py` | 99 | its own provenance string, `SPECULAR_NOTE`, `TRACKCOM_RENDER_FLAGS`, the CPU refusal, its `visual_fidelity` |
| `manipulator.py` | 97 | the 224 px default, `SHAPE_KNOB_SPECS`, `RENDER_INVARIANT_DIMS`, `STATE_DIMS_NOTE` |
| `cheetah/hopper/humanoid/walker.py` | 26-27 each | that domain's `visual_fidelity` entry and its cell |
| `finger.py` | 26 | `ALL_EXACT_FLOOR` and the three finger cells |
| `cartpole/reacher/ball_in_cup/acrobot/point_mass/pendulum.py` | 9-14 each | the cells, and what a reader needs to know about that domain |

Created besides: `tests/unit/test_envs_dmcontrol.py` (131 tests),
`scratch/step9/envs-dmcontrol/{oracle.py,perf.py,gpu.sbatch}`,
`examples/results/step9/envs-dmcontrol/{oracle.json,performance.json,t3_bake_vs_recompile.json,findings.md}`.

**`stable_envs/envs/dmcontrol.py` is REMOVED rather than kept as an alias shim, and it has to be.**
The brief asks for Workstream A's `swm_nav.py` pattern; that pattern works because `swm_nav` and
`navigation` are different names. Here the new home is a package with the SAME name, and a module
and a package cannot coexist in one directory — the package wins the import and the `.py` would be
unreachable dead code. So the package IS `stable_envs.envs.dmcontrol`, and the obligation the shim
existed to meet is met directly: the oracle reads the **79** module-level names off the baseline
source and checks each one resolves off the package. All 79 do except `_make_classes`, the private
factory that built the eighteen classes in one `type()` loop, which is declared absent with its
reason (§6) because the cells are now written where their domains are and there is no loop to name.
It was never referenced outside the module.

**The line count grew, 3,376 → 3,721 (+10%), and none of it is duplication.** `+80` of Step-9
machinery in `base.py` (the `num_envs=` constructor, `_layout`, `_close`, and the comment block
explaining the state grouping), `+146` for the package `__init__` (the family docstring, which the
old module carried at its top, plus the re-exports), `+~110` of per-module docstrings saying what
each module is and why it is separate, and `+~50` of per-domain module headers. The four hand-written
tables were **deleted** and replaced by collected views (§2). `suite.py` and `trackcom.py` are
byte-identical to the regions they came from apart from imports and one line.

## 2. API changes

**Additive.** Nothing a Step-8 caller did stopped working, and the packed row, the frames and every
declaration are unchanged.

```python
env = snv.DMCCartpoleSwingup(num_envs=64, device="cuda")   # num_envs=, batch= still accepted
st  = env.state                    # State: {"qpos": (nq,)} or {"qpos": (nq,), "episode": (k,)}
rows = env.pack(st)                # the frozen packed row, byte for byte
px  = env.render(st) == env.render(rows) == env.render()
env.step(a)                        # NotImplementedError: no batched simulator (see below)
env.close(); with snv.DMCWalkerWalk(...) as env: ...
```

**The public State is `qpos` plus the episode draw, grouped by what the two halves ARE.** `qpos` is
the simulator's configuration; `episode` is what `initialize_episode` wrote into the MODEL, which is
a different KIND of quantity and is the whole subject of the widening `_episode_slots` performs.
Eleven of the eighteen draw nothing per episode and get the single key alone, which is what contract
§3 means by no ceremony for a state that is one vector. **Nothing is hidden**: every packed column
belongs to one of the two keys, so no field has offset `None`, `_hidden_state` is never consulted,
and `pack`/`unpack` round-trip the frozen corpus layout bit for bit. **There is no `_live_state`
either**, and that is a measured statement rather than an omission: the packed row IS this family's
live state — `set_state` writes it and `render` reads it — so unlike classic control's float64
simulator there is no more precise buffer to answer from. The `state` property is **not** overridden
anywhere in this family.

**`state_layout` is an INSTANCE attribute and this family overrides nothing of the state machinery.**
Its width is `nq` plus however many model entries the task draws per episode, discovered by compiling
the model and running six episodes (`_episode_slots`), so it is not a per-class fact. The first round
of this patch carried an `_layout` override for that reason and filed it as R1; the foundation
**took R1** (`step9/integration` `f24d94c` made `Env._layout` and `Env._layout_dtypes` instance
methods reading `self.state_layout`), so the override is now **deleted** and `__init__` assigns
`state_layout` where the model is compiled. Not `state`, not `_layout`, not `_hidden_state`, not
`_live_state` — nothing. `test_the_public_state_is_qpos_plus_the_episode_draw_and_hides_nothing`
asserts `"_layout" not in vars(_DMCModel)` so the override cannot come back quietly. **R1 is CLOSED.**

**NO DYNAMICS, declared rather than omitted.** There is no batched simulator in this package for any
DM Control cell, so `has_dynamics` is False and `step`/`transition` raise `Env`'s precise refusal
naming the cell and the reason; `reset` and `render` always work. No simulator was invented and no
zero reward is fabricated. This is the same refusal Step 8 raised — the message is unchanged — and
`test_a_renderer_only_cell_refuses_step_and_transition_and_says_why` pins it for all eighteen.

**`close()` releases what the INSTANCE owns and nothing else.** `_DMCModel._close` drops the
per-instance CPU scene cache. The `mujoco.Renderer` in `_ENV_CACHE` and the captured MJWarp context
in `_TC_CACHE` are **not** released, which is the family's policy rather than an oversight: they are
shared by every instance at the same (domain, size) and releasing an EGL context corrupts every other
live one in the process. What `close()` buys is the contract's other half — a closed env refuses to
render rather than handing back frames drawn through something the caller believes is gone.

**Four hand-written tables became collected views**, which is the only behavioural code change
outside the constructor and it is a deletion. `DEFAULT_SIZES`, `CPU_UNCERTIFIED`, `TRACKCOM_BASES`
and `VISUAL_FIDELITY` were dicts keyed by class name standing BESIDE the classes they described — a
second place to change, and `CPU_UNCERTIFIED` was read at construction time by name lookup. They are
now derived from the cells (`default_size`, `CPU_UNCERTIFIED_REASON`, `__bases__`,
`visual_fidelity`) and re-exported under their Step-8 names and shapes, so every reader keeps working
and the table cannot disagree with the class. `DMControlTrackCom` reads
`type(self).CPU_UNCERTIFIED_REASON` instead of `CPU_UNCERTIFIED.get(type(self).__name__)`; the
refusal message is unchanged, and the oracle compares it.

**Each cell's `__module__` is pinned to `stable_envs.envs.dmcontrol`**, the package, rather than
left at `type()`'s default. `registry.spec` derives an env's FAMILY from the last component of
`__module__`, so the default would have split one family of eighteen into fourteen named after
domains and moved `list_families()`, `spec(name)["family"]`, `list_envs(family=...)` and the
registry row every Step-8 artifact recorded. The attribute is also accurate — the class is reachable
at that path. **This workstream therefore does NOT hit the registry-row `module` comparison
Workstream A's R2 ruling covers**: `module` is compared like every other column and is identical.

## 3. Step-8 IDs affected

**Behaviorally: none.** Every SWM id these cells serve is byte-identical. The eighteen cells are
unchanged in name, tier, decoder, `state_dim`, `reference_id`, module and registry row.

## 4. Targeted tests

`tests/unit/test_envs_dmcontrol.py`, **131 tests** (128 passed / 3 skipped locally), marker `unit`,
in two halves. The structural half needs no MuJoCo; the behavioural half constructs cells.

| test | what it pins |
|---|---|
| `the_import_path_is_the_package_and_it_carries_the_step8_names` | `stable_envs.envs.dmcontrol` is the package and 33 named Step-8 attributes resolve off it (the oracle checks all 79) |
| `every_cell_reports_the_package_as_its_module` | the registry row, `spec(name)["family"]` and `list_envs(family="dmcontrol")` did not move — and that the attribute is truthful, i.e. the class IS reachable there |
| `every_cell_is_written_in_its_own_domain_module` | plan §3's other half: the DECLARATION lives where the domain does |
| `the_package_tables_are_views_of_the_cells_and_not_a_second_copy` | each entry compared against the class it was collected from, not against a transcription |
| `a_cell_is_a_declaration_and_overrides_no_renderer_method` | no giant inheritance hierarchy: each leaf's own namespace holds no callables, and only quadruped carries an extra BASE |
| `the_two_tiers_are_the_only_thing_below_the_base` | the hierarchy is exactly three levels and stays three |
| `the_preserved_declarations_are_still_exactly_what_step_8_declared` | the 3.11 `reference_versions`/`measured_with` on all six visual cells, the one `ALL_EXACT_FLOOR`, `parity_interior_min` NOT relaxed anywhere, Manipulator's exclusions and nobody else's, the trackcom flags, both note suffixes reaching their cell's `visual_note`, and quadruped as the one cell not at camera 0 |
| `the_one_renderer_per_domain_size_bundle_policy_is_still_a_module_level_cache` | `_ENV_CACHE`/`_SCENE_CACHE`/`_TC_CACHE` are ONE object each across the package (a `from ... import` of a dict shares it; a re-declaration would not), and no tier closes one |
| `takes_the_step9_constructor_and_the_lifecycle` | `num_envs=` on every cell, `len`, idempotent `close`, a closed env refusing `render` — and swimmer's CPU refusal asserted rather than skipped |
| `the_public_state_is_qpos_plus_the_episode_draw_and_hides_nothing` | the two keys, every offset non-`None`, `pack(unpack(rows)) == rows`, `pack(env.state) == get_state()`, both hooks empty, and the keys naming the packed columns in order |
| `a_renderer_only_cell_refuses_step_and_transition_and_says_why` | `has_dynamics` False, both refusals naming the cell and the reason, `reset` still working |
| `render_takes_a_state_a_packed_row_or_nothing` | `render(State) == render(rows) == render()`, the shape, the dtype, and `observation_spec` |

Exact outputs, from the worktree root at HEAD `9257f96`:

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
1496 passed, 198 skipped, 112 deselected, 29521 warnings in 102.92s
```
Accounting: 1368 (the pre-migration number on this branch) + 128 mine = 1496. **No pre-existing test
changed its result.**

```
MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_variation_maps_dmc_control.py tests/unit/test_step8_gate_record.py \
  tests/unit/test_step8_provenance.py tests/swm tests/unit/test_envs_dmcontrol.py \
  tests/unit/test_dmcontrol_tendon.py tests/unit/test_dmcontrol_episode_quat.py \
  tests/unit/test_cpu3d.py tests/unit/test_declarations.py
436 passed, 4 skipped, 32719 warnings in 66.37s
```
(the four named suites alone are `200 passed`, the number this branch produced before the migration.)

**After merging `step9/integration` `b3f3a81`** (the R1 foundation change and Workstream H's
raycast3d/cpu3d consolidation), re-measured at HEAD: **1513 passed / 198 skipped** locally and
**436 passed / 4 skipped** in the SWM venv. The +17 is integration's own, 16 of it
`tests/unit/test_render_primitives.py`; my file is unchanged at 131 collected. **No pre-existing test
changed its result**, and the oracle re-ran identical — see §6.

The two tests the brief deselects — `test_repo_contracts.py::test_every_cited_scratch_path_resolves`
and `test_examples.py::...[robocasa_visual/render_mjwarp.py]` — were run anyway and fail on
`step9/integration` `86a6ba4` with the **identical** set of 14 unresolved citations and the same one
example. Neither is this workstream's: none of the 14 citations is a DM Control path, and no file I
touched cites a `scratch/` path that does not exist.

The oracle record's `head_sha` is `648f074`, the last commit that changed package code — the two
commits after it add the new test file and this document, and the commit that carries a record cannot
be the one it measured.

**Two pre-existing tests were edited**, both of them tests that read this family's SOURCE by path,
and neither claim is weakened:

* `test_declarations.py::_dmc_declaration_table` AST-parsed `stable_envs/envs/dmcontrol.py` and
  exec'd the assignments `VISUAL_FIDELITY` is built from. It now walks every module of the package,
  `_cells.py` first (the per-domain entries name the shared `_AT` strings) and merges the tables,
  skipping `__init__.py` because its `VISUAL_FIDELITY` is a VIEW of the classes rather than a
  declaration. Same entries, read off the source, never imported; an assertion was added that the
  walk found something, so a drifted glob fails instead of passing vacuously.
* `test_dmcontrol_tendon.py` substituted `D._chain_capsules_batch` and `D.catenary` on the module
  and compared `_fk`'s two paths. After the split those names must be substituted **where they are
  looked up** — `suite` for the first, `_tendon` for the second — and a patch of the package
  attribute would have left the real implementations running on both sides, i.e. the test would have
  compared the batched path against itself and passed. The substitution is now **counted** and the
  count asserted, so that failure mode is closed for good rather than fixed once.

## 5. Applicable certification checks

The corpus replay for all eighteen cells (§6), the T3 `floor.color` capability check (§6), the
Step-8 support-matrix rows, and `tests/unit/test_variation_maps_dmc_control.py` driven against
`stable_worldmodel` (**every Step-8 variation mapping still resolves**, unchanged). The GPU conformance suite, the
reproducibility suite, the fast-path suite and the MJWarp tier all ran on a CUDA host in job
**6469088** — 239 + 345 + 78 passed, **0 failed**, with the official-renderer parity probe reporting
`max_abs_diff: 0` — see §11. **ESCALATION 1 CLOSED.**

## 6. The oracle — default frames, variation/state comparison

`scratch/step9/envs-dmcontrol/oracle.py` loads each cell's corpus and pushes it through BOTH trees in
two separate interpreters — a detached `git worktree` of `21cb306` and this branch, the same venv —
then compares **35 keys** per cell:

| compared | result over all eighteen cells |
|---|---|
| frames, SHA-256 over the raw uint8 bytes, at the registered default size | **identical** (17 render on cpu; see below) |
| frames rendered from explicit packed rows | identical |
| `describe()` (tier, decoder, parity floors, style names, variations, reference versions) | identical |
| `style_names`, `style_dim`, the official default appearance | identical |
| `variation_space.names()` — the taxonomy, leaf for leaf | identical |
| `visual_fidelity` in full: measurements, thresholds, `draw_spread`, `measured_at` | identical |
| `reference_versions` / `measured_with` — the mujoco 3.11 provenance | identical |
| `parity_interior_min`, `parity_all_exact_min`, `parity_note`, `fidelity_tier`, `render_decoder` | identical |
| `RENDER_INVARIANT_DIMS`, `state_dims_note` — Manipulator's exclusions and their contract | identical |
| `SHAPE_KNOBS`, `(DOMAIN, TASK, CAMERA)`, the class docstring | identical |
| registry row **in full, `module` included** | identical |
| `state_dim` / `state_names`, `set_state`/`get_state` round-trip on the corpus | identical, exact |
| `sample_states(16, seed 0)` | identical |
| the CPU refusal, message included | identical |
| `default_supersample` (the ss=1 every parity number was measured at) | identical |

**Unexplained differences: zero** (`"unexplained_differences": {}`, `"identical": true` in
`examples/results/step9/envs-dmcontrol/oracle.json`). The one **expected** entry is
`_make_classes`, declared in the script with its reason rather than discovered in the diff.

**Re-run after merging `step9/integration` `b3f3a81`**: still `"identical": true`,
`"unexplained_differences": {}`, `"functional_ok": true`, and **every one of the seventeen cpu frame
digests below is unchanged**, down to the hex. That is worth stating as a result rather than as
housekeeping: the merge brought **Workstream H's raycast3d/cpu3d consolidation**, i.e. the two
backends that draw this entire family on CPU, and it moves no DM Control pixel. Deleting the
`_layout` override moves none either. Only `head_sha` changed in the record.

Per-cell default-frame digests (first 16 hex of the SHA-256 over 64 frames, `device="cpu"`):

| cell | frames | cell | frames |
|---|---|---|---|
| DMCAcrobotSwingup | `b0a1daa2adf19cc0` | DMCReacherEasy | `597a2005e9378c7b` |
| DMCBallInCupCatch | `4c33a6266a504ae6` | DMCReacherHard | `cebe25f1e17a52cf` |
| DMCCartpoleBalance | `02eacf7808d950eb` | DMCCheetahRun | `577f0adc52520070` |
| DMCCartpoleSwingup | `36b344fd3502e8f5` | DMCHopperHop | `4a57ea2f90203706` |
| DMCFingerSpin | `73a384069eb71654` | DMCHumanoidStand | `c805dee6755f717b` |
| DMCFingerTurnEasy | `e9260f3bf24cb720` | DMCQuadrupedWalk | `9a726a2eb3514303` |
| DMCFingerTurnHard | `51920494f3d65c94` | DMCWalkerWalk | `d8d30e1c869a9e1b` |
| DMCManipulatorBringBall | `f137a58113718c70` | DMCSwimmerSwimmer6 | *(refuses cpu — see below)* |
| DMCPendulumSwingup | `fa0747187cdc7f79` | | |
| DMCPointMassEasy | `63c4969f1a3da981` | | |

**Seventeen of the eighteen render on cpu.** `DMCSwimmerSwimmer6` refuses `device="cpu"` in BOTH
trees with the identical message — it is not an SWM id and has no CPU certification, so without the
refusal it would be DRAWN on CPU with nothing behind the frames. That is the declaration being
preserved, not a gap in the proof: the refusal itself is one of the 35 compared keys. Its frames are
compared on the CUDA arm (ESCALATION 1), which is also the only arm that exercises **MJWarp** — the
backend every `visual_fidelity` number in this family was measured with, and which a cpu comparison
never executes a line of (on cpu the six trackcom cells run `cpu3d` and the twelve parity cells run
`raycast3d`'s CPU path).

**Two cells have no frozen corpus** — `DMCPointMassEasy` and `DMCSwimmerSwimmer6` were never drawn
into `examples/results/step4/states/` (neither is an SWM id). For those the BASELINE arm draws 64
states from its own `sample_states(64, rng(0))`, writes them to a cache, and the migrated arm reads
THAT file — so both arms replay one array produced by the baseline, not two draws that happen to
agree. `sample_states_sha256` is compared on top. The Step-7 cells use the corpora frozen beside
their findings (`examples/results/step7/dmc-manipulator/corpus/states/`,
`examples/results/step7/dmc-quadruped/phase1/cell/`); the other fourteen use `step4/states/`.

**T3 `floor.color`** (`scratch/step8/capability/bake_vs_recompile.py`, probe 0, the three probe
cells `DMCCartpoleBalance`/`DMCPendulumSwingup`/`DMCReacherHard`): **VERDICT IDENTICAL**, texels and
frames, sha for sha, in two processes — so the quantised cache key, the texel substitution and the
in-place upload on the existing context all still do exactly what a full swm recompile does. The knob
is present on all **twelve** parity cells (the oracle's `variation_names`), which covers the 7 SWM
ids / 8 cells the brief names and the four more that carry it. Record:
`examples/results/step9/envs-dmcontrol/t3_bake_vs_recompile.json`.

**The Step-9 claims**, which have no baseline, are checked in the migrated tree for all eighteen
cells on `cpu`: `num_envs=` accepted, `pack(unpack(rows)) == rows`, `pack(state) == get_state()`,
no field with offset `None`, `has_dynamics` False, `step` and `transition` both refusing with the
contract's `NotImplementedError`, `reset` working, `render(State) == render(rows) == render()`, and
a closed env refusing to render. `"functional_ok": true`. The script runs the same checks on `cuda`
when one is present, which is the CUDA arm's second job.

## 7. Performance (plan §8)

`scratch/step9/envs-dmcontrol/perf.py`, the foundation's method: ALTERNATING subprocesses between the
two trees (A B A B …), 5 repeats × 9 timed iterations, `torch.set_num_threads(1)`,
`OMP/MKL_NUM_THREADS=1`, CPU, median of the per-repeat medians →
`examples/results/step9/envs-dmcontrol/performance.json`. One cell per BACKEND PATH, because within
a path every cell runs the same code.

| path | cell | baseline | migrated | delta |
|---|---|---|---|---|
| parity: baked plates + `raycast3d` analytic mover pass, 84 px, ss=1 | `DMCCartpoleBalance`, 16 envs | 12.79 ms | 12.73 ms | **−0.44 %** |
| visual on CPU: `cpu3d` whole-scene raytrace | `DMCCheetahRun`, 16 envs | 34.94 ms | 34.33 ms | **−1.75 %** |

Both deltas are inside run-to-run spread (the per-repeat lists overlap) and far under plan §8's 10 %
line. An earlier 3-repeat pass gave −3.6 % / +4.8 %, i.e. the same answer with a wider band. **There
is no `step` number** and that is not an omission: no DM Control cell has a `step`, so there is no
dynamics hot path to regress — which is also why this workstream does not pay the StepResult cost
Workstream A measured. GPU timings for two parity cells and one trackcom cell are in `gpu.sbatch`
(ESCALATION 1).

## 8. Resource lifetime

Unchanged, deliberately. `_ENV_CACHE` (one `mujoco.Renderer` per (domain, size), built once and
NEVER closed), `_SCENE_CACHE` and `_TC_CACHE` are module-level in `_model.py` and every module that
reads one reads THAT object — a `from ... import` of a dict shares it, and
`test_the_one_renderer_per_domain_size_bundle_policy_is_still_a_module_level_cache` asserts the
identity, which is the assertion that would fail if a future edit re-declared a cache in a second
module and quietly gave a second env a second EGL context. `_DMCModel._close` drops only the
per-instance CPU scene cache; neither tier defines `_close`. The suite's resource guard reports the
same counts as before the migration (16 EGL contexts created / 0 released over the unit run, +2 for
the new test file's own constructions — unchanged behaviour, a pre-existing Step-8 observation).

## 9. The §17 ten-item report

1. **files moved/removed/created** — §1. One file became a package of 22 modules; nothing importable
   was removed but `_make_classes`, declared.
2. **API changes** — §2. Additive: `num_envs=`, a two-key State, `close()`. Four hand-written tables
   became collected views. `state` is not overridden; `_layout` is (request R1).
3. **Step-8 IDs affected** — §3. Behaviorally none.
4. **targeted tests** — §4. 131 new, 1496 passing locally, 436 in the SWM venv; two pre-existing
   source-reading tests re-pointed at the package, one of them strengthened.
5. **applicable certification checks** — §5. Corpus replay and T3 done for all eighteen; the three
   GPU suites green on an L40S at the declared mujoco 3.11.0 (§11).
6. **default-frame comparison** — §6 (cpu) and §11 (cuda). Identical SHA-256 over all seventeen
   cpu-renderable corpora and over **all eighteen** on cuda, 2,240 frames in total.
7. **variation/state comparison** — §6. Style layout, default appearance, taxonomy, packed layout,
   round trip, reset distribution, every fidelity and provenance declaration: identical.
8. **performance comparison for hot paths** — §7 (cpu: −0.44 %, −1.75 %) and §11 (cuda: −0.24 %,
   −0.23 %, +0.40 %). Five hot paths across four backends, every one inside noise.
9. **resource-lifetime comparison** — §8: the one-Renderer policy is unchanged and now asserted.
10. **unexplained oracle differences** — **zero**.

## 10. Requests for the coordinator

| # | subject | ask |
|---|---|---|
| **R1** | `Env._layout` was a `classmethod` reading `cls.state_dim`, which this family only knows once a model is compiled, so `_DMCModel` overrode it. | **DONE by the foundation at `f24d94c`** — `_layout`/`_layout_dtypes` are instance methods. Merged here (no rebase); the override is deleted and `state_layout` is assigned in `__init__`. **CLOSED**, §2. |
| **R2** | `envs/dmcontrol.py` is **removed**, not aliased, because a module and a package cannot share a name (§1). The obligation is met differently: 79 Step-8 names checked off the package. | **ACCEPTED by the coordinator**; recorded for Workstream D (`ogbench.py` → `ogbench/`). **CLOSED.** |
| **R3** | Two tests outside my files read this family's SOURCE by path and had to be re-pointed (§4). The tendon one was a **latent vacuous pass** waiting for any package split. | **Noted and forwarded to Workstream D.** **CLOSED.** |
| **R4** | `SPECULAR_NOTE` is a DECLARED string carried into `DMCSwimmerSwimmer6.visual_note`, and it self-cites "see SPECULAR_NOTE in stable_envs/envs/dmcontrol.py", which is now a directory. Editing it would change `describe()` and break the oracle's byte-identity. | **Deferred to Wave 3** by the coordinator. `test_every_cited_scratch_path_resolves` only checks `scratch/` citations, so nothing fails today. |
| **ESC-1** | the CUDA arm | **DONE.** Job **6469088**, L40S on `gpu3001`, `$SCR/venvs/se-vis311`, mujoco 3.11.0 / torch 2.14.0+cu126, `ALLDONE9B-HEAD` reached. **CLOSED** — §11. |

**A's R9 is MOOT**, and I was wrong to raise it as a risk: `e1da2f6` ("renderer-only cells are
detected with `dynamics.has_dynamics`, not `hasattr(env, 'step')`") was already in this branch's base
`86a6ba4`. `render_worker.py:73` and `conformance.py:1075` both call `has_dynamics(env)` today. So
`tests/gpu -k "DMC or dmc-"` has no known reason to fail on the eighteen renderer-only cells.


---

## 11. The CUDA arm — job 6469088, `ALLDONE9B-HEAD`

**L40S on `gpu3001`**, driver 575.57.08, `$SCR/venvs/se-vis311` (mujoco **3.11.0**, torch
2.14.0+cu126, mujoco_warp present, tensordict/loguru already installed — the job's own preflight
confirms `deps ok`), tree `$SCR/se9/envs-dmcontrol`, a real clone of **`0f610cf`**. Every artifact is
committed under `examples/results/step9/envs-dmcontrol/gpu/` with `/oscar/scratch/sboughan` scrubbed
to `$SCR`.

That venv is the point rather than a detail: this family's certification semantics declare
mujoco 3.11.0 + mujoco-warp 3.11.0, and the default `se-step6` is 3.12.0, under which Step 4 recorded
three renderer FAILs for a texture-anchoring drift. The provenance file records `mujoco 3.11.0`, so
the numbers below are measured against the reference the declarations name.

| section | result | artifact |
|---|---|---|
| oracle, comparison arm `--device cpu` | **PASS**, `identical: true`, `unexplained_differences: {}`, `functional_ok: true`, 0 names missing | `gpu/oracle_head_cpu.json` |
| oracle, comparison arm **`--device cuda`** | **PASS**, `identical: true`, `unexplained_differences: {}`, `functional_ok: true`, 0 names missing | `gpu/oracle_head_cuda.json` |
| `tests/gpu/test_conformance.py -k "DMC or dmc-"` | **239 passed**, 103 skipped, 1429 deselected, **0 failed** | `gpu/pytest_conformance.txt` |
| `tests/gpu/test_fast_paths.py -k` | **78 passed**, 48 skipped, 321 deselected, **0 failed** | `gpu/pytest_fast_paths.txt` |
| `tests/gpu/test_reproducibility.py -k` | **345 passed**, 81 skipped, 1731 deselected, **0 failed** | `gpu/pytest_reproducibility.txt` |
| unit files on the CUDA host | **235 passed**, 5 skipped | `gpu/pytest_unit_dmcontrol.txt` |

**The `--device cuda` arm is the one that could only be settled here**, and it settles three things a
cpu run cannot:

1. **MJWarp was compared.** On cpu the six trackcom cells run `cpu3d`; on cuda they run MJWarp, which
   is the backend every `visual_fidelity` number in this family was measured with. Both trees'
   probes report `device: "cuda"`, and the frames are **identical between them**.
2. **`DMCSwimmerSwimmer6` has a frame digest at last** — `6f1387abd1f74c89` over 64 states. It
   refuses `device="cpu"` by declaration, so the local arm compared its refusal; here it compares its
   pixels. **All eighteen cells now have byte-identical frames in both trees on at least one device,
   and seventeen on both.**
3. **The fused mover pass was compared** for the twelve parity cells.

CUDA frame digests (first 16 hex of the SHA-256 over 64 frames; they differ from the cpu digests
because the backends differ, which is the declared CPU/GPU policy — "backend implementation may
differ by device, renderer semantics may not" — and what the oracle claims is identity BETWEEN THE
TWO TREES on each device, not between devices):

| cell | frames | cell | frames |
|---|---|---|---|
| DMCAcrobotSwingup | `f9956beb9b367e3f` | DMCReacherEasy | `7dc1d00b62634dab` |
| DMCBallInCupCatch | `715096942f107cb9` | DMCReacherHard | `a7b31f4be7815916` |
| DMCCartpoleBalance | `c2c52d85ea5c15f8` | DMCCheetahRun | `d11566bb3c19931b` |
| DMCCartpoleSwingup | `a21565d4d097b3bb` | DMCHopperHop | `a59aa37dedfea0d5` |
| DMCFingerSpin | `d68a9d210a535e54` | DMCHumanoidStand | `742cbbbcf31ebdcf` |
| DMCFingerTurnEasy | `5840d1659d9e74a1` | DMCQuadrupedWalk | `509b6b660ec88bf5` |
| DMCFingerTurnHard | `88efa20e53d48eb6` | **DMCSwimmerSwimmer6** | **`6f1387abd1f74c89`** |
| DMCManipulatorBringBall | `037325a065259bbf` | DMCWalkerWalk | `f0a208f0011bac7a` |
| DMCPendulumSwingup | `e18fd8599e6350c2` | | |
| DMCPointMassEasy | `e125aa3edd24378b` | | |

The functional checks ran on `cpu` **and `cuda`** for all eighteen: 18/18 `num_envs` accepted,
18/18 `pack`/`unpack` round-trip, 18/18 `has_dynamics` False with **both** `step` and `transition`
refusing by the contract, 18/18 `render(State) == render(rows) == render()`.

**`tests/gpu` is clean and A's R9 is confirmed fixed rather than merely absent.**
`test_reproducibility.py` — the file R9 said was broken for every renderer-only cell — reports
**345 passed, 0 failed** over a `-k` that selects all eighteen of mine, which are all renderer-only.
`e1da2f6` was already in this branch's base `86a6ba4`; both guards call `has_dynamics(env)`. The
conformance and reproducibility runs also print
`reference-renderer={'env': 'DMCAcrobotSwingup', 'size': 84, 'max_abs_diff': 0}` — parity against
dm_control's own renderer, **max absolute difference 0/255**, measured in this job.

### GPU timings (paired, same card, same job)

`perf.py --device cuda`, alternating subprocesses between a detached worktree of `21cb306` and the
branch, `torch.cuda.synchronize()` around every timed call, 5 repeats × 9 iterations, median of the
per-repeat medians, 16 envs:

| path | cell | baseline | migrated | delta |
|---|---|---|---|---|
| parity, fused mover pass | `DMCCartpoleBalance` | 3.5950 ms | 3.5864 ms | **−0.24 %** |
| parity, fused mover pass | `DMCReacherHard` | 4.1861 ms | 4.1766 ms | **−0.23 %** |
| visual, **MJWarp** | `DMCCheetahRun` | 0.5333 ms | 0.5355 ms | **+0.40 %** |

All three are **two orders of magnitude inside plan §8's 10 % bar**, and the per-repeat lists
overlap completely on all three. Nothing to explain and nothing to accept.

### Which head this evidence is for

The job ran on a clone of **`0f610cf`**; HEAD is now past the `step9/integration` merge. Stated
explicitly, as asked:

> **`git diff 0f610cf HEAD -- stable_envs/envs/dmcontrol` touches nothing device-specific.** It is
> one file, `base.py`, 18 insertions and 25 deletions, and all of it is: three docstring/comment
> paragraphs, and moving the *same* layout computation out of the deleted `_layout` method into an
> `__init__` assignment that produces the identical tuple. No renderer, no backend selection, no
> `device`/`cuda`/`cpu` branch, no import of a render module. The CUDA evidence above therefore
> **carries forward to the post-merge HEAD** for this package — and the CPU oracle re-run at HEAD is
> the independent confirmation, since it is byte-identical to the one taken at `0f610cf`.

One thing this job does **not** cover, said plainly rather than left to be inferred: the merge also
brought **Workstream H's raycast3d/cpu3d consolidation** into my tree, and the job's clone predates
it. H's own GPU oracle covers those files (`examples/results/step9/render-3d/oracle_gpu_*.json`, on
integration), and my CPU oracle at HEAD shows all seventeen cpu digests unchanged across that merge —
but **no GPU run has yet exercised H's consolidated backends on DM Control cells**. If the coordinator
wants that pinned to the merged head, `gpu.sbatch` is committed and re-runs in about 12 minutes.
