# Step 9 — Wave 1, Workstream A: classic control + simple navigation

**Workstream:** `step9/envs-classic-nav` · **Parent:** `step9/integration`, merged twice: `d07ffa8`
(Wave 0) and `b1585fc` (the per-field state dtype and the navigation shard mapping, this round) ·
**Step-8 baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Host:** Apple M5 Mac (`cis1b1zs2j3`), CPU only — no CUDA device in this ticket.
**Interpreters:** `/Users/samib/se6-venv` (torch 2.14 CPU, tensordict, loguru) and
`/Users/samib/se8-vm-venv` (+ `stable_worldmodel`, DM Control). `MUJOCO_GL=glfw`, `-p no:cacheprovider`.

**Result: PATCH.** Ten cells migrated onto the Wave-0 foundation. Revised after the coordinator's
rulings of 2026-09-17: `step9/integration` `b1585fc` merged in (no rebase), the per-field public dtype
adopted, ESCALATION 1 closed with the coordinator's own measurement, and one new
foundation defect found and filed (R8). The frozen Step-4 corpus of every
one of them renders to **byte-identical frames** through this branch and through a detached worktree
of `21cb306` in the same venv; `describe()`, the style layout, the default appearance, the variation
taxonomy, the fidelity declarations, the packed layout, `sample_states` and the registry row are
identical. **Unexplained oracle differences: zero.** Local unit suite **1355 passed**; SWM-venv subset
**270 passed / 57 skipped**, which is exactly what `step9/integration` alone produces on this host.

Cells: `CartPole`, `Pendulum`, `Acrobot`, `MountainCar`, `MountainCarContinuous` (classic control);
`TwoRoom`, `TwoRoomTarget`, `Piecewise`, `PiecewiseZones`, `SimplePointMaze` (navigation).

---

## 1. Files moved / created / removed

| before | after |
|---|---|
| `stable_envs/envs/swm_nav.py` (1,106 lines) | merged into `stable_envs/envs/navigation.py`; the file is now a 14-line `sys.modules` alias |
| `stable_envs/envs/simple_point_maze.py` (373 lines) | merged into the same module; a 15-line alias |
| `stable_envs/envs/classic_control.py` | edited in place (`StableEnv` → `Env`, the Step-9 block) |

Created: `stable_envs/envs/navigation.py` (1,527 lines), `tests/unit/test_envs_classic_nav.py`
(88 tests), `scratch/step9/envs-classic-nav/oracle.py`, `scratch/step9/envs-classic-nav/perf.py`,
`examples/results/step9/envs-classic-nav/{oracle.json,performance.json,findings.md}`.

Removed: **nothing importable.** `stable_envs.envs.swm_nav` and `stable_envs.envs.simple_point_maze`
are `sys.modules` ALIASES of the merged module rather than re-exports, so `SN._alpha_blend`,
`SN._ALPHA_FLOOR`, `SN._gaussian_dot`, `SN.IMG` (the four names `tests/unit/test_swm_nav_render.py`
and `tests/gpu/test_navdot_fused.py` reach for) and `DEFAULT_WALL_SHAPE`/`DEFAULT_WALL_POS`
(`scratch/spm_parity.py`) all still resolve, there is one module object so a monkeypatch through
either path is seen by both, and the oracle checks the identity. Wave 3 removes the two files.

**Duplication removed** (~85 lines), all of it genuine, none of it a shared-renderer rewrite:

| what was duplicated | now |
|---|---|
| the SWM reference bridge — `reference_env`, `reference_render` (the `options={"variation": []}` reset protocol), `reference_step`, `step_reference_reset` — written once in `swm_nav` and once in `simple_point_maze` | `_SWMPort`, with `reference_kwargs` and `reference_render_args` as the two per-cell hooks and `reference_set_state`/`reference_sim_state` as the two genuinely per-cell methods |
| the six `step_reference_*` declarations, identical in both files | on `_SWMPort` |
| `SimplePointMaze._style_for` — a hand-rolled batch-1 broadcast | `StyleTensor.for_batch`, which is the one operation that changes a style's batch **and refuses a per-environment style**, a check the hand-rolled copy did not have |
| five identical ten-line docstrings explaining why each concrete cell re-declares `step` | one note above `_SWMPort.step` and five three-line overrides |
| five hand-written `state_layout` declarations for classic control | derived in `_Classic.__init_subclass__` from `sim_state_names`/`render_from_sim`, which already carry the fact |

`navigation.py` is 1,527 lines against the 1,479 of the two files it replaces: −85 of duplication,
+~115 of Step-9 state/dynamics machinery, +~20 of merged header.

The two halves keep their own renderer (`render/navdot` for the four torch-rendered cells, `Scene2D`
for the matplotlib one) and share only `_SWMPort`. Merging the renderers was **not** attempted: the
brief forbids a shared-renderer rewrite and the two references are different kinds of program.

## 2. API changes

Additive, except `step`'s return type, which is the Wave-0 contract for every migrated cell:

```python
env  = snv.CartPole(num_envs=64, device="cpu")     # num_envs=, batch= still accepted (deprecated)
st   = env.state                                   # State: CartPole {"x","x_dot","theta","theta_dot"}
out  = env.step(a)                                 # StepResult TensorDict, carrying "state" only
nxt  = env.transition(st, a)                       # pure; bit-identical to step
traj = env.unroll(st, actions)                     # (T, num_envs, action_dim) actions, open loop
px   = env.render(st) == env.render(env.pack(st)) == env.render()
env.close(); with snv.TwoRoom(...) as env: ...
```

Public State keys, by cell:

| cell | State |
|---|---|
| CartPole | `{"x", "x_dot", "theta", "theta_dot"}` |
| Pendulum | `{"theta", "theta_dot", "torque"}` |
| Acrobot | `{"theta1", "theta2", "dtheta1", "dtheta2"}` |
| MountainCar, MountainCarContinuous | `{"position", "velocity"}` |
| TwoRoom, Piecewise, PiecewiseZones, SimplePointMaze | `{"agent": {"position"}}` |
| TwoRoomTarget | `{"agent": {"position"}, "target": {"position"}}` |

Classic control's keys are the reference's own `sim_state_names`, flat, one per dynamical variable:
these envs have no entities to group by, and renaming `x_dot` to `cart.velocity` would put a second
vocabulary between the user and the reference the numbers are certified against.

**`num_envs=`**: all ten cells already accepted it before this patch (they take `*args, **kwargs`, or
inherit `Env.__init__`), so **none of them was in the Wave-0 holdout list and
`tests/unit/test_foundation_api.py::test_the_number_of_classes_that_still_reject_num_envs_only_goes_down`
stays at 41** — the ratchet cannot go down for this workstream because it never counted these
classes. The 41 are `highway` (6), `maze2d` (4), `mujoco_locomotion` (4), `ogb_manip` (2) and
`ogbench` (25), i.e. Workstreams D and F. `test_envs_classic_nav.py` asserts the constructor on all
ten anyway, so the property is now pinned by construction and not only by signature.

**Hidden dynamical state.** Classic control's velocities (`x_dot`, `theta_dot`, `dtheta1`,
`dtheta2`, `velocity`) are declared in `state_layout` with offset `None` — real dynamical state the
packed row has never carried, exactly PushT's agent velocity — so they are part of the public State,
`transition` is pure, `unpack` fills them with zeros (which is what `set_state` writes and the
condition `reference_set_state` puts the official env in), and `pack` drops them, so **not one column
of the frozen corpus layout moved**. Pendulum is the exception in the other direction: its `torque`
is a DRAWN quantity that is not a velocity, which is why the layout is derived from
`render_from_sim` rather than from "the render dims come first".

**The classic-control State is float64.** This family's claim is near-bit-exactness against a
float64 reference (MountainCar tracks it to 3.3e-16 over 100 steps); the packed row is float32
because that is the renderer's contract. A State that rounded the dynamics to float32 would make
`transition` a **different, lower-precision simulator** from `step` — identical for one step taken
from a corpus state, drifting from the second. Since the coordinator's `b1585fc`, that is declared
the supported way: `state_layout` carries `torch.float64` as its fourth element, `unpack` and
`Env.state` honour it, and `pack` still writes the unchanged float32 row, so `pack(env.state) ==
get_state()` to the bit and nothing downstream of the renderer sees a different number.

`_Classic.state` is nevertheless **kept**, and the reason is measured rather than stylistic: the
fourth element fixes the DTYPE and not the PROVENANCE. `Env.state` rebuilds the drawn fields by
widening the packed float32 row, and that row is a lossy projection of the simulator buffer
(`_state = _sim[:, render_from_sim].to(float32)`), so `float64(float32(x))` is not `x`. Measured on
CartPole after five steps, the drawn `x` differs from the simulator's own value by **9.6e-08** — and
that is not cosmetic: `transition(Env.state.fget(env), a)` stops equalling `step(a)` from that point
on, i.e. the functional path quietly becomes a different simulator. Reading `_sim` is the only way to
hand back the number the next step will actually start from. `test_transition_still_equals_step_after_the_env_has_been_stepped`
pins the bit-equality MID-TRAJECTORY, which is where it would show. See §10, request R7.

**`_Classic.frame_shape`** (new): `Env.observation_spec` assumes a square frame, and CartPole,
MountainCar and MountainCarContinuous render at 400×600, so the spec was reporting `(600, 600, 3)`
for the buffer a consumer allocates. Found by the new render test; fixed in the family, not the base.

Unchanged and asserted: `state_dim`, `state_names`, the packed row and its column order,
`sample_states`, `set_state`/`get_state`, `render(rows)`, the whole style API, `scene`, every
reference-bridge behaviour, `describe()`, `variation_space`, the fidelity declarations, the
`use_step_graph` / `use_fused_render` switches and the CUDA-graph capture.

## 3. Step-8 IDs affected

**Behaviorally: none.** Eight of the 42 SWM ids are served by these ten cells, and all eight are
byte-identical (spellings from the accepted Step-8 matrix, asserted row for row by
`test_foundation_registry.py`):

| SWM id | cells |
|---|---|
| `swm/AcrobotControl-v1` | `Acrobot` |
| `swm/CartPoleControl-v1` | `CartPole` |
| `swm/MountainCarControl-v0` | `MountainCar` |
| `swm/MountainCarContinuousControl-v0` | `MountainCarContinuous` |
| `swm/PendulumControl-v1` | `Pendulum` |
| `swm/Piecewise-v0` | `Piecewise`, `PiecewiseZones` |
| `swm/TwoRoom-v1` | `TwoRoom`, `TwoRoomTarget` |
| `swm/SimplePointMaze-v0` | `SimplePointMaze` |

The two multi-cell ids still resolve to their two cells each and still refuse to pick one for the
caller.

## 4. Targeted tests

`tests/unit/test_envs_classic_nav.py`, 88 tests, marker `unit`, parametrised over all ten cells:

| test | what it pins |
|---|---|
| `takes_the_step9_constructor` | `num_envs=`, `len`, `close` idempotent, a closed env refuses `render` |
| `state_keys_are_the_declared_ones_and_cover_the_packed_row` | the public vocabulary, written out rather than derived; `pack(unpack(rows)) == rows` |
| `classic_velocities_are_hidden_state_the_packed_row_does_not_carry` | offset `None` exactly for the undrawn variables; `unpack` zeroes them; `_hidden_state` answers every one |
| `the_classic_public_state_is_float64_and_packs_back_to_the_float32_row` | the dtype decision, and that it costs the packed row nothing |
| `transition_is_bit_identical_to_the_stateful_step` | `torch.equal` on the packed row AND on every key |
| `transition_does_not_touch_the_env_and_unroll_is_repeated_transition` | purity (`get_state()` unchanged), `unroll == 4× transition`, **and `== 4 stateful steps`** — the horizon where a float32 State would have drifted while a one-step check passed |
| `render_takes_a_state_a_packed_row_or_nothing` | `render(State) == render(rows) == render()`, and `observation_spec` |
| `step_reports_a_state_and_does_not_fabricate_a_reward` | `list(out.keys()) == ["state"]` |
| `the_old_navigation_module_paths_are_the_merged_module` | the two shims are the same module object and carry the names the other tests reach for |
| `the_registered_default_frame_size_and_the_specs_agree` | `size`, `action_spec`, `state_spec` leaves |
| `the_shared_swm_reference_bridge_drives_simple_point_maze_the_old_way` | on a stub: `options={"variation": []}`, `u.state` float32, `render("rgb_array")` — see ESCALATION 1 |
| `the_fast_state_property_agrees_with_the_base_class_path` | `_Classic.state` vs `Env.state` after three steps |
| `a_classic_cell_whose_layout_does_not_cover_its_packed_row_fails_at_import` | the class-creation check that replaces the per-call hidden-field guard |

Two more tests were added after the coordinator's rulings, parametrised over all ten cells (+20):

| test | what it pins |
|---|---|
| `transition_still_equals_step_after_the_env_has_been_stepped` | the bit-equality MID-TRAJECTORY, which is where a State rebuilt from the lossy packed row stops matching (§2) |
| `a_step_result_is_not_a_view_of_the_env_that_produced_it` | a `StepResult` the caller keeps does not change when the env is stepped again — the silent corruption R8 describes |

Exact outputs, both from the worktree root, at the post-merge head:

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
1355 passed, 195 skipped, 112 deselected, 26619 warnings in 54.83s
```

Accounting: 1246 (Wave-0) + 108 mine + 1 (`b1585fc`'s own dtype test) = 1355. **No pre-existing test
changed its result** at any point.

```
MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_variation_maps_dmc_control.py tests/unit/test_variation_maps_newcells.py \
  tests/unit/test_variation_maps_fetch.py tests/unit/test_step8_gate_record.py tests/swm
270 passed, 57 skipped, 4088 warnings in 18.26s
```

That is **270/57 and not the 269/58 of the Wave-0 baseline**, and the difference is not mine: a
detached worktree of `step9/integration` alone produces `270 passed, 57 skipped` on this host, so one
test moved from skipped to passing somewhere between `d07ffa8` and `b1585fc`. My branch matches
integration exactly on this subset.

**One pre-existing test was edited**, and only in how it compares one column:
`tests/unit/test_foundation_registry.py::test_every_canonical_class_name_resolves_to_the_same_class_as_at_the_baseline`
compared the whole registry row against the baseline table, including `module`. Wave 1 moves files by
design (plan §3 renames `swm_nav`/`simple_point_maze` to `navigation`, `dmcontrol.py` to a package,
…), so the test now compares every semantic column exactly as before and compares `module` **by what
it guarantees**: if it changed, the class must still be THE SAME CLASS and still reachable under its
baseline import path. That is the property the row was protecting; nothing is weakened, and a class
that moved without leaving a shim now fails. **Workstreams B–F will all hit this test** — see §10,
request R2.

## 5. Applicable certification checks

The Step-4 frozen corpus replay for all ten cells (§6), the Step-8 support-matrix rows
(`test_foundation_registry.py`), and the SWM reference bridges driven live against
`stable_worldmodel` for four of the five navigation cells (§6, last paragraph). Gate A, the GPU
conformance suite, `check_step_matches_reference` and the tier measurements need a CUDA host and are
owed — ESCALATION 2.

## 6. The oracle — default frames, variation/state comparison

`scratch/step9/envs-classic-nav/oracle.py` loads each cell's frozen corpus
(`examples/results/step4/states/<Cell>.npz`, 64 states each, 640 states in all) and renders it
through BOTH trees in two separate interpreters — a detached `git worktree` of `21cb306` and this
branch, the same venv — then compares:

| compared | result over all ten cells |
|---|---|
| frames, SHA-256 over the raw uint8 bytes, at the registered default size | **identical** |
| frames rendered from explicit packed rows | identical |
| `describe()` (tier, decoder, `parity_interior_min`, style names, variations, reference versions) | identical |
| `style_names`, `style_dim`, the official default appearance | identical |
| `variation_space.names()` — the taxonomy, leaf for leaf | identical |
| `parity_interior_min`, `parity_all_exact_min`, `fidelity_tier`, `render_decoder` | identical |
| registry row (name, tier, decoder, state_dim, reference_id) | identical |
| `state_dim` / `state_names`, `set_state`/`get_state` round-trip on the corpus | identical, exact |
| `sample_states(16, seed 0)` | identical |

Per-cell default-frame digests (first 16 hex of the SHA-256 over 64 frames):

| cell | frames | cell | frames |
|---|---|---|---|
| CartPole | `4598212091e176d9` | TwoRoom | `ba79cc30ded03a73` |
| Pendulum | `feaf201ca40ca177` | TwoRoomTarget | `88533f44daab1f25` |
| Acrobot | `a8715ac880569d8c` | Piecewise | `2d41ae6076f430b7` |
| MountainCar | `fbe5504bcc987802` | PiecewiseZones | `e215ae82a1ef45d2` |
| MountainCarContinuous | `da0c0bc6622481ed` | SimplePointMaze | `e520afaf03cb2b8a` |

**Unexplained differences: zero** (`"unexplained_differences": {}`, `"identical": true` in
`examples/results/step9/envs-classic-nav/oracle.json`).

**The one EXPECTED difference**, declared in the script rather than discovered in the diff: the five
navigation cells' `registry_table()[i]["module"]` moved from `stable_envs.envs.swm_nav` /
`stable_envs.envs.simple_point_maze` to `stable_envs.envs.navigation`, because the coordinator's
brief and plan §3 mandate the merge. It is recorded under `expected_differences`, and the script
checks that both old import paths still resolve to the merged module. Consequences, all of them
location-only: `spec(name)["family"]` and `list_envs(family=...)` for those five now answer
`"navigation"`; `list_families()` loses `swm_nav` and `simple_point_maze` and gains `navigation`.
Two consumers outside my ownership read that field — see §10, requests R2 and R3.

The Step-9 claims, which have no baseline to compare against, are checked in the same script for all
ten cells on `cpu`: `transition` is pure (the env's buffers are unchanged after `transition`/`unroll`),
`transition` equals the stateful `step` **to the bit**, `unroll` equals repeated `transition` AND
equals four stateful steps, and `render(State)` equals `render(packed rows)`. `"functional_ok": true`.
The CUDA arm is owed (ESCALATION 2): `--python` on a CUDA host is the whole change, and classic
control's `step` is a captured CUDA graph there while `transition` is deliberately eager, which is the
one place the two could diverge — exactly the case the Wave-0 GPU run settled for PushT.

**Reference bridges, driven live** in `se8-vm-venv` against `stable_worldmodel`, after the
`_SWMPort` merge: `TwoRoom` 0.99988 of pixels exact, `TwoRoomTarget` 0.99974, `Piecewise` 0.99987,
`PiecewiseZones` 0.99994, max deviation 1/255 on all four — the declared `parity_interior_min`
0.999995 residual described in the module docstring (one ulp of `exp`), unchanged.

`SimplePointMaze` could not be driven on this host (neither venv has matplotlib), and **the
coordinator closed that gap**: they ran `scratch/spm_parity.py` on cpu in `se8-vm-venv` with
matplotlib installed, against the merged `_SWMPort` bridge, and measured

| check | result |
|---|---|
| `check_colors_apply_exactly` | **all 0** |
| parity, interior exact fraction | **1.0** — the declared bar, not relaxed |
| parity, all-pixel exact fraction | **0.99969** |
| parity, within 8/255 | **0.99974** |
| max absolute deviation | **241** (the anti-aliased agent rim and the four spine crossings; the residual the module docstring describes and quantifies) |
| `step` vs the official simulator | `exact_claim True`, **every error 0.0** |

Those numbers reproduce the module docstring's own measured table
(`interior 100.000%`, `all pixels 99.9690%`, `max 241/255`, `step exact_frac 1.000, max error 0.0`),
i.e. the merge changed nothing the reference can see. Coordinator-run evidence, recorded here rather
than reproduced: the only part of the script that did not run is its CUDA timing tail. **ESCALATION 1
CLOSED.**

## 7. Performance (plan §8)

`scratch/step9/envs-classic-nav/perf.py`, the foundation's method: ALTERNATING subprocesses between
the two trees (A B A B …), 5 repeats × 15 timed iterations, `torch.set_num_threads(1)`,
`OMP/MKL_NUM_THREADS=1`, CPU, median of the per-repeat medians →
`examples/results/step9/envs-classic-nav/performance.json`. One cell per family, because within a
family every cell shares the code the migration touched.

| path | baseline | migrated | delta |
|---|---|---|---|
| `CartPole.render(rows)`, 64 envs, 400×600 | 28.32 ms | 28.27 ms | **−0.19 %** |
| `TwoRoom.render(rows)`, 256 envs, 224×224 | 40.04 ms | 40.13 ms | **+0.24 %** |
| `CartPole.step(a)`, 64 envs | 0.0281 ms | 0.0403 ms | +43 % (**+12 µs**) |
| `TwoRoom.step(a)`, 256 envs | 0.0696 ms | 0.0974 ms | +40 % (**+28 µs**) |

**The render hot paths are unchanged** — both deltas are a quarter of a percent, i.e. inside
run-to-run spread, and render is what plan §8 is about and what the brief asked for.

The `step` deltas are above the 10 % line and are reported rather than buried. Mechanism, measured
(`env.state` 8.1 µs, `_step_result` 10.4 µs on CartPole): the whole of the increase is **building the
StepResult that contract §5 mandates** — a TensorDict per call where Step 8 handed back the packed
buffer itself. It is the same mechanism Wave 0 measured on PushT at **+1.0 %**; the ratio differs
because these are the cheapest `step`s in the repository (CartPole is four float64 expressions and a
cosine, ~28 µs for 64 environments) rather than a 10-substep contact solver, so a fixed ~15 µs of
TensorDict construction is 40 % of them and 0.1 % of PushT's. In absolute terms it is 12–28 µs per
call against a 28–40 ms render of the same states — 0.04–0.07 % of one frame — and it does not scale
with `num_envs`. One round of this was already paid back: `_Classic.state` originally built the State
twice per step (once through `Env.state`, once from `_sim`), which put CartPole at +153 %; reading
`_sim` directly removed 40 µs of it, and the per-call hidden-field guard that skipped is replaced by
a strictly stronger check at class creation. What remains is the contract, and I do not see a way to
pay less for it without making `step` return something other than a StepResult. **The absolute cost
is the TensorDict construction Wave 0 already measured on PushT** — the same `_step_result` and the
same `unpack`, +1.0 % there against a 10-substep contact solver and 28–46 % here against four float64
expressions and a cosine — so it is not a new cost, it is that same fixed cost seen against the
smallest denominator in the repository. **ACCEPTED by the coordinator with the mechanism recorded.**

## 8. Resource lifetime

`_Classic._close()` drops the captured step graph (`_step_graph = None`); `Env.close` calls it once
and is idempotent, and a closed env refuses `render`/`reset`/`step` with a message naming the call.
The navigation cells hold no external resource (their grids and masks are plain device tensors) and
need no `_close`. Correctness never depends on `__del__`. The suite's resource guard reports the same
counts as at baseline (16 EGL contexts created / 0 released over the unit run — unchanged, and a
pre-existing Step-8 observation). `SimplePointMaze.sample_states` no longer builds a `StyleTensor` by
hand, so the one place in this family that constructed a style outside `StyleTensor`'s own API is
gone.

## 9. The §17 ten-item report

1. **files moved/removed/created** — §1. Two files merged into one and kept as aliases, one edited in
   place, six created, nothing removed.
2. **API changes** — §2. Additive except `step`'s return type (the Wave-0 contract), plus one
   genuine bug fix (`frame_shape`).
3. **Step-8 IDs affected** — §3. Behaviorally none; the six ids these cells serve are byte-identical.
4. **targeted tests** — §4. 88 new, 1334 passing locally, 269 unchanged in the SWM venv.
5. **applicable certification checks** — §5. Step-4 replay done for all ten; Gate A and the GPU
   suites owed (ESCALATION 2).
6. **default-frame comparison** — §6. Identical SHA-256 over all ten corpora, 640 frames.
7. **variation/state comparison** — §6. Style layout, default appearance, taxonomy, packed layout,
   round trip and reset distribution all identical.
8. **performance comparison for hot paths** — §7: render −0.19 % / +0.24 %; `step` +12–28 µs, cause
   measured and attributed to the mandated StepResult.
9. **resource-lifetime comparison** — §8: explicit `_close` for the step graph; guard counts
   unchanged.
10. **unexplained oracle differences** — **zero**.

## 10. ESCALATIONS and requests — status after the coordinator's rulings (2026-09-17)

| # | subject | ruling | state |
|---|---|---|---|
| ESC-1 | `SimplePointMaze`'s live reference bridge unverifiable here (no matplotlib in either venv) | **CLOSED by the coordinator**, who ran `scratch/spm_parity.py` themselves | §6: colours all 0, interior 1.0, all-pixel 0.99969, within-8 0.99974, max 241, `step` exact with every error 0.0 — the module docstring's own table, reproduced |
| ESC-2 | the CUDA arm | **IN FLIGHT then DONE** (job 6464309, A40 `gpu1505`), plus my follow-up job for the two gaps it left | §11 |
| R1 | per-field dtype in `state_layout` | **DONE** on integration at `b1585fc` | merged here (no rebase); adopted — and found to be necessary but not sufficient, which is R7 |
| R2 | the registry-row `module` comparison | **ACCEPTED as the Wave-1-wide ruling** | no further change; B–F inherit it |
| R3 | `scratch/step4/universe.py::shard_of` | **DONE** on integration in the same commit (`"navigation": "swm"`) | merged here; verified present |
| R4 | `docs/api/envs.md` | **forwarded to Workstream M** | no action here |
| R5 | the `num_envs` ratchet cannot go down for this workstream | **noted** | unchanged at 41 |
| R6 | the five one-line `step` re-declarations | **deferred to Workstream K** | no action here |
| perf | `step` +12/+20 µs | **ACCEPTED with the mechanism recorded** | §7 |

### R7 (new) — the fourth element fixes the dtype, not the provenance

`b1585fc` is adopted and does exactly what it says: `state_layout`'s fourth element makes `unpack`
and `Env.state` return float64 for this family while `pack` keeps the float32 row. It does **not**
let `_Classic` delete its `state` override, and the reason is measurable rather than a matter of
taste. `Env.state` builds the DRAWN fields by widening the packed row, and the packed row is a lossy
projection of the simulator buffer (`_state = _sim[:, render_from_sim].to(float32)`), so
`float64(float32(x)) != x`. Measured on CartPole after five steps:

```
Env.state.fget(env)["x"]  vs  env.state["x"]      max |diff| = 9.56e-08
transition(env.state, a)          == step(a)      True
transition(Env.state.fget(env), a) == step(a)      False
```

So the base path silently becomes a different simulator from the second step on. **Ask:** a hook for
DRAWN fields symmetric with `_hidden_state` — e.g. `Env._live_state()` returning
`{key: authoritative tensor}` consulted by `Env.state` for any field the cell answers for — would let
this override go in Wave 3 and would close the same hole for any other cell whose packed row is a
narrowing of its simulator state (DM Control and MuJoCo are the obvious candidates). Until then the
override is contained: the layout and the hidden set are generated from one pair of class attributes,
checked at class creation, and `test_transition_still_equals_step_after_the_env_has_been_stepped`
pins the property the override exists for.

### R8 (new, and I would call it a BLOCKER) — `Env.unpack` stopped owning its output

`b1585fc` replaced `unpack`'s `r[:, off:off + w].clone()` with `r[:, off:off + w].to(dt)`. For a
field with no declared dtype, `dt` is `torch.float32` and `.to(torch.float32)` on a float32 tensor is
a **no-op that returns the same tensor**, so `unpack` now hands back a VIEW of the packed buffer.
Repro on this branch before my workaround, and it reproduces on PushT:

```python
env = snv.TwoRoom(num_envs=4, device="cpu"); env.set_state(env.sample_states(4))
out  = env.step(a)
snap = out["state"]["agent", "position"].clone()
env.step(a)
torch.equal(out["state"]["agent", "position"], snap)   # False
```

A caller who keeps the `StepResult` finds the value it holds has changed, with nothing raised and
nothing logged — the class of failure the Wave-0 review's T1 defect exists to prevent, arriving from
the other direction. `unpack`'s own docstring promises a fresh State ("`unpack(pack(s))` is the same
environment"), and every migrated cell whose State is float32 is affected; classic control happens
not to be, because it builds its State from `_sim`.

**Fix (one token, in a file I do not own):** `r[:, off:off + w].to(dt)` → `...to(dt).clone()`, or
`.clone().to(dt)`. I did **not** edit `env.py`. `_SWMPort.step` clones the packed row before
unpacking it, with a comment pointing here and saying to delete the clone once `unpack` clones again,
and `test_a_step_result_is_not_a_view_of_the_env_that_produced_it` pins the property for these ten
cells. **PushT and every future migrated cell are still exposed.**

## 11. The CUDA arm (ESCALATION 2)

### Job 6464309 — coordinator-launched, `A40` on `gpu1505`, sha `aa1608e`, venv `$SCR/venvs/se-step6`

Artifacts: `examples/results/step9/envs-classic-nav/gpu/oracle_6464309.json` and
`provenance_6464309.txt` (absolute `/oscar/scratch/sboughan` paths scrubbed to `$SCR`).

**The oracle is identical on the CUDA host** — `"identical": true`, `"unexplained_differences": {}`,
the same single expected `registry_row.module` difference, both old import paths resolving — and the
functional checks ran on `cpu` **and `cuda`** for all ten cells:

| claim | cpu | cuda |
|---|---|---|
| `transition_is_pure` | 10/10 | 10/10 |
| `transition_equals_step_bitwise` | 10/10 | **10/10** |
| `unroll_equals_repeated_transition` | 10/10 | 10/10 |
| `unroll_equals_four_stateful_steps` | 10/10 | 10/10 |
| `render_state_equals_render_rows` | 10/10 | 10/10 |
| `stateful_path_was_a_cuda_graph` | 0/10 (expected) | **5/10 — every classic-control cell** |

The bolded pair is the claim that could only be settled on a GPU and the one this escalation was
about: on `cuda`, `_Classic.step` is a **replayed CUDA graph** (`_stepgraph.capture`) while
`transition` is deliberately eager, and the two agree **to the bit** for all five cells. The five
navigation cells report no graph, which is correct — their `step` is not captured. This is the
classic-control analogue of what the Wave-0 GPU run settled for PushT.

**Two gaps that job left**, both now closed by my own follow-up (below), and neither of them a
result that was claimed and wrong — they were results that were not produced:

1. **The comparison arm ran on `device="cpu"` in both trees.** The probe hardcoded it. The fused
   render paths — `render/gfxdraw_classic` and `render/navdot` — are CUDA-only, so **not one line of
   them was executed**, let alone diffed baseline-vs-branch. Every frame digest in §6 and in that
   job's record is from the per-primitive/torch path.
2. **The `tests/gpu` subset produced no pytest summary.** All that reached the log for that section
   was an EGL teardown traceback (`OpenGL.raw.EGL._errors.EGLError` from `mujoco.egl.__del__`), so
   the subset's result is **indeterminate**, not passing. `COMPLETED` was not evidence.

Its unit-test run on the CUDA host did report: `88 passed` (the count at `aa1608e`; the file is 108
tests now).

One incidental observation, recorded because it would otherwise look like a discrepancy: on that
host `Pendulum`'s 64-frame digest is `2698c4a43016c321` where this Mac produces `feaf201ca40ca177`.
The other nine agree across hosts. Pendulum is the one cell whose frame contains a pygame
`smoothscale` blit (the torque-arrow sprite), so a different pygame build gives different bytes. It
is **identical between the two trees on each host**, which is the only thing the oracle claims.

### Job 6464378 — mine, at the post-merge head

`scratch/step9/envs-classic-nav/gpu.sbatch`: both oracle arms (`--device cpu` and, for the first
time, `--device cuda`, which is what puts the fused kernels' output under the byte comparison), the
five `tests/gpu` files with **every run writing its own results file** and its summary echoed, and
the unit tests. Results below.

