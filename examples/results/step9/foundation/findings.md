# Step 9 — Wave 0 (foundation): canonical skeleton + PushT migrated end to end

**Workstream:** `step9/foundation` · **Branch HEAD:** see below · **Baseline:** Step-8
`21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`, integration parent `502b533`
**Host:** Apple M5 Mac (`cis1b1zs2j3`), CPU only — no CUDA device in this ticket.
**Interpreters:** `/Users/samib/se6-venv` (torch 2.14 CPU, + `tensordict` 0.14.2, `loguru` 0.7.3
installed for this work) and `/Users/samib/se8-vm-venv` (SWM + DMC, + `tensordict` 0.14.2).

**Result: PATCH.** The Wave-0 skeleton exists, PushT runs on it end to end, the frozen Step-4 PushT
corpus renders to **byte-identical frames** through the migrated tree and through a worktree of the
baseline, and the local unit suite is green with 116 new tests.

---

## 1. Files moved / created / removed

Moved (with `git mv`, so history follows):

| before | after |
|---|---|
| `stable_envs/core/env.py` | `stable_envs/env.py` (+ the registry half split into `registry.py`) |
| `stable_envs/core/style.py` | `stable_envs/_style.py` |
| `stable_envs/compat/worldmodel.py` | `stable_envs/_interop/worldmodel.py` |
| `stable_envs/compat/variation_maps/*` | `stable_envs/_interop/variation_maps/*` |

Created:

- `stable_envs/env.py` — `Env` (the former `StableEnv`, same class object), `State` (a TensorDict
  alias), `pack`/`unpack`, the four specs, `reset`/`step`/`transition`/`unroll`, `close` and the
  context manager, plus every piece of Step-8 machinery the certification harness relies on
  (`check_states`, `render_states_arg`, `require_contiguous_out`, the fidelity/decoder declarations,
  `describe`, `__init_subclass__` render wrapping, `StyleTensor` use). Nothing was duplicated: the
  machinery moved under the new `Env`.
- `stable_envs/dynamics.py` — the `Dynamics` protocol, `has_dynamics`, `no_dynamics` (the precise
  renderer-only refusal), `unroll_from_transition`.
- `stable_envs/specs.py` — `Spec` + `CompositeSpec`, `sample`/`validate`/`zeros`/`to`. No torchrl.
- `stable_envs/registry.py` — `make`/`spec`/`list_envs`/`list_families`, `register` and the
  declaration checks, `registry_table`/`format_registry`, the 42 SWM-id alias table, auto-discovery
  (`discover`, `import_errors`).
- `stable_envs/_config.py` — `snv.set`/`snv.get_config`, the loguru logger, one-shot deprecations.
- `stable_envs/__about__.py` — `__version__` 0.1.0 and the packaging metadata.
- `stable_envs/interop.py` — the public face of `_interop`: `VariationMap`, `VARIATION_MAPS`,
  `style_from_variation`, `variation_from_style`, `check_mapping`, `swm_variation_space`,
  `swap_renderer`, `StableEnvGym`, `BatchedEnvPool`, `attach_batched_pool`, `register_gym_ids`,
  `pusht_state`, `style_from_swm_env`, `variation_map`, `INFO_KEYS`.
- `stable_envs/_interop/__init__.py`, `stable_envs/compat/__init__.py` (shim), rewritten
  `stable_envs/core/__init__.py`, `stable_envs/core/env.py`, `stable_envs/core/style.py` (shims), rewritten `stable_envs/__init__.py`.
- Tests: `tests/unit/test_foundation_specs.py`, `..._state.py`, `..._pusht.py`,
  `..._api.py`, `..._registry.py`, `..._imports.py`.
- Scripts: `scratch/step9/foundation/pusht_oracle.py`, `scratch/step9/foundation/pusht_perf.py`,
  `scratch/step9/foundation/demo_wave0.py`.
- Artifacts under `examples/results/step9/foundation/`: `pusht_oracle.json`,
  `registry_baseline.json`, `performance.json`, `findings.md`.

Removed: **nothing**. `stable_envs.core.env`, `stable_envs.core.style`,
`stable_envs.compat.worldmodel` and `stable_envs.compat.variation_maps.*` all still import and yield
the SAME objects (the `compat` submodules are `sys.modules` aliases, not copies, so a monkeypatch or a
mutation through either path is seen by both). Wave 3 removes them, never before the oracle is green.

## 2. API changes

New (contract §§1–7):

```python
import stable_envs as snv
env = snv.PushT(num_envs=64, device="cuda", size=None, supersample=1, backend="auto", seed=0)
state = env.reset(seed=0)            # State = TensorDict, batch [num_envs]
out   = env.step(actions)            # StepResult TensorDict; PushT carries only "state"
nxt   = env.transition(state, a)     # pure in `state`
traj  = env.unroll(state, actions)   # (T, num_envs, ...) actions, open loop -> (T, num_envs) states
px    = env.render(state)            # State or packed rows
env.close();  with snv.PushT(...) as env: ...
env.state_spec / action_spec / observation_spec / variation_spec
env.pack(state) -> (n, state_dim);  env.unpack(rows) -> State;  env.state_names unchanged
snv.make / snv.spec / snv.list_envs / snv.set / snv.get_config / snv.import_errors / snv.__version__
```

Kept working, warning once each (one loguru WARNING per alias per process): `snv.StableEnv`,
`snv.REGISTRY`, `snv.register`, and `batch=` wherever `num_envs=` is now meant (including
`snv.make(name, batch=...)`). `num_envs` sits in the same POSITIONAL slot `batch` had, so positional
call sites are unaffected.

Changed behavior, deliberately and only here: **`step` now returns a StepResult TensorDict** instead
of the packed `(B, state_dim)` buffer, for cells migrated to the new API (today: PushT). Every
in-package caller was updated to read `env.get_state()` instead, which is the packed row on both the
old and the new path: `stable_envs/rollout.py`, `stable_envs/_interop/worldmodel.py` (`StableEnvGym`,
`BatchedEnvPool`), `stable_envs/testing/conformance.py`, and `tests/gpu/test_reproducibility.py`.
`hasattr(env, "step")` as "does this cell have a simulator" became `dynamics.has_dynamics(env)`,
because every `Env` now defines `step` in order to raise the precise refusal instead of an
`AttributeError` from inside a loop; duck-typed stand-ins keep the old answer.

PushT specifics: the State is `{"agent": {"position", "velocity"}, "block": {"position", "angle"}}`;
the agent velocity that lived in `_agent_v` is public state, which is what makes `transition` pure.
`step`'s docstring states that PushT has no reward and no termination in stable-envs and that the keys
are omitted rather than fabricated. `close()` drops the captured CUDA graph and is idempotent.

Unchanged, and asserted: `state_dim`, `state_names`, the packed row and its column order,
`sample_states`, `set_state`, `get_state`, `render(rows)`, the whole style API, `scene`, the
reference-bridge staticmethods, `describe()`, `variation_space`, the fidelity declarations.

**Wave-1 scope (disclosed after the architecture review, defect D4): 41 of the 81 registered classes
still reject `num_envs=`.** Their own `__init__` declares `batch=` and forwards positionally to
`Env.__init__`, so they accept the Step-8 spelling, never reach the deprecation branch, and raise
`TypeError: got an unexpected keyword argument 'num_envs'` for the Step-9 one. Only PushT was in
Wave-0 scope; migrating the rest is family-agent work:

| family | classes | names |
|---|---:|---|
| `highway` | 6 | Highway, Intersection, Merge, Parking, Racetrack, Roundabout |
| `maze2d` | 4 | PointMazeLarge, PointMazeMedium, PointMazeOpen, PointMazeUMaze |
| `mujoco_locomotion` | 4 | MujocoInvertedDoublePendulum, MujocoInvertedPendulum, MujocoPusher, MujocoReacher |
| `ogb_manip` | 2 | OGBCube, OGBScene |
| `ogbench` | 25 | Antmaze{Giant,Large,Medium,Teleport}{Navigate,Stitch,Explore}, Humanoidmaze{Giant,Large,Medium}{Navigate,Stitch}, Pointmaze{Giant,Large,Medium,Teleport}{Navigate,Stitch} |

(Counted on this interpreter, where 81 classes register. The other 40 inherit `Env.__init__` and take
both spellings today.) `tests/unit/test_foundation_api.py::test_the_number_of_classes_that_still_reject_num_envs_only_goes_down`
pins the number at 41 from the class SIGNATURES — no construction, so it is enforced on any machine —
and fails if it ever grows, so a new env cannot be added on the old constructor.

## 3. Step-8 IDs affected

**Behaviorally: none.** The 42 SWM ids are unchanged; the migration touches one cell, `PushT`, which
serves `swm/PushT-v1` and `swm/PushT-Discrete-v1`, and its frames, digests, style layout, registry row
and support-matrix row are byte-identical (§5). The 42 ids gained an *alias table* in
`registry.py` (`SWM_IDS`), copied verbatim from the accepted Step-8 support matrix and asserted
against it row for row; the two symbolic ids (`swm/CraftaxClassicSymbolic-v1`,
`swm/CraftaxSymbolic-v1`) resolve to a refusal that says they are symbolic rather than "unknown env",
and the multi-cell ids (`swm/CartpoleDMControl-v0`, `swm/OGBMaze-v0`, `swm/Piecewise-v0`,
`swm/TwoRoom-v1`) refuse to pick a cell for the caller.

## 4. Targeted tests

New, all marked `unit`, 121 tests (116 at review time, 5 added for the review changes):

| file | what it pins |
|---|---|
| `test_foundation_specs.py` | Spec/CompositeSpec: shape/dtype/bounds/names, seeded sampling, validation messages, the refusal to sample an unbounded spec |
| `test_foundation_state.py` | State keys, the packed layout is the Step-8 one, `pack(unpack(rows))` identity, zero-filled hidden field, `render(State) == render(rows) == render()`, reset determinism by seed, the four specs, `num_envs`/`len` |
| `test_foundation_pusht.py` | stateful `step` returns only `state`; `transition` pure and **bit-equal** to `step`; the hidden velocity changes the answer; `unroll` = repeated `transition`; the renderer-only refusal; the `StepResult` helper; **a cell with hidden state and no `_hidden_state()` hook fails loudly instead of reporting zeros** |
| `test_foundation_api.py` | the public namespace (incl. `StepResult`), `__dir__` advertising only the documented names, the three deprecated names + `batch=` each warning exactly once, config keys and typos, `default_device`, `close` idempotence, a closed env's refusal, the context manager, the 41-class `num_envs=` holdout pin |
| `test_foundation_registry.py` | the 42 SWM ids against `examples/results/step8/support_matrix.json`; every canonical class name against the baseline registry table; `list_envs`/`spec` without constructing; discovery records its failures |
| `test_foundation_imports.py` | the import graph, parsed: core modules import no backend/env/interop at module scope; `render/*` never imports `envs/*` or SWM; only interop reaches for `stable_worldmodel`; no core cycle; `import stable_envs` needs no optional dependency |

Exact outputs (both commands run from the worktree root):

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
1246 passed, 195 skipped, 112 deselected, 26619 warnings in 85.07s (0:01:25)
```

Baseline for that command on this machine, measured before any edit: `1125 passed, 195 skipped, 112
deselected` (the brief's "1130" is 5 higher; 1125 is what `21cb306`/`502b533` produces in this venv).
1125 + 121 new = 1246, i.e. **no pre-existing test changed its result**. (The review was run against
1241; the five extra tests are the review round's.)

```
MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_variation_maps_dmc_control.py tests/unit/test_variation_maps_newcells.py \
  tests/unit/test_variation_maps_fetch.py tests/unit/test_step8_gate_record.py tests/swm
269 passed, 58 skipped, 4088 warnings in 26.18s
```

The identical command in a worktree of the baseline `21cb306` gives `269 passed, 58 skipped in 26.93s`.

One pre-existing test was EDITED, and only for the file it reads:
`tests/unit/test_declarations.py::test_no_env_shadows_an_attribute_the_base_class_owns` parses the
base class's source to learn which `self.<attr>` it owns; the path became `stable_envs/env.py` and the
root of the inheritance walk became `{"Env", "StableEnv"}`. The assertion is unchanged.

## 5. Certification checks, default frames, variation/state comparison — the oracle

`scratch/step9/foundation/pusht_oracle.py` renders the frozen Step-4 corpus
(`examples/results/step4/states/PushT.npz`, 64 states) through BOTH trees in two separate
interpreters — a detached `git worktree` of `21cb306` and this branch, same venv — and compares:

| compared | result |
|---|---|
| frames, SHA-256 over the raw uint8 bytes (`render()` at the registered default 96 px) | identical — `39620967bc6eba0b1814b7b00543e63d56ce81380231ffc83f795fb84d58c67b` |
| frames rendered from explicit rows | identical |
| `describe()` | identical (tier `parity`, decoder `renderer`, `parity_interior_min` 1.0) |
| style layout: `style_names`, `style_dim` (29), the official default appearance | identical |
| registry row (name, tier, decoder, state_dim, reference_id, module) | identical |
| `state_dim`/`state_names`, `set_state`/`get_state` round-trip on the corpus | identical, exact |
| `sample_states(16, seed 0)` | identical |

**Unexplained oracle differences: zero** (`"unexplained_differences": {}`, `"identical": true` in
`examples/results/step9/foundation/pusht_oracle.json`).

The two claims that are new in Step 9 and have no baseline to compare against are checked in the same
script, per device: `transition` equals the stateful `step` **to the bit**, `unroll` equals repeated
`transition`, and `render(State)` equals `render(rows)`. On this host that is cpu only
(`"stateful_path_was_a_cuda_graph": false`). **The CUDA arm is now done** and was run by the
coordinator on Oscar (commit `779e7a9`, job 6463991, `gpu1503`, RTX A6000,
`examples/results/step9/foundation/gpu/pusht_oracle.json`): `"identical": true`,
`"unexplained_differences": {}`, the same frames SHA-256, and on `cuda`
`transition_equals_step_bitwise: true` WITH `stateful_path_was_a_cuda_graph: true` — i.e. the eager
functional path reproduces the replayed graph to the bit, which was the one place the two could have
diverged. PushT GPU test subset: 35 passed.

## 6. Performance (plan §8)

`scratch/step9/foundation/pusht_perf.py`, paired by ALTERNATING subprocesses between the two trees
(A B A B …), 5 repeats × 15 timed iterations, `torch.set_num_threads(1)`, PushT `num_envs=256`, 96 px,
CPU, median of the per-repeat medians → `examples/results/step9/foundation/performance.json`:

| path | baseline | migrated | delta |
|---|---|---|---|
| `render(states)` | 59.15 ms | 57.11 ms | **−3.5 %** |
| `step(action)` | 18.15 ms | 18.34 ms | **+1.0 %** |

Both are inside run-to-run spread (the baseline render repeats span 56.8–63.1 ms). `step` is the path
that could have cost something — it now builds a StepResult (an `unpack` plus a velocity clone) where
it used to hand back the buffer — and at 256 envs that is ~0.19 ms against a 10-substep solver. No
change is anywhere near the 10 % block threshold. GPU timings remain owed (ESCALATION 1).

## 7. Resource lifetime

`close()` is explicit, idempotent, and never required for correctness from `__del__`; PushT's `_close`
drops the captured CUDA graph and its pinned target buffer. A closed env refuses `render`, `reset` and
`step` with a message that names the call and says how to get a working env back. The suite's own
resource guard reports the same counts as at baseline (16 EGL contexts created / 0 released over the
unit run — unchanged, and a pre-existing Step-8 observation, not something this patch introduces).

## 8. The plan §11 demonstration, executed

```
$ MUJOCO_GL=glfw PYTHONPATH=$PWD /Users/samib/se6-venv/bin/python scratch/step9/foundation/demo_wave0.py
state            dict_keys(['position', 'velocity']) ['agent', 'block'] [64]
step             ['state'] [64]
next_state       True (== step, to the bit)
trajectory       [3, 64]
pixels           (64, 96, 96, 3) torch.uint8
specs            [('agent', 'position'), ('agent', 'velocity'), ('block', 'position'), ('block', 'angle')] (2,) (96, 96, 3)
registry         PushT | envs: 81
closed           True
```

(`device="cpu"`; the snippet in the plan says `"cuda"`, which this Mac does not have.)

## 9. The §17 ten-item report

1. **files moved/removed/created** — §1. Moved 4, created 13, removed 0.
2. **API changes** — §2. Additive except `step`'s return type for migrated cells; all Step-8 names
   still resolve, four of them warning once each.
3. **Step-8 IDs affected** — §3. Behaviorally none; `swm/PushT-v1` and `swm/PushT-Discrete-v1` are the
   ids whose cell was migrated and both are byte-identical.
4. **targeted tests** — §4. 116 new, 1241 passing locally, 269 in the SWM venv.
5. **applicable certification checks** — the Step-4 frozen corpus replay for PushT (§5) and the
   Step-8 support-matrix row (asserted in `test_foundation_registry.py`). Gate A, the GPU conformance
   suite and the tier measurements need a CUDA host: owed, ESCALATION 1.
6. **default-frame comparison** — identical SHA-256 over the 64 default frames (§5).
7. **variation/state comparison** — `style_names`, `style_dim`, the default appearance, `state_dim`,
   `state_names` and the corpus round-trip are all identical (§5).
8. **performance comparison for hot paths** — §6: render −3.5 %, step +1.0 %, CPU, paired.
9. **resource-lifetime comparison** — §7: explicit `close`, idempotent, closed env refuses; guard
   counts unchanged.
10. **unexplained oracle differences** — **zero**.

## 10. ESCALATIONS — and the architecture review's rulings

Independent review (`examples/results/step9/reviews/review_foundation.md`, reviewed SHA `aeaaf6c`):
**APPROVE-WITH-CHANGES**. Every ruling below is the reviewer's; the "now" column says what this branch
did about it.

| # | escalation | ruling | now |
|---|---|---|---|
| 1 | GPU arm of the oracle owed | ACCEPT (pending) | **DONE** — coordinator commit `779e7a9`, Oscar job 6463991 on `gpu1503` (RTX A6000): identical, and `transition == step` to the bit on `cuda` against the replayed CUDA graph; PushT GPU subset 35 passed. GPU *timings* are still owed. |
| 2 | `rollout.py` not moved out of the core | **DEFER-TO-K** | No change. The prohibition the contract makes — no `env.rollout(policy)` — is honoured and was verified by the reviewer (`hasattr(env, "rollout")` is False; no core module imports `rollout`). The move is a ~470-call-site rename; Workstream K owns it. |
| 3 | Step-8 top-level surface behind a lazy `__getattr__` | **ACCEPT, with D3 as change-now** | **D3 FIXED** — `__dir__` is built from `__all__` plus the registered classes, so the legacy names resolve but are no longer advertised. Pinned by `test_dir_advertises_the_documented_namespace_and_the_environments_only`. |
| 4 | `StyleTensor` in `stable_envs/_style.py` | **ACCEPT** | No change. |
| 5 | `interop.py` public over `_interop/` private | **ACCEPT** | No change. |
| 6 | discovery imports `envs` at call time | **ACCEPT** (tidy the self-re-entry in Wave 3) | No change; the re-entrancy note is carried to Wave 3 below. |
| 7 | two pre-existing test files edited | **ACCEPT** (diffed line by line; nothing weakened) | No change. |
| 8 | `tensordict` + `loguru` as core dependencies | **ACCEPT** | No change. |

### Review defects, and what was done

| id | what | state |
|---|---|---|
| **D1** | the `Env` docstring's rename sentence was inverted (a sed artifact): it is `StableEnv` that was renamed and that remains the alias | **fixed**, commit `4f7c1d8` |
| **D2** | `StepResult` was used as a type name in five places and defined nowhere | **fixed**, commit `5f49e15`: `StepResult = TensorDict` beside `State`, exported from the package and the core shim, and built by `Env._step_result(state, **extras)` — one place that fixes the batch size and device and that DROPS a `None` reward/done, so a cell with no such semantics gets an absent key rather than a zero |
| **D3** | `__dir__` advertised the retained backend names | **fixed**, commit `4f7c1d8` |
| **D4** | 41 of 81 classes still reject `num_envs=`, undisclosed | **disclosed** in §2 above with the per-family list, and pinned by a signature-based test that can only let the number go down |
| **T1** | `PushT.state` overrode the property to splice the hidden velocity back in — the one template hazard that would produce a SILENT wrong answer when copied (zeros reported, `transition(env.state, a) != step(a)`) | **landed**, commit `5f49e15`: `Env._hidden_state()` returns `{key: tensor}` for the offset-`None` fields and `Env.state` splices them in; a cell that declares such a field and does not answer gets a `NotImplementedError` naming the field and the hook. PushT's 12-line property became a 2-line declaration. Re-checked against the oracle (below). |
| **T3/T4** | `step` built its result in both branches; double reshape | **folded into `5f49e15`** — one `_step_result` return; the explicit `(B, 2)` reshape is KEPT and now says why (the captured graph's target buffer would silently BROADCAST a one-row target). |
| **T5** | `has_dynamics` inlined a third time in `registry.spec()` | **not taken** (reviewer marked it optional); it is two lines of duplication with a test over it. Recommend folding it into Wave 1's first registry edit. |
| **D5/D6** | stray space before a colon; the same duplication | **not taken**, cosmetic. |

### Post-change verification (this round)

- Oracle re-run on cpu and `examples/results/step9/foundation/pusht_oracle.json` regenerated: the only
  key that changed against the reviewed record is `head_sha`. `"identical": true`,
  `"unexplained_differences": {}`, frames SHA-256 `39620967…8c67b`, `ORACLE: PASS`. T1 touches the
  `state` property, which is why the oracle was re-run rather than reasoned about.
- `tests/unit`: **1246 passed, 195 skipped, 112 deselected** (1241 at review + 5 new).
- SWM venv subset: **269 passed, 58 skipped**, unchanged.

### Carried to Wave 3 (reviewer's cleanup list, no action now)

`env.py` is two things in one file (459 lines of Step-8 certification machinery before `class Env`) and
could lose them to a `_checks.py`; `registry.discover()` re-enters itself once through
`envs/__init__.py` (correct, all `sys.modules` hits, but set the flag before the walk); the retained
`reference_*` staticmethods make the PushT template ~100 lines longer to scroll past.

### Merge note (reviewer)

`git diff 7209f14 aeaaf6c` shows `examples/results/step9/dashboard.yaml` as −18 lines: this branch was
cut from `21cb306` and carries its own dashboard while `step9/integration` has since added the Wave-0
contract and the ownership table. Divergent artifact, not a regression — keep the integration branch's
dashboard at merge.

### The eight escalations in full (as filed, before the review)

1. **The GPU arm of the oracle rule is owed.** Contract §10 asks for "a local CPU test and one GPU
   check (PushT is a CPU-capable cell: run both)". This ticket is explicitly local-Mac-only and the
   host has no CUDA device, so `transition == step` is proved on cpu and the CUDA path — where `step`
   is a replayed graph and `transition` is deliberately eager — is unproved. The script already runs
   both arms (`--python` on a CUDA host is the whole change); the coordinator should run
   `scratch/step9/foundation/pusht_oracle.py` on Oscar before Wave 1, together with the GPU
   conformance/reproducibility suites and a GPU copy of §6's timings.
2. **`rollout.py` was not moved.** Contract §5 says policies/`sample` move out of the package core to
   `examples/` or `interop.py`. `Env.sample` is asserted by
   `tests/unit/test_rollout_api.py::test_an_env_level_sample_is_the_module_level_one` ("StableEnv no
   longer offers the shorthand") and the module is imported by name in tests and examples, so moving
   it is a call-site migration (Workstream K) rather than a Wave-0 edit. What the contract forbids IS
   honoured: there is no `env.rollout(policy, ...)`, and `rollout.py` is not imported by any core
   module (asserted in `test_foundation_imports.py`). Recommend Wave 1 or K owns the move.
3. **The Step-8 top-level surface is retained behind a lazy `__getattr__`.** Contract §1 says nothing
   from `render/`, `testing/` or `interop.py` should be at top level. `snv.Scene2D`, `snv.raster2d`,
   `snv.render_states`, `snv.spaces`, `snv.StyleTensor`, `snv.check_states`, `snv.registry_table`, …
   have hundreds of call sites in tests, examples and `scratch/`, so they still resolve — lazily,
   absent from `__all__` and absent from the documented namespace, and removable in Wave 3 in one
   edit of two dicts. They do not warn (only the four names the contract names do), because warning
   on `snv.spaces` would print noise from code Workstream K is about to rewrite anyway.
4. **`StyleTensor` lives in `stable_envs/_style.py`.** Plan §3's target tree names no style module.
   Inlining 255 lines of it into `env.py` would make the "keep it small" rule worse, so it is a
   private module at the package root (`stable_envs.core.style` still re-exports it). Flagging it
   because the file name is a coordinator-owned decision.
5. **`interop` is a public module over a private package.** Contract §6/plan §2 ask for one
   `interop.py`; the material is ~2,450 lines (the SWM facade plus four variation-map tables), which
   the same plan's §16 would not call a small obvious module. `stable_envs/interop.py` is the single
   public face; `stable_envs/_interop/` holds the implementation, exactly as the brief's "or a small
   private package … if it is genuinely too large" allows.
6. **Auto-discovery imports `stable_envs.envs` at CALL time.** Contract §6 keeps discovery in the
   registry and §8 forbids the registry importing `envs/`. Both hold only if the import is dynamic:
   `registry.discover()` walks the package with `importlib`/`pkgutil`, so the STATIC import graph has
   no registry→envs edge (asserted) while the behavior — a populated registry on `import stable_envs`,
   and `import_errors()` for the families whose optional dependency is missing — is unchanged.
7. **Two pre-existing test files were edited**, both because they name something the migration moved:
   `tests/unit/test_declarations.py` (the base class's file path and the inheritance-walk root) and
   `tests/gpu/test_reproducibility.py` (reads `get_state()` instead of `step`'s return). Neither
   assertion was weakened.
8. **`tensordict` and `loguru` were installed into both venvs** and added to `pyproject.toml`'s core
   dependencies, as the brief authorises (`tensordict` 0.14.2, `loguru` 0.7.3).
