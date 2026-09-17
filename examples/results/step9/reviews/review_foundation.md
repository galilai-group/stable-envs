APPROVE-WITH-CHANGES

# Step 9 Wave 0 — independent architecture review (`step9/foundation`)

**Reviewed SHA:** `aeaaf6cfac5296c33c4e4d8f1caf0fdcc14afb65` (branch `step9/foundation`)
**Delta reviewed:** `git diff 7209f14 aeaaf6c`; baseline milestone `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
(verified: `git merge-base --is-ancestor 21cb306 aeaaf6c` → true; 8 commits over baseline).
**Reviewer host/venvs:** Apple M5 Mac `cis1b1zs2j3`, CPU only. `/Users/samib/se6-venv`,
`/Users/samib/se8-vm-venv`. `MUJOCO_GL=glfw`, `-p no:cacheprovider`. Nothing was edited, committed,
stashed or pushed in the review worktree (`git status --porcelain` empty before and after); Oscar was
not touched.

**Verdict rationale in one paragraph.** Every binding item of `WAVE0_CONTRACT.md` is met and I
verified the load-bearing ones myself rather than through the author's tests: I re-ran the PushT
oracle and reproduced the committed JSON key-for-key (only `head_sha` differs), I parsed the import
graph with my own AST walk and found no cycle and no backwards arrow, and both required suites
reproduce the stated counts exactly. The architecture is genuinely good: the core is six small
modules, the spec vocabulary is one dataclass used four times, the dependency arrows point one way,
and the packed row that every frozen digest rests on did not move. The changes I am asking for are
all small, none of them is a behaviour defect, and the reason they should land *before* Wave 1 is
precisely that this branch becomes the thing 40 cells copy: a docstring that says the opposite of
what happened, a type name used in five places and defined nowhere, and a `__dir__` that advertises
the backend machinery the contract says to hide are cheap now and expensive after they are cloned.

---

## 1. The §17 ten-item report — one line each

| # | item | reviewer finding |
|---|---|---|
| 1 | files moved/removed/created | Confirmed against `git diff --stat`: 4 `git mv` renames (history follows; `compat/worldmodel.py → _interop/worldmodel.py` shows 98% similarity), 13 created, 0 removed. Accurate. |
| 2 | API changes | Confirmed. Additive except `step`'s return type for migrated cells. All four deprecated names resolve and warn; measured exactly 4 warnings over 3× repetition (warn-once holds). |
| 3 | Step-8 IDs affected | Confirmed behaviourally none. `SWM_IDS` is 42 rows, 2 symbolic; `swm/PushT-v1` and `swm/PushT-Discrete-v1` are byte-identical (§6 below). |
| 4 | targeted tests | Confirmed: 116 new, all `pytestmark = pytest.mark.unit`. Both required suites reproduced (§ Commands). |
| 5 | applicable certification checks | Step-4 frozen corpus replay + Step-8 support-matrix row, both green. GPU conformance/tiers/Gate A **pending coordinator** (Oscar). |
| 6 | default-frame comparison | **I reproduced it.** SHA-256 over the 64 default frames is `39620967bc6eba0b1814b7b00543e63d56ce81380231ffc83f795fb84d58c67b` on both the baseline worktree and this branch. |
| 7 | variation/state comparison | Confirmed identical: `style_names`, `style_dim`=29, default appearance, `state_dim`=5, `state_names`, corpus round-trip exact, `sample_states(16, seed 0)`. |
| 8 | performance | Confirmed within noise: render −3.46 %, step +1.02 %, paired alternating subprocesses. See §5. |
| 9 | resource lifetime | Confirmed: `close()` idempotent (called twice, no error), closed env refuses `render`/`reset`/`step`, context manager closes on exit, `PushT._close` drops the graph. No `__del__` dependency anywhere. |
| 10 | unexplained oracle differences | **Zero, independently reproduced.** My run: `"unexplained_differences": {}`, `"identical": true`, `ORACLE: PASS`. |

---

## 2. Contract conformance, module by module

| contract § | subject | verdict | evidence |
|---|---|---|---|
| §1 | public namespace | **PASS** (one nit) | `snv.__all__` = `['Env','State','Spec','CompositeSpec','Dynamics','make','spec','list_envs','list_families','import_errors','set','get_config','SWM_IDS','__version__']` — 14 names, no `render/`, `testing/` or `interop` member. `__version__` from `__about__.py` = 0.1.0. Nit: `__dir__` DOES list the legacy backend names (D3). |
| §1 | deprecated aliases warn once | **PASS** | Loguru sink captured exactly 4 records over 3 iterations × 4 aliases (`StableEnv`, `REGISTRY`, `register`, `batch=`). Legacy lazy names (`snv.Scene2D`, `snv.spaces`) correctly do not warn. |
| §2 | constructor / batch / device | **PASS for PushT; repo-wide gap disclosed here (D4)** | `snv.PushT(num_envs=64, device=…, size=…, supersample=…, backend=…, seed=…)`; `env.num_envs`, `env.device` is a `torch.device`, `env.size`, `len(env)==num_envs`. Short batches unchanged (`set_state`/`_live_rows` untouched). **41 of 81 registered classes still reject the keyword `num_envs=`** — Wave-1 work, not a Wave-0 defect, but undisclosed in findings §2. |
| §3 | State / pack / unpack | **PASS** | State keys `[('agent','position'),('agent','velocity'),('block','angle'),('block','position')]`, batch `[num_envs]`. Packed row byte-identical to Step 8: `rows_sha256` and `roundtrip_exact: true` agree across trees in my own oracle run. Agent velocity is public state with offset `None`; `pack` drops it, `unpack` zero-fills it. `state_names` unchanged. |
| §4 | specs | **PASS** | One frozen `Spec` dataclass + `CompositeSpec`; `sample/validate/zeros/to` all present. `import torchrl` → ImportError (not a dependency). `state_spec` is `CompositeSpec`, `action_spec`/`observation_spec` `(96,96,3)` uint8 / `variation_spec` `(29,)` are `Spec`. No per-kind hierarchy. |
| §5 | lifecycle | **PASS** | `reset(seed)` → State and sets the held state; `step(a)` → TensorDict with keys `['state']`, batch `[4]`, no fabricated reward/done; `transition`/`unroll` pure and **bit-equal** to `step` (reproduced, cpu); `close()`/`__enter__`/`__exit__` correct; `hasattr(env,"rollout")` is `False` — no `env.rollout(policy)` anywhere in the package. |
| §6 | registry | **PASS** | `make`/`spec`/`list_envs`/`list_families`; `spec("swm/PushT-v1")` answers without constructing; symbolic ids and multi-cell ids raise their specific refusals; `discover()` records import errors per family. Classes hold the semantics; registry holds names + the alias table. |
| §7 | config/logging | **PASS** | Exactly the three keys `verbose`, `default_device`, `warn_on_deprecated`; unknown key raises `KeyError`; loguru logger bound with `extra={"package":"stable_envs"}` and — correctly — **does not add or remove global loguru handlers**, which is the right call for a library. |
| §8 | dependency direction | **PASS — independently parsed, no circularity** | My own AST walk (module-scope imports only, relative imports resolved, `from . import X` resolved to `pkg.X`): `env → {_config, _style, dynamics, spaces, specs}`; `dynamics → {}`; `specs → {}`; `registry → {_config, env}`; `_config → {}`; `__about__ → {}`. **No core module imports `render/`, `envs/`, `interop`/`_interop`.** No `render/*` module imports `envs/*` or `stable_worldmodel` at any depth. `stable_worldmodel` is imported lazily only, by `_interop/*` and four env modules. **Cycle search over the whole package: zero cycles.** `Env.render` imports `render.raster2d` inside the method, which is the correct way to keep the arrow at call time. |
| §9 | markers | **PASS** | `unit`, `integration`, `certification` added to `pyproject.toml` alongside existing `gpu`, `slow`; all 6 new test files carry `pytestmark = pytest.mark.unit`. (`integration`/`certification` are declared but not yet used — expected at Wave 0.) |
| §10 | oracle rule | **PASS on CPU; GPU arm pending coordinator** | See §3 below. |

---

## 3. Oracle — re-run by the reviewer

```
cd /Users/samib/stable-envs-step9-foundation
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/foundation/pusht_oracle.py \
  --out <scratchpad>/oracle
```
```
{
 "baseline_sha": "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc",
 "head_sha": "aeaaf6cfac5296c33c4e4d8f1caf0fdcc14afb65",
 "identical": true,
 "unexplained_differences": {},
 "functional_checks": [
  {"device": "cpu", "transition_equals_step_bitwise": true,
   "unroll_equals_repeated_transition": true,
   "stateful_path_was_a_cuda_graph": false,
   "render_state_equals_render_rows": true}
 ]
}
ORACLE: PASS
```
Diffed my JSON against the committed `examples/results/step9/foundation/pusht_oracle.json`: the only
differing top-level key is `head_sha` (the committed record was written one doc-commit earlier).
Frames SHA-256 identical on both arms: `39620967…8c67b`. The script's methodology is sound — it runs
the two trees in **separate interpreters** against a detached `git worktree` of the baseline SHA with
the same venv, so any difference would be ours; it compares 14 named facts and it cleans its worktree
up (`git worktree list` returned 24 entries before and after, working tree clean).

**GPU arm: PENDING COORDINATOR.** Contract §10 also wants one CUDA check, and on CUDA `step` is a
replayed graph while `transition` is deliberately eager — the one path a functional reimplementation
could diverge on. This host has no CUDA device, the script already runs both arms unchanged, and the
coordinator is running it on Oscar in parallel. Not a BLOCK, per the review assignment; it must be
green before Wave 1 merges.

---

## 4. Behaviour preservation

**Pre-existing tests edited: exactly two files, both call-site adaptations, neither weakened.**

- `tests/unit/test_declarations.py:463` — the parsed path becomes `stable_envs/env.py`. The
  `assert {"_state","device","_live_rows"} <= owned` self-check is unchanged and still passes, which
  proves the new path is the right file.
- `tests/unit/test_declarations.py:478` — inheritance-walk root `{"StableEnv"}` → `{"Env","StableEnv"}`.
  Not a weakening: `StableEnv is Env` (the same class object, verified via `stable_envs/core/env.py:17`),
  so the transitive closure is identical; the extra name only keeps the walk working under either
  spelling.
- `tests/gpu/test_reproducibility.py:1270,1295` — `graphed.step(a)` / `eager.step(a)` then compare
  `get_state()`. `step` previously returned `self._state`, the same buffer `get_state()` returns, so
  this is the identical comparison. Both assertions (`worst` tolerance and the `== 0.0` post-`set_state`
  assertion) are character-for-character unchanged.

**Call-site adaptations in the package: confirmed adaptations only.**

- `stable_envs/testing/conformance.py:1117` — comment only; the code path already read through the
  caller-supplied `get_sim`. Zero semantic change.
- `stable_envs/rollout.py:117` — `hasattr(env,"step")` → `has_dynamics(env)`. Necessary, because every
  `Env` now defines `step` in order to raise the precise refusal; `has_dynamics` falls back to
  `hasattr` for duck-typed stand-ins, so non-`Env` objects keep the old answer.
- `stable_envs/_interop/worldmodel.py:558,565,821,828` — same two substitutions. I traced the
  `BatchedEnvPool` aliasing: `prev = get_state().clone()` then `new = get_state()` is exactly what
  `new = step(...)` used to be (the live buffer), and the subsequent `torch.where`/`set_state`/
  `get_state().cpu().numpy().copy()` sequence is untouched. Semantics preserved.

**Suites (run by the reviewer, from the worktree root):**

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
→ 1241 passed, 195 skipped, 112 deselected, 26619 warnings in 85.73s (0:01:25)      [expected 1241 ✓]

MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_variation_maps_dmc_control.py tests/unit/test_variation_maps_newcells.py \
  tests/unit/test_variation_maps_fetch.py tests/unit/test_step8_gate_record.py tests/swm
→ 269 passed, 58 skipped, 4088 warnings in 30.32s                          [expected 269 / 58 ✓]
```
1125 (author's measured baseline) + 116 new = 1241. No pre-existing test changed its result.

---

## 5. Performance

`examples/results/step9/foundation/performance.json`. Method read and **accepted**: alternating
subprocesses (A B A B …) so drift lands on both arms, 5 repeats × 15 timed iterations, median of the
per-repeat medians, `torch.set_num_threads(1)` plus `OMP_NUM_THREADS=MKL_NUM_THREADS=1`, `num_envs=256`,
96 px, CPU, baseline arm a detached worktree of `21cb306` under the same interpreter. That is a
genuinely paired design, not two sequential runs relabelled.

| path | baseline median | migrated median | delta | reviewer call |
|---|---|---|---|---|
| `render(states)` | 59.15 ms (repeats 56.83–63.12) | 57.11 ms (56.76–58.48) | **−3.46 %** | Within noise. The baseline spread alone is 11 %, so a −3.5 % point estimate carries no signal; the honest reading is "no measurable change". |
| `step(action)` | 18.15 ms (17.76–18.25) | 18.34 ms (17.92–18.49) | **+1.02 %** | Within noise; the ranges overlap almost completely. ~0.19 ms on a 10-substep solver, consistent with the `unpack` + velocity clone the StepResult now costs. |

Neither is near the plan §8 10 % block threshold, and the sign of the render number is favourable
rather than a regression that needs explaining. **CPU performance preserved.** GPU timings owed with
the Oscar arm (**pending coordinator**).

---

## 6. Template quality — is `envs/pusht.py` the right shape for 40 cells to copy?

**Broadly yes.** The Step-9 surface a family agent must add is genuinely small and well-placed: two
class-level declarations (`state_layout`, `state_bounds`, 8 lines), `transition`, a `_close`, and a
one-line change to what `step` returns. Everything else — specs, `pack`/`unpack`, `unroll`, `reset`,
`close`, the context manager, the refusal for renderer-only cells — is inherited. The `state_layout`
offset-`None` convention is the single best idea in the patch: it lets a hidden dynamics variable
become public state without moving one column of the frozen corpus layout, and it is declarative, so
a family agent states a fact rather than writes a method. The docstring work is exemplary and the
refusal messages say what is missing *and* what to do instead.

**Concrete, cheap, behaviour-preserving simplifications (recommend before Wave 1):**

- **T1 — the `PushT.state` property override is the one copy-hazard.** `stable_envs/envs/pusht.py:517-528`
  re-derives `Env.state` to splice `_agent_v` back in. The base class already knows which fields carry
  offset `None`; it does not know where the live value lives, so every cell with hidden dynamics must
  remember to write this property. Forget it, and `env.state` silently reports zeros and
  `transition(env.state, a) != step(a)` — a wrong answer, not a crash. Cheap fix: a `_hidden_state()`
  hook on `Env` returning `{key: tensor}` (default `{}`) that `Env.state` splices in. ~10 lines added
  to `env.py`, ~12 removed from `pusht.py`, and the hazard becomes a declaration.
- **T2 — `StepResult` is named in five places and defined nowhere.** `stable_envs/env.py:1019`,
  `envs/pusht.py:555`, `testing/conformance.py:1120`, `_interop/worldmodel.py:565`,
  `tests/gpu/test_reproducibility.py:1270`. Contract §5 uses the word too. Either add
  `StepResult = TensorDict` beside `State = TensorDict` and export it, or stop using the name. Leaving
  it undefined guarantees 40 agents hand-roll `TensorDict({"state": …}, batch_size=[…], device=…)`
  slightly differently. A `Env._step_result(state, reward=None, done=None)` helper would be better
  still and is ~6 lines.
- **T3 — `step` builds the same StepResult in two branches.** `envs/pusht.py:565` and `:575` are the
  identical `return TensorDict({"state": self.state}, …)`. Assign once and return once.
- **T4 — double reshape in `step`.** `envs/pusht.py:561`: `self._target(action).reshape(self.B, 2)`,
  where `_target` already did `.reshape(-1, 2)`. Harmless but it is the line every cell will copy.
- **T5 — `has_dynamics` is computed three ways.** `dynamics.has_dynamics` (the real one),
  `Env.has_dynamics` (delegates, fine), and an inlined duplicate in `registry.spec()` at
  `stable_envs/registry.py:~318` (`any(getattr(cls,m,None) is not getattr(Env,m,None) …)`). Make
  `spec()` call a class-level helper so the two cannot drift.

**Wave-3 cleanup (not now):**

- `env.py` is 1066 lines, of which the first 459 are module-level Step-8 certification machinery
  (`check_states`, `render_states_arg`, `require_contiguous_out`, `_nonfinite_report`,
  `reference_versions_of`, the tier/decoder constants) before `class Env` even starts. Contract §6 says
  "keep it small". Nothing is duplicated and nothing is coupled wrongly — it simply did not get split.
  A later `_checks.py` (or moving it under `testing/`) would leave `env.py` at ~600 lines of actual
  environment vocabulary.
- The retained certification machinery on the class (`describe`, the fidelity/decoder declarations, the
  `reference_*` staticmethods in PushT at `envs/pusht.py:587-690`) is ~100 lines of the template that a
  family agent must scroll past. It is not duplicated with the new `Env` and it is load-bearing for
  Step 8, so keeping it was right; flagging it only so Wave 3 knows the file is two things.
- `registry.discover()` re-enters itself: `stable_envs/__init__.py` calls `discover()`, which
  `importlib.import_module("stable_envs.envs")`, whose `__init__` calls `discover()` again (the
  `_DISCOVERED` flag is only set at the end of the outer call). It is correct and the second pass is all
  `sys.modules` hits, but it is a re-entrancy a reader will trip over. Set the flag before the walk, or
  drop the `discover()` call from `envs/__init__.py`.

**Hidden coupling:** I looked for it and did not find any. `Env.render` imports the rasterizer inside
the method with a comment saying why; the `compat` shim installs `sys.modules` aliases rather than
copies, so `VARIATION_MAPS` is one dict and a monkeypatch through either path is seen by both (I
confirmed the shim code does `sys.modules.setdefault`, not a re-import); `Env._se_dynamics_base = Env`
is set in `env.py` rather than in `dynamics.py`, which keeps that arrow one-way.

---

## 7. Defects (file:line)

| id | severity | where | what |
|---|---|---|---|
| **D1** | doc, fix now | `stable_envs/env.py:463` and `:468` | The `Env` class docstring says "Step 9 renamed this from ``Env``" and "``Env`` remains as a deprecated alias". Both should say `StableEnv` — a sed artifact that inverts the sentence. This is the first docstring a contributor reads and the first one 40 cells will paraphrase. |
| **D2** | API vocabulary, fix now | `stable_envs/env.py:1019` + 4 more | `StepResult` used as a type name in 5 places, defined nowhere. See T2. |
| **D3** | contract §1 nit, fix now | `stable_envs/__init__.py:~118` (`__dir__`) | `__dir__` returns `_LEGACY_MODULES` and `_LEGACY_ATTRS`, so tab-completion and `dir(snv)` advertise `snv.Scene2D`, `snv.raster2d`, `snv.testing`, `snv.render`, `snv.spaces` — exactly the backend machinery §1 and plan §5 say not to expose. Keeping them *resolvable* is right (escalation 3); listing them is not. One-line fix: drop the two legacy dicts from `__dir__`. |
| **D4** | disclosure, Wave-1 | `stable_envs/env.py:546` vs. 41 env classes | Contract §2's headline constructor holds for PushT and the 40 classes that do not override `__init__`; **41 of 81 registered classes still declare `__init__(self, batch=…)` and raise `TypeError: got an unexpected keyword argument 'num_envs'`.** Correct for Wave 0 (only PushT was in scope) but findings §2 does not say it, and those 41 also accept `batch=` *without* the deprecation warning because they pass it to `super()` positionally. Record it as explicit Wave-1 family-agent work. |
| **D5** | style, defer | `stable_envs/env.py:557` | `if num_envs not in (1, batch) :` — stray space before the colon. Also: passing `num_envs=1` explicitly together with `batch=5` silently yields 5 rather than raising. Cosmetic. |
| **D6** | duplication, defer | `stable_envs/registry.py` (`spec()`), `dynamics.py:has_dynamics` | See T5. |

No behaviour defect was found. No weakened test, threshold, assertion or check was found.

---

## 8. Rulings on the author's eight escalations

| # | escalation | ruling | reasoning |
|---|---|---|---|
| 1 | GPU arm of the oracle owed | **ACCEPT (pending coordinator)** | Correctly scoped and correctly disclosed; the script needs no change, only a CUDA host. Must be green before Wave 1 merges — on CUDA `step` is a replayed graph and `transition` is eager, which is the only place these two could disagree. |
| 2 | `rollout.py` not moved | **DEFER-TO-K** | The prohibition the contract actually makes — no `env.rollout(policy)` — is honoured and I verified it (`hasattr(env,"rollout")` is False; no core module imports `rollout`). Moving the module is a ~470-call-site rename with a test that asserts the current arrangement; that is Workstream K's shape of work, not Wave 0's. |
| 3 | Step-8 top-level surface behind lazy `__getattr__` | **ACCEPT, with D3 as CHANGE-NOW** | Keeping the names resolvable is the only way ~470 call sites survive until K, and they are absent from `__all__`, lazy, and removable in one edit of two dicts. But they must not be *advertised*: drop them from `__dir__` (one line). Not warning on them is the right call — noise from code K is about to rewrite. |
| 4 | `StyleTensor` in `stable_envs/_style.py` | **ACCEPT** | Inlining 255 lines into an already-1066-line `env.py` would make the "keep it small" rule worse, and a leading underscore says exactly what its status is. The right decision; flagging it to the coordinator was also right, since the file name is coordinator-owned. |
| 5 | `interop.py` public over `_interop/` private | **ACCEPT** | ~2,450 lines across a facade and four variation-map tables is precisely the case the brief's "or a small private package … if it is genuinely too large" anticipates. One public face, one `__all__`, nothing in the core imports it. |
| 6 | discovery imports `envs` at call time | **ACCEPT (with the re-entrancy note)** | This is the only way §6 (discovery in the registry) and §8 (registry imports nothing from `envs/`) can both hold, and I confirmed by my own parse that the static graph has no `registry → envs` edge while behaviour is unchanged. Tidy the self-re-entry (Wave-3 note above) but the design is right. |
| 7 | two pre-existing test files edited | **ACCEPT** | I diffed both line by line. Both are name/path adaptations forced by the migration; every assertion, tolerance and message is unchanged, and `{"Env","StableEnv"}` is a superset of a set whose two members are the same object. Nothing weakened. |
| 8 | `tensordict` + `loguru` as core deps | **ACCEPT** | Contract §3 and §7 mandate both by name. `tensordict>=0.6` is pure torch and pulls no RL framework; I confirmed `torchrl` is absent from the environment and from the import graph, which is the thing §4 was protecting. |

---

## 9. Required before Wave 1 (the "-WITH-CHANGES")

1. Fix **D1** (`env.py:463,468` — the inverted rename sentence).
2. Resolve **D2**: define and export `StepResult` (and preferably an `Env._step_result(...)` helper), or
   remove the name from all five docstrings.
3. Fix **D3**: drop `_LEGACY_MODULES`/`_LEGACY_ATTRS` from `stable_envs/__init__.__dir__`.
4. Record **D4** in the dashboard as explicit Wave-1 family-agent scope (`num_envs=` for the 41 classes
   that override `__init__`).
5. Land **T1** (the `_hidden_state()` hook) if the coordinator agrees it is cheap — it is the one
   template hazard I expect to produce a real, silent defect when copied. T3/T4/T5 are optional polish.
6. Green GPU oracle arm from Oscar (**pending coordinator**, not this review's responsibility).

Items 1–4 are edits to files the coordinator owns (`__init__.py`, `env.py`) plus a dashboard line; none
touches behaviour and none needs a re-run of the oracle. Item 5 does touch behaviour-adjacent code and
should be re-checked against `scratch/step9/foundation/pusht_oracle.py` if taken.

## 10. Merge note

`git diff 7209f14 aeaaf6c` shows `examples/results/step9/dashboard.yaml` as −18 lines: the branch was
cut from `21cb306` and carries its own 68-line dashboard, while `step9/integration` has since added the
Wave-0 contract and the shared-file ownership table. That is a divergent artifact file, not a
regression — expect and resolve it at merge, keeping the integration branch's dashboard.
