# Test redundancy audit

Reviewed against `d424517f7766a16bcb3554469b0f42704c7d0fc5` on 2026-10-06.
The initial audit recommended consolidation without changing tests. The follow-up
below records completed consolidation separately. Craftax and archived evidence
are excluded from recommendations.

## Method and limits

Parsed all 194 tracked Python files under `tests/` (1,420 `test_*` definitions,
including nested definitions, not collected pytest cases). Excluding Craftax-named
files leaves 184 files and 1,290 definitions. Checked same-scope name
collisions and identical test bodies with docstrings removed. Compared near-matching
bodies, then inspected fixtures, globals, decorators, inputs, outputs, and thresholds
for the candidates below. Similar text alone is not proof of redundant coverage.

No same-scope duplicate function/class definitions were found. Two pairs of test
bodies are identical, but each pair reads different fixtures. No blanket deletion
is justified by the mechanical scan. This was a static whole-tree scan plus focused
semantic review, not a proof that every remaining assertion is necessary.

## Findings

### 1. Classic frozen replay is implemented three times

The two [older integration modules](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/integration/test_native_classic_gate.py)
and [remaining-classic module](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/integration/test_native_classic_remaining_gate.py)
contain five parameterized cases. They replay the same files, reset seeds, actions,
State fields, and task outputs as
[`test_frozen_source_state_and_tasks`](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/contracts/test_saved_native.py).
The shared reader additionally verifies source identity, schema, shape, and dtype.

There are two differences to preserve or resolve before retirement:

- The older checks use exact `np.array_equal`; the shared frozen-corpus check permits
  absolute float64 error up to `1e-12`, with zero relative tolerance, for documented
  cross-machine roundoff. Other dtypes remain exact. Deleting the older checks would
  remove their stricter float64 condition, not merely duplicated Python.
- Older calls use default `step` capture and therefore render each step, but never
  assert on frames. The shared saved-trajectory test explicitly compares captured
  frames with direct renders, which is more useful coverage of that behavior.

**Recommendation:** retire the two old modules after making their precision policy
explicit. Keep exact immediate restoration and live same-runtime source comparisons.
Keep the shared full-trajectory replay and separate capture/restore checks. This can
remove two files and five repeated replay cases; it must not weaken a gate silently.

Local verification: both old modules and both shared saved-corpus modules pass on
this Mac. Five CUDA cases were skipped; this audit did not run a new GPU gate.

### 2. Classic State tests duplicate implementation, not family coverage — consolidated

At the audit checkpoint, the first tests in [classic semantics](https://github.com/galilai-group/stable-envs/blob/d424517f7766a16bcb3554469b0f42704c7d0fc5/tests/unit/test_native_classic_semantics.py)
and [remaining classic semantics](https://github.com/galilai-group/stable-envs/blob/d424517f7766a16bcb3554469b0f42704c7d0fc5/tests/unit/test_native_classic_remaining_semantics.py)
perform almost the same schema, dtype, ownership, rendering, and projection-rejection
checks. Their environment parameter lists are disjoint. One selects the first State
key; the other explicitly selects a string key.

**Completed:** one parameterized test now covers all five environments in
`tests/unit/test_native_classic_semantics.py`, with the mutated field named in each
case. Distinct action, reward, physics-default, and dtype-transition tests moved
unchanged into that module. The second file was removed. Collection remains
**12 cases**, and all 12 pass both before and after consolidation. No tolerance or
assertion was removed; this does not retire the separate frozen replay tests.

### 3. DMC oracle replay shares a small reader/assertion helper — completed

[Continuation](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_dmc_continuation_oracle.py) and
[Humanoid](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_dmc_humanoid_oracle.py) reconstruct State from `state/`
arrays and compare all non-State output leaves in nearly identical loops.
[First-family](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_dmc_first_family_oracle.py) and
[Reacher](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_reacher_oracle.py) use a similar loop with an explicit
task-key list. Aligned DMC has another version.

**Completed at the October 6 follow-up:** `tests/helpers/reference/oracle.py`
shares only State extraction and exact output comparison. All five modules retain
adapter construction, action dimensions, source/version pins, hashes, output-key
sets and NaN policies. NPZ handles close on failures as well as success.

All 18 original cases pass before and after extraction. The final tests also
restore reached source checkpoints at steps 7 and 24 into a different candidate
history, compare the immediate State, and replay every remaining transition.
Full-output corpora check both key sets; the old actual-output-only loop could
miss a field that the candidate omitted. Eleven helper checks reject missing/extra
outputs, dtype/shape/value corruption and inappropriate NaN handling, and preserve
owned 64-bit RNG values. The final 29-case CPU selection passes without skips.

The two mechanically identical hash-test pairs are aligned/continuation DMC and
OGBMaze/Reacher. Each uses a different module-level `ROOT`, so neither pair is
redundant coverage. They may share an assertion helper; all fixture checks remain.

### 4. DMC adapter ownership tests repeat a common contract

[Continuation](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_dmc_continuation_adapter.py) and
[Humanoid](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_dmc_humanoid_adapter.py) both mutate qpos, warmstart,
and RNG snapshots; reject incomplete State; and reject wrong-width or NaN actions.
They test different adapters and action dimensions.

**Recommendation:** consider a small parameterized contract after the oracle cleanup.
Keep every adapter and invalid-input case. A public wrapper test alone does not
replace a direct adapter boundary test.

### 5. Fetch had an obsolete prose assertion

[`test_envs_fetch.py`](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_envs_fetch.py) required the exact docstring
phrase “NO DYNAMICS.” That no longer describes registered Fetch classes. The cleanup
replaces this one assertion with an AST check of `_FetchSemantics`/`FetchBase`
composition and retains the check that `FetchBase` supplies no stepping method.
The first documentation batch did not remove redundant tests.

### 6. Discovery tested docstring wording instead of failure behavior — replaced

`test_import_contracts.py` required one of two phrases in the discovery docstring.
It now checks the explicit private-module mapping and injects both missing-dependency
and unexpected import errors. Both must be recorded without escaping; their
diagnostics must remain distinguishable. No Craftax environment is constructed.

### 7. Visibility-collapse negative control patched a nonexistent method — repaired

`test_task_object_visibility.py` tried to read `FakeEnv.render` before injecting
the faulty renderer, but the fixture and conformance checker use `_render`. The
test now patches that hook. It first verifies zero per-row error for the clean
fixture, then requires positive error after row 0's appearance is broadcast to
every row. All seven visibility checks pass; their assertions and thresholds stay
unchanged. The file's documentation was also rewritten without historical narratives.

## Similar-looking tests to keep

- Shared [State](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/contracts/test_state.py),
  [simulation](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/contracts/test_simulation.py),
  [task](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/contracts/test_task.py), and
  [online](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/contracts/test_online.py) tests share setup but test different
  contracts. State tests restore into another history; simulation checks reached
  fields; task checks source outputs; online checks capture ordering and validity.
  Common setup may be optimized later without merging these acceptance axes.
- [Saved trajectory](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/contracts/test_saved_trajectory.py) versus live-source
  tests: disk restoration, frozen provenance, and current source agreement are
  different witnesses. A candidate agreeing with itself is insufficient.
- [Input validation](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_input_validation.py) versus
  [DataLoader integration](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_loader_idiom.py): the singleton-list
  test passes the wrapper into the API; the loader test unpacks it and checks a whole
  epoch, including the short final batch. Neither subsumes the other.
- [Slice dispatch](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_step_capture.py) versus family/process tests:
  the counter fixture exposes exact call counts, skipped lanes, and in-flight aborts;
  real families exercise their own State and execution adapters.
- [Compact PushT State tests](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_state.py) versus
  [public PushT tests](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/unit/test_pusht_public_semantics.py): these instantiate
  different implementations. The public tests add disk checkpoints, replay-backed
  continuation, and atomic failure behavior.
- GPU rendering checks and CPU State/task checks are separate coverage. Neither
  renderer self-consistency nor shared helper identity proves source pixel parity.

## Validation and next work

The targeted documentation checks pass: **336 passed, 84 skipped**. The audit's
classic replay, classic contract, and loader checks pass: **40 passed, 5 skipped**.
Executable ASTs for the five edited production files are unchanged after removing
docstrings and the reviewed generated Fetch `__doc__` value.

The broader declaration/navigation run also exposed nine existing failures, confirmed
against an isolated copy of the starting commit: two require missing Matplotlib;
two concern registry declarations/attribute ownership; two concern the context-cap
guard; three fail while extracting DMC declarations because
`_FROZEN_STAND_VISUAL_FIDELITY` is absent from the extractor's namespace. Diagnose
these separately; failing does not make a test redundant.

Next cleanup order: resolve the old classic replay precision policy and consolidate
its duplicate modules. Classic State and DMC replay helpers are combined.
For each change, compare collected parameter cases and assertion coverage, run both
old/new checks before deleting code, and preserve source pins and negative controls.
Do not reduce the Cartesian slice cases merely because their bodies are shared.
