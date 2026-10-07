# Testing strategy and infrastructure review

Review date: 2026-10-04. Baseline: `sami-dev` at `c229a921`, plus the
Rocket device-guard/private-helper candidate under acceptance. This document
records the testing approach used in recovery and proposes an incremental
cleanup. The implementation status below distinguishes migrated pieces from
remaining work. Craftax is
outside this work. Existing evidence and acceptance thresholds stay unchanged.

## Implementation checkpoint

Validation follow-up: the [fresh RTX 3090 gate](research/testing_cleanup_cuda_20261004/README.md)
passed all nine required cases without skips, closing the four CUDA and one
Matplotlib-dependent skips from the local cleanup run.

The initial cleanup now provides:

- A 38-registration coverage inventory backed by the accepted matrix, with an
  explicit four-registration Craftax exclusion and separate current-run status.
- Shared overflow-safe RGB MSE, explicitly labeled SD, typed leaf diagnostics,
  write-once evidence directories, hash verification and raw failure bundles.
- Named render-projection round-trip checks; the misleading old spelling is only
  a compatibility alias. Full State retains its independent semantic checks.
- Explicit native, FetchReach and Rocket source profiles across State, simulation,
  task and online State capture, with source pins and complete required-case
  accounting. Missing dependencies and partial selections cannot pass.
- Read-only native corpus replay without the full source package, plus a Rocket
  saved-pixel consumer checked against the preserved historical verifier on valid
  and invalid evidence. Historical drivers and accepted artifacts are unchanged.
- Git-index-scoped repository checks and contributor commands in [Testing](testing.md).

This implements the shared foundation and representative migration from steps
1–3. It does **not** replace all family acceptance drivers: their existing boundary,
contact, Variation, process and live CPU/CUDA render gates remain required. The
inventory points to those gates and preserves their exact accepted scopes.
The first step-4 cohort adds a sealed numeric State/task trajectory format and a
source-only generator for all five classic-control defaults. Disk replay checks
restore reached states and compare future outputs; migration checks retain the old
corpora and require byte-identical arrays. The native live profile now has 56
required cases, covering all five registrations. Together with FetchReach and
Rocket, **7/38 registrations** have shared live-source regression profiles.
This is migration coverage, not a change to the 38 accepted registration scopes.
The [corpus cohort report](research/testing_corpus_20261004/README.md) records
local and Oscar validation, preserved failed controls and the measured float64
roundoff policy for cross-machine frozen trajectories.

Remaining step-4/5 work includes source pixels and render configuration in saved
corpora, other family profiles, benchmark consumers, and removal of duplicate
legacy orchestration only after equivalent outcomes are demonstrated. No simulator
or renderer API changes are part of this cohort.

## Shared-profile expansion

The [38-registration migration](research/testing_shared38_20261004/README.md)
expands the four shared State, simulation, task and online-State contracts to
all non-Craftax defaults. Registration and action-domain data live in one table;
family bridges retain independent source capture. CPU and CUDA runs explicitly
check tensor placement, required-case coverage and exact live source equality.
The five DMC reference implementations are consolidated into one, with old/new
source-oracle equivalence evidence across all twelve defaults. Existing boundary,
contact, Variation, rendering and process acceptance drivers remain in place.

Prefer adding registration data and cases to these contracts. Introduce another
bridge only for a source API that needs it. Remove duplicate helpers after proving
that their old obligations are preserved; a smaller file count alone is not proof.
The remaining cleanup is to move the specialized cases into shared consumers,
not to remove their stronger assertions merely because every registration now
has the common regression checks.

## Source helper and native acceptance cleanup

The next bounded cohort moves the independent Fetch, PushT, Rocket and Cube/Scene
witnesses into `tests/helpers/reference/`. Shared profiles no longer import
research drivers or execute archived Python. Benchmark consumers use the same
helpers and include their hashes in new reports. Historical outputs are preserved.

The two native source serializers are one five-domain oracle. The two native
acceptance drivers now delegate to shared case bodies; the native profile accounts
for 16 additional specialized cases (72 total). Duplicate live integration wrappers
are removed, while their frozen-corpus assertions remain. This includes the original
seeds, batches, horizons, terminal rewards, invalid actions, precision controls,
Variation, snapshot ownership and different-history replay obligations.

The [cohort report](research/testing_source_cleanup_20261004/README.md) records
old/new equivalence, CPU results and the pending CUDA gate. Other families' specialized
acceptance orchestration, process-adapter duplication, certification patching and
installation/CI alignment remain follow-up work. No production API or runtime changes
are part of this cohort.

## Recommendation

Keep the public environment API. Consolidate test orchestration around separate
contracts for **State, simulation, tasks, rendering, and online execution**.
Use small family-specific source bridges and shared comparison, scenario,
reporting, and evidence utilities. Adding a task should usually add cases and
expected semantics; adding a renderer should usually reuse a reached-state
corpus. Neither should require copying an entire acceptance script.

The current suite has useful building blocks. The main problem is that shared
unit/conformance checks and stronger family recovery scripts are separate paths
to evidence. Their names and reports do not always describe the same claim.

## What we currently do, and should preserve

### Research-to-test migration rule

Review research by the failure it can detect, not its directory name or age.
Before archiving an active gate, identify its maintained replacement and compare
their case sets, assertions, source independence, tolerances, and rejected controls.
Keep source-only corpus generators and physical edge cases when generic contracts
do not cover them. Preserve original reports and executed code in the archive;
do not rewrite historical hashes to match a refactor.

The October 6 review identifies these reusable methods and their current homes:

| Method and research example | Maintained coverage | Still required before retiring the research gate |
| --- | --- | --- |
| Reached-State replay: DMC `freeze_*_oracle.py` generators and recovery gates | Five DMC oracle modules use `tests/helpers/reference/oracle.py`. All 12 domains restore disk checkpoints at steps 0, 7 and 24, compare immediate full State, then every remaining transition and declared output. | Keep independent source generation, manifest hashes, registration identities and family lifecycle/Variation witnesses. Frozen replay does not replace live-source tests. |
| Different-history restore and owned snapshots: `replay_feasibility.py` | Shared `test_state.py`, saved trajectories, and PushT/Rocket public semantics tests | Preserve contact and controller witnesses; equal qpos or immediate pixels cannot establish continuation. Old feasibility reports describe their historical prototype, not current acceptance. |
| Forward-only source rollout into pixels: Rocket `online_source_gate.py` | `rocket_pixels.py` supplies independently witnessed traces to the shared `pixels.py` checker. `test_rocket_source_pixels.py` preserves all four scenarios, dtypes, horizons and selectors, and adds independent State/frame selection. | Other families still need source adapters and validated edge cases. Keep the historical Rocket driver for its recorded evidence; never restore source lanes from candidate State. |
| Hidden mutation after rejection: Rocket `rejected_controls_gate.py` | `test_rocket_replay_semantics.py` checks bad actions/State against an untouched destination and its next transition; PushT has corresponding atomicity checks. | Retain the deliberately failing source witness: immediate numeric equality can hide forces that change the next reward. A successful counterexample script is not acceptance. |
| Clone/handle ownership: `clone_ownership.py` | Disk State ownership and cross-instance continuation are covered by saved/public tests. | Native clone close-order and fresh-process deserialization remain research diagnostics. Keep them until an independent native clone API is actually supported and tested. Pickle success alone proves nothing. |
| Process transport: DMC/Fetch `*_source_transport_gate.py` | Shared slice/masking tests cover dispatch; family process tests and source transport gates cover real engines. | Preserve mixed terminal lanes, RNG Gaussian caches, reset-next-step behavior and all output leaves. Serial agreement does not certify process/pipeline routes. |
| Thin geometry and state-dependent pixels: Rocket coverage probes and Cube/Scene renderer gates | Renderer component tests, frozen golden frames and family source-render gates | Preserve source comparisons at reached/contact states and appearance extremes. Candidate golden equality checks refactor stability, not agreement with ground truth. |

The DMC consolidation preserves zero numeric tolerances and each corpus's NaN
policy. Full-output corpora now require equal output-key sets, so an omitted output
cannot disappear from the comparison. Eleven helper checks cover omission, extra
fields, dtype/shape/value changes, empty output corpora, NaN policy, and owned
64-bit RNG storage. The original 18 cases pass before and after consolidation;
with helper checks and the extra restore positions, the selection passes 29 cases
on macOS with MuJoCo 3.12.0. This adds State/simulation regression coverage, not a
new pixel-parity, CUDA or registration-acceptance claim.

No research driver or evidence file was removed. Saved Rocket verification now
distinguishes a specified historical Git revision from current worktree hashes.
It reads recorded code without checking it out or executing it, retains the
report format's original thresholds, and never marks a saved report as fresh
acceptance. The historical 204-frame report and corrupted controls agree with the
unchanged legacy verifier running against its recorded code snapshot. See
[the contributor commands and adapter contract](testing.md#online-source-pixels).

The next source-pixel cohort covers all five classic-control registrations with
the same checker. The native profile grows from 72 to 128 required cases: 56 new
pixel cases span two seeds, native action dtypes, horizons 1/8, reached checkpoints
and mixed time-limit lanes. The adapter uses source-owned full-State serialization
and forward-only source rollouts, and checks rendering for State mutation.
Required-case accounting includes these tests and the pinned pygame version.
No benchmark driver is retired: this B2 correctness cohort does not replace its
large-batch, process, performance or Variation coverage. Shared source-pixel
migration covered six registrations at that checkpoint; full-State/simulation
regression profiles continue to cover all 38 non-Craftax registrations.

The parallel extension implements Navigation, DMC and nine non-PickAndPlace Fetch
source-pixel adapters. Native and Navigation pass their complete fresh CPU/CUDA
profiles. Seven DMC registrations pass each required case, while five approximate
moving-camera renderers exceed the unchanged 0.001 per-frame limit. The complete
DMC profile remains FAIL; a sibling's result cannot satisfy a failing registration.
Fetch's isolated-source OpenGL profile passes all 336 required cases on each
device. Validated shared source-pixel migration is now 25/38 registrations; five
DMC registrations fail the pixel limit and eight still need adapters. The
[Oscar record](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/fixtures/testing/parallel-source-pixels-oscar-validation.json)
keeps passing cases, failed controls and runtime/code provenance separate. No
profile failure is converted to PASS because some registrations succeeded.

The shared checker accepts nested task keys and explicit per-output NaN sentinels
for DMC. Tests reject undeclared NaNs, including NaNs in reward when another output
permits them. Fetch source traces run in a separate interpreter: upstream GLFW
teardown destroys every window in the process, so sharing it with cached candidate
contexts corrupts later images. Each source lane's GL context must be current
before rendering. This isolation changes ownership, not the source code or oracle.
PickAndPlace's render mutation, OGBench's source/device scope, and PushT remain
separate follow-ups; no historical gate or evidence file was removed.

### Acceptance axes

1. **Establish authority.** Trace the pinned registration, factory, wrappers,
   simulator/controller, task rules, RNG, lifecycle, and renderer. Record actual
   installed versions and source/asset hashes. Read source before choosing State
   fields or interpreting apparent discrepancies.
2. **Drive independent source trajectories.** Advance source and candidate from
   matching seeds/configuration with identical actions. Compare authoritative
   physical, task, RNG and lifecycle fields, rewards, success and both done flags.
   A source engine must not use the candidate serializer as its oracle.
3. **Test restoration through future behavior.** Save a reached State, disturb an
   independent destination with another history, restore, then compare immediate
   state and future transitions. Include contact, controller, terminal, reset,
   horizon and RNG witnesses; matching one picture is insufficient.
4. **Compare actual pixels.** Match configuration and reached states; compare our
   CPU/GPU output to the applicable source CPU/GPU renderer. Retain raw uint8
   pairs and per-frame errors, and verify reported metrics independently.
5. **Exercise the public online API.** Compare action sequences, selected snapshots
   and frames, validity masks and final State to forward-running source lanes.
   Check single actions and action chunks, plus serial/process/pipelined routes.
6. **Verify evidence before promotion.** Require registration-specific coverage,
   exact executed code identity, complete successful gates, and resolved artifact
   hashes. A scheduler success or a sibling's pass does not establish acceptance.

The recent Rocket investigation illustrates why these axes must stay separate:
State continuation matched while a GPU coverage rule omitted thin fin pixels.
Later, a saved exact-replay checkpoint failed across NumPy BLAS profiles despite
matching package versions. These require different diagnoses and different tests.
See the [Rocket investigation](research/rocket_lowres_coverage_checkpoint.md).

## Findings in the current code

| Existing code | Finding | Improvement |
|---|---|---|
| `stable_envs/testing/conformance.py` | Useful reusable measurements, but 1,750 lines mix pixels, legacy render rows, variation and simulator comparison. `check_state_roundtrip()` calls `_set_render_state`/`_get_render_state`, despite describing public get/set. | Name it as render-projection round-trip; give full State its own checks. Split by axis incrementally, retaining thin compatibility imports where needed. |
| `tests/helpers/reference/semantics.py` | Already compares typed full-State leaves, reward/done, reset and continuation independently of pixels. | Build on it. Add structured per-leaf diagnostics and explicit comparison policies; retain assertions in tests rather than public measurement functions. |
| `tests/unit/test_full_state_contract.py` | Meaningful aliasing/restoration tests, but the main parametrization covers seven named classes, not all accepted registrations. Family scripts supply additional coverage elsewhere. | Drive applicability from an explicit registration manifest; a required registration missing from an interpreter must block that profile. |
| `tests/unit/test_step_capture.py`, `test_step_masking.py` | Strong synthetic coverage of slices, ownership and terminal routing. Masking also compares against a frozen older dispatcher. | Keep these cheap regression controls, but label them internal controls. Independent forward source trajectories remain necessary for semantic acceptance. |
| `tests/helpers/reference/certification.py` | Keeps historical SWM identity out of production, but patches classes and Rocket reference caches in the test process. | Supply explicit reference fixtures/bridges for new tests; gradually remove reliance on process-global patches and test order. |
| `scripts/research_rocket_production/public_gate.py`, `online_source_gate.py`, and `scripts/research_replay/pusht_public_gate.py` | Strong case coverage, but standalone drivers repeat setup, comparison, hashes, raw outputs and report assembly. | Extract common runner/evidence pieces; leave physical witnesses, source capture and task rules with the family. Preserve executed historical drivers unchanged. |
| `stable_envs/testing/provenance.py` | Checks declared source package versions, with clear blocked results for mismatches. | Extend new-run provenance to assets, wrappers, schema, hardware/driver and replay math profile. Unknown authority must not establish new acceptance. |
| `tests/conftest.py`, `tests/gpu/_shared.py`, supervisor and resource guard | Empty-run protection, optional-dependency handling, crash isolation and frame assertions already exist. | Reuse them. Add required-case accounting so a partially skipped required family cannot hide behind unrelated passing tests. |
| `docs/testing.md` | Much of the page documents the older renderer-focused suite and historical machines/counts. Its State terminology can be read as a stronger simulator guarantee. | Separate current contributor instructions from historical measurements; link each check to the precise axis it certifies. |
| `tests/unit/test_repo_contracts.py` | Checks the entire working directory: existing ignored `scratch/` and untracked `docs/recovery/` make repository checks fail independently of this change. | Test packaging/navigation against a clean staged artifact; keep a separate working-directory hygiene diagnostic. Never delete unrelated work to make a test green. |

Full State legitimately includes invisible fields, such as RNG and counters.
“Every dimension changes pixels” can test declared visual degrees of freedom;
it must never reject valid full State or encourage dropping future-relevant fields.
Likewise, changing arbitrary full-State leaves can construct impossible contact or
RNG states. Use reached checkpoints and source-supported interventions for
positive tests; use deliberate corruption for negative validation tests.

## Proposed division of responsibility

Keep the existing dependency direction: public measurement primitives in
`stable_envs/testing/`; assertions and acceptance policy in `tests/`; thin CLI
drivers in `scripts/`. Do not make production environments import test fixtures,
SWM certification code, or a new testing-only API.

Suggested organization, introduced only as migrated checks need it:

```text
stable_envs/testing/
  conformance.py          # compatibility exports during migration
  metrics.py              # pixels and typed-tree differences; no pass thresholds
  corpus.py               # numeric snapshots, projections, actions, source pixels
  provenance.py           # source/candidate/runtime identity
  evidence.py             # immutable run writing and verification
  supervisor.py           # retain existing crash isolation
tests/
  contracts/
    test_state.py         # ownership, restore, fresh instance/process, rejection
    test_simulation.py    # independent source reset/transition continuation
    test_task.py          # task outcomes, boundaries, lifecycle/configuration
    test_render.py        # matched-state fidelity and renderer invariants
    test_online.py        # public sequence/capture/executor behavior
  helpers/reference/
    cases.py              # explicit per-registration coverage requirements
    <family>.py           # source factories, capture and family witnesses
```

Keep this as composition of ordinary fixtures/functions and small data records;
there is no need for a deep inheritance hierarchy or one giant configurable test.
A source bridge should provide construction, seeded reset, ordinary stepping,
independent state capture, source rendering and capability metadata. Optional
source restoration is allowed only when independently proven; otherwise replay a
source prefix. A bridge must not call candidate export/restore to manufacture the
expected answer. Shared upstream physics is expected for delegated adapters;
independence concerns the serialization, orchestration and oracle, not writing a
second physics engine.

Each registration's case record should name its source identity, public class,
constructor/task configuration, source bridge, devices/backends, action dtypes,
seeds and scenario IDs, leaf comparison policy, image thresholds, and required
gates. Family defaults reduce repetition; siblings still execute their own cases.
CPU execution, GPU execution, CUDA tensor delivery and GPU rasterization are
separate capabilities. Do not infer them from `device="cuda"`.

## Contracts and coverage

| Axis | Required witnesses | What does not prove it |
|---|---|---|
| `get_state()` | Schema/shape/dtype, independent source fields, detached ownership; snapshots survive subsequent step/reset/Variation calls | Returning a TensorDict or matching a rendered pose |
| `set_state()` | Reached-State round-trip; fresh destination with different history; future suffix; RNG/reset/lifecycle; input not retained by alias; invalid input fails atomically | Same-instance immediate round-trip alone |
| Simulation | Seeded reset, normal actions, dtype/scaling boundaries, long continuation, contacts/controllers; all applicable state leaves | Candidate versus candidate or solver distribution alone where exact replay is claimed |
| Task | Reward/success/done, time-limit ownership, exact threshold and adjacent representable values, task IDs, reset options and persistent configuration | Shared physics passing for another task |
| Render | Full State versus RenderState; frozen reached-state source pairs; CPU/GPU/backend comparisons; purity; singleton/batch/chunk/order invariance | Reset-only images, source-less kernel agreement, or a favorable mean hiding bad frames |
| Online | `[B,A]` and `[B,T,A]`; T=1 and T=8; independent state/frame slices; terminal lanes; ownership while workers run; error/close cleanup | Pure renderer FPS or comparing only internal execution routes |

Online selectors include `None`, all, final (`slice(-1, None)`), sparse,
reverse, empty/out-of-range and invalid zero-stride selections. Test states and
frames independently, not only equal selectors. Cover no terminal lanes, mixed
termination/truncation, all lanes stopping, and a subsequent public call. Reject
invalid actions/selectors before mutation. Process tests need real workers and
shared-buffer lifetime/error witnesses in addition to synthetic pipeline counters.

Test restore in a fresh process under the declared runtime profile. Cross-device,
cross-host and cross-version portability are additional claims, not automatic
consequences of numeric serialization. Record unsupported combinations explicitly.
For replay backends, include short, long and terminal histories and measure how
snapshot size/export/restore cost grows with history length.

## Reusable corpora and pixel policy

Generate versioned corpora from independent source trajectories once, then reuse
them for render development. Store registration/configuration, seeds, exact action
dtypes and bytes, checkpoint indices, numeric State and RenderState where available,
source pixels, schema/projection version and provenance. Avoid opaque live handles
and pickle-dependent engine objects. Corpus verification is read-only; regeneration
always writes a new directory and never overwrites accepted evidence.

Maintain two complementary paths: a small fixed corpus containing known failures
for quick iteration, and a broader live-source gate for acceptance. A frozen corpus
alone can be overfit and cannot establish new source/version compatibility. A
renderer-only edit should not require reconstructing every trajectory just to
recover the same pictures; final integration still checks the current public path.

For uint8 RGB, normalized per-frame MSE is
`sum((ours - source)**2) / (H * W * 3 * 255**2)`, using widened integer
subtraction/accumulation. Record mean, population SD (explicitly `ddof=0`), worst
frame, quantiles and the number over threshold. Preserve channel order, dimensions,
camera, source backend and fidelity setting; do not align, blur, crop or recolor
images to improve the score. Interior agreement and SSIM help diagnosis but do not
excuse full-frame MSE failures. The current target is **each required frame <=
0.001**, with 0.0005 reported as the tighter target; exact CPU paths retain exact
requirements. Do not change existing accepted gates during harness migration.

Where source supports both devices, report ours CPU/source CPU, ours GPU/source
GPU, and source CPU/source GPU. Also report cross-pairs and ours CPU/GPU when
useful for diagnosis. If source has only CPU rendering, name that comparison;
source pixels delivered to CUDA are not a native source GPU renderer.

Every failure should immediately retain the first divergent step/leaf or worst
frame, source/ours/difference images, minimal numeric checkpoint/action prefix,
configuration, and a replay command. Progress records should survive crashes;
report partial coverage without granting acceptance.

## Performance is another axis

Use the same correctness-passing corpus and configuration for paired benchmarks.
Report these measurements separately:

- **Renderer only:** load saved states before timing; include equivalent render
  work and declare whether transfers/output materialization are included.
- **State operations:** export, restore, projection and device transfer; retain
  raw per-operation samples and history length for replay backends.
- **Simulation only:** valid transitions per second through the real public
  adapter, with the capture/return contract declared.
- **Online:** wall time through step plus requested frame delivery, with batch,
  action horizon, workers, frame/state slices and valid counts recorded.

Warm up compilation and allocation outside steady-state timing, synchronize the
measured CUDA device at appropriate boundaries, and report startup separately.
Compare the best supported source batching/parallelism on the same device with
matched resources and outputs. Label serial source controls separately. Record
CPU model, cores/threads, GPU, utilization, memory and contention. Sweep batches
instead of choosing a favorable one silently; summarize independent timed repeats
with mean/SD and retain raw durations. Frames from one timed batch are not
independent timing samples. Correctness jobs may share resources; performance
measurements require controlled contention. No sum of isolated stage times proves
pipeline throughput or overlap.

## Runner, evidence and CI

Reuse the current supervisor/node checks for native crashes. A future thin runner
should select explicit registrations, axes and profiles, validate prerequisites,
run cases, and write a versioned result manifest. It should not embed Oscar account
names in test logic. Local CPU and Oscar CUDA launchers invoke the same checks.

Each result identifies registration/case/axis, candidate file hashes and commit,
source hashes/versions/assets, runtime/math profile, parameters, measured values,
comparison policy, raw artifact hashes, and outcome. Distinguish PASS, FAIL,
BLOCKED, SKIPPED and NOT_APPLICABLE. Require the expected case set, not a hardcoded
pytest pass count or a successful scheduler exit. Fresh certification rejects
missing required source pins, cases or artifacts. Historical evidence is referenced
with its original identity, never relabeled as freshly run.

Use four levels: dependency-light CPU contracts on PRs; pinned family source
integration jobs for affected code; required CPU/CUDA acceptance across supported
profiles before promotion/release; controlled performance jobs for regression
tracking. Keep a small mandatory GPU corpus per renderer backend and include
rendering on a noncurrent CUDA device where multiple devices are supported.
CPU-only import smoke checks must prove optional GPU helpers remain optional.
Craftax is excluded explicitly from this project's case set, not silently skipped.

## Incremental implementation plan

1. **Clarify claims and inventory coverage.** Map existing tests/evidence to the
   axes above; distinguish legacy render-row round-trip from full-State restore.
   Keep historical reports intact. Acceptance: every in-scope registration has
   explicit required cases and unsupported/blocked capabilities.
2. **Extract comparisons and evidence writing.** Consolidate typed-tree/pixel
   metrics, source identity, immutable output and failure bundles. First migrate
   one Rocket driver and one native family driver. Acceptance: both old and new
   verifiers agree on saved pass and failure artifacts, including missing data.
3. **Parameterize contracts using source bridges.** Start with one native task,
   Fetch, and replay-based Rocket/PushT to expose real differences. Add independent
   source witnesses rather than forcing all backends through a fake common engine.
   Acceptance: each registration retains its original semantic/visual requirements;
   required missing dependencies cannot produce a green certification.
4. **Separate corpus generation from execution.** Migrate render regressions and
   benchmarks to verified corpora. Acceptance: saved-state reruns reproduce metrics,
   known thin-triangle/contact failures remain caught, and live-source gates still
   cover current integration and non-reset states.
5. **Migrate remaining registrations and contributor docs.** Retire duplicated
   active launch logic only after parity of outcomes; archive historical scripts.
   Acceptance: a new task supplies configuration/edge cases, a new renderer reuses
   corpus/contracts, and a new simulator adds one source bridge plus family
   witnesses. No user-facing State/step API redesign is needed.

Prioritize steps 1–3 before building a general framework. Keep thresholds reviewed
separately from refactors. This proposal changes how we organize and expose proof;
it does not itself certify any environment or promise an online speedup.

## Validation of this review

The tracked-tree repository/navigation checks passed **10/10** in an isolated
snapshot including this page. Running those checks directly in the working tree
also encounters the unrelated ignored/untracked files described above. No files
were removed to obtain the clean-tree result.

The strict docs build reached four existing archive-link warnings: links from
`research/navigation_online_profile.md`, `research/pusht_saved_rendering.md`,
`research/semantic_recovery_evidence/fetch_hybrid_20261001/README.md`, and a missing
trace archive linked from
`research/semantic_recovery_evidence/renderer_profiles_20260928/README.md`.
It therefore did not pass globally; the new review introduced no reported link
warning. These historical evidence files were left unchanged. This is a code and
documentation audit, not a new full-registry conformance or performance run.
