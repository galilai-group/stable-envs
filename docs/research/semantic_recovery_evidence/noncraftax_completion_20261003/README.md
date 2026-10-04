# Non-Craftax completion work, 2026-10-03

Starting branch: `sami-dev`, commit
`f88bd4546878797504463dbd32bce736fd2979e4`. This campaign does not promote any
registration: **39/42 VERIFIED**, or **35/38 excluding Craftax**. Stale blocker
descriptions in the three incomplete rows are updated to the active replay work. Craftax
is excluded from further work. No PushT or Rocket promotion is implied by a
private implementation test, a queued job, or a scheduler completion code.

## PushT: private source-aligned task and replay adapter

`pusht_task_sources.json` records the pinned source files and their hashes.
The private task has no stable-worldmodel runtime import and uses official
Pymunk 7.2.0. The existing public PushT registration is unchanged. The new
candidate is deliberately unregistered until its public State/render/action
integration passes independent gates for both registrations.

Completed prerequisites:

| Evidence | Scope |
| --- | --- |
| `pusht_replay/result.json` | 72 independent seeded replay cases; histories through 10,240 actions; 2,304 future transitions and 2,376 frame pairs |
| `pusht_reset_anchors/result.json` | 24 typed reset recipes, including `reset(None)`, all-Variation resets and explicit start/goal; 2,400 future transitions |
| `pusht_task/result.json` | Migrated task passes all 96 replay/reset witnesses and all 256 discrete action centers |
| `pusht_adapter/result.json` | 72 cases, 61,200 future transitions and 15,912 frame pairs; numeric disk-State restoration, separate continuous/discrete cases, B1/B16, seeds 0/17/42, prefixes 0/1/100/300/1,025/10,240 |
| `pusht_adapter_variation/result.json` | All 8 supported agent shapes × 7 supported block shapes, independently for both action spaces; persistent initial values and action markers across reset |

Read each report's `status`; partial reports are not passes. Raw NPZ files are
numeric and load with `allow_pickle=False`. `verify_pusht_candidate.py` checks
the completed cohorts, artifact hashes, exact image/observation pairs and the
finite padded State representation. The task migration reuses the first two
cohorts' witnesses, so their counts must not be added as independent coverage.
The shape gate asserts task equality live but records only source task columns;
the main batch gate retains both task columns for independent comparison.

The private State contains a resolved reset recipe, typed Variation values and
initial values, owned PCG64 states, typed action history, and current diagnostic
fields. Kinematic infinite mass/inertia are represented by a finite zero plus
an explicit boolean flag. No pointer, engine handle or serialized Python object
appears in State. Snapshot leaves own their storage.

`history.actions` is `[B,T,2]` float64 with original action dtype codes;
`history.lengths` is `[B]` and `history.valid` is `[B,T]`. `T` is the longest
current finite prefix, not an imposed TimeLimit. Shorter lanes have canonical
zero padding. Discrete actions preserve their two polar indices and source
dequantization. Continuous actions preserve float32/float64 arithmetic and are
validated against `[-1,1]` without clipping. Invalid whole-batch actions and
invalid States leave every destination lane unchanged. Restoration validates,
reconstructs all destination engines, then swaps the complete batch.

Costs are explicit: action recording is amortized O(1); State export and
restoration are O(T). Restoration uses ordinary authoritative steps to recover
hidden contact caches. Per-case `restore_seconds` is diagnostic, not a formal
throughput measurement. Full State includes the cached goal image as a numeric
diagnostic, so its constant storage cost also depends on resolution.

`pusht_replay_cost/cost.csv` reports seven measured samples after a warmup for
get/set State at B1/B16 and histories 0/100/300/1,025/10,240, with median, p95,
mean, sample SD, operations/s and numeric/serialized bytes. These are **local
Apple M5 Pro CPU measurements**, not Oscar online speedups. For continuous
PushT at 10,240 actions, median get/set is 3.46 ms / 821 ms at B1 and
33.6 ms / 13,233 ms at B16. Discrete B16 is 46.6 ms / 13,926 ms. Ordinary
stepping does not call restoration. Process parallelization and public
rendered branching still require separate work and measurement.

The instrumented profile in `pusht_replay_profile/` attributes most cumulative
restore time to the source Python controller/substep path (26.9 of 33.1 seconds
under cProfile), with validation secondary (4.1 seconds). These instrumented
times must not replace the uninstrumented timing CSV. No approximate dynamics
or unchecked restoration shortcut was introduced.

The narrow dynamic-history collation hook is implemented and has passed 25
local PushT tests using real replay States through the shared step executor.
Together with fixed-shape capture/masking checks, the run passed 243 tests;
101 unavailable-device/dependency cases were skipped locally and are not
counted as CUDA passes. Oscar job `6992506` covers CPU/CUDA State storage and
the shared fixed-shape CUDA tests. The public PushT registration is unchanged.
See `pusht_history/local.xml` and its frozen candidate manifest.

`pusht_render_projection_v3/` proves a source-derived immediate render recipe:
matching reset geometry, current body poses, action marker, and numeric contact
segments reproduce 3,776 frames exactly across 118 cases, independently for
continuous/discrete actions and every supported shape pairing. The render-only
destination never steps or replays after reset. Pymunk's public debug-draw
callback exposes the segments without private ABI access. Reindexing shapes
after installing body poses is also required. The stale reset contact cache
produces different pixels in 66 cases (maximum normalized MSE 0.000164525).
This is source-only feasibility evidence; the small immediate-output cache and
production CPU/GPU projection still need implementation and acceptance.
The first probe's wrong import is preserved; v2 passed, and v3 repeats the
probe with complete source-file/version provenance. These repeated cohorts
must not be added as independent coverage.

Remaining before public PushT acceptance: wire the history collation into the
production source-aligned public class; finish State to
RenderState projection and CPU/CUDA renderer gates across source Variations;
exercise public single-action/chunk/terminal/reset paths; measure replay costs
and online source comparisons. No new public `step_sequence` or `unroll` API
is proposed. Existing fixed-shape families must retain their behavior.

## Rocket: current public gates

`rocket/` freezes the submitted candidate and launch scripts. Public semantic
step calls explicitly use `states=None, frames=None`; rendering checks are
separate. The public gate now exercises CPU TinyRenderer as well as CUDA and
requires every measured normalized RGB MSE to be at most 0.001, in addition to
the original frozen visual thresholds. Source/production hashes are recorded.

The uploaded `public_gate.py` is preserved as `rocket/public_gate_frozen.py`.
The working script was subsequently formatted; its AST was checked identical.
Do not treat that formatting-only byte difference as a fresh executed gate.

CPU job `6990800` also ran the persistent-Variation, independent-source suffix,
and Rocket unit gates after its public gate. GPU task `6990749_1` completed
with exit 0 on gpu2111. Array task `6990749_0` was canceled while pending and
replaced by the CPU-only job; it is not a failed correctness result. The GPU
task was moved from `3090-gcondo` to `gpu`/`norm-gpu`, retaining RTX 3090, four
CPUs and the exact candidate. This avoids duplicating its computation.

The current CPU public gate passed all 204 checkpoints. Raw frames were fetched
and independently verified: normalized MSE is exactly zero at every checkpoint,
and all 41 frozen visual groups pass. See `rocket/cpu-render-verification.json`.
This does not replace the companion Variation/suffix/unit gates or CUDA gate.

The companion CPU reports also passed: 90 persistent-Variation cases and 108
independent-source suffix cases. Job `6990800` subsequently exited 4 because
pytest could not collect without `tests/fixtures/regression/support_matrix.json`,
which the uploaded bundle had omitted. This failure is preserved in
`rocket/cpu-6990800/unit.log`. Retry `6992056` extracted the original candidate
unchanged, supplied that hash-checked fixture and passed all 18 tests with no
skips, exiting 0. `rocket/cpu-companion-verification.json` records this closure.
The original failed job is not relabeled as successful.

The GPU raw frames were independently recomputed in
`rocket/gpu-frame-verification.json`: 204 checkpoints, mean normalized MSE
0.0001606725 ± 0.0001436219 (one sample SD), maximum 0.0004282208. All 41
original frozen visual groups pass. CPU MSE remains exactly zero. The verifier
allows only float64 reduction roundoff when comparing a recomputed metric to
the producer's metric (the observed maximum difference was 2.71e-19); neither
the 0.001 per-frame limit nor the frozen renderer policy was changed.

Formal descriptive replay costs are running in array `6992404` (B1/B16,
CPU/CUDA rendering, RTX 3090 node profile). The previous performance harness
implicitly rendered during calls labeled unrendered; it now explicitly passes
`states=None, frames=None` and separately requests each render cadence.
It reports raw samples, mean/sample SD, median/p95, bytes and construction
costs. These repeated single-action branch calls are not an action-chunk
throughput benchmark. The frozen correctness candidate is unchanged.

Separate online source gates `6992588` (CPU) and `6992589` (CUDA) cover B2,
T1/T8, float32/64, flight/contact/success/horizon cases and four State/frame
selectors at 96 pixels. Source lanes are never restored. These checks are
required separately from the completed renderer batch/chunk comparison.
No Rocket promotion is recorded while this closure work is pending.

CPU online job `6992588` completed with exit 0: all 64 cases passed, with
224 pixel-exact frame pairs, mixed per-lane terminal boundaries, exact
State/task/lifecycle outputs and owned selected snapshots. The independent
verifier checks the full case matrix, raw frames, source/production hashes and
scheduler terminal record; see `rocket/online/cpu-verification.json`.
CUDA job `6992589` remains a separate required gate. Remaining GPU cost tasks
wait for the short correctness jobs via scheduler dependencies.

`rocket/performance/partial-cpu-b1.json` is explicitly a **RUNNING, incomplete**
snapshot, not an accepted performance table. Its six completed scenarios show
B1 restore increasing from a 163 ms median at reset to 7.61 s at the maximum
history; get-State medians are about 3.9 ms. A default 480-pixel State occupies
768,719 bytes, including its 691,200-byte named goal-image cache. Ordinary
stepping does not replay history. Complete B1/B16 and GPU reports remain pending.

## OGBench: source worker failure and inspection repair

`ogbench_prior/` preserves terminal evidence from the prior campaign:
Cube/Scene jobs `6976161_1/2` timed out while a worker exception containing a
ctypes pointer could not cross Gym's error queue. The diagnostic wrapper now
transports a plain RuntimeError containing the original traceback.

`ogbench_diagnostic/terminal/` preserves the resulting failures. The source
render leaves `GL_INVALID_VALUE`; Mesa tracing in job `6991422_1` identifies
`glLight`. The pinned `ogbench/manipspace/descriptions/floor_wall.xml` contains
`cutoff="100"`. This is the same upstream spotlight issue already documented
by `tests/unit/test_mjwarp_spot_semantics.py`. The GL error surfaced at our next
PyOpenGL call (`glGetString`), even though that call returned `Mesa`.

The repair in `ogbench_gl_identity/` records the known pending error before
querying vendor/renderer. It accepts that provenance only for Cube/Scene's
specific out-of-range cutoff; unexpected error codes and query failures still
fail. It does not alter source lights, frames, physics or error-checking policy.
The source/public frame threshold remains 0.001, and P1/P2 source transport
still requires exact pixels. Mesa tracing uses the documented
[`MESA_DEBUG` option](https://docs.mesa3d.org/envvars.html).

CPU array `6991509` passed both families: 30 cases and 192 compared frames per
family. Cube normalized MSE is 0.0000157465 ± 0.0000189204 (one sample SD),
maximum 0.0000911108; Scene is exactly zero in this bounded cohort. All recorded
source workers exited cleanly. The logs retain nonfatal MuJoCo GLFW destructor
warnings at interpreter shutdown. See `cpu-verification.json` and `cpu-mse.csv`.

Paired CPU/CUDA array `6991508` was canceled before starting and replaced by
`6991787` with four CPUs and 32 GB, sufficient for this B2 correctness gate and
compatible with the general GPU queue's CPU quota alongside Rocket. The first
attempt to reduce the pending job's CPU request left inconsistent Slurm TRES
metadata, so it was replaced rather than used. Source/candidate files and gates
are unchanged. The new array uses `gpu`/`norm-gpu`, RTX 3090, one concurrent task.
No GPU source comparison or speedup is accepted from the CPU-only reports.

Cube task `6991787_1` subsequently passed both device gates and exited 0.
GPU mean normalized MSE is 0.00000951894 ± 0.0000150610, maximum 0.0000953622,
over 192 compared frames. Source P1/P2 pixels and task/state/ownership/lifecycle
checks are exact. This uses the explicit OpenGL backend at B2/T1,T8; it does not
certify production P8 throughput or the default accelerated renderer. See
`cube-paired-verification.json`. Scene task `6991787_2` also completed with
exit 0; both CPU and GPU MSE are exactly zero across 192 frame pairs per device.
`ogbench_gl_identity/paired-verification.json` and `paired-mse.csv` validate
both families, candidate/source hashes, terminal status and worker closure.

## Remaining DMC source comparisons

`dmc_remaining/` freezes array `6990477` for reacher, cheetah, hopper, walker,
ball_in_cup, finger, manipulator, quadruped and humanoid. It uses RTX 3090 on
`3090-gcondo`, B64, T1/T8 and separate software-GL/NVIDIA-GL source comparisons.
Quadruped keeps its existing local execution route; this campaign does not
silently promote its process route. The held unrelated saturation array
`6794187_[30-41]` was left untouched.

## Reproduce local checks

Use a Python environment containing the versions recorded by the reports and
official Pymunk 7.2.0, with `SDL_VIDEODRIVER=dummy` for headless Pygame.

```sh
python -m pytest tests/unit/test_pusht_replay.py tests/unit/test_ogbench_source_frames.py tests/unit/test_dmc_source_summary.py -q
python scripts/research_replay/verify_pusht_candidate.py --evidence docs/research/semantic_recovery_evidence/noncraftax_completion_20261003 --output /tmp/pusht-verification-new.json
```

The verifier refuses to overwrite its output. Gate scripts likewise require a
fresh output directory; preserve these witnesses and historical accepted reports.
