# OGBench online execution: export checkpoint

This continues from `f0716916cf285f969a1159f6904eac254a926dc6` on
`repair/render-config-fixes`. Scope is the three accepted historical registrations:
`swm/OGBMaze-v0` → `PointMazeLargeNavigate`, `swm/OGBCube-v0` → `OGBCube`, and
`swm/OGBScene-v0` → `OGBScene`. Dataset siblings are not covered by these gates.
The semantic matrix remains 39 VERIFIED / 3 INCOMPLETE; Craftax, PushT and Rocket
are outside this implementation cohort.

## Source and execution audit

All three registrations retain scalar CPU simulation. Maze delegates to
`ogbench.locomaze.maze.make_maze_env` with the accepted point/large/states default.
It swaps each lane's owned legacy NumPy RNG around source calls and restores the
caller's global RNG afterward. This requires process isolation for concurrent
owners; a shared thread pool cannot simply wrap the current loop.

Cube/Scene construct `CompatibleCube`/`CompatibleScene` over the unmodified
OGBench 1.2.1 classes. Their accepted compatibility layer owns reset/variation
RNG and registration-specific deltas; upstream stepping, control and IK remain
authoritative. A full State has 139 leaves / 9,397 bytes per Cube lane and 132
leaves / 10,485 bytes per Scene lane in the measured local dependency profile.
Maze has 14 leaves / 5,145 bytes. Actual versions and adapter hashes are recorded
in each report; cross-version State signatures are not assumed compatible.

Cube/Scene `set_state` validates the entire batch, builds replacement worlds,
and swaps them into the live adapter only after every replacement restores
exactly. It is intentionally much more expensive than `step`. Normal stepping
does not call it. This transaction must survive future process partitioning.

## Measured bottleneck and bounded change

The baseline constructed a TensorDict and transferred tensors for each lane,
then recursively stacked the results. The candidate gathers the same NumPy
values into fresh batch columns and constructs/transfers each column once.
The shared `owned_columns` helper checks consistent field shapes/dtypes and
owns its arrays; later steps, resets or exports cannot overwrite a returned
snapshot. No State field, RNG representation, reset/step algorithm, public API,
renderer or atomic restore behavior changes.

At local CPU B16, Cube adapter stepping measured 15.3 ms and separate export
8.8 ms; the diagnostic without export measured 6.9 ms. Its batch restore cost
3.5 seconds. These are separately measured component diagnostics, not additive
timing partitions or renderer FPS. Local profiles were not on an exclusive
host; use the paired comparisons for change attribution.

The completed Oscar Maze baseline on `gpu1503` / RTX A6000 confirms the device
transfer cost at B64: full adapter step took 72.4 ms with CPU outputs versus
104.6 ms with CUDA outputs; export alone measured 28.9 versus 61.5 ms. Physics
is CPU in both cases. This motivates transferring columns rather than each
lane's 14 fields. It is not an upstream-source speedup comparison.

The paired local Cube public-API comparison at B16/T8 improved from
**1,089 ± 3 to 1,849 ± 9 steps/s**, with every State retained and frames disabled.
Export throughput improved from 1,862 ± 34 to 6,674 ± 344 States/s in that timing
cell. Six samples use two reversed strategy-order blocks; reset, restore,
construction and warmup are outside the measured call. Local CPU measurements
do not close CUDA or frame acceptance.

All three local paired public comparisons completed. At B16/T8, with every
State retained and frames disabled:

| Registration | Previous steps/s ± SD | Column export steps/s ± SD | Speedup |
| --- | ---: | ---: | ---: |
| OGBMaze | 7,202 ± 72 | 10,274 ± 51 | 1.43× |
| OGBCube | 1,089 ± 3 | 1,849 ± 9 | 1.70× |
| OGBScene | 877 ± 2 | 1,297 ± 9 | 1.48× |

The [local comparison CSV](semantic_recovery_evidence/ogbench_online_20261002/local/comparison.csv)
contains all 24 paired B1/16, T1/8 step/export cells. It explicitly marks frames
unchecked. These improvements are in single-process execution.

## Correctness and pending acceptance

The new exact-export tests passed on all three local CPU registrations; their
three CUDA cases are reserved for Oscar. They compare against the frozen
pre-change row exporters from `f0716916`, including cached Gaussian RNG, eight
steps, restore/reset continuation, snapshot ownership, modified outputs, and
Cube/Scene task/geometry variations. Another 19 existing tests passed, covering
the independent Maze source corpus, global-RNG isolation, continuation,
behaviorally relevant solver/controller fields and atomic restore failures.

The public comparison exercises one/eight actions and all/final/reverse/empty/
disabled slices, retained outputs across subsequent step/reset, restored-State
equality, unseeded continuation and manipulation variations. CPU software-GL
and GPU EGL run in separate processes. Acceptance requires every registration
on both devices, exact frames, tested source hashes, and all paired timing cells.
MSE here compares the old and new export paths, not a new upstream fidelity gate.

Oscar closure update (preserved in
[`ogbench_online_closure_20261002`](semantic_recovery_evidence/ogbench_online_closure_20261002/)):

- Baseline `6963334`: Maze completed. Cube/Scene timed out after four completed
  B1/16 CPU/CUDA cells each. Array `6964316` measures only their missing B64
  cells on the same frozen source, one task per registration/device.
- `6963434` was canceled while pending after correcting the canonical Maze name.
- `6963669` failed test collection because the source archive omitted the
  reference-helper support-matrix fixture. Logs are preserved. Repaired bundle
  V3 adds only that fixture; all six tests collect from the isolated archive.
- `6964281`: Maze passed independently on CPU/software-GL and CUDA/EGL, including
  all public State/frame gates and timing cells. Its actual GPU was RTX 3090.
  Cube/Scene remain running. No sibling acceptance or process default is implied.

## Next execution stage

After export gates pass, benchmark persistent pure-process owners against the
local path at equal CPU budgets. Reuse bounded request handling and owned typed
State transport, with explicit family hooks. Cube/Scene restoration needs a
prepare/discard/commit transaction: stage replacements on all owners, discard
all staged worlds after any ordinary validation/restore failure, and commit
only after every owner is ready. Unexpected worker death must fail closed.
Maze needs isolated legacy RNG and complete nested task/observation outputs.

Then measure B64/128/256 and larger justified batches, one/eight actions,
State export/IPC/transfer, render-only FPS, selected-frame pipeline throughput
and same-device upstream baselines. Preserve active-lane stopping and the
existing slice-only public API. This export checkpoint does not complete the
three-family scaling work or the broader 35-registration goal.

Cube closure: `6964281_1` independently passed both CPU/software-GL and
CUDA/EGL gates and all paired timing cells, on an RTX A6000 host. Its
[terminal evidence](semantic_recovery_evidence/ogbench_export_cube_20261003/)
validates the frozen exporter source; the newer shared masking change is
separate. Maze and Cube exporter acceptance are complete; Scene remains live.

## Complete exporter closure and Maze process candidate (October 3)

All six independent CPU/CUDA exporter gates now pass. The consolidated
[48-row comparison and terminal evidence](semantic_recovery_evidence/ogbench_export_complete_20261003/)
preserve actual hardware: Maze RTX 3090, Cube RTX A6000, Scene RTX A5000.
Exact regression frame MSE is zero. This closes the exporter stage only.

The [Maze process candidate](semantic_recovery_evidence/ogbench_maze_process_20261003/)
has nine local CPU tests passing, including the independent source oracle,
RNG continuation, owned shared-column outputs and all-shard restore validation.
Its default remains local; RTX 3090 job 6964833 must establish public CPU/CUDA
correctness and local/P4/P8 throughput. Cube/Scene process ownership remains
unimplemented pending staged-world restore and parent metadata hooks.

## Cube/Scene process candidate (October 3)

The [private manipulation process candidate](semantic_recovery_evidence/ogbench_manipulation_process_20261003/)
now stages replacement worlds on every worker before distributed restore commits.
The parent retains one metadata validator; style and variation anchors use
explicit adapter hooks. Eight new CPU process tests and 44 existing manipulation
regressions pass. Both families survive a forced late-shard restore failure
without mutating the live batch. Defaults remain local.

RTX 3090 array 6966296 independently gates Cube and Scene on CPU/software-GL
and CUDA/EGL, then compares local/P4/P8 at B16/64, T1/8. No process performance
or GPU acceptance is claimed while pending. Full component, source and combined
pipeline comparisons remain subsequent work.
