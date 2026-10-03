# DMC process execution candidate

This follows the column-export checkpoint `c291296c` on
`repair/render-config-fixes`. DMC public construction still selects its accepted
local adapter. The process strategy is explicitly injected only by acceptance
tests and the benchmark until registration-specific gates and resource-matched
throughput establish an appropriate default. The semantic matrix is unchanged.

## Implementation

`DMCProcessAdapter` retains persistent CPU owners of the existing accepted
dm_control implementations. It partitions lanes into bounded process shards,
keeps source lane seeds/order/dtypes, and transports full typed State and nested
task columns in shared memory. Actions/commands use pipes. A single CPU prototype
discovers task shapes and is closed before the live workers are created; there
is no duplicate parent-side batch or renderer in a worker.

The existing Fetch request lifecycle was moved, without changing its protocol,
into `ProcessRequests`. Both families use the same one-request limit, worker
failure propagation and cleanup. Family construction, payloads, validation and
configuration remain separate. No public State/RenderState or simulator contract
was redesigned, and no new public flag or sequence API was introduced.

DMC's private pipeline hook uses the existing slice-based dispatcher. It receives
owned outputs, submits the next known action, then renders the preceding State
on the caller's thread. CPU/CUDA outputs remain independent of worker storage.
Unselected intermediate States need not be exported; the final full State is
always captured. One-action and final-frame-only calls retain their normal
dependency order and do not claim temporal rendering overlap.

The accepted DMC wrappers never report public terminal lanes; their internal
reset boundaries remain part of source semantics. The process candidate rejects
partial active masks rather than inventing a lane-stop policy. Reacher density
changes preserve current State. Finger/Manipulator density changes rebuild and
reset the **whole** batch: even an unchanged shard must follow that source path
when another shard changes. A private force flag preserves this behavior across
owners. Repeating an unchanged density configuration still does nothing.

Whole-batch shape/dtype/finite validation precedes shared-buffer writes. Requests
during lookahead are refused before touching buffers; worker failures abort and
reap every owner rather than returning partial State. Aligned DMC adapters also
check their closed flag before task-only advancement.

## Current evidence

Local process/State tests passed 31 cases, with 24 CUDA cases reserved for Oscar.
The shared-code Fetch regression passed 185 local cases with four CUDA skips.
On Oscar, the initial process source passed **69 DMC CPU/CUDA tests** and
**188 Fetch regression tests**, with one optional Matplotlib skip. The final
source adds only the closed-adapter guard and its regression test; its focused
CPU/CUDA suite passed **29 tests**. Source manifests and earlier source overrides
are preserved in [the evidence directory](semantic_recovery_evidence/dmc_process_20261002/).

The public gate compares the local column exporter against two process shards,
including uneven B5 ownership, one/eight actions, float32 device actions,
reverse/empty/disabled slices, cached Gaussian RNG, mixed internal reset
boundaries, checkpoint/reset replay, owned outputs and exact frames. It confirms
that workers can finish a following transition while the parent still holds the
preceding State for rendering. CPU gates require Mesa software OpenGL; CUDA gates
require NVIDIA EGL, in separate processes. Frame MSE here is against the local
accepted library path, not a fresh upstream renderer certification.

All twelve registrations passed independently on both CPU and CUDA in final
array **`6961927`**: 24 reports, each with 12 cases, exact frame equality
(normalized MSE 0), and successful physics and lighting variation checks.
The [gate CSV](semantic_recovery_evidence/dmc_process_20261002/v3/gate-summary.csv)
records the actual host, GPU and GL renderer. The summarizer also verifies the
tested source hashes against this checkout. This closes process-candidate
correctness; default selection and throughput acceptance remain pending.

Earlier array `6961809` passed 22 device gates but its Manipulator job failed
before physics because Xvfb reset after the last GL client disconnected. Its
failure is preserved. The final coordinator uses Xvfb `-noreset`, which prevents
that server reset ([X.Org server manual](https://xorg.freedesktop.org/archive/X11R7.5/doc/man/man1/Xserver.1.html)).
Production code did not change between these two arrays; the final harness also
adds the two public variation cases for every registration/device.

The baseline array `6960514`, column-export retry `6961949` and process throughput
array `6961930` remain queued on `3090-gcondo` under its group GPU quota. The latter
two replace pending arrays `6961314` and `6961810`, respectively, so that they use
the corrected display coordinator. General-GPU measurements from `6961970` are
kept separately and are not relabeled as RTX 3090 performance.

## Initial performance evidence

The [local CSV](semantic_recovery_evidence/dmc_process_20261002/local/comparison.csv)
contains actual public `step` measurements, with every State retained and frames
disabled, at B64. Physics startup, restore, warmup and correctness checks are
outside timing. Six samples span two reversed strategy-order blocks. These are
Mac measurements; they are neither source-renderer speedups nor cluster results.

| Task / horizon | Local steps/s ± SD | Two processes | Four processes |
| --- | ---: | ---: | ---: |
| Acrobot / 1 | 17,380 ± 792 | 27,400 ± 1,572 | 42,430 ± 2,345 |
| Acrobot / 8 | 18,743 ± 351 | 31,679 ± 472 | 51,126 ± 1,293 |
| Humanoid / 1 | 5,882 ± 78 | 9,894 ± 377 | 16,131 ± 877 |
| Humanoid / 8 | 5,295 ± 23 | 9,398 ± 55 | 16,518 ± 104 |

The pending Oscar sweep compares local/four/eight process execution at B64/256,
T1/8, with CPU and CUDA output separately. Its allocations each have eight CPUs;
processes remain single-threaded for simulation and are pinned within that
allocation. This first comparison determines process strategy before larger-batch
saturation and combined rendering experiments. No DMC process default is promoted
from these two local rows or from a sibling's success.

Available-host Linux array `6961970` completed both Acrobot and Humanoid, with
eight allocated CPU cores per job and NVIDIA L40S devices. The
[48-row comparison](semantic_recovery_evidence/dmc_process_20261002/v3/comparison-general.csv)
contains B64/256, T1/8, local/four/eight processes and CPU/CUDA outputs. Each
comparison stays on its recorded host. At B256/T8:

| Task / output | Local steps/s ± SD | Four processes | Eight processes | Eight-process speedup |
| --- | ---: | ---: | ---: | ---: |
| Acrobot / CPU | 8,007 ± 16 | 28,437 ± 195 | 49,711 ± 317 | 6.21× |
| Acrobot / CUDA | 7,921 ± 50 | 27,433 ± 580 | 46,045 ± 776 | 5.81× |
| Humanoid / CPU | 2,008 ± 8 | 7,922 ± 16 | 14,706 ± 72 | 7.32× |
| Humanoid / CUDA | 2,028 ± 7 | 7,734 ± 56 | 14,240 ± 63 | 7.02× |

Physics is CPU in every row, all States are retained, and frames are disabled.
CUDA indicates output placement. These measurements support continuing the
process strategy; they do not replace the RTX 3090 sweep, establish combined
pipeline FPS, or select defaults for the other ten DMC registrations.

## Follow-on families and remaining work

The current OGBench audit confirms scalar owners and per-lane TensorDict export.
Maze also swaps the process-global NumPy RNG, so independent processes fit its
ownership constraints. Cube/Scene have a stronger restore path: validate all
rows, build staged replacement simulators, and swap only after every row restores
exactly. A distributed adapter must preserve that batch transaction, not commit
successful shards before discovering another shard's failure. These family hooks
do not require redesigning the shared State contract or public API.

Classic control also exports a TensorDict per scalar lane, while its physics can
be very cheap. Profile and remove export overhead before deciding whether a pool
wins. Navigation already performs tensor-batched stepping and should retain it.
These observations are routing audits, not completed implementations or new
performance claims. Craftax remains deferred; PushT/Rocket remain blocked on
their independent correctness prerequisites.

Remaining DMC work: collect matched resource throughput, choose measured
defaults, then measure larger batches, State/IPC/
transfer components, renderer-only FPS, selected-frame pipeline throughput, and
same-device source baselines. The full 35-registration goal remains active.

October 2 follow-up: baseline `6960514` is complete. Process throughput array
`6961930` has completed additional registrations; final report reconciliation
is pending. Its Acrobot/Pendulum tasks failed opening their X11 display before
the renderer proof. Retry `6964379` repeats only those tasks on unchanged frozen
source and writes `benchmark-retry2`. Failed reports, retry launcher and scheduler
state are preserved in [the closure](semantic_recovery_evidence/dmc_baseline_closure_20261002/).
The exporter acceptance array `6961949` remains pending. Defaults are unchanged.

## RTX 3090 throughput closure, ten registrations

The [new 240-row CSV](semantic_recovery_evidence/dmc_process_throughput_20261003/comparison.csv)
independently validates ten registrations on CPU/CUDA, using the frozen
`f0716916` implementation. Acrobot/Pendulum are excluded pending `6964379`;
no sibling result fills their rows. At B256/T8, all States and frames disabled:

| Domain | Local CPU-output steps/s | P8 CPU-output steps/s | CPU-output ratio | Local CUDA-output steps/s | P8 CUDA-output steps/s | CUDA-output ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ball_in_cup | 2,723 | 17,406 | 6.39× | 2,665 | 17,334 | 6.50× |
| cartpole | 3,505 | 22,364 | 6.38× | 3,492 | 20,345 | 5.83× |
| cheetah | 4,202 | 25,470 | 6.06× | 4,215 | 23,256 | 5.52× |
| finger | 2,480 | 16,721 | 6.74× | 2,440 | 15,778 | 6.47× |
| hopper | 3,221 | 20,564 | 6.39× | 3,182 | 16,618 | 5.22× |
| humanoid | 1,239 | 8,365 | 6.75× | 1,231 | 8,186 | 6.65× |
| manipulator | 1,450 | 10,296 | 7.10× | 1,461 | 9,824 | 6.72× |
| quadruped | 1,632 | 11,176 | 6.85× | 1,612 | 10,729 | 6.65× |
| reacher | 4,194 | 27,352 | 6.52× | 4,088 | 26,104 | 6.39× |
| walker | 2,517 | 17,141 | 6.81× | 2,548 | 16,245 | 6.38× |

The CSV retains sample SD and every B64/256, T1/8, local/P4/P8 cell. P8 is
1.92–7.10× local across these workloads and at least 1.075× P4. Every comparison
shares its recorded eight-CPU allocation and RTX 3090 host. Physics stays CPU;
these are adapter improvements, not render FPS or upstream speedups. The
results support a large-batch process route after remaining exporter, retry
and unpatched-default gates close; the public default is still local.

## Complete bounded process sweep and default candidate (October 3)

Acrobot/Pendulum retries 6964379 now pass independently. The consolidated
[288-row CSV and terminal evidence](semantic_recovery_evidence/dmc_process_complete_20261003/)
cover all twelve registrations, both output devices and every B64/256, T1/8,
local/P4/P8 cell. Source hashes match the original f0716916 implementation.

[Isolated default candidate 6966465](semantic_recovery_evidence/dmc_default_route_20261003/)
now tests unpatched public construction at B64 for the eleven registrations
whose separate exporter gates also pass. Quadruped remains excluded pending
its exporter frame diagnostic. Public constructors in this checkout remain local;
the candidate activation is preserved as five source overrides, ready to apply
only after required gates. Larger batches, components, source and full pipeline
benchmarks remain necessary.

## Default routing accepted (October 3)

All [22 independent default gates](semantic_recovery_evidence/dmc_default_acceptance_20261003/)
now pass. The tested constructor integrations activate bounded processes for
eleven DMC registrations at Linux B>=64 with >=4 available CPUs, up to eight
owners. Small/non-Linux/restricted/daemon contexts remain local. Quadruped stays
local pending its exporter frame diagnostic. The public API is unchanged.

This enables the measured execution strategy; it does not complete the remaining
large-batch, component, source or combined pipeline measurements.
