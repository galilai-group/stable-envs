# Reproducible State-operation benchmarks

These measurements describe the accepted implementations without optimizing them
or adding a promotion gate. Every accepted historical registration is a separate
case, including shared Fetch and Craftax adapters. The executable catalog comes
from the committed semantic implementation map; incomplete cells are refused.

Run from the checkout with its accepted family dependency profile and pinned SWM
source on `PYTHONPATH`:

```sh
PYTHONPATH=.:/path/to/pinned-stable-worldmodel OMP_NUM_THREADS=1 \
  python scripts/benchmarks/state_ops.py \
  --registration swm/CartPoleControl-v1 --device cpu --batches 1 16 \
  --warmup 10 --iterations 50 --output /fresh/output/directory
```

Output directories must not exist. The harness writes incremental `result.json`
records, preserves raw nanosecond samples, and returns nonzero for incomplete
runs. Use one measurement process per machine/allocation; do not overlap CPU
builds or unrelated local benchmarks with reported CPU timing. Smoke-test runs
are development checks, not reportable throughput. Never overwrite acceptance
collectors or their sealed outputs.

## Measurement boundaries

Each case times construction separately. Each operation records its first call,
ten warmups, at least fifty measured calls and, for fast operations, additional
samples up to 1000 until 0.25 seconds of measured operation time. Counts are CLI
parameters recorded with results; expensive larger batches may use a documented
smaller count. The main harness's first-call field means first **measured** call:
checkpoint preparation has already exercised step/restore. `cold_craftax.py`
therefore measures constructor, first explicit reset/step/get/set and a second
step in separate fresh public/oracle processes for every Craftax registration
and device. Constructor/reset may internally invoke other operations; exact call
order is recorded. Interpreter startup and the initial hardware probe are
excluded. JIT compilation is included when first triggered and is not reported
as warm steady-state cost. Every call is materialized: CUDA is
synchronized before/after timing, JAX outputs are blocked, and returned values
remain live. Report median and p95 latency, median-derived calls/second, mean,
coefficient of variation and raw samples. Large variance is descriptive evidence,
not permission to discard inconvenient samples.

Batch size 1 is mandatory; public batches 16 (or smaller where measured resource
cost warrants it) are additional cases. No benchmark constructs a fake batched
adapter from scalar APIs. The authoritative oracle and its native snapshot
baseline are scalar; they are compared only with public batch 1. Operations per
second count full calls, while environment operations/second multiply by batch
size. Neither quantity is rendered FPS.

The common checkpoint follows seed 17 and two deterministic sampled actions.
Oracle step uses the same actions and prefix. Public step restores that checkpoint
outside the timer; oracle step resets and replays its two-step prefix outside the
timer because the oracle has no exhaustive restore interface. Timed ordinary
step itself never restores. Reset uses seed 17. Preparation is excluded and
recorded explicitly. Same-instance and independent-instance restore follow eight
substantially different public actions; independent targets are separately
constructed with seed 42. Branch-cycle cost includes snapshot, eight divergent
steps, restore and two continuation steps. It is also measured directly rather
than estimated by adding separate medians.

Rendering is a separate operation on an explicit full State. Public
`to_render_state` is measured where available; lack of that public API is reported
as unsupported, not replaced by a private packed renderer row. CPU renderer
refusals are retained and required CUDA measurements run on Oscar. Symbolic
Craftax has no renderer. Authoritative pixel-Craftax `step` intrinsically produces
a pixel observation: its timing explicitly includes that required output, even
though the benchmark makes no extra render call. This differs from exhaustive
State-only stepping and is not a like-for-like observation-production comparison.

State sizes include both numeric tensor payload bytes and an uncompressed NPZ
encoding with named keys and dtype/shape metadata. No pickle is used. Serialization
is outside operation timing. Versions, hardware, device, execution mode, batch,
actions, construction cost and reproduction command accompany each case.

## Native baseline interpretation

| Family | Closest native mechanism | Classification | Exhaustive comparison? |
| --- | --- | --- | --- |
| Classic control | No native snapshot API | `NONE` | No |
| Simple navigation | No native snapshot API | `NONE` | No |
| DMC | `Physics.get_state/set_state` | `NATIVE_PARTIAL` | Omits RNG, task/lifecycle/model changes and warm-start fields |
| Fetch / OGBench | MuJoCo `mjSTATE_INTEGRATION` get/set | `NATIVE_PARTIAL` | Omits wrapper/task/RNG/configuration and output caches |
| Craftax | Explicit copy/install of upstream JAX engine-state pytree | `NATIVE_PARTIAL` | Omits wrapper key, lifecycle and immediate-output caches |
| Rocket diagnostic | PyBullet `saveState/restoreState` | `OPAQUE_SAME_PROCESS_NON_EQUIVALENT` | Known different-history failure; never an exhaustive baseline |
| PushT research | Typed body-cache and active-contact-order accessors | `NATIVE_PARTIAL` | Research subset only; complete restoration still fails |

Native ratios are explicitly marked non-equivalent lower-bound comparisons. The
harness does not relabel shallow references, body pose copies or opaque handles
as exhaustive snapshots. `NATIVE_EQUIVALENT` requires source and behavioral proof,
not just a similarly named method.

## Adding future coverage

After a registration is accepted, add its canonical class and evidence to the
implementation map; the benchmark catalog picks it up independently. Supply its
accepted dependency profile and run a fresh smoke test. If the generic action or
oracle factory is insufficient, add a bounded family hook in this research
harness, with an explicit configuration and source citation. Audit native
snapshot semantics before adding a native operation. Keep failed and unsupported
operations visible. Validate timing accounting with
`pytest -q tests/unit/test_state_ops_benchmark.py`.

Reconstruction-heavy restoration and large State/step ratios should be recorded
for the later post-42/42 optimization pass. No production fast paths or shared
State-contract changes belong in this initial baseline.

`report_state_matrix.py` combines completed per-registration JSON files into
machine-readable and Markdown tables. Pass GPU result roots in chronological
order so a completed, corrected profile supersedes its earlier failed attempt;
retain both raw runs. The report refuses missing CPU or GPU registration coverage.
It does not substitute a sibling registration's measurement.
