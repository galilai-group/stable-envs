# Fetch source online comparison — prepared, not measured

Base `18e2883b58542d3d1351b841ecd0c214404e3868`. This checkpoint prepares an
independent source comparison for all 12 Fetch registrations. It changes benchmark
code, checks and documentation only. Production routing, State semantics, public
API and rendering are unchanged. No new FPS, MSE, CUDA correctness or universal
speedup is claimed.

## What will be measured

Each registration gets one eight-physical-core RTX 3090 allocation. Source serial,
source eight-process and our public eight-process implementation run in separate
interpreters, on both CPU and CUDA output paths. CPU rendering uses software GL;
GPU rendering uses NVIDIA EGL. Parent Torch uses eight threads on CPU and one on
CUDA; source workers use one thread. The initial bounded batch is 64, with T1/T8,
six timing samples and no intermediate State history. Mandatory final library
State remains inside timing. Broader batches need actual memory/runtime evidence
before expanding this campaign; this does not replace the existing Fetch
large-batch library-control sweeps.

For Reach/Push/Slide plain, Dense and Dict, the coordinator first runs that exact
registration's independent B4/P1/P2 image gate on CPU and CUDA. Rendered timing
requires complete passing reports with current code/source/dependency hashes,
matching registration/device, per-frame MSE <= 0.001, source render nonmutation,
full State/tasks and clean worker exits. Source rasterization remains 480x480,
followed by the original source resize to 64x64. No internal-resolution shortcut
or modified source observation computation is used.

Those nine registrations measure no/final/all frame selections. Each measured
B64 cell again checks source tasks, full State, image MSE and output ownership
against an independent source before timing. A child owns the oracle GL contexts
and is closed before constructing the measured population. Source P1 timing is
a real serial parent source loop, **not** the P1 IPC oracle. Source P8 uses the
existing shared task/image transport. Source image readback and result transfer
to the requested device remain inside timing.

The three PickAndPlace registrations measure **no-frame execution only**. Their
source render callback changes continuation State, as established previously.
No images are generated for those rows, and FPS/MSE are blank. CUDA rows there
mean CUDA output delivery from CPU physics; they do not establish a rendered
online comparison or GPU physics speedup. This leaves their rendered protocol
explicitly unresolved.

Untimed preparation restores a fixed seeded initial State before each sample.
Source snapshots/restoration are independent full source fields and are never
counted in online timing. This is a controlled default-variation short-chunk
benchmark; it is not a sustained autoregressive-policy or arbitrary-variation
benchmark. Source computes its native observations/info before the harness
projects the task fields shared with the library API.

Separate library components measure stepping without State export (including
process IPC/task transfer), full `get_state`, owned CPU-to-device State copying,
and prepared-State rendering where the rendered protocol is admissible. CPU
copying performs real owned copies too. These operations are **not additive
estimates** of overlapped pipeline runtime. P1/P8 no-frame online rows provide
source stepping/packaging costs without a synthetic source-State export charge.

## Executed local evidence

`source-state-final-v2.json` is the authoritative final-code CPU prerequisite:
96 cases independently validate the snapshot-to-public-State projection across
all 12 registrations, T1/T8, float32/64 and ordinary/mixed limits. Existing source
transport, exact tasks/full State, restored replay, unseeded reset RNG, terminal
padding/held State and ownership checks also pass. All 48 workers exit cleanly.
No GL or timing is performed by this prerequisite.

`frozen.xml/log` runs 68 tests from the actual upload bundle. They cover owned
128-bit RNG/controller projection, frame selection/ownership, single-frame MSE
rejection, stale/sibling/wrong-device/partial image evidence, gate-before-timing
ordering and strict CSV joins. Synthetic collector fixtures are explicitly test
data, not benchmark results. In particular, they prove that a faster P8 source
is selected over P1 and that no-frame results do not get invented FPS/MSE.

`synthetic-final.log/xml` retains the earlier failed check: TensorDict `.to()`
does not accept `copy=True` in this profile. The final driver instead copies
individual tensors and packs an owned TensorDict. The corrected synthetic run
and frozen run pass. No production State code was modified to resolve this.
The CPU profile remains isolated MuJoCo 3.11.0 with Robotics 1.4.2; dependency
provenance is in `fetch_source_prerequisite_20261003`.

## Reports, reproduction and pending acceptance

`summarize_fetch_source.py` requires all six strategy/device reports, identical
host/allocation/dependency/candidate provenance, distinct strategy PIDs, exact
source State/task hashes, expected cell counts, six finite positive samples,
GL identities and the required component measurements. It emits online and
component CSVs with mean/sample SD. Speedup is against the fastest measured
same-device serial/P8 source; CPU source is not substituted for GPU source.
The collector checks application evidence; terminal SLURM accounting and logs
must also be reviewed before accepting or publishing a completed campaign.

`candidate_manifest.json` freezes 262 files, `source_manifest.json` pins the five
actual upstream dependencies at `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
Rebuild a new archive without overwriting an existing one:

```bash
python3 docs/research/semantic_recovery_evidence/fetch_source_pipeline_20261003/prepare.py \
  --checkout . --source /path/to/pinned/stable-worldmodel --output /tmp/new-fetch-source-pipeline
```

The local frozen bundle is `/private/tmp/fetch-source-pipeline-bundle-v1-20261003`.
`run.sh` requests 12 tasks, at most three concurrently, on 3090-gcondo. It uses
the previously accepted `se6-verify-fetch311` profile and requires MuJoCo 3.11.0
at runtime. Each rendered registration runs its own image gates before the
three timing strategies; strategy order rotates across registrations. This
integrated campaign includes the earlier standalone image campaign's checks,
so do not submit both merely to duplicate them.

**NOT_UPLOADED / NOT_SUBMITTED.** Oscar still rejects SSH keyboard-interactive
authentication. No previous job has been restarted or declared finished. The
whole timing driver, GPU copy path, public B64 source comparisons and actual
CSV collection remain unexecuted until Oscar is available. This is a prepared
benchmark with tested prerequisites, not completed performance acceptance.

The broad goal remains open: collect outstanding OGB/classic jobs, run the staged
DMC/Navigation/OGBench source comparisons, measure remaining registrations and
broader batches, resolve retained regressions and PickAndPlace lifecycle
accounting, and publish only independently supported results. Craftax is deferred;
PushT/PushT-Discrete/Rocket remain correctness-blocked. Matrix: 39 VERIFIED /
3 INCOMPLETE; this retains the existing semantic meaning, not a speedup claim.
