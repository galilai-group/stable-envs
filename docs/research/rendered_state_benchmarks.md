# Rendered end-to-end State benchmark extension

This research extension preserves the earlier measurements and their immutable
report. It adds a sixteen-step composed rollout at B1 with four render cadences:
every step, every fourth step, final frame only, and no rendering. The four
workloads are ordinary step/render, step/get/render, restore then step/render,
and snapshot/eight divergent steps/restore/rendered continuation. Each uses the
same starting State and prescribed actions. Preparation is outside the timer;
complete branch work is inside. Ten warmups and fifty timed samples retain
materialized outputs and synchronize GPU/JAX work.

`rendered_state_ops.py --derive <prior-summary.json> --output <fresh-directory>`
first derives the requested ratios for all 37 renderable registrations. Two
limits are explicit: the earlier non-Craftax CPU dynamics ran on Apple M5 while
GPU rendering ran on Oscar, so adding those medians is cross-host arithmetic,
not a measured latency prediction. Also, `render(full State)` already includes
projection. The literal requested sum and the version without double-counting
projection are both recorded. This yields 60 renderer/device rows covering 37
registrations: 25 have matched existing component measurements, while 35 are
explicitly cross-host arithmetic. No old artifact is rewritten.

The composed collector measures components again on the **same host/profile**.
Where public projection exists, it times `to_render_state` followed by
`render(RenderState)` exactly once. Otherwise public `render(State)` includes its
private projection and separate projection cost is zero, explicitly classified
as included rather than an absent computational cost. The main additive estimate
uses these matched, nonoverlapping components. Both extra-get and extra-set
ratios divide by step + projection + render; their one-frame slowdown factors
are one plus those ratios. Rollout predictions scale step/get by the horizon,
render/projection by actual cadence, and set by the number of restores.

The collector separately compares a step after restore with the same action at
a fresh reset/prefix checkpoint, and project/render immediately after restore
with project/render on the warm fixed snapshot. These measurements reveal
restoration-related initialization and invalidation costs. Ratios and sample
variance are descriptive; no semantic acceptance gate or performance threshold
is introduced. Full rollout measurements are compared with sums of component
medians, which need not equal a median of a composed call.

Pixel Craftax's delegated step intrinsically constructs its upstream pixel
observation. The explicit render in this workload is additional API work; it is
not subtracted or relabelled as pure physics time.

CPU Craftax and all available GPU renderer registrations use the accepted
family dependency profiles. Scripts refuse existing output directories and
retain failed runs. The profile dispatcher uses a clean base PYTHONPATH before
launching each family's interpreter. Raw per-case JSON is consolidated in the
final report rather than duplicating previous native archives.

## Reproduction and provenance

The collector and dispatcher are pinned at
`f66d8d2f844441ea9ddc9ae15c7a4460ea7a398a`; their source hashes are retained in
`semantic_recovery_evidence/branch_replay_final/source-identities.json`.
The CPU raw records have a null `source_commit` field because that optional
environment variable was unset. The collector files were unchanged from this
commit during CPU collection; the supplemental source hashes document that
identity without rewriting the raw results. The Oscar archive explicitly sets
the same commit and uses oracle commit
`176d54532b3c0a251a4f509b74f2fdacc143a63d`.

CPU reproduction with a fresh output directory:

```sh
PYTHONPATH=. SWM_ROOT=/Users/samib/stable-worldmodel CRAFTAX_JAX_BACKEND=cpu JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 STATE_BENCH_SOURCE_COMMIT=f66d8d2f844441ea9ddc9ae15c7a4460ea7a398a .venv/bin/python scripts/benchmarks/run_rendered_matrix.py --only 'Craftax.*Pixels' --output /private/tmp/new-rendered-cpu --profile craftax=/private/tmp/craftax-recovery-312/bin/python
```

The exact Oscar batch script and archive hashes are in
`semantic_recovery_evidence/rendered_state_submission/`. The existing job's
partition/QOS was changed from `3090-gcondo`/`cs-3090-gcondo` to `gpu`/`norm-gpu`
when the former was blocked by its group GPU quota. This was the same job ID,
not a duplicate submission. It retained the RTX 3090 constraint and accepted
family profiles.

After retrieving and checking the terminal archive, report reproduction uses:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/branch-replay-mpl PYTHONPATH=.:/private/tmp/native-nav-py312-deps .venv/bin/python scripts/benchmarks/report_rendered_state.py --cpu docs/research/semantic_recovery_evidence/rendered_state_cpu --gpu docs/research/semantic_recovery_evidence/rendered_state_gpu_6703024/results --output /private/tmp/new-rendered-report
```

Output paths must be fresh. Rerunning the report reads immutable raw samples;
rerunning the collector creates new measurements and must not replace evidence.
