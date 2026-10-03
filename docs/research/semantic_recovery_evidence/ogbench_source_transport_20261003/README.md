# OGBench parallel source transport prerequisite

At integration base `bff0435dffbf6eccf4a754470b9fcfc0f9ba3c2c`, the benchmark-only
parallel source now passes independently for Maze, Cube and Scene. Production
adapters, source algorithms, process routing and public APIs are unchanged.
This prepares an equal-CPU-budget source baseline; it measures no speedup.

`ogbench_parallel_source.py` reuses the existing benchmark `SourceShard` and
`ParallelSource` scaffolding with Gymnasium `AsyncVectorEnv`, spawn, shared
observation buffers, `copy=True`, and disabled automatic resets. Each persistent
process owns a scalar source shard. Source task outputs travel in typed shared
columns; returned CPU tensors retain owned arrays. No thread pool is introduced.

Maze uses the pinned SWM wrapper with the accepted lane-owned legacy NumPy RNG.
Cube/Scene use the already accepted source entropy witness. These are the same
authoritative call paths audited in the [direct-source prerequisite](../ogbench_source_prerequisite_20261003/).
Source stepping still computes ordinary observations/info before projecting the
library's accepted numeric task fields. Array aggregation is NumPy; no extra
Torch conversion is inserted into each source lane's physics call.

Full source snapshot/restore RPCs use ordinary pickling and are **untimed oracle
preparation**, not shared-State throughput measurements. Maze snapshots use its
independent numeric State fields. Cube/Scene retain independent source engine,
cache, model, Python and entropy snapshots in the existing model configuration.
They are not claimed interchangeable with the candidate's public State schema.

## Executed evidence

`local-v2.json` and `local-v2.log` contain all **12 passing CPU cases**:
three registrations × T1/T8 × float32/float64, each B2/P2. For every case:

- Actual worker task outputs and complete source snapshots equal serial source.
- Restoring the initial source snapshot reproduces the tasks and endpoint exactly.
- Another changed-action call moves the physics but does not mutate retained
  output tensors or earlier snapshots.
- Parent NumPy global RNG remains unchanged. Both source workers have distinct
  PIDs, one Torch thread each, the expected lane assignment, and no renderer.

All **six worker-pool closures** record two exited workers with exit code zero.
The checker verifies shared memory, owned copies and disabled container auto-reset.
The parent additionally guards MuJoCo renderer construction with an assertion;
the source paths do not call render, and each worker's dedicated renderer field
is confirmed absent after stepping. No images, GL context or CUDA output is tested.

`source_manifest.json` pins seven SWM files at
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. The report records exact candidate,
loaded source/dependency hashes and installed versions; `review.json` checks the
complete case grid, pool shutdowns and all hashes. Local V1 is preserved as a
preliminary run; V2 adds explicit worker-exit evidence and matches the committed
gate script. Neither report contains performance timings.

Reproduce into a new output path:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/samib/se7-ogb-venv/bin/python \
  scripts/benchmarks/online_stepping/ogbench_source_transport_gate.py \
  --source-root /Users/samib/stable-worldmodel \
  --source-manifest docs/research/semantic_recovery_evidence/ogbench_source_transport_20261003/source_manifest.json \
  --output /tmp/new-ogbench-source-transport.json
```

## Scope still open

The helper deliberately supports nonterminal, no-frame workloads only. If any
lane terminates or truncates, it fails the benchmark rather than stepping beyond
the public chunk's active-lane boundary. This does not validate terminal handling;
that must be implemented and independently gated before broader source workloads.
Invalid inputs or a failed benchmark are not transactional source restores.

The full comparison still requires rendered source buffers, matched-state MSE,
CPU/software-GL and GPU/NVIDIA-GL evidence, CUDA delivery, larger batches,
resource measurements and six-sample mean/SD timings on Oscar. The previous
source cleanup finding also remains: calling source `close()` alone does not
prove its dedicated renderer has been released. Rendering must use measured
resource lifetime or an exited preparation process before the timed population.
None of these requirements is closed by this CPU transport gate. Existing Oscar
jobs and previously accepted evidence are untouched.
