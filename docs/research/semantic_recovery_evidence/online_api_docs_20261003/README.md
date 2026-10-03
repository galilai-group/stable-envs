# Current online API documentation and routing reconciliation

This checkpoint updates documentation and public examples against the accepted
implementation at `78f0312ab05f33781f656b0d052c4b8991cfec84`. No production code,
State schema, routing default, renderer or semantic acceptance status changes.

The 42-row stepping backlog is reconciled against the committed semantic matrix:
39 VERIFIED / 3 INCOMPLETE. `noncraftax-routes.csv` records all 38 non-Craftax
registrations, including the 35 accepted registrations and the three explicit
correctness exclusions. Within the accepted group, 27 have eligible process
routes, three use native Torch batching and five retain local execution. Four
Craftax registrations remain deferred. These counts do not count source speedups.

The existing Maze local/P4/P8 benchmark already had 16 accepted frozen-candidate
rows in job 6964833. Its backlog field incorrectly said NOT_YET_MEASURED. It now
says PARTIAL_ADAPTER_SCALING_MEASURED, with the old-candidate scope retained.
No old report or acceptance evidence was regenerated.

## Executed validation

The existing Fetch, DMC, OGBench and classic routing test files pass **251 tests**
with no skips. These mock OS/affinity/daemon and constructor selection; `cuda`
arguments are strings passed through mocked factories, not CUDA execution. The
JUnit XML and log are preserved. This is a consistency check for documentation,
not a replacement for previously accepted registration-specific Oscar gates.

`examples/action_sequences.py` runs on CartPole CPU and checks exact sparse-frame
selection against an all-frame run from the same State. It demonstrates `[B,A]`,
`[B,T,A]`, independent State/frame slices, validity, endpoint reuse and histories
disabled without an extra `get_state` after stepping. Construction is guarded
by `main`, so its structure is safe to adapt to spawn-based process routes.

`examples/batched_env.py` now defaults to accepted CartPole and runs on CPU.
It renders three saved States directly, producing `(3,400,600,3)`, without trying
to resize the live simulator through `set_state`. Its metadata lookup uses the
current `reference_id`, and it no longer calls removed public compact-packing
methods. The first two validation logs retain the obsolete metadata/packing
failures; `batched-env-v3.log` is the successful final run and ends `closed=True`.

Both example files pass Ruff. Commands used:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/samib/se7-ogb-venv/bin/python -m pytest -q \
  tests/unit/test_fetch_execution.py tests/unit/test_dmcontrol_execution.py \
  tests/unit/test_ogbench_execution.py tests/unit/test_classic_execution.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/samib/se6-venv/bin/python examples/action_sequences.py
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/samib/se6-venv/bin/python examples/batched_env.py --device cpu
```

## Documentation and remaining scope

The main dynamics guide now lists Cube/Scene and classic Acrobot process routes,
describes family-specific snapshot work honestly, and distinguishes raw task
rates, State export, prepared-State rendering and complete-call online FPS. It
explains that final-only and all-frame FPS have different image denominators,
and that library-control gains are not upstream speedups.

The variation-storage example uses full State. The examples index explicitly
marks the legacy training scripts as needing migration from compact rows and
`set_state(...).render()` chaining. Those scripts were not migrated or validated
in this checkpoint, and no private RenderState API was made public to support
them. Broader example migration and the full online performance objective remain
unfinished. GPU/source benchmarks and pending Oscar result collection still
require working cluster authentication.
