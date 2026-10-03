# OGBench source chunk stopping — CPU adapter/dispatcher prerequisite

At base `d82f679b3c9bb35c517509e17cfadebe916d70a4`, the benchmark source helper
now handles lanes that terminate inside a chunk. This changes benchmark code
only. Production adapters, the shared dispatcher, renderer and public API are
unchanged. This checkpoint does **not** validate public environment construction,
images, CUDA or performance.

Each chunk begins with all lanes active. A terminal/truncated transition remains
valid; subsequent actions in that lane are not consumed. Task columns are padded
with zero and `valid` records executed transitions. A subsequent call begins with
ordinary source lifecycle behavior, without a container-invented reset. The
helper still has no rendering path. This replaces its previous explicit rejection
of terminal trajectories; the previous nonterminal-only evidence stays frozen.

## Executed checks

`local-v3.json` / `local-v3.log` contain **24 passing CPU B2/P2 cases**:
Maze/Cube/Scene × T1/T8 × float32/float64 × one/both lanes already at their goal.
Fixtures move goal targets to the current physical configuration in the source
and candidate worlds; no source step or termination function is patched. The
genuine source transition reports termination on the first action.

The comparison uses serial source, persistent shared-memory source workers, and
the real accepted CPU adapters passed through the unchanged `_stepping.step`
dispatcher. A small render-free fixture supplies metadata and delegates State
and step operations to those adapters. It is explicitly **not** a public Env
constructor or a renderer substitute. The adapter path exercises the same action
shape normalization, State histories, padding and validity logic used by public
stepping, with rendering disabled.

Every case checks exact source tasks and complete source snapshots, agreement
with adapter/dispatcher outputs, held terminal State histories, validity, output
ownership, and another call after termination. The accepted independent
engine/RNG witness compares source and candidate after both calls. At T8, one
reached lane yields 9 valid transitions across B2; two reached lanes yield 2,
rather than reporting 16 nominal action slots as executed work. At T1 there are
two valid transitions. Six pools close with two zero-exit workers each.

`nonterminal-regression.json` / `.log` rerun all **12 existing B2/P2 cases** with
the updated helper. Source snapshot/task equality, restored replay, global RNG,
owned outputs and six clean pool closures still pass. Neither run measures
timings. The parent forbids MuJoCo renderer construction; worker inspection
confirms no renderer and one Torch thread per owner.

## Retained failures and correction

- `local-v1.json` / `.log`: the real public Maze constructor attempted a renderer
  bake, despite `frames=None` in the planned later calls. The local renderer guard
  stopped construction before any test case. Public construction remains pending
  for an Oscar compute node with the proper GL setup. The driver now offers an
  explicit Linux-only `--public` mode for that subsequent gate.
- `local-v2.json` / `.log`: eight Maze cases passed before an exact Cube source
  snapshot comparison failed. The accepted independent witness reads the source
  `action_space` property, incrementing a diagnostic `access_count`. Calling it
  between source replay operations made the serial counter differ from workers.
  `witness-counter.json` confirms the sole change is this counter (+1) in both
  Cube and Scene. The driver moves the witness after the exact comparisons;
  **no snapshot field or equality check was removed**. V3 passes all 24 cases.

Current driver/helper and loaded dependency hashes are in the reports. The
seven source files remain pinned at stable-worldmodel
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. Installed versions are OGBench 1.2.1,
MuJoCo 3.12.0, dm-control 1.0.45, Gymnasium 1.3.0, NumPy 2.5.3, Torch 2.14.0 and
TensorDict 0.14.2. Ruff and `git diff --check` pass.

Reproduce each result into a new path:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/samib/se7-ogb-venv/bin/python \
  scripts/benchmarks/online_stepping/ogbench_source_terminal_gate.py \
  --source-root /Users/samib/stable-worldmodel \
  --source-manifest docs/research/semantic_recovery_evidence/ogbench_source_terminal_20261003/source_manifest.json \
  --output /tmp/new-ogbench-terminal.json
```

Use `ogbench_source_transport_gate.py` with the same arguments and another new
output path for the nonterminal regression. Do not add `--public` on the Mac.

The full source performance campaign still requires real public constructors,
source and candidate images, verified CPU/GPU GL, shared frame transport, CUDA
output, memory/resource evidence, and same-allocation batch/horizon timing
sweeps. Synthetic goal fixtures are correctness checks, not a representative
timing workload. No source speedup, renderer acceptance or registration promotion
is inferred here. Existing Oscar jobs remain uncollected while authentication
is unavailable.
