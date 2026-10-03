# Classic-control export evidence

Parent `6ce4e82931a629b68218bbe5331296bdc0a946d2`.
See [the report](../../classic_online_execution.md) for interpretation.

`local/profile` contains the five pre-change CPU component profiles. Reconstruct
their source using `local/baseline-source-overrides.json` plus the current
profiling script. `local/paired` contains the five complete public CPU gate and
paired timing reports; `local/comparison.csv` is their strict summary.

The initial CartPole report is retained as FAIL because the harness requested
size 64, while that renderer requires its native 400×600 output. Its exact script
is in `local/classic-column-local-v1-source-20261002.json`. The corrected reports
use the native default resolution and pass exact frame comparisons.

`column-v1` is the frozen Oscar source/launcher for array `6964063`, canceled
while pending because reference-helper imports required a missing fixture.
`column-v2` adds that fixture, with unchanged production and benchmark source;
all ten tests collect from its isolated archive. Its array is `6964284`. The source
manifest hashes each archived file and the archive. Reconstruct a source file
from `source_overrides.json` when present, otherwise the committed path at this
checkpoint. Source/reference provenance and the local test/valid-transition
observations are in `reference-provenance.json`.

Local test command (observed CLI output; no JUnit file):

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/samib/se6-venv/bin/python -m pytest -q tests/unit/test_classic_state_export.py tests/unit/test_native_classic_semantics.py tests/unit/test_native_classic_remaining_semantics.py
```

Result: 17 passed, five CUDA skips, 0.28 seconds. The frozen reference's AST
matches the parent's method with only its function name changed. The imported
PCG limb helper is unchanged.

Reproduce the CPU summary with a new output path:

```sh
python3 scripts/benchmarks/online_stepping/summarize_classic_state_export.py \
  docs/research/semantic_recovery_evidence/classic_online_20261002/local/paired \
  /tmp/classic-cpu.csv --source . --devices cpu
```

The default invocation requires both CPU and CUDA gates, all five registrations,
native frames, complete timing cells, six samples and source hashes matching
the checkout. GPU acceptance remains pending. Later results must be preserved
in a new closure directory, without overwriting these files.
