# OGBench online export evidence

Parent: `f0716916cf285f969a1159f6904eac254a926dc6`.
See [the checkpoint report](../../ogbench_online_execution.md) for scope and limitations.

For each frozen stage, reconstruct each file in `source_manifest.json` from
`source_overrides.json` when present, otherwise from this checkpoint's committed
source. Each manifest also hashes its uploaded archive. Binary source archives
remain on scratch and are not required to reconstruct the source.

- `profile`: original library source, diagnostic script; job `6963334`. Maze
  report and its terminal log are complete. Cube/Scene are still running.
- `column-v1`: candidate and first public harness; job `6963434` was canceled
  while pending after local smoke found a wrong canonical name for Maze.
- `column-v2`: same production candidate, corrected Maze name; job `6963669`,
  queued at checkpoint. CPU software-GL and GPU EGL use separate processes.
- `local`: all three CPU baseline profiles, complete paired CPU comparisons,
  and the failed initial Maze smoke (retained). Frames were disabled in the
  local paired runs. The failed smoke corresponds to `column-v1` source.
  The earlier successful Cube paired run used the two formatting/name versions
  preserved in `ogbench-column-local-source-overrides-20261002.json`; its source
  and reference hashes are therefore not replaced with the later files' hashes.

The frozen row functions in `tests/helpers/reference/ogbench_state_export.py`
were verified against the parent commit's AST, allowing only the documented
function renames and the direct call to the extracted manipulation row function.
`reference-provenance.json` records the source and reference hashes. Production
field/RNG helper declarations imported by that reference did not change.

Local test observations from this checkpoint (CLI outputs, no local JUnit file):

```text
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/samib/se7-ogb-venv/bin/python -m pytest -q tests/unit/test_ogbench_state_export.py
3 passed, 3 CUDA skipped; 4.39 s

PYTHONPATH=. OMP_NUM_THREADS=1 /Users/samib/se7-ogb-venv/bin/python -m pytest -q tests/unit/test_ogbench_semantics_oracle.py tests/unit/test_ogbench_manipulation_cpu.py -k 'not controlled_oracle and not reward_routes and not histories_oracle and not size_dtype and not child_streams and not identical_genuine and not metadata_reads'
19 passed, 70 deselected; 25.59 s
```

These exclude the full manipulation upstream-oracle sweep; they do not certify
new physics or fresh upstream image parity. Existing accepted source evidence
remains unchanged. The export tests cover every field against the pre-change
exporter plus continuation/ownership. The GPU array runs each registration's
CPU/CUDA export tests before its public frame and timing gates.

After the final six reports arrive, validate with a fresh output path:

```sh
python3 scripts/benchmarks/online_stepping/summarize_ogbench_state_export.py \
  /path/to/column-v2/results /tmp/ogbench-column-comparison.csv --source .
```

The summarizer requires all six PASS reports, independent device/GL proof,
exact frame equality, expected cases and all paired timing cells, six samples
per mode, consistent hashes, and matching checked-out source. Missing reports
or skipped frame checks cannot close acceptance. All output files are immutable
once committed; collect later results into a new closure directory.
