# DMC process candidate evidence

Base: `c291296c642090dc77f86902b2f7d55afcb38d1d`, branch
`repair/render-config-fixes`. The candidate remains private; public construction
continues to select the local DMC adapter. No semantic-matrix promotion occurs.

## Frozen sources and jobs

For each `v1`, `v2`, `v3` stage, `source_manifest.json` hashes the archived source.
Read a path from that stage's `source_overrides.json` when present; otherwise read
the committed path in this checkout. This reconstructs every staged source file
exactly without committing binary source archives. Overrides are complete UTF-8
file contents. Launch scripts and original terminal logs are preserved.

| Job | Stage | Result / scope |
| --- | --- | --- |
| 6961714 | v1 | Completed: 69 DMC tests; 188 Fetch tests, one optional Matplotlib skip |
| 6961716 / 6961719 | v1 | Pending gate/performance arrays canceled before running; superseded |
| 6961806 | v2 | Completed: 29 focused CPU/CUDA State/closed-adapter tests |
| 6961809 | v2 | 11 task jobs passed both devices; Manipulator failed before physics due to Xvfb display reset |
| 6961810 | v2 | Pending throughput array canceled; superseded |
| 6961927 | v3 | All 12 jobs completed; all 24 independent CPU/CUDA public gates passed |
| 6961930 | v3 | RTX 3090 process throughput, queued at checkpoint |
| 6961970 | v3 | Completed: separate Acrobot/Humanoid throughput, 48 cells; actual L40S hosts recorded in each report |

V2 adds the aligned-adapter closed guard and regression test. V3 leaves
production source unchanged, adds public physics/lighting variation checks, and
keeps Xvfb alive with `-noreset` after probe clients disconnect. V2 failures and
partial reports remain intact. The final correctness CSV uses only V3 reports.

The older column-export array `6961314` was canceled while pending and replaced
by `6961949`. `exporter-run-v4.sh` runs the unchanged frozen column-export library
with the V3 coordinator, writing a new `results-v4` directory. The legacy
profiling array `6960514` remains queued. These exporter jobs have their own
acceptance criteria and are not closed by process-vs-column correctness.

## Verification and interpretation

From the repository root, with a new output filename:

```sh
python3 scripts/benchmarks/online_stepping/summarize_dmc_process.py \
  docs/research/semantic_recovery_evidence/dmc_process_20261002/v3/gates \
  /tmp/dmc-process-gates.csv --source .
```

This verifies all 24 reports, per-device GL implementation, all 12 cases,
physics/visual variations, State/RNG/lifecycle/ownership, completed lookahead,
exact frames and matching tested source hashes. Source hashes must agree across
all reports and with the checked-out production files. It refuses to overwrite
an output. `SHA256SUMS.json` inventories the preserved evidence, excluding itself.

MSE 0 here means process execution reproduces the accepted local-library
renderer for these cases. It does not recertify upstream renderer fidelity.
Performance is public `step`, all States retained, frames disabled; physics is
CPU for both CPU and CUDA **output** devices. Restore, warmup and construction
are excluded. Six samples per cell use two reversed strategy-order blocks.
These are not renderer FPS, pure physics-only rates, or upstream speedups.
Local Mac and available-host Linux results are separate from the queued RTX 3090
comparison. No default execution policy is selected from sibling results.
