# Cube/Scene full-State validator accepted into production

The exact GPU-tested candidate is now applied to the production adapter. Full
State is transferred and copied once per field rather than once per lane/field;
the existing semantic validation tail is unchanged. No shared API, renderer,
routing, State schema, RNG or lifecycle change. Every production file in the
frozen candidate manifest matches the integrated production tree.

Both registrations independently pass 12 public gate cases on CPU and CUDA,
including State/tasks, lifecycle, RNG, ownership, malformed render/restore
rejection and validation while workers step. Candidate frames equal the previous
library frames exactly. Each completed cluster unit suite has 88 passes and no
skips; local integration has 78 passes and ten CUDA cases already covered on
Oscar. Strict collectors verify all 12 component pairs and 24 pipeline pairs per
registration. Source-image gates remain separate and are not promoted here.

| Registration | Batch | Previous GPU online FPS | Candidate GPU online FPS | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Cube | 64 | 201.93 ± 10.34 | 634.31 ± 14.96 | 3.141× |
| Cube | 128 | 215.48 ± 6.14 | 772.78 ± 14.29 | 3.586× |
| Scene | 64 | 199.29 ± 11.61 | 560.86 ± 4.11 | 2.814× |
| Scene | 128 | 212.55 ± 0.77 | 648.60 ± 2.16 | 3.051× |

These are internal library controls, **not upstream speedups**. They use eight
actions, all intermediate States and all frames, six samples per strategy.
CPU rendered rates remain approximately 4.2–4.4 FPS. All cells, including Scene
no-frame T1 mean regressions of 4.6–6.8 percent with overlapping variability, are
retained. This does not establish a universal no-regression or source-performance
claim. The 35-registration source-comparison tally stays at eleven.

Scene's original CPU subprocess completed successfully and the launcher printed
`Scene cpu PASS`; the enclosing job later timed out during CUDA timing. The
completed, unchanged CPU output is retained. Retry 6976164 completed the entire
CUDA sweep with exit 0. CPU and CUDA ran on different nodes, recorded per row;
every baseline/candidate pair remains on the same node and allocation. Final EGL
destructor warnings are retained in the raw log and did not fail the process.
The original timed-out output remains in the preceding checkpoint.

Reproduce tables with `summarize_ogbench_validation_columns.py <Cube-or-Scene>
<new-output> --source <matching checkout>`. Frozen source and reference provenance
remain in `ogbench_validation_columns_20261003`; current runtime/test hashes match
those measured in all four complete reports. `acceptance.json` records scope and
limitations. No semantic-matrix promotion or modification of old evidence occurs.
