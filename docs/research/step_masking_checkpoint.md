# Shared output-padding checkpoint

The dispatcher now reuses its all-active routing decision to skip empty-mask
writes. Cloning still owns every task output; mixed/finished lanes retain zero
padding, held endpoint States and the existing slice behavior. The all-active
path reuses its CUDA scalar decision. The first transition is known all-active; later
decisions already route serial execution or process submission.

The [local CSV](semantic_recovery_evidence/step_masking_20261003/local/comparison.csv)
contains all three native registrations, B1/16/64, T1/8, all/no State history
and final-frame pipeline cases. Every registration passes 16 exact public
State/task/frame comparisons against the frozen c39880f9 dispatcher, including
early stopping, reverse/empty slices and retained-output ownership. This is
library-regression MSE 0, not a fresh upstream fidelity survey.

At B64 with all States retained and frames disabled (Mac CPU):

| Environment | Actions | Previous steps/s ± SD | Candidate steps/s ± SD | Ratio |
| --- | ---: | ---: | ---: | ---: |
| TwoRoom | 1 | 183,725 ± 7,117 | 196,298 ± 1,944 | 1.07× |
| TwoRoom | 8 | 194,774 ± 1,508 | 191,303 ± 1,696 | 0.98× |
| Piecewise | 1 | 222,863 ± 12,235 | 249,897 ± 3,404 | 1.12× |
| Piecewise | 8 | 242,154 ± 2,396 | 244,315 ± 3,879 | 1.01× |
| SimplePointMaze | 1 | 4,940 ± 387 | 26,524 ± 260 | 5.37× |
| SimplePointMaze | 8 | 4,087 ± 312 | 4,702 ± 101 | 1.15× |

SimplePointMaze benefits most for single actions because its source-required
500×500 RGB `info.goal` made empty-mask indexing expensive. Eight-action calls
still retain a large image history and pay its copying/stacking costs. This
change does not claim a universal improvement: TwoRoom B64/T8 measured 0.98×.
CPU measurements of the two small native tasks overlapped the SimplePointMaze
local run; treat small differences as diagnostic, pending isolated cluster pairs.

Validation: 226 local tests passed, 76 CUDA cases skipped; Ruff passed. Tests
cover serial and immediately completing pipelines, all-active/mixed/all-ended
lanes, nested image outputs, borrowed serial buffers, complete owned pipeline
outputs and output mutation isolation. Oscar arrays `6964504` (native pairs and
302 shared/native tests) and `6964505` (all twelve Fetch CPU/CUDA pipeline gates)
are pending on RTX 3090. No GPU acceptance is claimed yet.

All scripts, source manifests, JUnit output and job handles are in
[the evidence folder](semantic_recovery_evidence/step_masking_20261003/).
The next native work is to close those gates, profile retained-image history
cost, and audit exact cached-goal restoration dependencies before changing its
Matplotlib regeneration. The source info payload must remain intact.

## Sparse inactive-row follow-up (October 3)

Profiling the T8 case corrects the earlier explanation: image copying/stacking
was not the only remaining cost. With reset seed 17, one of 64 lanes ends after
its first transition. Boolean padding scans large image outputs for that ended
lane. The dispatcher now computes inactive row indices once and uses index_fill_.
Output ownership, padded zeros, State capture and source info payloads are unchanged.
CUDA nonzero may synchronize on mixed lanes, so GPU benefit requires measurement.

[Fresh sequential CPU pairs](semantic_recovery_evidence/step_masking_sparse_20261003/local/comparison.csv)
against frozen c39880f9 report, at B64 with all States and frames disabled:

| Environment | Actions | Previous steps/s | Candidate steps/s ± SD | Ratio |
| --- | ---: | ---: | ---: | ---: |
| TwoRoom | 1 | 200,409 | 212,559 ± 4,885 | 1.06× |
| TwoRoom | 8 | 200,595 | 202,903 ± 5,198 | 1.01× |
| Piecewise | 1 | 243,176 | 264,770 ± 3,929 | 1.09× |
| Piecewise | 8 | 257,086 | 263,143 ± 6,170 | 1.02× |
| SimplePointMaze | 1 | 5,522 | 27,831 ± 926 | 5.04× |
| SimplePointMaze | 8 | 4,126 | 20,226 ± 220 | 4.90× |

The CSV includes control SDs, all 54 workloads, and valid-transition counts.
SimplePointMaze T8 counts 505 transitions, not 512. Its source goal-image history
is still 384 MB. Single cProfile calls locate 82 ms of direct masking cost before
and 2 ms of index_fill_ afterward; those profiles are diagnostics, not paired rates.

All three registrations pass 16 exact CPU State/task/frame cases; 226 shared
tests pass and 76 CUDA cases are skipped locally. The still-pending original
arrays 6964504/6964505 were canceled before execution; immutable replacement
arrays 6966161/6966162 validate native CPU/CUDA and all twelve Fetch pipeline
gates on RTX 3090. [Source manifests, profiles and launchers](semantic_recovery_evidence/step_masking_sparse_20261003/)
preserve both attempts. No GPU acceptance is claimed yet.

## Oscar sparse-padding closure (October 3)

Arrays 6966161/6966162 passed: all six native CPU/CUDA gates, 302 shared/native
tests and 38 Fetch adapter tests. All twelve Fetch registrations independently
pass CPU/CUDA State/task/replay checks and exact CUDA frame checks. CPU Fetch
frames were not exercised by that harness and remain blank in the gate CSV.
[The 108-row native comparison and 24 Fetch gates](semantic_recovery_evidence/step_masking_sparse_closure_20261003/)
preserve raw samples, SDs, source hashes and terminal evidence.

At B64/T8, SimplePointMaze improves 1,164→2,537 valid steps/s on CPU (2.18×),
and 23,307→24,867 on CUDA (1.07×). GPU benefit is modest compared with the
Mac CPU result. These are same-host library-dispatcher comparisons, not source
speedups or fresh upstream image MSE. Further family integration/default gates
remain independent.
