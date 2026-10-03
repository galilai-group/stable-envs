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
