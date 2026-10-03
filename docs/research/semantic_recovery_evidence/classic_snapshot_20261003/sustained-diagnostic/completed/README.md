# Sustained CUDA diagnostic closure

Both array tasks 6970936 completed on gpu2102 (Pendulum raw6970937, Acrobot
raw6970936), exit 0:0. Each registration has eight fresh-process reports:
four cells, two strategies, six windows each, 128 consecutive transitions per
lane at B1024. Strict summaries verify all recorded source hashes, eight CPU
cores, RTX 3090, unchanged normal-GC thresholds, independent process IDs,
exact full-trajectory fingerprints and complete final State after every window.
No timing samples were removed. Reset durations are recorded separately.

| Domain | T | States | Frames | Baseline steps/s ± SD | Candidate steps/s ± SD | Ratio |
|---|---:|---|---|---:|---:|---:|
| Pendulum | 1 | none | final | 20,717 ± 351 | 20,924 ± 193 | 1.010 |
| Pendulum | 8 | all | final | 29,453 ± 421 | 29,454 ± 420 | 1.000 |
| Pendulum | 8 | none | none | 30,913 ± 401 | 48,650 ± 1,093 | 1.574 |
| Pendulum | 8 | none | final | 29,723 ± 379 | 43,799 ± 858 | 1.474 |
| Acrobot | 1 | none | final | 32,074 ± 787 | 32,262 ± 346 | 1.006 |
| Acrobot | 8 | all | final | 69,501 ± 703 | 68,630 ± 1,420 | 0.987 |
| Acrobot | 8 | none | none | 86,657 ± 655 | 98,918 ± 543 | 1.141 |
| Acrobot | 8 | none | final | 71,028 ± 498 | 78,734 ± 1,178 | 1.108 |

For final frames at T8, emitted FPS is steps/s divided by eight. Pendulum's
earlier isolated-call regressions do not persist in these sustained cases;
this does not erase those measurements or conclusively identify their cause.
Each Pendulum strategy records one generation-2 collection across each cell's
six windows; Acrobot records zero. Acrobot's all-State mean remains 1.3% lower.
CPU sustained follow-up is separate and pending. Production remains unchanged.
These are library candidate/control comparisons, not source speedups.
