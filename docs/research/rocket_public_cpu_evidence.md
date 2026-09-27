# Rocket public candidate: CPU evidence

The matrix remains **39 VERIFIED / 3 INCOMPLETE**. These results support the
Rocket candidate; they do not promote it. The final production revision is
`d2516cb67b0ef23b0dedc2f378d6e1c6a93f93d8`. Oscar job 6732132 covers that
revision's public GPU/CPU checkpoints, persistent configurations and independent
source suffixes; job 6732153 depends on its success for descriptive performance.

| Evidence directory | Completed coverage | Revision qualification |
|---|---:|---|
| `rocket_public_cpu_0059c137` | 204 checkpoints | Earlier production hashes captured at startup; predates persistent scalar precision fix |
| `rocket_source_suffixes_0059c137` | 108 branches / 1,296 future steps | Earlier production hashes; independent source reconstructed for every suffix |
| `rocket_persistent_variation_v1` | 90 cases | Scalar precision fix included; predates external configuration sampler change |
| `rocket_schema_f87cafd8` | 187 numeric leaves | Descriptive inventory; no correctness claim |
| `rocket_public_unit_d2516cb6` | 18 passed tests | Final candidate production hashes and test hash recorded |

Paths are under `semantic_recovery_evidence/`. Each directory contains compact
raw logs, provenance and SHA256SUMS. The startup production hashes, rather than
the directory names, identify the implementation each test actually covered.
CPU checks used the local Apple M5 with official PyFlyt 0.29.0 and PyBullet 3.2.7.
These CPU timings are diagnostic observations, not matched Oscar speedups.

The trajectory gate covers seeds 0/17/42, float32/float64 controls, free flight,
mixed controls, crash and continued terminal calls, success, horizon and all
Variation paths. The source-supported high-ceiling configuration reaches the
1,202-action bound; it is a diagnostic configuration, not an ordinary default
flight. The default fall crashes much earlier. A successful restore must match
all captured source diagnostics, immediate source pixels and valid continuation.
The independent-source suffix gate replays a new oracle prefix for each branch.
Persistent-configuration cases cover all leaf minima, maxima and defaults, plus
Python float / float32 / float64 height values with nontrivial precision.

At the registered 480×480 resolution, B1 State is 768,719 bytes; an uncompressed
NPZ is 825,183 bytes. The named immediate goal cache occupies 691,200 bytes.
The small-resolution CPU gate deliberately uses a smaller cache and therefore
reports smaller State sizes. Neither size implies omission of the goal cache.

Oscar jobs 6731765 and 6731919 were cancelled after 14:08 and 20:44 respectively
to cover subsequent candidate revisions. Their logs and terminal accounting are
preserved in directories ending `_superseded`; neither completed acceptance.
They join the previously recorded superseded job 6731648. Final acceptance must
resolve against job 6732132 and the unchanged final production hash inventory.

No shared State or renderer contract changed. All prior accepted evidence and
the frozen Rocket renderer remain unchanged. No push occurred.
