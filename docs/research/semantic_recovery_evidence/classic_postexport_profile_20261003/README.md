# Post-export classic-control component profiles

All five tasks in array 6966952 completed on RTX 3090 hosts with eight allocated
CPUs, one Torch/BLAS thread, and source frozen at 1dff9793. Forty independent
batch/device cells cover B1/64/256/1024 and CPU/CUDA output. Every cell passes
exact restore/step replay. Six samples per operation and separate cProfile
captures are preserved. The strict summarizer validates complete coverage,
source hashes, terminal PASS reports and finite samples before writing 160 rows.

CPU physics is used for both output devices. The diagnostic without export
substitutes an already owned State to isolate task/physics cost, includes that
mock's overhead, and is never a valid transitioned State. These are adapter
component timings, not public pipeline FPS or source speedups. Separately
measured operations are not additive timing partitions. Restore is outside the
step timing and is reported explicitly.

At B1024 with CPU output, milliseconds per batch (mean ± sample SD):

| Registration | Step including State | Task/physics diagnostic | State export | State restore |
| --- | ---: | ---: | ---: | ---: |
| swm/AcrobotControl-v1 | 90.58 ± 0.21 | 66.54 ± 0.25 | 24.86 ± 0.16 | 92.62 ± 0.28 |
| swm/CartPoleControl-v1 | 39.43 ± 0.18 | 11.02 ± 0.12 | 28.93 ± 0.13 | 106.15 ± 1.28 |
| swm/MountainCarContinuousControl-v0 | 29.85 ± 0.11 | 5.47 ± 0.06 | 23.76 ± 0.06 | 91.37 ± 0.57 |
| swm/MountainCarControl-v0 | 37.66 ± 0.15 | 16.28 ± 0.20 | 21.26 ± 0.09 | 81.33 ± 0.35 |
| swm/PendulumControl-v1 | 45.32 ± 2.56 | 19.20 ± 0.17 | 25.82 ± 0.29 | 95.44 ± 0.56 |

Acrobot's profile is dominated by Gymnasium RK4/derivative execution; bounded
processes are a strong next candidate there. Continuous MountainCar's physics
is much cheaper, and State row conversion/assembly dominates its profile. The
exporter still builds a NumPy array for every field of every lane before copying
into columns. Avoiding those temporary arrays is a concrete local optimization
candidate before selecting a pool for cheap tasks. Neither candidate is accepted
by these profiles; both require independent correctness and paired timing gates.

Reproduce with summarize_classic_profile.py results NEW_OUTPUT --source
FROZEN_CHECKOUT. Reconstruct that checkout from source_manifest.json's base and
verify its hashes. No production code or semantic status changed for this run.
