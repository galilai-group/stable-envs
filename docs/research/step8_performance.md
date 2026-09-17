# Step 8 — final performance and pipeline accounting

**Generated** by `scratch/step8/perf/write_doc.py` from `examples/results/step8/performance/tiers.json`, which `scratch/step8/perf/build_performance.py` builds from committed Step-6 and Step-7 artifacts. Do not edit this page by hand: re-run the two scripts.

Step-8 baseline SHA: `43ac95238f7472137619f3e69e7636e11dd088b1`.

## What this page is

One CPU tier and one GPU tier for every one of the 42 registered `stable_worldmodel` ids, with the numbers each tier rests on, the host and job those numbers were taken on, and the reference rule that was applied. Nothing here was re-measured. The tiers are **copied** from the Step-7 record and re-derived from the underlying artifacts only as a cross-check; where a copied tier and its own evidence disagree, both are printed and the disagreement is explained rather than resolved.

### No optimization was reopened

Step 8 does not reopen Step-6 optimization, and this workstream did not. No renderer, harness or shared file was touched; no benchmark was run; no SLURM job was submitted; no Oscar tree was read or written. No frozen threshold was changed — in particular the `BATCH_SCALES` bar stays at `fps(4096) >= 20 x fps(1)` and the Rocket and Fetch rows keep the `GPU_PARITY` classification that bar produces. Every tier below is either a number somebody already measured and committed, or an honest `UNMEASURED` / `MISSING-UNSUPPORTED_BY_DESIGN`. Where a better tier is plainly reachable with work — a shared CPU triangle backend for the three GPU-only cells, a quiet-node Quadruped CPU run — that work is named as deferred and not started.

## The two performance claims, kept separate

They answer different questions and neither may be quoted in place of the other (`docs/research/certification.md` §6.1):

| | conservative certification microbenchmark | production pipeline |
|---|---|---|
| what it compares | our renderer against the **official renderer** of the same env, and against **one core** of in-memory JPEG-q90 decode of *that env's own frame* | **stored image -> load -> decode -> H2D** against **compact state -> load -> render -> device-resident pixels** |
| what it is for | the frozen tier ladders `S_cpu` / `cores_cpu` / `R` / `cores` and the qualifiers `BATCH_SCALES`, `NO_LARGE_BATCH_REGRESSION`, `HOST_BOUND` | whether regenerating beats reading in a realistic training loop |
| where the numbers are | the tier tables on this page, from `examples/results/step8/performance/tiers.json` | the Step-6 Lance/JPEG comparison, `examples/results/step6/lance/comparison2.{json,md}` |
| cores budget | **1 thread** on CPU; the all-core number is supplemental and never certifies | 8 Lance workers against our 8-way arm, and separately 1 worker against 1 thread |
| JPEG quality | q90, decoded from memory | q95, as `stable_worldmodel` and Lance write it |

The microbenchmark is deliberately the harsher of the two on the stored side: it charges the stored path nothing for storage, for the loader, or for getting the frame onto the device. The pipeline comparison charges both sides the whole path and ends both on the same side of the PCIe bus.

## Tier accounting — all 42 registered ids

CPU: **24 MEASURED**, **13 UNMEASURED**, **3 MISSING-UNSUPPORTED_BY_DESIGN**, **2 UNSUPPORTED_BY_DESIGN** (symbolic). GPU: **40 MEASURED**, **0 UNMEASURED**, **2 UNSUPPORTED_BY_DESIGN** (symbolic).

`status` says only whether a measurement **exists**. A tier is a grade, not a gate: `CPU_REGRESSION` is a well-measured tier, and `CPU_MISSING [RAISES_BY_DESIGN]` is a supported answer — the env raises on `device="cpu"` instead of silently rendering on CUDA.

| SWM id | primary cell | class | CPU tier | CPU status | CPU flags | GPU tier | GPU status | GPU flags |
|---|---|---|---|---|---|---|---|---|
| `swm/AcrobotControl-v1` | `Acrobot` | R | `CPU_REGRESSION` | MEASURED | `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FASTER` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/AcrobotDMControl-v0` | `DMCAcrobotSwingup` | S | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FAST` | MEASURED | `BATCH_SCALES`, `HOST_BOUND`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/BallInCupDMControl-v0` | `DMCBallInCupCatch` | S | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FAST` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/CartPoleControl-v1` | `CartPole` | R | `CPU_PARITY` | MEASURED | `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FASTER` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/CartpoleDMControl-v0` | `DMCCartpoleBalance` | S | `CPU_PARITY` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FAST` | MEASURED | `BATCH_SCALES`, `HOST_BOUND`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/CheetahDMControl-v0` | `DMCCheetahRun` | S | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_EXCEPTIONAL` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/CraftaxClassicPixels-v1` | `CraftaxClassic` | R | `CPU_FASTER` | MEASURED | `BELOW_G4`, `REFERENCE_SPLIT_JOB` | `GPU_FASTER` | MEASURED | `BATCH_SCALES`, `REFERENCE_SPLIT_JOB` |
| `swm/CraftaxClassicSymbolic-v1` | — | n/a | `n/a` | UNSUPPORTED_BY_DESIGN | `SYMBOLIC` | `n/a` | UNSUPPORTED_BY_DESIGN | `SYMBOLIC` |
| `swm/CraftaxPixels-v1` | `CraftaxFull` | R | `CPU_FASTER` | MEASURED | `BELOW_G4` | `GPU_FASTER` | MEASURED | — |
| `swm/CraftaxSymbolic-v1` | — | n/a | `n/a` | UNSUPPORTED_BY_DESIGN | `SYMBOLIC` | `n/a` | UNSUPPORTED_BY_DESIGN | `SYMBOLIC` |
| `swm/FetchPickAndPlace-v3` | `FetchPickAndPlace` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchPickAndPlaceDense-v3` | `FetchPickAndPlaceDense` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchPickAndPlaceDict-v3` | `FetchPickAndPlaceDict` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchPush-v3` | `FetchPush` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchPushDense-v3` | `FetchPushDense` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchPushDict-v3` | `FetchPushDict` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchReach-v3` | `FetchReach` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchReachDense-v3` | `FetchReachDense` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchReachDict-v3` | `FetchReachDict` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchSlide-v3` | `FetchSlide` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchSlideDense-v3` | `FetchSlideDense` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FetchSlideDict-v3` | `FetchSlideDict` | S | `CPU_UNMEASURED` | UNMEASURED | `CPU_BACKEND_MISSING`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/FingerDMControl-v0` | `DMCFingerTurnHard` | S | `CPU_REGRESSION` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FAST` | MEASURED | `BATCH_SCALES`, `HOST_BOUND`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/HopperDMControl-v0` | `DMCHopperHop` | S | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_EXCEPTIONAL` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/HumanoidDMControl-v0` | `DMCHumanoidStand` | S | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FAST` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/ManipulatorDMControl-v0` | `DMCManipulatorBringBall` | S | `CPU_REGRESSION` | MEASURED | — | `GPU_FAST` | MEASURED | `HOST_BOUND`, `HOST_FROM_PROSE` |
| `swm/MountainCarContinuousControl-v0` | `MountainCarContinuous` | R | `CPU_PARITY` | MEASURED | `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FASTER` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/MountainCarControl-v0` | `MountainCar` | R | `CPU_PARITY` | MEASURED | `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FASTER` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/OGBCube-v0` | `OGBCube` | S | `CPU_MISSING` | MISSING-UNSUPPORTED_BY_DESIGN | `RAISES_BY_DESIGN` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL` |
| `swm/OGBMaze-v0` | `PointmazeLargeNavigate` | S | `CPU_FAST` | MEASURED | `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_EXCEPTIONAL` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/OGBScene-v0` | `OGBScene` | S | `CPU_MISSING` | MISSING-UNSUPPORTED_BY_DESIGN | `RAISES_BY_DESIGN` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL` |
| `swm/PFRocketLanding-v0` | `PFRocketLanding` | S | `CPU_MISSING` | MISSING-UNSUPPORTED_BY_DESIGN | `RAISES_BY_DESIGN` | `GPU_PARITY` | MEASURED | `BATCH_SCALES_FAIL` |
| `swm/PendulumControl-v1` | `Pendulum` | R | `CPU_PARITY` | MEASURED | `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FAST` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/PendulumDMControl-v0` | `DMCPendulumSwingup` | S | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FAST` | MEASURED | `BATCH_SCALES`, `HOST_BOUND`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/Piecewise-v0` | `Piecewise` | R | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FASTER` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/PushT-Discrete-v1` | `PushT` | R | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_EXCEPTIONAL` | MEASURED | `BATCH_SCALES`, `NO_REFERENCE`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/PushT-v1` | `PushT` | R | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_EXCEPTIONAL` | MEASURED | `BATCH_SCALES`, `NO_REFERENCE`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/QuadrupedDMControl-v0` | `DMCQuadrupedWalk` | S | `CPU_UNMEASURED` | UNMEASURED | `LOAD_QUARANTINED` | `GPU_EXCEPTIONAL` | MEASURED | `BATCH_SCALES`, `CENSUS_ENTRY_MISSING` |
| `swm/ReacherDMControl-v0` | `DMCReacherHard` | S | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FAST` | MEASURED | `BATCH_SCALES`, `HOST_BOUND`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/SimplePointMaze-v0` | `SimplePointMaze` | R | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FAST` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/TwoRoom-v1` | `TwoRoom` | R | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_FASTER` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |
| `swm/WalkerDMControl-v0` | `DMCWalkerWalk` | S | `CPU_FASTER` | MEASURED | `BELOW_G4`, `TIER_COPIED_FROM_GENERATED_TABLE` | `GPU_EXCEPTIONAL` | MEASURED | `BATCH_SCALES`, `TIER_COPIED_FROM_GENERATED_TABLE` |

### The numbers under the CPU tiers

`S_cpu` = our f/s / the official renderer's f/s, both at **1 thread on the same host**. `cores_cpu` = our f/s / one core of q90 in-memory decode of this env's own frame. `CPU_FAST` is `S_cpu >= 5` **and** `cores_cpu >= 1`.

| SWM id | CPU tier | S_cpu | cores_cpu | our f/s | official f/s | 1-core decode f/s | batch | reference | host / job |
|---|---|---|---|---|---|---|---|---|---|
| `swm/AcrobotControl-v1` | `CPU_REGRESSION` | 0.96 | — | 382.16 | 399.16 | — | 64 | OFFICIAL_SAME_HOST | node1829.oscar.ccv.brown.edu / — |
| `swm/AcrobotDMControl-v0` | `CPU_FASTER` | 8.28 | 0.047 | 311.03 | 37.57 | 6,684 | 64 | OFFICIAL_SAME_HOST | node2317.oscar.ccv.brown.edu / — |
| `swm/BallInCupDMControl-v0` | `CPU_FASTER` | 3.61 | 0.007 | 55.19 | 15.30 | 7,453 | 64 | OFFICIAL_SAME_HOST | node2317.oscar.ccv.brown.edu / — |
| `swm/CartPoleControl-v1` | `CPU_PARITY` | 1.20 | — | 546.71 | 456.07 | — | 64 | OFFICIAL_SAME_HOST | node1829.oscar.ccv.brown.edu / — |
| `swm/CartpoleDMControl-v0` | `CPU_PARITY` | 1.43 | 0.036 | 777.11 | 543.62 | 21,768 | 64 | OFFICIAL_SAME_HOST | cis1b1zs2j3.ad.brown.edu / — |
| `swm/CheetahDMControl-v0` | `CPU_FASTER` | 4.93 | 0.017 | 115.56 | 23.45 | 6,831 | 64 | OFFICIAL_SAME_HOST | node2333.oscar.ccv.brown.edu / — |
| `swm/CraftaxClassicPixels-v1` | `CPU_FASTER` | 2.80 | 0.263 | 1,151 | 213.44 | 4,367 | 64 | OFFICIAL_SPLIT_JOB | node2356 / node2335 / 6395527 (throughput), 6409933 (reference) |
| `swm/CraftaxPixels-v1` | `CPU_FASTER` | 2.40 | 0.342 | 1,469 | 613.40 | 4,294 | 64 | OFFICIAL_SAME_HOST | node2339.oscar.ccv.brown.edu / 6409963 |
| `swm/FingerDMControl-v0` | `CPU_REGRESSION` | 0.28 | 0.006 | 118.97 | 419.11 | 20,350 | 16 | OFFICIAL_SAME_HOST | cis1b1zs2j3.ad.brown.edu / — |
| `swm/HopperDMControl-v0` | `CPU_FASTER` | 6.86 | 0.028 | 187.77 | 27.39 | 6,758 | 256 | OFFICIAL_SAME_HOST | node2333.oscar.ccv.brown.edu / — |
| `swm/HumanoidDMControl-v0` | `CPU_FASTER` | 4.22 | 0.006 | 45.82 | 10.86 | 7,551 | 64 | OFFICIAL_SAME_HOST | node2333.oscar.ccv.brown.edu / — |
| `swm/ManipulatorDMControl-v0` | `CPU_REGRESSION` | 0.70 | 0.007 | 15.50 | 22.23 | 2,172 | 1 | OFFICIAL_SAME_HOST | node2315.oscar.ccv.brown.edu / 6396735 |
| `swm/MountainCarContinuousControl-v0` | `CPU_PARITY` | 1.15 | — | 473.60 | 410.99 | — | 64 | OFFICIAL_SAME_HOST | node1829.oscar.ccv.brown.edu / — |
| `swm/MountainCarControl-v0` | `CPU_PARITY` | 1.12 | — | 465.71 | 415.60 | — | 256 | OFFICIAL_SAME_HOST | node1829.oscar.ccv.brown.edu / — |
| `swm/OGBMaze-v0` | `CPU_FAST` | 249.96 | 2.152 | 5,535 | 22.14 | 2,573 | 256 | OFFICIAL_SAME_HOST | node1946.oscar.ccv.brown.edu / — |
| `swm/PendulumControl-v1` | `CPU_PARITY` | 1.96 | — | 503.57 | 256.35 | — | 64 | OFFICIAL_SAME_HOST | node1829.oscar.ccv.brown.edu / — |
| `swm/PendulumDMControl-v0` | `CPU_FASTER` | 5.60 | 0.033 | 210.42 | 37.59 | 6,315 | 64 | OFFICIAL_SAME_HOST | node2317.oscar.ccv.brown.edu / — |
| `swm/Piecewise-v0` | `CPU_FASTER` | 3.21 | 0.372 | 1,243 | 387.35 | 3,340 | 16 | OFFICIAL_SAME_HOST | node2334.oscar.ccv.brown.edu / — |
| `swm/PushT-Discrete-v1` | `CPU_FASTER` | 5.51 | 0.245 | 1,981 | 359.54 | 8,092 | 1024 | OFFICIAL_SAME_HOST | node2334.oscar.ccv.brown.edu / — |
| `swm/PushT-v1` | `CPU_FASTER` | 5.51 | 0.245 | 1,981 | 359.54 | 8,092 | 1024 | OFFICIAL_SAME_HOST | node2334.oscar.ccv.brown.edu / — |
| `swm/ReacherDMControl-v0` | `CPU_FASTER` | 3.40 | 0.018 | 121.33 | 35.66 | 6,769 | 16 | OFFICIAL_SAME_HOST | node2317.oscar.ccv.brown.edu / — |
| `swm/SimplePointMaze-v0` | `CPU_FASTER` | 4.91 | 0.216 | 241.53 | 49.22 | 1,118 | 16 | OFFICIAL_SAME_HOST | node2334.oscar.ccv.brown.edu / — |
| `swm/TwoRoom-v1` | `CPU_FASTER` | 4.32 | 0.411 | 1,359 | 314.96 | 3,309 | 16 | OFFICIAL_SAME_HOST | node2334.oscar.ccv.brown.edu / — |
| `swm/WalkerDMControl-v0` | `CPU_FASTER` | 2.52 | 0.008 | 63.08 | 25.03 | 7,572 | 16 | OFFICIAL_SAME_HOST | node2333.oscar.ccv.brown.edu / — |

### The numbers under the GPU tiers

`R` = our f/s at the certified batch / the official renderer's f/s on the same host. `cores` = our f/s / one core of q90 decode. Class S (3-D scene) certifies `GPU_FAST` at `R >= 4` and `cores >= 1`; class R (2-D raster/tile) at `R >= 500` and `cores >= 1`. `BATCH_SCALES` is `fps(4096) / fps(1)` against a bar of 20 and is **required for any tier above `GPU_PARITY`**. `cpu_frac >= 1.5` makes the row `HOST_BOUND`: it is then a measurement of that node's processor under that node's load and may not be quoted as a device capability.

| SWM id | GPU tier | R | cores | fps(1) | fps(best) | BATCH_SCALES | NO_LARGE_BATCH_REGR. | cpu_frac | reference | host / job |
|---|---|---|---|---|---|---|---|---|---|---|
| `swm/AcrobotControl-v1` | `GPU_FASTER` | 422.06 | 121.93 | 88.43 | 160,041 | 1,810 | 7.675 | 0.992 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/AcrobotDMControl-v0` | `GPU_FAST` | 93.00 | 5.13 | 169.10 | 36,066 | 213.28 | 1.600 | 2.134 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/BallInCupDMControl-v0` | `GPU_FAST` | 28.62 | 1.39 | 180.75 | 10,112 | 55.95 | 1.174 | 0.993 | OFFICIAL_SAME_HOST | gpu2105.oscar.ccv.brown.edu / — |
| `swm/CartPoleControl-v1` | `GPU_FASTER` | 392.96 | 167.68 | 93.49 | 185,798 | 1,987 | 8.308 | 0.993 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/CartpoleDMControl-v0` | `GPU_FAST` | 139.35 | 7.72 | 168.19 | 54,132 | 307.89 | 1.779 | 1.509 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/CheetahDMControl-v0` | `GPU_EXCEPTIONAL` | 255.29 | 11.84 | 1,331 | 94,570 | 71.04 | 1.028 | 0.992 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/CraftaxClassicPixels-v1` | `GPU_FASTER` | 315.98 | 15.40 | — | 46,187 | 238.00 | 1.093 | — | OFFICIAL_SPLIT_JOB | gpu2107 / gpu2109 / 6395847 (throughput), 6409934 (reference) |
| `swm/CraftaxPixels-v1` | `GPU_FASTER` | 100.96 | 21.82 | — | 94,344 | — | — | — | OFFICIAL_SAME_HOST | gpu2109.oscar.ccv.brown.edu / 6410052 |
| `swm/FetchPickAndPlace-v3` | `GPU_PARITY` | 5.82 | 4.37 | 622.25 | 5,159 | 8.29 | 1.006 | 0.995 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchPickAndPlaceDense-v3` | `GPU_PARITY` | 6.06 | 4.37 | 630.14 | 5,157 | 8.18 | 1.013 | 0.994 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchPickAndPlaceDict-v3` | `GPU_PARITY` | 5.71 | 5.30 | 637.49 | 5,124 | 8.04 | 1.013 | 0.994 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchPush-v3` | `GPU_PARITY` | 6.41 | 4.40 | 616.71 | 5,223 | 8.42 | 1.006 | 0.995 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchPushDense-v3` | `GPU_PARITY` | 6.08 | 5.30 | 633.14 | 5,188 | 8.19 | 1.013 | 0.995 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchPushDict-v3` | `GPU_PARITY` | 6.18 | 5.27 | 638.39 | 5,185 | 8.12 | 1.013 | 0.995 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchReach-v3` | `GPU_PARITY` | 6.19 | 4.16 | 631.00 | 5,181 | 8.21 | 1.007 | 0.995 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchReachDense-v3` | `GPU_PARITY` | 5.43 | 5.14 | 663.54 | 5,177 | 7.80 | 1.012 | 0.995 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchReachDict-v3` | `GPU_PARITY` | 5.90 | 5.81 | 663.05 | 5,176 | 7.80 | 1.011 | 0.991 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchSlide-v3` | `GPU_PARITY` | 6.94 | 5.44 | 628.52 | 5,289 | 8.37 | 1.007 | 0.995 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchSlideDense-v3` | `GPU_PARITY` | 6.08 | 5.58 | 637.55 | 5,252 | 8.24 | 1.013 | 0.995 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FetchSlideDict-v3` | `GPU_PARITY` | 5.97 | 5.42 | 642.38 | 5,257 | 8.18 | 1.014 | 0.995 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/FingerDMControl-v0` | `GPU_FAST` | 81.75 | 4.46 | 186.15 | 30,451 | 163.59 | 1.400 | 1.769 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/HopperDMControl-v0` | `GPU_EXCEPTIONAL` | 259.08 | 14.25 | 1,340 | 94,822 | 70.75 | 1.033 | 0.992 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/HumanoidDMControl-v0` | `GPU_FAST` | 231.37 | 9.46 | 1,154 | 79,097 | 68.56 | 1.037 | 0.991 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/ManipulatorDMControl-v0` | `GPU_FAST` | 32.04 | 3.72 | 136.97 | 8,396 | 61.30 | 1.017 | 1.603 | OFFICIAL_SAME_HOST | — / — |
| `swm/MountainCarContinuousControl-v0` | `GPU_FASTER` | 307.78 | 129.79 | 84.14 | 143,708 | 1,708 | 7.264 | 0.993 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/MountainCarControl-v0` | `GPU_FASTER` | 308.43 | 135.04 | 83.94 | 143,876 | 1,714 | 7.308 | 0.993 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/OGBCube-v0` | `GPU_PARITY` | 3.20 | 1.88 | 1,162 | 6,230 | 5.36 | — | — | OFFICIAL_SAME_HOST | gpu2109.oscar.ccv.brown.edu / 6407605 |
| `swm/OGBMaze-v0` | `GPU_EXCEPTIONAL` | 120.84 | 58.35 | 256.09 | 196,982 | 769.18 | 3.280 | 0.993 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/OGBScene-v0` | `GPU_PARITY` | 3.62 | 2.05 | 832.86 | 5,412 | 6.50 | — | — | OFFICIAL_SAME_HOST | gpu2109.oscar.ccv.brown.edu / 6407605 |
| `swm/PFRocketLanding-v0` | `GPU_PARITY` | 241.51 | 2.61 | 238.14 | 3,374 | 14.17 | 1.253 | 1.027 | OFFICIAL_SAME_HOST | gpu2106.oscar.ccv.brown.edu / 6410105 |
| `swm/PendulumControl-v1` | `GPU_FAST` | 655.81 | 129.83 | 88.26 | 174,171 | 1,973 | 8.330 | 0.993 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/PendulumDMControl-v0` | `GPU_FAST` | 119.00 | 5.57 | 145.39 | 44,897 | 260.75 | 1.497 | 1.697 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/Piecewise-v0` | `GPU_FASTER` | 434.87 | 93.08 | 1,198 | 292,173 | 243.79 | 1.404 | 0.992 | OFFICIAL_SAME_HOST | gpu2105.oscar.ccv.brown.edu / — |
| `swm/PushT-Discrete-v1` | `GPU_EXCEPTIONAL` | — | 112.82 | 404.54 | 843,711 | 2,086 | 11.136 | 0.993 | NO_REFERENCE | gpu2101.oscar.ccv.brown.edu / — |
| `swm/PushT-v1` | `GPU_EXCEPTIONAL` | — | 112.82 | 404.54 | 843,711 | 2,086 | 11.136 | 0.993 | NO_REFERENCE | gpu2101.oscar.ccv.brown.edu / — |
| `swm/QuadrupedDMControl-v0` | `GPU_EXCEPTIONAL` | 129.94 | 13.61 | — | 44,374 | 73.24 | 1.126 | 0.995 | OFFICIAL_SAME_HOST | gpu2109.oscar.ccv.brown.edu / — |
| `swm/ReacherDMControl-v0` | `GPU_FAST` | 71.42 | 3.52 | 142.88 | 25,895 | 181.24 | 1.150 | 1.754 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |
| `swm/SimplePointMaze-v0` | `GPU_FAST` | 711.78 | 31.52 | 209.59 | 42,142 | 201.07 | 1.899 | 0.994 | OFFICIAL_SAME_HOST | gpu2101.oscar.ccv.brown.edu / — |
| `swm/TwoRoom-v1` | `GPU_FASTER` | 182.82 | 29.57 | 653.72 | 95,691 | 146.38 | 1.101 | 0.994 | OFFICIAL_SAME_HOST | gpu2105.oscar.ccv.brown.edu / — |
| `swm/WalkerDMControl-v0` | `GPU_EXCEPTIONAL` | 241.78 | 11.50 | 1,286 | 89,310 | 69.24 | 1.025 | 0.992 | OFFICIAL_SAME_HOST | gpu2104.oscar.ccv.brown.edu / — |

### The rows that are not measured, and why

| SWM id | device | tier | status | kind | why, from the record |
|---|---|---|---|---|---|
| `swm/CraftaxClassicSymbolic-v1` | CPU | `n/a` | UNSUPPORTED_BY_DESIGN | — | this id produces no frame by design (the upstream env is symbolic), so there is no renderer to tier; the live probe recorded the raise verbatim |
| `swm/CraftaxClassicSymbolic-v1` | GPU | `n/a` | UNSUPPORTED_BY_DESIGN | — | this id produces no frame by design (the upstream env is symbolic), so there is no renderer to tier; the live probe recorded the raise verbatim |
| `swm/CraftaxSymbolic-v1` | CPU | `n/a` | UNSUPPORTED_BY_DESIGN | — | this id produces no frame by design (the upstream env is symbolic), so there is no renderer to tier; the live probe recorded the raise verbatim |
| `swm/CraftaxSymbolic-v1` | GPU | `n/a` | UNSUPPORTED_BY_DESIGN | — | this id produces no frame by design (the upstream env is symbolic), so there is no renderer to tier; the live probe recorded the raise verbatim |
| `swm/FetchPickAndPlace-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchPickAndPlaceDense-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchPickAndPlaceDict-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchPush-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchPushDense-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchPushDict-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchReach-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchReachDense-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchReachDict-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchSlide-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchSlideDense-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/FetchSlideDict-v3` | CPU | `CPU_UNMEASURED` | UNMEASURED | backend_missing | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `swm/OGBCube-v0` | CPU | `CPU_MISSING` | MISSING-UNSUPPORTED_BY_DESIGN | raises_by_design | OGBCube has no CPU backend (CPU3D-OGB-1): this scene is 193,113 visible triangles at a median projected edge of ~0.7 px, and the certification CPU bar leaves under a nanosecond per triangle against a 24 ns/triangle vertex-setup floor -- 9.9-12.9x short before a single pixel is filled. See examples/r |
| `swm/OGBScene-v0` | CPU | `CPU_MISSING` | MISSING-UNSUPPORTED_BY_DESIGN | raises_by_design | OGBScene has no CPU backend (CPU3D-OGB-1): this scene is 196,633 visible triangles at a median projected edge of ~0.7 px, and the certification CPU bar leaves under a nanosecond per triangle against a 24 ns/triangle vertex-setup floor -- 9.9-12.9x short before a single pixel is filled. See examples/ |
| `swm/PFRocketLanding-v0` | CPU | `CPU_MISSING` | MISSING-UNSUPPORTED_BY_DESIGN | raises_by_design | device='cpu' raises honestly. Measured feasibility: a naive single-thread torch rasteriser draws 486 triangles at 480 px at 1.92 f/s = 0.056x the official 34.25 f/s (S_cpu bar 171) and 0.0016 cores (bar 1,191) - both bars missed by two to three orders. A shared CPU triangle backend is deferred to St |
| `swm/QuadrupedDMControl-v0` | CPU | `CPU_UNMEASURED` | UNMEASURED | load_quarantined | 73 polls, load1 34.06-43.59, rule: certify_shard_cpu3d --max-load 12; waited 6 h at 5-min polls |

## `BATCH_SCALES` — the Rocket and Fetch edge case

**Sami's ruling, applied: the threshold is not changed in Step 8.** Both cells keep the classification the frozen bar produces. What follows explains the mechanism and, separately and clearly labelled, proposes a taxonomy for Step 9 to consider.

The bar was frozen in Step 2 as a **batching detector**, not a throughput bar: a renderer that loops over the batch in Python scores about 1x, and the bar (20x) was set below the observed minimum across 129 measured rows (22.8x), so it should fail only a renderer that is not actually batching. The metric is a **ratio to `fps(1)`**, and that is the whole difficulty: it measures how much a cell *improves* from a batch of one, not how fast it ends up. A cell whose **B=1** number is already high — because its per-call overhead is small relative to its frame, or because its scene is uploaded once and reused — has little headroom left to show, and can miss the ratio while batching perfectly well in absolute terms.

| cell | fps(1) | fps at the certified batch | BATCH_SCALES | bar | R | cores | resulting tier |
|---|---|---|---|---|---|---|---|
| `PFRocketLanding` | 238.14 | 3,374 (B=4096) | **14.17x** | 20x | 241.51 | 2.61 | `GPU_PARITY` [`BATCH_SCALES_FAIL`] |
| `FetchPickAndPlace` (and its 11 siblings) | 622.25 | 5,159 (B=4096) | **8.29x** | 20x | 5.82 | 4.37 | `GPU_PARITY` [`BATCH_SCALES_FAIL`] |
| `OGBCube` | 1,162 | 6,230 | **5.36x** | 20x | 3.20 | 1.88 | `GPU_PARITY` [`BATCH_SCALES_FAIL`] |
| `OGBScene` | 832.86 | 5,412 | **6.50x** | 20x | 3.62 | 2.05 | `GPU_PARITY` [`BATCH_SCALES_FAIL`] |

**Rocket.** 238.14 f/s at B=1 is unusually high for a class-S cell — the nvdiffrast raster backend draws 486 triangles at 480 px and the scene is rebuilt per call from a compact state, so a single frame is nearly free. Scaling to 3,374 f/s is a factor of 14.17, under the bar, even though the cell renders 241.51x the official PyBullet TinyRenderer on the same node and clears `NO_LARGE_BATCH_REGRESSION` at 1.253. The cap is the correct application of the frozen rule (certification.md §6.7) and it is kept.

**Fetch.** The same shape for a different reason: the twelve Fetch ids share one mjwarp scene whose model upload dominates at small batch, so B=1 already reaches 622.25 f/s at 480 px and the curve is flat from B=256 (5,129 f/s) to B=4096 (5,159 f/s) — 8.29x overall. The flatness that fails the ratio is the same flatness that passes `NO_LARGE_BATCH_REGRESSION` (1.006). Fetch is therefore `GPU_PARITY`, although on the class-S ladder R = 5.82 (bar 4) and cores = 4.37 (bar 1) would otherwise make it `GPU_FAST`. Classification retained.

**PROPOSAL, for Step 9 — not adopted, not applied to any row on this page.** The ratio conflates two things a certification could separate:

1. *does this renderer batch at all* — the question the bar was written for. A per-frame Python loop is detectable far more directly, e.g. `fps(64) >= 4 x fps(1)` **or** `us_per_frame(4096) <= 0.5 x us_per_frame(1)`, which a looping renderer fails and a flat, already-saturated renderer passes.
2. *does this renderer reach its device's throughput* — which is what `R` and `cores` already measure, on the batch the row certifies at.

A Step-9 taxonomy could keep the 20x bar as the default and add one narrow escape: a cell may hold a tier above `GPU_PARITY` with `BATCH_SCALES < 20` if it demonstrates saturation instead — `fps(best) >= 0.9 x` the best over the whole sweep at some batch <= 256, and `NO_LARGE_BATCH_REGRESSION` holds — recorded as a distinct qualifier such as `BATCH_SATURATES_EARLY` so nothing is hidden. That is a **proposal**; adopting it is a change to a frozen Step-2 threshold and needs Sami's explicit approval, a re-judgement of every row measured under the old bar, and a re-freeze. Step 8 does none of that.

## Quadruped CPU, and the three CPU-unsupported cells

**`swm/QuadrupedDMControl-v0` CPU stays `CPU_UNMEASURED` [`LOAD_QUARANTINED`].** The only CPU tier run for this cell is `examples/results/step7/dmc-quadruped/phase1/tier_cpu/QUARANTINED_LOAD_node2301_6398084.json`: 73 polls, load1 34.06-43.59, rule: certify_shard_cpu3d --max-load 12; waited 6 h at 5-min polls. No trustworthy quiet-node measurement exists in the committed artifacts, and Step 8 runs no benchmark, so nothing is filled in. Local M5 numbers exist in the Step-7 workstream tree and are explicitly **not** certification numbers (they are neither the certification host nor a 1-thread batch allocation). The CPU path itself runs — W4 recorded 25/25 CPU identity — so this is a missing *measurement*, not a missing *backend*.

*Is a new run needed?* Only to convert one `UNMEASURED` into a tier. It is not needed for certification truthfulness, it does not block closure, and it costs one `batch`/`normal` allocation that waits for the node load to drop below the pre-registered guard of 12. Recommendation: schedule it as a single quiet-node job when Oscar is quiet, as deferred work, not as Step-8 scope.

**`CPU_MISSING [RAISES_BY_DESIGN]` is kept on `swm/OGBCube-v0`, `swm/OGBScene-v0` and `swm/PFRocketLanding-v0`.** These three raise on `device="cpu"` rather than falling back to CUDA, which is the honest behaviour the backend policy requires, and the status word says so: `MISSING-UNSUPPORTED_BY_DESIGN`, never a plain failure. Each carries a measured reason — for the OGB cells a triangle budget short of the CPU bar by 10-70x depending on the rate assumed, for Rocket a single-thread torch rasteriser at 0.056x the official renderer — and each names a shared CPU triangle backend as deferred work. No projection is offered in place of a tier.

## Production pipeline — stored JPEG against compact state

Source: `examples/results/step6/lance/comparison2.json` / `.md` (the Step-6 `lance` workstream), 36 registered configs, each at **its own certified resolution**, JPEG **q95**, chunk 256, using `examples/bench_lance.py`'s own timing functions verbatim. This is a different claim from every tier above and is never substituted for one.

| arm | what it measures | result over the 36 rows |
|---|---|---|
| **GPU, end to end** | ours: compact state -> render -> pixels **on device**; theirs: stored JPEG -> 8-worker load+decode -> H2D | **30 of 36** rows win; `gpu_pipeline_ratio` median **1.44**, range 0.18–46.21 |
| **GPU, against nvJPEG** | the same, with the stored side decoding on the GPU | **35 of 36** win (only `DMCBallInCupCatch` does not, at 0.72) |
| **CPU, 8-way against 8 Lance workers** | both sides end in **host** memory | **1 of 24** measurable rows wins (`CraftaxClassic`, 6.96); median **0.11** |
| **frozen conservative column** | our f/s / one core of q90 decode (`cores_cpu >= 1`) | 1 of 36 clears it — the same `CraftaxClassic` |

The shape of the answer: **on a GPU, regenerating from state beats reading stored frames for most of the corpus; on a CPU, it does not.** The GPU win is largest exactly where the frame is big and the state is small (`Pendulum` 46x, `CartPole` 44x, `Acrobot` 41x) and reverses on the small-frame DMC cells (84x84), where the stored path decodes a 2-3 kB JPEG faster than a ray-cast can draw the scene.

**Provenance.** CPU arms: `node2317` (Xeon Platinum 8268, 8 of 48 cores, load ~35) job 6366909, and the instrumented re-run on `node1956` job 6367705. GPU arms: `gpu2104` (AMD EPYC 7532 + RTX 3090) jobs 6366975, 6367697 and 6368790. Per-cell artifact and batch for every row are in T5 of `comparison2.md`.

**Limits — read these before quoting any of it.** They are the Step-6 workstream's own and they are not softened here:

- **No cold storage anywhere.** The Lance table is written in-process moments before it is read, on every arm; nothing here measures a cold NFS or GPFS read, and an unprivileged job cannot drop caches. 512 frames per config is not a corpus, and Lance's behaviour at TB scale is untested.
- **The GPU denominator is CPU-bound on a weaker CPU.** The loader+H2D arm ran on an EPYC 7532 at load 54 while the CPU Lance arms ran on a faster Xeon; the stored path is decoder-bound, so `gpu_pipeline_ratio` **flatters our renderer** by roughly the ratio of those two processors.
- **Asymmetries, sized in the source:** q95 vs the q90 bar (~4% in f/s, not the ~20% the byte counts suggest); 8 cores vs the 1-thread certification budget; output in host memory vs on device (not charged to the stored path anywhere); min vs median reduction statistics; warm page cache on every stored arm.
- **The 8-way CPU number is supplemental** per certification.md §5.3 and never certifies.
- `DMCCheetahRun`, `DMCHopperHop` and `DMCWalkerWalk` are **visual-tier** cells; their wins are wins at the visual tier, not parity claims.

## Cross-checks

**Against the Step-7 record** (`examples/results/step7/closure2/coverage_matrix.json`): every one of the 42 ids' CPU and GPU tiers on this page is the tier that record carries, and for every row where the underlying artifact can be re-judged by the frozen evaluator (`scratch/step3/certify.py`, Step 2's ladder) the re-judgement **reproduces the copied tier**. Recorded disagreements: **0**.

The four cells Step 7 added — `DMCManipulatorBringBall`, `DMCQuadrupedWalk`, `OGBCube` and `OGBScene` — could not be re-judged at all before Workstream E merged: the Step-3 census had no entry for them, so the frozen evaluator could not assign a backend class and returned `unknown` instead of a tier (the Step-7 carryover "Step 8 must regenerate the census"). E's regenerated census (88 -> 100 entries) assigns all four **class S**, and the re-judgement now reproduces their Step-7 tiers exactly. That carryover is closed, and every measured row on this page is re-judgeable from committed evidence in this tree.

**Against the Step-8 support matrix** (`examples/results/step8/support_matrix.json`, sha256 `7adb020c27099741…`, Workstream E, merged at `4210793`): 42 of 42 ids compared, **0 tier disagreements**. The two pages reach the tier column by different code — the matrix through its own generator, this page through the underlying measurement artifacts — and they agree on every id.

What does differ is **vocabulary on rows that have no tier** (7 status words, 4 placeholder strings), and one of the two is worth acting on:

| id(s) | this page | support matrix | reading |
|---|---|---|---|
| `swm/CraftaxClassicSymbolic-v1`, `swm/CraftaxSymbolic-v1` | `UNSUPPORTED_BY_DESIGN`, tier `n/a` | `UNMEASURED`, tier `unknown` | **the symbolic ids should not read `UNMEASURED`.** They produce no frame by design, so there is nothing to measure; `UNMEASURED` invites the reader to think a run is owed. Reported to the certification workstream; not changed here, because that file is theirs. |
| `swm/OGBCube-v0`, `swm/OGBScene-v0`, `swm/PFRocketLanding-v0` (CPU) | `MISSING-UNSUPPORTED_BY_DESIGN` | `UNSUPPORTED_BY_DESIGN` | same claim, different spelling. This page uses the three-word vocabulary the Step-8 closure criteria name, which keeps "the device raises by design" distinct from "this id renders nothing at all". |

**Superseded measurements.** Kept rather than deleted, because a superseded number is still a number somebody may find:

| cell | device | superseded | by | why |
|---|---|---|---|---|
| `OGBCube` | gpu | `examples/results/step7/ogb-manipulation/m4/m4_tiers.json` | `examples/results/step7/ogb-manipulation/m3/fidelity/remeasure.json` | m4 measured the renderer before the W5 closure follow-on (PlateRig, float32 round-trip states, role-split reference cache); it reads R 2.995 / cores 1.714 / BATCH_SCALES 5.02 against the re-measured 3.201 / 1.875 / 5.362. Same tier, different digits |
| `OGBScene` | gpu | `examples/results/step7/ogb-manipulation/m4/m4_tiers.json` | `examples/results/step7/ogb-manipulation/m3/fidelity/remeasure.json` | as OGBCube |
| `PFRocketLanding` | gpu | `examples/results/step7/rocket-landing/tier/gpu2109.json (job 6406061) and examples/results/step7/rocket-landing/tier/job6408817/gpu2109.json` | `examples/results/step7/rocket-landing/tier/job6410105/gpu2106.json` | three tier jobs on the same code path; the earliest read R 248.38 / BATCH_SCALES 14.56 and the last R 241.51 / 14.17. All three fail the 20x bar and all three land GPU_PARITY |
| `CraftaxClassic` | cpu | `docs/research/step6_optimization.md (generated Step-6 tier table: CPU_EXCEPTIONAL, src `s8`)` | `examples/results/step7/closure2/declarations/ogbmanip.json` | the class carries performance_record_cpu, measured in Step 7 against the JITTED official renderer; it SUPERSEDES the Step-6 tilemap tier, which was measured on an observation that could not represent night and had no reference at all |
| `CraftaxClassic` | gpu | `docs/research/step6_optimization.md (generated Step-6 tier table: GPU_EXCEPTIONAL, src `s9`)` | `examples/results/step7/closure2/declarations/ogbmanip.json` | as the CPU side: performance_record_gpu supersedes the Step-6 NO_REFERENCE tier |

**Reference rule.** A ratio exists only where the official renderer was timed **on the same host**; otherwise the row is `NO_REFERENCE` and certifies on the cores ladder alone. Two rows are flagged `REFERENCE_SPLIT_JOB`: both Craftax Classic arms declare their throughput and their reference from *different jobs on different nodes* (CPU: 6395527 on node2356 and 6409933 on node2335; GPU: 6395847 on gpu2107 and 6409934 on gpu2109). The Step-7 workstream recorded this deliberately — the jitted reference needed a different venv — but it is a weaker provenance than the same-host rule asks for and the flag says so on the row. `DMCManipulatorBringBall` GPU is flagged `HOST_FROM_PROSE`: the committed economics artifact carries the three timings but not its own host or SLURM job, which are named only in the workstream's prose (`gpu2107`, job 6395157).

## Every artifact this page rests on

| artifact | sha256 | what it supplies |
|---|---|---|
| `docs/research/certification.md` | `32717032cb0c5abf…` | frozen Step-2 tier definitions (prose) |
| `docs/research/step6_optimization.md` | `f45702be5e558cc6…` | Step-6 generated tier tables + src legend |
| `examples/results/step3/certification.json` | `e821aff8ba140cff…` | tier evidence: DMCCartpoleBalance cpu; tier evidence: DMCFingerTurnHard cpu; tier evidence: FetchPickAndPlace cpu; tier evidence: FetchPickAndPlaceDense cpu; tier evidence: FetchPi |
| `examples/results/step6/cpu-3d/sweep/after/cpu_1thread_paired_after_node2333.json` | `606879cd1b273e3c…` | tier evidence: DMCCheetahRun cpu; tier evidence: DMCHopperHop cpu; tier evidence: DMCHumanoidStand cpu; tier evidence: DMCWalkerWalk cpu |
| `examples/results/step6/gfxdraw/sweep/cpu_1thread_paired_after_node1829.json` | `4718ae3d457eff9b…` | tier evidence: Acrobot cpu; tier evidence: CartPole cpu; tier evidence: MountainCar cpu; tier evidence: MountainCarContinuous cpu; tier evidence: Pendulum cpu |
| `examples/results/step6/lance/renderer_gpu_after_fetch311_gpu2104_6366975.json` | `d9ae21527938d143…` | tier evidence: FetchPickAndPlace gpu; tier evidence: FetchPickAndPlaceDense gpu; tier evidence: FetchPickAndPlaceDict gpu; tier evidence: FetchPush gpu; tier evidence: FetchPushDen |
| `examples/results/step6/lance/renderer_gpu_after_gpu2104_6366975.json` | `913f326ab59822c1…` | tier evidence: Acrobot gpu; tier evidence: CartPole gpu; tier evidence: DMCAcrobotSwingup gpu; tier evidence: DMCCartpoleBalance gpu; tier evidence: DMCCheetahRun gpu; tier evidenc |
| `examples/results/step6/raster2d/sweep/cpu_1thread_after_pusht.json` | `3626b0ed3fd6be13…` | tier evidence: PushT cpu |
| `examples/results/step6/raster2d/sweep/cpu_1thread_after_swm.json` | `404bf37b6e7e17fa…` | tier evidence: Piecewise cpu; tier evidence: SimplePointMaze cpu; tier evidence: TwoRoom cpu |
| `examples/results/step6/raster2d/sweep/order_check/gpu_after_first.json` | `b74eeaa63398f53e…` | tier evidence: PushT gpu; tier evidence: SimplePointMaze gpu |
| `examples/results/step6/raster2d/swmnav2b/sweep/gpu_after.json` | `c7d2a607ec4c4e0f…` | tier evidence: Piecewise gpu; tier evidence: TwoRoom gpu |
| `examples/results/step6/raycast3d/sweep/gpu_3090_after_branch.json` | `b86b31789e9a62ff…` | tier evidence: DMCBallInCupCatch gpu |
| `examples/results/step6/raycast3d/sweep/r6/cpu_1thread_after_branch.json` | `39a364282224c2ea…` | tier evidence: DMCAcrobotSwingup cpu; tier evidence: DMCBallInCupCatch cpu; tier evidence: DMCPendulumSwingup cpu; tier evidence: DMCReacherHard cpu |
| `examples/results/step6/raycast3d/sweep/r6_repeat/cpu_1thread_after_branch.json` | `532efc78b3c7c5ea…` | tier evidence: PointmazeLargeNavigate cpu |
| `examples/results/step7/closure2/coverage_matrix.json` | `6540eeab19050fbe…` | Step-7 record: per-id tiers, cells, pixel/symbolic |
| `examples/results/step7/closure2/declarations/ogbmanip.json` | `982546bd278cec7c…` | class declaration: CraftaxClassic cpu; class declaration: CraftaxClassic gpu; class declaration: CraftaxFull cpu; class declaration: CraftaxFull gpu |
| `examples/results/step7/dmc-manipulator/cputier/cpu_tier_branch_node2315.json` | `51bd3e2430e9ec94…` | tier evidence: DMCManipulatorBringBall cpu |
| `examples/results/step7/dmc-manipulator/economics/bench_vs_decode_se-step6-3090.json` | `5a03c7f10ebdf312…` | tier evidence: DMCManipulatorBringBall gpu |
| `examples/results/step7/dmc-quadruped/phase1/tier_cpu/QUARANTINED_LOAD_node2301_6398084.json` | `a4fe1d9c41a87e17…` | cpu quarantine record: DMCQuadrupedWalk |
| `examples/results/step7/dmc-quadruped/phase1/tier_gpu/tiers_B_branch_gpu2109_6398085.json` | `5a2df4aa5d3c8754…` | tier evidence: DMCQuadrupedWalk gpu |
| `examples/results/step7/ogb-manipulation/m3/fidelity/remeasure.json` | `f343dff01f08becb…` | cpu raise record: OGBCube; cpu raise record: OGBScene; tier evidence: OGBCube gpu; tier evidence: OGBScene gpu |
| `examples/results/step7/rocket-landing/tier/job6410105/gpu2106.json` | `155b94c16bfb3e4d…` | tier evidence: PFRocketLanding gpu |
| `examples/results/step8/support_matrix.json` | `7adb020c27099741…` | cross-check: Workstream E support matrix (tier columns) |
| `scratch/census/census_macos-probe.json` | `d9327a0952b19daa…` | backend-class census (Step-1 declarations) |
| `scratch/census/census_oscar-3090-fetch.json` | `55ace7358a566323…` | backend-class census (Step-1 declarations) |
| `scratch/census/census_step8-craftax-gpu.json` | `7df2868c51e381c0…` | backend-class census (Step-1 declarations) |
| `scratch/census/census_step8-craftax.json` | `fe0cb64cc79855cc…` | backend-class census (Step-1 declarations) |
| `scratch/census/census_step8-fetch311.json` | `541eaa07a5f56600…` | backend-class census (Step-1 declarations) |
| `scratch/census/census_step8-ogb.json` | `c8e0d2df6a94a8c3…` | backend-class census (Step-1 declarations) |
| `scratch/census/census_step8-rocket.json` | `3176155775403e3f…` | backend-class census (Step-1 declarations) |
| `scratch/census/census_step8-step5.json` | `969b1229e2cc4371…` | backend-class census (Step-1 declarations) |
| `scratch/census/census_step8-step6.json` | `644e82a6780f59f3…` | backend-class census (Step-1 declarations) |
| `scratch/census/census_step8-vis311.json` | `b2702549ed990dfb…` | backend-class census (Step-1 declarations) |
| `scratch/certification.json` | `7ba99f51090c432b…` | frozen Step-2 certification spec (ladders, bars, qualifiers) |

## Open items handed forward

| item | classification | owner |
|---|---|---|
| Quadruped CPU tier: one quiet-node `batch`/`normal` run under the load-12 guard would convert the last `CPU_UNMEASURED` on a supported CPU path | `DEFERRED_OPTIMIZATION` (measurement, not code) | Step 9 |
| A shared CPU triangle/span backend would give OGBCube, OGBScene and PFRocketLanding a real CPU tier instead of a by-design raise | `DEFERRED_OPTIMIZATION` | Step 9 |
| Fetch has no CPU backend at all (`CPU_BACKEND_MISSING` on all twelve ids) | `DOCUMENTED_LIMITATION` | Step 9 |
| `BATCH_SCALES` taxonomy proposal above | `STEP9_CLEANUP` (a threshold change needs Sami) | Step 9 |
| Step-3 census had no entry for the four Step-7 cells | **CLOSED** by Workstream E's regenerated census (88 -> 100), merged at `4210793` | — |
| The support matrix reports the two symbolic ids' tiers as `UNMEASURED` | `DOCUMENTED_LIMITATION` (reporting) | certification workstream |

