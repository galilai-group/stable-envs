# Oscar reconnection: completed evidence and new source campaigns

Checkpoint started at `40c39110a91609a2d881efcfdfa941cfd0aef6be` on `repair/render-config-fixes`, with a clean, pushed worktree. No production change or semantic promotion is made here. Matrix remains **39 VERIFIED / 3 INCOMPLETE**; 35 accepted non-Craftax registrations are in scope. Earlier evidence is preserved unchanged.

## Completed classic comparison

Array 6971892: all five tasks COMPLETED, exit 0. Each report matches its scheduler job/host and the frozen candidate; all 120 source comparison rows reproduce byte-for-byte through the strict collector. Parent Torch uses eight threads on CPU and one on CUDA; each source worker uses one. Every strategy has the same eight-core allocation. Timing includes public stepping, mandatory final State, selected rendering and output delivery, with six samples per cell. State histories are disabled.

CPU results below use B256, eight actions and all frames. Values are mean ± one sample SD; the ratio is ours divided by the faster of source serial/P8. All displayed MSEs are zero. These are bounded-workload results, not universal performance acceptance.

| Environment | Source serial FPS | Source P8 FPS | Ours FPS | Ours / best source |
| --- | ---: | ---: | ---: | ---: |

| acrobot | 96 ± 3 | 317 ± 85 | 955 ± 89 | 3.018× |

| cartpole | 340 ± 80 | 2,068 ± 81 | 1,771 ± 51 | 0.856× |

| mountain_car | 427 ± 9 | 1,344 ± 23 | 1,346 ± 67 | 1.002× |

| mountain_car_continuous | 461 ± 4 | 2,075 ± 12 | 1,658 ± 31 | 0.799× |

| pendulum | 96 ± 5 | 302 ± 68 | 1,605 ± 24 | 5.316× |


Full cells and SDs: [all-cells.csv](classic-verified/all-cells.csv). Classic source pygame has no GPU renderer; CUDA rows compare our GPU with source CPU rendering plus CUDA delivery. They must not be called native source-GPU speedups. Host/run differences prevent attributing changes from the historical one-thread campaign solely to threading.

## Retained optimization evidence

All seven classic isolated-cell tasks (6971656) are now joined with successful terminal accounting and cluster logs. Their summaries reproduce byte-for-byte. The continuous MountainCar CPU B1024/T8/all States/final frame snapshot candidate remains 0.8642× baseline and is **not promoted**. Other six ratios are 0.9925–1.0399.

Cube validator candidate 6970790_0 completed and passes both device gates: 12 component pairs and 24 online pairs. CUDA B64/T8/all frames improves 201.93 ± 10.34 → 634.31 ± 14.96 FPS; B128 improves 215.48 ± 6.14 → 772.78 ± 14.29 FPS. These are **previous-library-validator controls, not source speedups**. CPU rendered throughput remains approximately 4.4 FPS.

Scene 6970790_1 timed out. Its completed CPU report and incomplete CUDA report are preserved, not promoted. Retry 6976164 runs only CUDA with the exact same frozen source and new outputs. Its CPU/CUDA unit suite passed 88 tests and its public CUDA gate passed; the latest saved sweep has eight of 24 timing cells, so full sweep completion remains required. `scene-retry-progress/` is explicitly partial.

## New submissions

Submitter 6976159 completed successfully. Uploaded archive hashes are in `uploads.json`; exact launcher, submission output and job IDs are preserved under `submitter/` and `launch/`. Each array is throttled to three simultaneous RTX 3090 tasks on `3090-gcondo`. Held job 6794187_[30-41] was untouched.

| Campaign | Job | Scope |
| --- | --- | --- |
| Fetch | 6976160 | Twelve registrations; nine require independent CPU/CUDA image gates, PickAndPlace three no-frame only |
| OGBench source image gates | 6976161 | Maze/Cube/Scene; source P1/P2 and public images, full State/tasks, lifecycle and renderer nonmutation |
| DMC source comparison | 6976162 | Acrobot/Pendulum/Cartpole, source P1/P8 versus public adapter |
| Navigation source comparison | 6976163 | TwoRoom/Piecewise/SimplePointMaze, source P1/P8 versus native Torch |
| Scene validator CUDA retry | 6976164 | Candidate versus previous validator; no source-performance claim |


Nine Fetch rendered jobs failed their CPU image gates on the first nonempty selection: Reach MSE 0.001199–0.001351, Push 0.001318–0.001435, Slide 0.001456–0.001684. All exceed the per-frame 0.001 gate. Their timing did not proceed; failures are not filtered out. The three PickAndPlace jobs completed successfully and all six per-strategy/device reports per registration passed the strict collector; all online/component CSVs reproduce byte-for-byte. They measure only no-frame stepping because upstream render changes its continuation. Their FPS and MSE fields intentionally remain blank; see `pick-verified/`.

Fetch sampling diagnostic 6976210 failed on source import ordering before measurement. Probe 6976233 produced all six device reports, but two tasks failed during concurrent xvfb-run cleanup. Retry 6976275 used the established isolated-display launcher and then failed on diagnostic report-path handling after its CPU runs. All attempts are preserved. Final retry **6976285 completed all three tasks with exit 0**, producing six CPU/CUDA reports. It compares 64px raster, 2×/4× supersampling and 480px raster followed by source-style linear resize. All 192 measured frame errors were recomputed from the saved image arrays. See [fetch-sampling.csv](fetch-sampling.csv). Matching the source sampling path reduces MSE to roughly 0.00006–0.00019 in these two-step/four-lane probes, while even 4× supersampling retains some Slide errors above 0.001. These diagnostics have no throughput measurements and cannot promote registrations. Production renderers are unchanged.

OGBench Maze passed all 30 CPU cases, but its CUDA gate stopped when P1/P2 **source renderers** differed by one intensity level in 72 of 24,576 channels. That failure needs investigation; source transport/image equivalence is not waived. Cube/Scene source gates and DMC comparisons remain uncollected at this checkpoint.

Navigation TwoRoom and Piecewise completed. Their strict collectors reproduce 24 source-comparison rows and 12 component rows per registration. At B64/T8/all frames, CPU speedups versus best source are 1.736× and 1.436×, with normalized image MSE 1.40e-10 and 1.19e-10. CUDA-output ratios are 5.970× and 7.733× against source CPU plus CUDA delivery, not native GPU source. See `navigation-verified/`. SimplePointMaze remains pending collection.


This checkpoint does not close the overall online performance goal. Fresh reports and scheduler terminal evidence must be collected before updating these statuses or promoting any candidate.

The new [42-registration online coverage CSV](coverage/coverage.csv) separates seven bounded rendered comparisons, three no-frame comparisons, nine image-gate failures, the source-P1/P2 image discrepancy, and uncollected/missing campaigns. It does not replace the historical offline renderer table or imply universal renderer acceptance. Reproduce it in a new output directory using `make_coverage.py --evidence <this directory> --output <new directory>`.
