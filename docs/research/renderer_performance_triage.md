# Renderer performance triage — 2026-09-28

Analysis of the completed 1,000-frame campaign; **no new timing runs** and no
production changes. [Complete ranking](semantic_recovery_evidence/renderer_timing_triage_20260928/timing_ranking.csv).

## Priorities

1. **Full-State projection and validation: Craftax, Cube, Scene, Rocket.**
   Highest measured GPU public-call latency: full Craftax 81.49 ± 1.69 ms,
   Classic 42.31 ± 2.34, Cube 40.76 ± 1.31, Rocket 37.19 ± 5.99, Scene
   31.41 ± 0.50. Separate RenderState-only batch-1 probes measured Cube
   18.59 ms, Rocket 5.49 ms and Scene 7.45 ms. These probes used different
   input distributions; Scene also used a different GPU model. Subtraction is
   not a measured cost breakdown. Nevertheless, the code provides concrete
   reasons to profile this boundary: Cube and Rocket call exhaustive simulator
   validation when projecting State; Cube does per-leaf GPU finite checks and
   CPU copies. Craftax validates and decodes full State, then validates and
   decodes RenderState again. Its render boundary copies Torch tensors to
   NumPy, creates JAX inputs, renders lanes individually, transfers images to
   NumPy and then back to Torch. GPU Craftax was slower than CPU in these runs;
   a controlled same-host breakdown is needed before attributing the difference.
   Preserve all validation and nonmutation guarantees while reducing duplicate
   work and transfers; this does not require changing the shared State contract.

2. **BallInCup CPU is the largest observed avoidable CPU gap.** 210.67 ± 16.74
   ms versus source 57.69 ms, 3.65× observed time and 152.98 ms extra per frame.
   Both hosts use Xeon Platinum 8268 CPUs with one Torch thread. Finger is
   another candidate: 79.60 versus 41.03 ms (1.94×). Inspect primitive ray
   casting, tendon construction and shading separately before assigning cause.

3. **Simple classic-control rendering is unexpectedly expensive.** MountainCar
   Continuous CPU 21.19 versus source 3.75 ms (5.65×); discrete MountainCar
   21.31/6.21 (3.43×); Acrobot 25.58/7.81 (3.27×); CartPole 19.98/8.03
   (2.49×); Pendulum 13.80/5.98 (2.31×). Their small-batch GPU rendering also
   costs 8–16 ms; a GPU tensor does not establish a speedup over the source CPU
   renderer. We do not yet have broad batched throughput results for them.

4. **GPU fidelity throughput: Scene then Cube.** At batch 256, Scene's optional
   GL path is 1.214 ms/frame versus MJWarp mean 0.644, median 0.498. One
   baseline sample was 1.960 ms/frame; the other nine were about 0.494–0.507.
   Thus the prior +89% mean-based penalty is a noisy estimate: median fidelity
   cost is about 2.44× baseline. Keep all samples and repeat the paired probe
   before accepting a precise regression percentage. Cube is steadier:
   fidelity 0.550 versus baseline 0.332 ms/frame, 1.66×. Cube GL is 1.669,
   about 5.0× baseline on this same GPU. Rocket's corrected rasterization is
   not a major regression: 0.443 versus 0.436 ms/frame (+1.75%). These are
   prebuilt RenderState throughput tests, separate from priority 1.

5. **Fetch and OGBMaze have small-batch GPU gaps worth profiling next.** Fetch
   takes about 1.65–3.42 ms versus source 0.26–0.47 ms, usually on different
   GPU models. FetchPickAndPlaceDict uses L40S on both sides: 1.661 versus
   0.279 ms (5.94× observed), with reset-only coverage. Maze uses RTX A6000 on
   both sides: 6.651 versus 1.597 ms (4.17×). These remain different hosts and
   runs, so they are prioritization signals, not controlled speedup estimates.

## Absolute CPU cost versus excess over source

Scene and Cube have the slowest CPU rendering (350 and 310 ms/frame), but their
source CPU renderers are already similarly slow (361 and 286 ms). BallInCup's
153 ms excess is a clearer implementation target. Other CPU gaps include Rocket
140.40/92.54 ms (+47.86), Humanoid 120.46/86.78 (+33.68), and Quadruped
103.83/80.43 (+23.39). Fetch CPU results vary between siblings (some ratios near
1×, others near 2×); verify the discrepancy under controlled conditions before
claiming a family-wide regression.

## Limits and next measurement

All main-table timings are warm, batch 1, include full-State projection where
applicable, and exclude reset/step/get_state/set_state. The table does not
measure simulator speed, clone/replay cost or complete training throughput.
CPU device models match, but hosts and run conditions differ. GPU models often
differ. Matching a device model still does not make two runs controlled pairs.
Source Craftax's render call returns a cached observation; its apparent
hundreds/thousands-fold ratio is not a valid rendering comparison. It must be
benchmarked with the upstream state-to-pixels computation timed separately.

The next useful experiment is a same-host, same-device breakdown into (a) State
projection and validation, (b) data transfer/conversion, and (c) rasterization,
followed by end-to-end measurements at batches 1, 8, 64 and 256. Use identical
States, resolutions and appearance distributions, synchronized warmed timings,
and unchanged MSE/nonmutation gates. Start with full Craftax, Rocket, Cube and
BallInCup; add a simple control environment to expose fixed call overhead.
