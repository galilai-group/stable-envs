# Step 3 — the current implementation, measured

!!! info "Raw measurements, classified by the frozen Step 2 rules"
    Nothing here moved a threshold. Every label is computed by
    `scratch/step3/certify.py` from `scratch/certification.json`, which was frozen before any
    of these numbers existed; the evaluator contains no bound of its own and a unit test fails
    if one appears. Raw shards are under `examples/results/step3/`, the merged artifact is
    `merged.json`, and the labels are `certification.json` — every row below can be recomputed
    without rerunning a benchmark.

    **No renderer was changed.** Step 3 measures; a slow row is a result, not a defect to fix here.

## The conclusion

> **GPU rendering is broadly successful. The dominant unfinished engineering problem is CPU rendering — including a missing CPU backend for the whole mjwarp family.**

All 34 matched rows now carry a GPU tier and none is a regression. On the CPU not one row reaches `CPU_FAST`, 14 are outright regressions against the renderer they replace, every measured row loses to reading a stored frame off disk (`BELOW_G4`), and the 16 that are unmeasured are unmeasured because **no CPU renderer exists for them** rather than because a benchmark failed. `ENVIRONMENT_DONE` is therefore 0, and CPU is the reason for all 34.

## Hosts

| shard | device | host | CPU | threads | cores | notes |
|---|---|---|---|---|---|---|
| `m5_18thread.json` | cpu | `cis1b1zs2j3.ad.brown.edu` | Apple M5 Pro | 18 | 18 | Darwin 25.6.0 |
| `m5_1thread.json` | cpu | `cis1b1zs2j3.ad.brown.edu` | Apple M5 Pro | 1 | 18 | Darwin 25.6.0 |
| `oscar_fetch311_1thread.json` | cpu | `gpu2101.oscar.ccv.brown.edu` | AMD EPYC 7532 32-Core Processor | 1 | 64 | SLURM 6113848 |
| `oscar_x86_1thread.json` | cpu | `node2349.oscar.ccv.brown.edu` | Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz | 1 | 48 | SLURM 6107221 |
| `oscar_x86_fetch_1thread.json` | cpu | `gpu2108.oscar.ccv.brown.edu` | AMD EPYC 7413 24-Core Processor | 1 | 48 | SLURM 6107396 |
| `oscar_x86_mjwarp_1thread.json` | cpu | `gpu2106.oscar.ccv.brown.edu` | AMD EPYC 7532 32-Core Processor | 1 | 64 | SLURM 6107880 |
| `3090_fetch.json` | NVIDIA GeForce RTX 3090 | `gpu2108.oscar.ccv.brown.edu` | AMD EPYC 7413 24-Core Processor | 1 | 48 | SLURM 6107396 |
| `3090_fetch311.json` | NVIDIA GeForce RTX 3090 | `gpu2101.oscar.ccv.brown.edu` | AMD EPYC 7532 32-Core Processor | 1 | 64 | SLURM 6113848 |
| `3090_full.json` | NVIDIA GeForce RTX 3090 | `gpu2107.oscar.ccv.brown.edu` | AMD EPYC 7532 32-Core Processor | 1 | 64 | SLURM 6107219 |
| `smoke.json` | NVIDIA GeForce RTX 3090 | `gpu2116.oscar.ccv.brown.edu` | AMD EPYC 7413 24-Core Processor | 1 | 48 | SLURM 6107165 |

## CPU

`S_cpu` is the stable-envs renderer over the official renderer **on the same host**, at the single-thread budget Step 2 certifies on. `cores_cpu` is the same renderer over one core of JPEG-q90 decode of that env's own frame — the G4 economics bar.

| swm id(s) | renderer | host | best B | official f/s | stable-envs f/s | S_cpu | cores_cpu | tier | notes |
|---|---|---|---|---|---|---|---|---|---|
| AcrobotControl-v1 | `Acrobot` | `cis1b1zs2j3` | 256 | 1,033 | 128 | 0.12 | 0.05 | **CPU_REGRESSION** | BELOW_G4 |
| AcrobotControl-v1 | `Acrobot` | `node2349` | 64 | 353 | 70 | 0.20 | 0.06 | **CPU_REGRESSION** | BELOW_G4 |
| CartPoleControl-v1 | `CartPole` | `cis1b1zs2j3` | 256 | 1,129 | 174 | 0.15 | 0.06 | **CPU_REGRESSION** | BELOW_G4 |
| CartPoleControl-v1 | `CartPole` | `node2349` | 64 | 424 | 91 | 0.21 | 0.08 | **CPU_REGRESSION** | BELOW_G4 |
| CraftaxClassicPixels-v1 | `CraftaxClassic` | `cis1b1zs2j3` | 64 | 1,046 | 5,243 | 5.01 | 0.49 | **CPU_FASTER** | BELOW_G4 |
| CraftaxClassicPixels-v1 | `CraftaxClassic` | `node2349` | 16 | — | 2,274 | — | 0.44 | **CPU_REGRESSION** | NO_REFERENCE BELOW_G4 |
| AcrobotDMControl-v0 | `DMCAcrobotSwingup` | `cis1b1zs2j3` | 256 | 523 | 573 | 1.09 | 0.03 | **CPU_PARITY** | BELOW_G4 |
| AcrobotDMControl-v0 | `DMCAcrobotSwingup` | `node2349` | 64 | 38 | 267 | 7.05 | 0.04 | **CPU_FASTER** | BELOW_G4 |
| BallInCupDMControl-v0 | `DMCBallInCupCatch` | `cis1b1zs2j3` | 16 | 306 | 46 | 0.15 | 0.00 | **CPU_REGRESSION** | BELOW_G4 |
| BallInCupDMControl-v0 | `DMCBallInCupCatch` | `node2349` | 16 | 13 | 21 | 1.61 | 0.00 | **CPU_PARITY** | BELOW_G4 |
| CartpoleDMControl-v0 | `DMCCartpoleBalance` | `cis1b1zs2j3` | 64 | 544 | 777 | 1.43 | 0.04 | **CPU_PARITY** | BELOW_G4 |
| CartpoleDMControl-v0 | `DMCCartpoleBalance` | `node2349` | 64 | 41 | 370 | 9.10 | 0.05 | **CPU_FASTER** | BELOW_G4 |
| CartpoleDMControl-v0 | `DMCCartpoleSwingup` | `cis1b1zs2j3` | 256 | 522 | 785 | 1.50 | 0.04 | **CPU_PARITY** | BELOW_G4 |
| CartpoleDMControl-v0 | `DMCCartpoleSwingup` | `node2349` | 64 | 37 | 372 | 10.06 | 0.06 | **CPU_FASTER** | BELOW_G4 |
| CheetahDMControl-v0 | `DMCCheetahRun` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'warp' |
| CheetahDMControl-v0 | `DMCCheetahRun` | `gpu2106` | 1,024 | 385 | — | — | — | **CPU_UNMEASURED** | blocked: device="cpu" was not honoured: 77,457 f/s is 1.01x this env's own CUDA |
| CheetahDMControl-v0 | `DMCCheetahRun` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Found no NVIDIA driver on your system. Please check that |
| FingerDMControl-v0 | `DMCFingerTurnHard` | `cis1b1zs2j3` | 16 | 419 | 119 | 0.28 | 0.01 | **CPU_REGRESSION** | BELOW_G4 |
| FingerDMControl-v0 | `DMCFingerTurnHard` | `node2349` | 16 | 29 | 57 | 1.96 | 0.01 | **CPU_PARITY** | BELOW_G4 |
| HopperDMControl-v0 | `DMCHopperHop` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'warp' |
| HopperDMControl-v0 | `DMCHopperHop` | `gpu2106` | 1,024 | 391 | — | — | — | **CPU_UNMEASURED** | blocked: device="cpu" was not honoured: 78,729 f/s is 1.03x this env's own CUDA |
| HopperDMControl-v0 | `DMCHopperHop` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Found no NVIDIA driver on your system. Please check that |
| HumanoidDMControl-v0 | `DMCHumanoidStand` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'warp' |
| HumanoidDMControl-v0 | `DMCHumanoidStand` | `gpu2106` | 1,024 | 355 | — | — | — | **CPU_UNMEASURED** | blocked: device="cpu" was not honoured: 66,567 f/s is 1.03x this env's own CUDA |
| HumanoidDMControl-v0 | `DMCHumanoidStand` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Found no NVIDIA driver on your system. Please check that |
| PendulumDMControl-v0 | `DMCPendulumSwingup` | `cis1b1zs2j3` | 256 | 503 | 314 | 0.62 | 0.01 | **CPU_REGRESSION** | BELOW_G4 |
| PendulumDMControl-v0 | `DMCPendulumSwingup` | `node2349` | 16 | 38 | 145 | 3.79 | 0.02 | **CPU_FASTER** | BELOW_G4 |
| ReacherDMControl-v0 | `DMCReacherHard` | `cis1b1zs2j3` | 16 | 541 | 206 | 0.38 | 0.01 | **CPU_REGRESSION** | BELOW_G4 |
| ReacherDMControl-v0 | `DMCReacherHard` | `node2349` | 16 | 34 | 99 | 2.87 | 0.01 | **CPU_FASTER** | BELOW_G4 |
| WalkerDMControl-v0 | `DMCWalkerWalk` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'warp' |
| WalkerDMControl-v0 | `DMCWalkerWalk` | `gpu2106` | 1,024 | 378 | — | — | — | **CPU_UNMEASURED** | blocked: device="cpu" was not honoured: 75,122 f/s is 1.02x this env's own CUDA |
| WalkerDMControl-v0 | `DMCWalkerWalk` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Found no NVIDIA driver on your system. Please check that |
| FetchPickAndPlace-v3 | `FetchPickAndPlace` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchPickAndPlace is not in stable_envs.REGISTRY in this interpreter |
| FetchPickAndPlace-v3 | `FetchPickAndPlace` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchPickAndPlace-v3 | `FetchPickAndPlace` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchPickAndPlace-v3 | `FetchPickAndPlace` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchPickAndPlaceDense-v3 | `FetchPickAndPlaceDense` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchPickAndPlaceDense is not in stable_envs.REGISTRY in this interpre |
| FetchPickAndPlaceDense-v3 | `FetchPickAndPlaceDense` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchPickAndPlaceDense-v3 | `FetchPickAndPlaceDense` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchPickAndPlaceDense-v3 | `FetchPickAndPlaceDense` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchPickAndPlaceDict-v3 | `FetchPickAndPlaceDict` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchPickAndPlaceDict is not in stable_envs.REGISTRY in this interpret |
| FetchPickAndPlaceDict-v3 | `FetchPickAndPlaceDict` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchPickAndPlaceDict-v3 | `FetchPickAndPlaceDict` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchPickAndPlaceDict-v3 | `FetchPickAndPlaceDict` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchPush-v3 | `FetchPush` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchPush is not in stable_envs.REGISTRY in this interpreter |
| FetchPush-v3 | `FetchPush` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchPush-v3 | `FetchPush` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchPush-v3 | `FetchPush` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchPushDense-v3 | `FetchPushDense` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchPushDense is not in stable_envs.REGISTRY in this interpreter |
| FetchPushDense-v3 | `FetchPushDense` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchPushDense-v3 | `FetchPushDense` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchPushDense-v3 | `FetchPushDense` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchPushDict-v3 | `FetchPushDict` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchPushDict is not in stable_envs.REGISTRY in this interpreter |
| FetchPushDict-v3 | `FetchPushDict` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchPushDict-v3 | `FetchPushDict` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchPushDict-v3 | `FetchPushDict` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchReach-v3 | `FetchReach` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchReach is not in stable_envs.REGISTRY in this interpreter |
| FetchReach-v3 | `FetchReach` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchReach-v3 | `FetchReach` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchReach-v3 | `FetchReach` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchReachDense-v3 | `FetchReachDense` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchReachDense is not in stable_envs.REGISTRY in this interpreter |
| FetchReachDense-v3 | `FetchReachDense` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchReachDense-v3 | `FetchReachDense` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchReachDense-v3 | `FetchReachDense` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchReachDict-v3 | `FetchReachDict` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchReachDict is not in stable_envs.REGISTRY in this interpreter |
| FetchReachDict-v3 | `FetchReachDict` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchReachDict-v3 | `FetchReachDict` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchReachDict-v3 | `FetchReachDict` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchSlide-v3 | `FetchSlide` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchSlide is not in stable_envs.REGISTRY in this interpreter |
| FetchSlide-v3 | `FetchSlide` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchSlide-v3 | `FetchSlide` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchSlide-v3 | `FetchSlide` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchSlideDense-v3 | `FetchSlideDense` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchSlideDense is not in stable_envs.REGISTRY in this interpreter |
| FetchSlideDense-v3 | `FetchSlideDense` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchSlideDense-v3 | `FetchSlideDense` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchSlideDense-v3 | `FetchSlideDense` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| FetchSlideDict-v3 | `FetchSlideDict` | `cis1b1zs2j3` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: FetchSlideDict is not in stable_envs.REGISTRY in this interpreter |
| FetchSlideDict-v3 | `FetchSlideDict` | `gpu2101` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: RuntimeError: Expected all tensors to be on the same device, but found |
| FetchSlideDict-v3 | `FetchSlideDict` | `gpu2108` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: AssertionError at mujoco_utils.py line 142 |
| FetchSlideDict-v3 | `FetchSlideDict` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ModuleNotFoundError: No module named 'gymnasium_robotics' |
| MountainCarControl-v0 | `MountainCar` | `cis1b1zs2j3` | 256 | 1,074 | 130 | 0.12 | 0.05 | **CPU_REGRESSION** | BELOW_G4 |
| MountainCarControl-v0 | `MountainCar` | `node2349` | 16 | 426 | 71 | 0.17 | 0.07 | **CPU_REGRESSION** | BELOW_G4 |
| MountainCarContinuousControl-v0 | `MountainCarContinuous` | `cis1b1zs2j3` | 256 | 1,005 | 130 | 0.13 | 0.05 | **CPU_REGRESSION** | BELOW_G4 |
| MountainCarContinuousControl-v0 | `MountainCarContinuous` | `node2349` | 16 | 423 | 70 | 0.17 | 0.07 | **CPU_REGRESSION** | BELOW_G4 |
| PendulumControl-v1 | `Pendulum` | `cis1b1zs2j3` | 256 | 794 | 129 | 0.16 | 0.05 | **CPU_REGRESSION** | BELOW_G4 |
| PendulumControl-v1 | `Pendulum` | `node2349` | 16 | 267 | 69 | 0.26 | 0.07 | **CPU_REGRESSION** | BELOW_G4 |
| Piecewise-v0 | `Piecewise` | `cis1b1zs2j3` | 1,024 | 1,860 | 1,472 | 0.79 | 0.15 | **CPU_REGRESSION** | BELOW_G4 |
| Piecewise-v0 | `Piecewise` | `node2349` | 16 | — | 494 | — | 0.15 | **CPU_REGRESSION** | NO_REFERENCE BELOW_G4 |
| Piecewise-v0 | `PiecewiseZones` | `cis1b1zs2j3` | 256 | 1,013 | 534 | 0.53 | 0.06 | **CPU_REGRESSION** | BELOW_G4 |
| Piecewise-v0 | `PiecewiseZones` | `node2349` | 16 | — | 261 | — | 0.08 | **CPU_REGRESSION** | NO_REFERENCE BELOW_G4 |
| OGBMaze-v0 | `PointmazeLargeNavigate` | `cis1b1zs2j3` | 1,024 | 812 | 2,238 | 2.76 | 0.30 | **CPU_FASTER** | BELOW_G4 |
| OGBMaze-v0 | `PointmazeLargeNavigate` | `node2349` | 64 | 21 | 994 | 48.01 | 0.42 | **CPU_FASTER** | BELOW_G4 |
| OGBMaze-v0 | `PointmazeLargeStitch` | `cis1b1zs2j3` | 1,024 | 649 | 2,222 | 3.42 | 0.30 | **CPU_FASTER** | BELOW_G4 |
| OGBMaze-v0 | `PointmazeLargeStitch` | `node2349` | 64 | 21 | 984 | 47.67 | 0.41 | **CPU_FASTER** | BELOW_G4 |
| PushT-Discrete-v1, PushT-v1 | `PushT` | `cis1b1zs2j3` | 1,024 | 1,036 | 385 | 0.37 | 0.02 | **CPU_REGRESSION** | BELOW_G4 |
| PushT-Discrete-v1, PushT-v1 | `PushT` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ValueError: Expected a cuda device, but got: cpu |
| SimplePointMaze-v0 | `SimplePointMaze` | `cis1b1zs2j3` | 64 | 121 | 10 | 0.08 | 0.00 | **CPU_REGRESSION** | BELOW_G4 |
| SimplePointMaze-v0 | `SimplePointMaze` | `node2349` | — | — | — | — | — | **CPU_UNMEASURED** | blocked: ValueError: Expected a cuda device, but got: cpu |
| TwoRoom-v1 | `TwoRoom` | `cis1b1zs2j3` | 1,024 | 1,351 | 729 | 0.54 | 0.07 | **CPU_REGRESSION** | BELOW_G4 |
| TwoRoom-v1 | `TwoRoom` | `node2349` | 16 | — | 320 | — | 0.09 | **CPU_REGRESSION** | NO_REFERENCE BELOW_G4 |
| TwoRoom-v1 | `TwoRoomTarget` | `cis1b1zs2j3` | 1,024 | 1,001 | 582 | 0.58 | 0.06 | **CPU_REGRESSION** | BELOW_G4 |
| TwoRoom-v1 | `TwoRoomTarget` | `node2349` | 16 | — | 223 | — | 0.06 | **CPU_REGRESSION** | NO_REFERENCE BELOW_G4 |

### CPU thread scaling

The all-core column is a **scaling** result and is never the certified number. Step 2 certifies at one thread so a fast node cannot stand in for a fast renderer — and this table is where that rule earns itself.

| renderer | host | 1 thread | all-core | renderer speedup | official 1 thread | official all-core | baseline moved |
|---|---|---|---|---|---|---|---|
| `Acrobot` | `cis1b1zs2j3` | 128 | 776 (18t) | **6.06×** | 1,033 | 1,064 | — |
| `CartPole` | `cis1b1zs2j3` | 174 | 979 (18t) | **5.63×** | 1,129 | 1,123 | — |
| `CraftaxClassic` | `cis1b1zs2j3` | 5,243 | 17,571 (18t) | **3.35×** | 1,046 | 1,046 | — |
| `DMCAcrobotSwingup` | `cis1b1zs2j3` | 573 | 1,624 (18t) | **2.83×** | 523 | 545 | — |
| `DMCBallInCupCatch` | `cis1b1zs2j3` | 46 | 131 (18t) | **2.87×** | 306 | 309 | — |
| `DMCCartpoleBalance` | `cis1b1zs2j3` | 777 | 2,320 (18t) | **2.99×** | 544 | 532 | — |
| `DMCCartpoleSwingup` | `cis1b1zs2j3` | 785 | 2,426 (18t) | **3.09×** | 522 | 529 | — |
| `DMCFingerTurnHard` | `cis1b1zs2j3` | 119 | 343 (18t) | **2.88×** | 419 | 490 | — |
| `DMCPendulumSwingup` | `cis1b1zs2j3` | 314 | 962 (18t) | **3.07×** | 503 | 489 | — |
| `DMCReacherHard` | `cis1b1zs2j3` | 206 | 613 (18t) | **2.98×** | 541 | 451 | — |
| `MountainCar` | `cis1b1zs2j3` | 130 | 818 (18t) | **6.29×** | 1,074 | 1,003 | — |
| `MountainCarContinuous` | `cis1b1zs2j3` | 130 | 775 (18t) | **5.96×** | 1,005 | 1,056 | — |
| `Pendulum` | `cis1b1zs2j3` | 129 | 829 (18t) | **6.42×** | 794 | 740 | — |
| `Piecewise` | `cis1b1zs2j3` | 1,472 | 5,299 (18t) | **3.60×** | 1,860 | 337 | **5.5× slower** |
| `PiecewiseZones` | `cis1b1zs2j3` | 534 | 2,813 (18t) | **5.27×** | 1,013 | 215 | **4.7× slower** |
| `PointmazeLargeNavigate` | `cis1b1zs2j3` | 2,238 | 8,013 (18t) | **3.58×** | 812 | 493 | **1.6× slower** |
| `PointmazeLargeStitch` | `cis1b1zs2j3` | 2,222 | 8,006 (18t) | **3.60×** | 649 | 511 | **1.3× slower** |
| `PushT` | `cis1b1zs2j3` | 385 | 936 (18t) | **2.43×** | 1,036 | 1,026 | — |
| `SimplePointMaze` | `cis1b1zs2j3` | 10 | 25 (18t) | **2.53×** | 121 | 116 | — |
| `TwoRoom` | `cis1b1zs2j3` | 729 | 3,031 (18t) | **4.16×** | 1,351 | 171 | **7.9× slower** |
| `TwoRoomTarget` | `cis1b1zs2j3` | 582 | 2,488 (18t) | **4.28×** | 1,001 | 126 | **8.0× slower** |

!!! warning "The all-core column degrades some references, which is why it cannot certify"
    Giving torch every core makes the *official* renderer of `TwoRoomTarget`, `TwoRoom`, `Piecewise`, `PiecewiseZones` **4.7–8.0× slower**, not faster. Those references are themselves torch arithmetic on small tensors (`swm_nav`'s envs draw a uint8 image with torch ops), so a thread pool costs more than it buys and the baseline collapses. Smaller drifts in the column above are node noise on a shared machine, not this mechanism.

    An all-core `S_cpu` for those rows would therefore look excellent while measuring nothing but a damaged denominator. Step 2 froze the certified budget at one thread before any of this was known; the table above is the reason that was right.

## GPU — RTX 3090

`R` is the stable-envs GPU renderer over the official CPU renderer measured on the **same node**. `cores` is the same over one core of decode there. Both bars are class-dependent and frozen, and two qualifiers cap a row regardless of how large `R` is: `BELOW_G4` (`cores < 1`) caps at `GPU_FASTER`, and a failed `BATCH_SCALES` caps at `GPU_PARITY` because Step 2 makes it required above that tier.

| swm id(s) | renderer | class | best B | official f/s | 3090 f/s | R | cores | tier | qualifiers |
|---|---|---|---|---|---|---|---|---|---|
| AcrobotControl-v1 | `Acrobot` | R | 4,096 | 340 | 15,216 | 44.7 | 15.41 | **GPU_FASTER** | — |
| CartPoleControl-v1 | `CartPole` | R | 4,096 | 381 | 17,444 | 45.8 | 16.23 | **GPU_FASTER** | — |
| CraftaxClassicPixels-v1 | `CraftaxClassic` | R | 4,096 | — | 144,738 | — | 35.50 | **GPU_FAST** | NO_REFERENCE |
| AcrobotDMControl-v0 | `DMCAcrobotSwingup` | S | 4,096 | 283 | 27,808 | 98.3 | 4.74 | **GPU_FAST** | — |
| BallInCupDMControl-v0 | `DMCBallInCupCatch` | S | 4,096 | 262 | 2,459 | 9.4 | 0.41 | **GPU_PARITY** | BELOW_G4 BATCH_SCALES_FAIL |
| CartpoleDMControl-v0 | `DMCCartpoleBalance` | S | 4,096 | 287 | 43,642 | 152.3 | 7.31 | **GPU_FAST** | — |
| CartpoleDMControl-v0 | `DMCCartpoleSwingup` | S | 4,096 | 287 | 42,389 | 147.6 | 6.51 | **GPU_FAST** | — |
| CheetahDMControl-v0 | `DMCCheetahRun` | S | 4,096 | 280 | 76,589 | 273.6 | 14.24 | **GPU_EXCEPTIONAL** | — |
| FingerDMControl-v0 | `DMCFingerTurnHard` | S | 4,096 | 282 | 22,042 | 78.3 | 3.67 | **GPU_FAST** | — |
| HopperDMControl-v0 | `DMCHopperHop` | S | 4,096 | 287 | 76,735 | 267.1 | 14.65 | **GPU_EXCEPTIONAL** | — |
| HumanoidDMControl-v0 | `DMCHumanoidStand` | S | 4,096 | 264 | 64,493 | 244.3 | 11.97 | **GPU_EXCEPTIONAL** | — |
| PendulumDMControl-v0 | `DMCPendulumSwingup` | S | 4,096 | 289 | 36,127 | 124.9 | 5.27 | **GPU_FAST** | — |
| ReacherDMControl-v0 | `DMCReacherHard` | S | 4,096 | 281 | 25,207 | 89.7 | 3.77 | **GPU_FAST** | — |
| WalkerDMControl-v0 | `DMCWalkerWalk` | S | 4,096 | 284 | 73,503 | 258.5 | 11.82 | **GPU_EXCEPTIONAL** | — |
| FetchPickAndPlace-v3 | `FetchPickAndPlace` | S | 1,024 | 906 | 5,171 | 5.7 | 4.36 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchPickAndPlaceDense-v3 | `FetchPickAndPlaceDense` | S | 4,096 | 848 | 5,162 | 6.1 | 4.95 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchPickAndPlaceDict-v3 | `FetchPickAndPlaceDict` | S | 4,096 | 830 | 5,163 | 6.2 | 4.96 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchPush-v3 | `FetchPush` | S | 4,096 | 854 | 5,224 | 6.1 | 5.27 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchPushDense-v3 | `FetchPushDense` | S | 1,024 | 930 | 5,221 | 5.6 | 5.30 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchPushDict-v3 | `FetchPushDict` | S | 4,096 | 918 | 5,192 | 5.7 | 4.12 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchReach-v3 | `FetchReach` | S | 1,024 | 910 | 5,212 | 5.7 | 4.16 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchReachDense-v3 | `FetchReachDense` | S | 4,096 | 971 | 5,185 | 5.3 | 4.10 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchReachDict-v3 | `FetchReachDict` | S | 4,096 | 914 | 5,184 | 5.7 | 4.12 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchSlide-v3 | `FetchSlide` | S | 1,024 | 845 | 5,295 | 6.3 | 4.28 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchSlideDense-v3 | `FetchSlideDense` | S | 4,096 | 893 | 5,267 | 5.9 | 5.40 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| FetchSlideDict-v3 | `FetchSlideDict` | S | 4,096 | 786 | 5,264 | 6.7 | 4.23 | **GPU_PARITY** | BATCH_SCALES_FAIL |
| MountainCarControl-v0 | `MountainCar` | R | 4,096 | 359 | 15,182 | 42.3 | 17.89 | **GPU_FASTER** | — |
| MountainCarContinuousControl-v0 | `MountainCarContinuous` | R | 4,096 | 364 | 15,160 | 41.6 | 18.26 | **GPU_FASTER** | — |
| PendulumControl-v1 | `Pendulum` | R | 4,096 | 205 | 17,647 | 85.9 | 20.41 | **GPU_FASTER** | — |
| Piecewise-v0 | `Piecewise` | R | 4,096 | — | 54,747 | — | 23.50 | **GPU_FAST** | NO_REFERENCE |
| Piecewise-v0 | `PiecewiseZones` | R | 4,096 | — | 46,951 | — | 18.45 | **GPU_FAST** | NO_REFERENCE |
| OGBMaze-v0 | `PointmazeLargeNavigate` | S | 4,096 | 1,391 | 95,962 | 69.0 | 43.60 | **GPU_EXCEPTIONAL** | — |
| OGBMaze-v0 | `PointmazeLargeStitch` | S | 4,096 | 1,383 | 96,013 | 69.4 | 44.69 | **GPU_EXCEPTIONAL** | — |
| PushT-Discrete-v1, PushT-v1 | `PushT` | R | 4,096 | 364 | 679,732 | 1,865.3 | 104.67 | **GPU_FAST** | — |
| SimplePointMaze-v0 | `SimplePointMaze` | R | 4,096 | — | 41,250 | — | 49.63 | **GPU_FAST** | NO_REFERENCE |
| TwoRoom-v1 | `TwoRoom` | R | 4,096 | — | 39,448 | — | 12.56 | **GPU_FAST** | NO_REFERENCE |
| TwoRoom-v1 | `TwoRoomTarget` | R | 4,096 | — | 23,744 | — | 7.67 | **GPU_FASTER** | NO_REFERENCE |

### The same renderer on two 3090 nodes

Where a shard overlapped another, the same env was measured twice on different nodes. Step 2 warned that a GPU row does not automatically travel; this is how far it moved here, on rows that are NOT host-bound.

| renderer | nodes | slower f/s | faster f/s | spread |
|---|---|---|---|---|
| `PushT` | gpu2107, gpu2116 | 679,732 | 789,888 | **1.16×** |

## The H200 → RTX 3090 prediction, tested

Step 2 pre-registered **5.13×** on the grounds that the rasterizer is output-bandwidth-bound. That was a prediction and this is the test of it. A `HOST_BOUND` row is expected NOT to scale with the device, so the two populations are reported apart.

| renderer | class | host-bound | H200 f/s | 3090 f/s | H200/3090 |
|---|---|---|---|---|---|
| `Acrobot` | R | no | 31,630 | 15,216 | **2.08×** |
| `CartPole` | R | no | 35,973 | 17,444 | **2.06×** |
| `CraftaxClassic` | R | no | 568,670 | 144,738 | **3.93×** |
| `DMCAcrobotSwingup` | S | no | 48,877 | 27,808 | **1.76×** |
| `DMCBallInCupCatch` | S | no | 4,378 | 2,459 | **1.78×** |
| `DMCCartpoleBalance` | S | no | 76,456 | 43,642 | **1.75×** |
| `DMCCartpoleSwingup` | S | no | 74,802 | 42,389 | **1.76×** |
| `DMCCheetahRun` | S | no | 124,345 | 76,589 | **1.62×** |
| `DMCFingerTurnHard` | S | no | 40,348 | 22,042 | **1.83×** |
| `DMCHopperHop` | S | no | 124,436 | 76,735 | **1.62×** |
| `DMCHumanoidStand` | S | no | 101,359 | 64,493 | **1.57×** |
| `DMCPendulumSwingup` | S | no | 67,711 | 36,127 | **1.87×** |
| `DMCReacherHard` | S | no | 48,812 | 25,207 | **1.94×** |
| `DMCWalkerWalk` | S | no | 116,662 | 73,503 | **1.59×** |
| `FetchPickAndPlace` | S | no | 8,653 | 5,171 | **1.67×** |
| `FetchPickAndPlaceDense` | S | no | 8,654 | 5,162 | **1.68×** |
| `FetchPickAndPlaceDict` | S | no | 8,655 | 5,163 | **1.68×** |
| `FetchPush` | S | no | 8,744 | 5,224 | **1.67×** |
| `FetchPushDense` | S | no | 8,746 | 5,221 | **1.68×** |
| `FetchPushDict` | S | no | 8,749 | 5,192 | **1.68×** |
| `FetchReach` | S | no | 8,686 | 5,212 | **1.67×** |
| `FetchReachDense` | S | no | 8,678 | 5,185 | **1.67×** |
| `FetchReachDict` | S | no | 8,682 | 5,184 | **1.67×** |
| `FetchSlide` | S | no | 8,958 | 5,295 | **1.69×** |
| `FetchSlideDense` | S | no | 8,953 | 5,267 | **1.70×** |
| `FetchSlideDict` | S | no | 8,956 | 5,264 | **1.70×** |
| `MountainCar` | R | no | 31,111 | 15,182 | **2.05×** |
| `MountainCarContinuous` | R | no | 31,137 | 15,160 | **2.05×** |
| `Pendulum` | R | no | 38,396 | 17,647 | **2.18×** |
| `Piecewise` | R | no | 187,975 | 54,747 | **3.43×** |
| `PiecewiseZones` | R | no | 141,119 | 46,951 | **3.01×** |
| `PointmazeLargeNavigate` | S | no | 242,332 | 95,962 | **2.53×** |
| `PointmazeLargeStitch` | S | no | 242,672 | 96,013 | **2.53×** |
| `PushT` | R | no | 1,181,957 | 679,732 | **1.74×** |
| `SimplePointMaze` | R | no | 70,746 | 41,250 | **1.72×** |
| `TwoRoom` | R | no | 117,007 | 39,448 | **2.97×** |
| `TwoRoomTarget` | R | no | 76,785 | 23,744 | **3.23×** |

| population | n | median H200/3090 |
|---|---|---|
| S | 25 | **1.68×** |
| R | 12 | **2.13×** |
| device bound | 37 | **1.75×** |

## Input pipeline (product C — never a certification number)

The renderer sweep times states already resident. This times the road to that point. `ceiling` is the no-loader arm; the others are the fraction of it each idiom reaches.

| env | device | B | ceiling f/s | default_collate | batch_sampler | collate_batched | render share of ceiling |
|---|---|---|---|---|---|---|---|
| `PushT` | NVIDIA GeForce RTX 3090 | 256 | 65,069 | 0.83 | 0.96 | 0.96 | 0.98 |
| `PushT` | NVIDIA GeForce RTX 3090 | 1,024 | 226,033 | 0.63 | 0.93 | 0.93 | 0.99 |
| `PushT` | NVIDIA GeForce RTX 3090 | 4,096 | 754,202 | 0.36 | 0.84 | 0.85 | 0.99 |
| `DMCReacherHard` | NVIDIA GeForce RTX 3090 | 256 | 20,867 | 0.94 | 0.99 | 0.99 | 0.99 |
| `DMCReacherHard` | NVIDIA GeForce RTX 3090 | 1,024 | 25,422 | 0.94 | 0.99 | 0.99 | 1.00 |
| `DMCReacherHard` | NVIDIA GeForce RTX 3090 | 4,096 | 25,962 | 0.94 | 0.99 | 0.99 | 1.00 |
| `DMCHumanoidStand` | NVIDIA GeForce RTX 3090 | 256 | 66,452 | 0.84 | 0.96 | 0.96 | 0.99 |
| `DMCHumanoidStand` | NVIDIA GeForce RTX 3090 | 1,024 | 69,943 | 0.85 | 0.98 | 0.98 | 0.99 |
| `DMCHumanoidStand` | NVIDIA GeForce RTX 3090 | 4,096 | 71,221 | 0.85 | 0.98 | 0.98 | 1.00 |

## Blocked measurements

A blocker is **not** a performance tier. Every row here was never reached, so there is no claim to make about it in either direction — Step 2 forbids turning "we could not build this here" into `REGRESSION` or `unsupported`.

26 earlier blocker(s) are omitted because a later shard measured the same env on the same device; they are kept in `certification.json` as history.
A further 16 env(s) are omitted because their CPU failure is an **implementation gap and not a blocker** — see `CPU_BACKEND_MISSING` below.

| what failed | where | rows |
|---|---|---|
| `AssertionError at mujoco_utils.py line 142` | cpu | 12 (FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict…) |
| `ModuleNotFoundError: No module named 'gymnasium_robotics'` | cpu | 12 (FetchPickAndPlace, FetchPickAndPlaceDense, FetchPickAndPlaceDict…) |
| `ModuleNotFoundError: No module named 'warp'` | cpu | 4 (DMCCheetahRun, DMCHopperHop, DMCHumanoidStand…) |
| `FetchPickAndPlace is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchPickAndPlace) |
| `FetchPickAndPlaceDense is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchPickAndPlaceDense) |
| `FetchPickAndPlaceDict is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchPickAndPlaceDict) |
| `FetchPush is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchPush) |
| `FetchPushDense is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchPushDense) |
| `FetchPushDict is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchPushDict) |
| `FetchReach is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchReach) |
| `FetchReachDense is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchReachDense) |
| `FetchReachDict is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchReachDict) |
| `FetchSlide is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchSlide) |
| `FetchSlideDense is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchSlideDense) |
| `FetchSlideDict is not in stable_envs.REGISTRY in this interpreter` | cpu | 1 (FetchSlideDict) |

### Fetch: the conflict was real, and a matched stack dissolved it

Step 3 concluded that no mujoco version could host both halves of Fetch, having tested two combinations:

```text
mujoco 3.13  + gymnasium-robotics      -> AssertionError at mujoco_utils.py:142
mujoco 3.11  + mujoco_warp HEAD        -> AttributeError: no attribute 'mjtCtrlInput'
```

**That conclusion was wrong, and Step 3B corrects it.** Both failures came from mixing a pinned mujoco with a mujoco_warp built for a different one. mujoco_warp publishes a `v3.11.0` tag built against mujoco 3.11, and the matched pair holds:

```text
mujoco 3.11.0 + mujoco-warp 3.11.0 + gymnasium-robotics 1.4.2
    mujoco          3.11.0      imports
    mujoco_warp     3.11.0      imports
    FetchPush-v4                reset OK
```

The intermediate failures on the way there (`No module named 'packaging'`, then `'pettingzoo'`) were self-inflicted: installing `gymnasium-robotics --no-deps` to stop it dragging mujoco forward also starved it of pure-python dependencies. Resolving its full dependency set turned out not to move mujoco at all — **the version conflict Step 1 and Step 3 both recorded does not exist for this combination**, and the cost of assuming it did was 12 rows reported as unmeasurable.

All **12 Fetch rows now carry fresh RTX 3090 measurements**, over a state corpus written by the GPU pass and re-read by the CPU pass so both see byte-identical states.

#### Fetch has no CPU render path

The CPU pass of the same matched stack fails on all 12, and not on a dependency:

```text
RuntimeError: Expected all tensors to be on the same device,
              but found at least two devices, cuda:0 and cpu!
  at stable_envs/envs/fetch.py:811 in _upload
```

The Fetch renderer allocates its mjwarp buffers on CUDA regardless of the requested device, and its own upload path then meets the CPU state tensor. This is the same conclusion as the four mjwarp DM Control rows above — mjwarp is GPU-only — reached the honest way round: **Fetch raises where DM Control silently returned device frames.** So the 12 rows are `CPU_UNMEASURED` because there is no CPU renderer to measure, which is a fact about the backend and not a dependency blocker and not a regression.

### `device="cpu"` is silently ignored by the mjwarp backend

The four mjwarp-backed DM Control envs (`DMCCheetahRun`, `DMCHopperHop`, `DMCHumanoidStand`, `DMCWalkerWalk`) were first blocked on a CPU-only node with `RuntimeError: Found no NVIDIA driver`, which is ambiguous — it could mean the renderer has no CPU path, or just that the node had no driver. Re-running the **CPU** sweep on a GPU node separated the two, and the answer was not the expected one:

```text
asked for device="cpu", on a node with a 3090 present:
    DMCCheetahRun     77,457 f/s   = 1.01x its own CUDA throughput
    DMCHopperHop      78,729 f/s   = 1.03x
    DMCHumanoidStand  66,567 f/s   = 1.03x
    DMCWalkerWalk     75,122 f/s   = 1.02x
```

A single CPU core does not match a 3090 to within 3 %. The render ran on the GPU: the mjwarp path has no CPU implementation and **does not raise when asked for one** — it quietly returns device frames. Every other renderer in this campaign honours the flag, which is why the four stood out at all.

Two consequences, and the second is the reason this has its own section:

1. those four rows have **no CPU throughput**, and are `CPU_UNMEASURED` rather than    fast or slow — there is nothing on the CPU to be either;
2. taken at face value they would have certified **`CPU_EXCEPTIONAL`** on an `S_cpu`    above 200, putting 3090 numbers in the CPU column of a document whose whole    purpose is keeping the two apart. `scratch/step3/certify.py` now refuses any CPU    label above half its own env's CUDA throughput, and the raw shard is kept as the    evidence for the refusal rather than deleted.

### `CPU_BACKEND_MISSING` — the mjwarp family has no CPU renderer

These are **not** dependency or setup blockers, and Step 3B is what settles the difference:

```text
dependency / setup blocker   a working implementation may exist; this interpreter
                             could not reach it   (an install fixes it)

CPU_BACKEND_MISSING          the renderer has no CPU path at all
                             (only writing one fixes it)
```

Step 3 filed all 16 of these as setup problems. That was wrong. The matched `mujoco 3.11.0 + mujoco-warp 3.11.0 + gymnasium-robotics 1.4.2` stack **builds and measures every one of them on the GPU** — so the dependency question is closed — and the CPU pass still fails, inside the renderer:

```text
Fetch (12 rows)        RuntimeError: Expected all tensors to be on the same device,
                       cuda:0 and cpu, at stable_envs/envs/fetch.py:811 in _upload

DM Control (4 rows)    no exception at all: device="cpu" returns frames at 1.01-1.03x
                       the env's own CUDA throughput, because the render ran on the GPU
```

| env | GPU tier | CPU |
|---|---|---|
| `DMCCheetahRun` | GPU_EXCEPTIONAL | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `DMCHopperHop` | GPU_EXCEPTIONAL | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `DMCHumanoidStand` | GPU_EXCEPTIONAL | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `DMCWalkerWalk` | GPU_EXCEPTIONAL | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchPickAndPlace` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchPickAndPlaceDense` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchPickAndPlaceDict` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchPush` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchPushDense` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchPushDict` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchReach` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchReachDense` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchReachDict` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchSlide` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchSlideDense` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |
| `FetchSlideDict` | GPU_PARITY | `CPU_UNMEASURED` + `CPU_BACKEND_MISSING` |

The tier stays `CPU_UNMEASURED` — Step 2's definition is untouched and there is no measurement to tier. What changes is that these rows are now legible as **renderer work that has not been done**, rather than as benchmark plumbing that failed.

## Headline counts over the 34 matched rows

**CPU** (a row with several renderers takes its weakest):

```text
CPU_EXCEPTIONAL        0
CPU_FAST               0
CPU_FASTER             2
CPU_PARITY             2
CPU_REGRESSION         14
CPU_UNMEASURED         16
```

All 16 `CPU_UNMEASURED` rows are `CPU_BACKEND_MISSING`: no CPU renderer exists for them. None is an unrun or failed benchmark.

**GPU** (a row with several renderers takes its weakest):

```text
GPU_EXCEPTIONAL        5
GPU_FAST               10
GPU_FASTER             6
GPU_PARITY             13
GPU_REGRESSION         0
GPU_UNMEASURED         0
```

```text
ENVIRONMENT_DONE               0   (needs full coverage + CPU_FAST + GPU_FAST)
RENDERER_ABSENT                6   (the Step 1 renderer gaps)
NOT_APPLICABLE_SYMBOLIC        2
```

## Optimization queue

Ranked by measured failure mode and by how many registered ids ride on it. **This is the output of Step 3, not a plan** — nothing here was acted on.

| # | renderer | swm rows | CPU | GPU | bottleneck | evidence |
|---|---|---|---|---|---|---|
| 1 | `DMCBallInCupCatch` | 1 | CPU_REGRESSION | GPU_PARITY | below decode break-even | 0.41 cores of decode on the GPU · also cpu:BELOW_G4 gpu:BATCH_SCALES_FAIL gpu:BELOW_G4 |
| 2 | `FetchPickAndPlace` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 11.6x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 3 | `FetchPickAndPlaceDense` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 7.7x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 4 | `FetchPickAndPlaceDict` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 7.6x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 5 | `FetchPush` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 8.0x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 6 | `FetchPushDense` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 7.7x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 7 | `FetchPushDict` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 7.8x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 8 | `FetchReach` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 7.8x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 9 | `FetchReachDense` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 7.7x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 10 | `FetchReachDict` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 7.6x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 11 | `FetchSlide` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 8.1x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 12 | `FetchSlideDense` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 7.9x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 13 | `FetchSlideDict` | 1 | CPU_UNMEASURED | GPU_PARITY | launch-bound / does not batch | fps(4096) is only 7.8x fps(1); capped at GPU_PARITY · also cpu:CPU_BACKEND_MISSING gpu:BATCH_SCALES_FAIL |
| 14 | `PushT` | 2 | CPU_REGRESSION | GPU_FAST | CPU implementation too slow | S_cpu = 0.37 · also cpu:BELOW_G4 |
| 15 | `Acrobot` | 1 | CPU_REGRESSION | GPU_FASTER | CPU implementation too slow | S_cpu = 0.12 · also cpu:BELOW_G4 |
| 16 | `CartPole` | 1 | CPU_REGRESSION | GPU_FASTER | CPU implementation too slow | S_cpu = 0.15 · also cpu:BELOW_G4 |
| 17 | `CraftaxClassic` | 1 | CPU_FASTER | GPU_FAST | CPU implementation too slow | S_cpu = 5.01 · also cpu:BELOW_G4 gpu:NO_REFERENCE |
| 18 | `DMCAcrobotSwingup` | 1 | CPU_PARITY | GPU_FAST | CPU implementation too slow | S_cpu = 1.09 · also cpu:BELOW_G4 |
| 19 | `DMCCartpoleBalance` | 1 | CPU_PARITY | GPU_FAST | CPU implementation too slow | S_cpu = 1.43 · also cpu:BELOW_G4 |
| 20 | `DMCCartpoleSwingup` | 1 | CPU_PARITY | GPU_FAST | CPU implementation too slow | S_cpu = 1.50 · also cpu:BELOW_G4 |
| 21 | `DMCFingerTurnHard` | 1 | CPU_REGRESSION | GPU_FAST | CPU implementation too slow | S_cpu = 0.28 · also cpu:BELOW_G4 |
| 22 | `DMCPendulumSwingup` | 1 | CPU_REGRESSION | GPU_FAST | CPU implementation too slow | S_cpu = 0.62 · also cpu:BELOW_G4 |
| 23 | `DMCReacherHard` | 1 | CPU_REGRESSION | GPU_FAST | CPU implementation too slow | S_cpu = 0.38 · also cpu:BELOW_G4 |
| 24 | `MountainCar` | 1 | CPU_REGRESSION | GPU_FASTER | CPU implementation too slow | S_cpu = 0.12 · also cpu:BELOW_G4 |
| 25 | `MountainCarContinuous` | 1 | CPU_REGRESSION | GPU_FASTER | CPU implementation too slow | S_cpu = 0.13 · also cpu:BELOW_G4 |
| 26 | `Pendulum` | 1 | CPU_REGRESSION | GPU_FASTER | CPU implementation too slow | S_cpu = 0.16 · also cpu:BELOW_G4 |
| 27 | `Piecewise` | 1 | CPU_REGRESSION | GPU_FAST | CPU implementation too slow | S_cpu = 0.79 · also cpu:BELOW_G4 gpu:NO_REFERENCE |
| 28 | `PiecewiseZones` | 1 | CPU_REGRESSION | GPU_FAST | CPU implementation too slow | S_cpu = 0.53 · also cpu:BELOW_G4 gpu:NO_REFERENCE |
| 29 | `PointmazeLargeNavigate` | 1 | CPU_FASTER | GPU_EXCEPTIONAL | CPU implementation too slow | S_cpu = 2.76 · also cpu:BELOW_G4 |
| 30 | `PointmazeLargeStitch` | 1 | CPU_FASTER | GPU_EXCEPTIONAL | CPU implementation too slow | S_cpu = 3.42 · also cpu:BELOW_G4 |
| 31 | `SimplePointMaze` | 1 | CPU_REGRESSION | GPU_FAST | CPU implementation too slow | S_cpu = 0.08 · also cpu:BELOW_G4 gpu:NO_REFERENCE |
| 32 | `TwoRoom` | 1 | CPU_REGRESSION | GPU_FAST | CPU implementation too slow | S_cpu = 0.54 · also cpu:BELOW_G4 gpu:NO_REFERENCE |
| 33 | `TwoRoomTarget` | 1 | CPU_REGRESSION | GPU_FASTER | CPU implementation too slow | S_cpu = 0.58 · also cpu:BELOW_G4 gpu:NO_REFERENCE |
| 34 | `DMCCheetahRun` | 1 | CPU_UNMEASURED | GPU_EXCEPTIONAL | GPU-only backend, no CPU path | the GPU meets its bars; there is no CPU renderer to measure at all · also cpu:CPU_BACKEND_MISSING cpu:DEVICE_NOT_HONOURED |
| 35 | `DMCHopperHop` | 1 | CPU_UNMEASURED | GPU_EXCEPTIONAL | GPU-only backend, no CPU path | the GPU meets its bars; there is no CPU renderer to measure at all · also cpu:CPU_BACKEND_MISSING cpu:DEVICE_NOT_HONOURED |
| 36 | `DMCHumanoidStand` | 1 | CPU_UNMEASURED | GPU_EXCEPTIONAL | GPU-only backend, no CPU path | the GPU meets its bars; there is no CPU renderer to measure at all · also cpu:CPU_BACKEND_MISSING cpu:DEVICE_NOT_HONOURED |
| 37 | `DMCWalkerWalk` | 1 | CPU_UNMEASURED | GPU_EXCEPTIONAL | GPU-only backend, no CPU path | the GPU meets its bars; there is no CPU renderer to measure at all · also cpu:CPU_BACKEND_MISSING cpu:DEVICE_NOT_HONOURED |

