# GPU vs official CPU renderer

Measured on `NVIDIA H200` / `INTEL(R) XEON(R) PLATINUM 8558`, torch 2.11.0+cu130, 16 cores visible.
Supersample: `native` — with `native`, every env renders at its own `default_supersample`, which is the picture `se.make(name)` gives and the one every parity number in this repo is measured under. Each row below carries the RESOLVED value; forcing a number instead makes 44 of the classes report a cheaper picture than they ship. Every row below was measured by the current code, which records the value it resolved.
Median of 25 full-batch renders (GPU) and 200 sequential frames (CPU).

Host 1-minute load average 7.08 at the start and 7.20 at the end, over 16 visible cores. Each GPU row carries `cpu%`, the process CPU seconds per wall second of the timed loop. Near 0 the device is the limit and the row travels between nodes; at or above 100% the host is busy for the whole render and the row is a measurement of this node's processor under this node's load — the ray-cast families sit there (DMCBallInCupCatch measures 196%, and the same configuration has produced 3,664 frames/s idle and 271 contended). The CPU column does not reproduce under contention at all: the same sweep on a busier host measured MiniGridKeyCorridor's official renderer at 37 frames/s against 425, and elsewhere in this repo one official renderer has gone from 422 frames/s idle to 3 with three other jobs resident. Read every speedup as an order of magnitude, and discard a baseline taken on a loaded node rather than averaging it in.

Registry at run time (64 envs): Acrobot, AntmazeGiantNavigate, AntmazeGiantStitch, AntmazeLargeExplore, AntmazeLargeNavigate, AntmazeLargeStitch, AntmazeMediumExplore, AntmazeMediumNavigate, AntmazeMediumStitch, AntmazeTeleportExplore, AntmazeTeleportNavigate, AntmazeTeleportStitch, BabyAIGoToLocal, CartPole, CraftaxClassic, Crafter, Highway, HumanoidmazeGiantNavigate, HumanoidmazeGiantStitch, HumanoidmazeLargeNavigate, HumanoidmazeLargeStitch, HumanoidmazeMediumNavigate, HumanoidmazeMediumStitch, Intersection, Merge, MiniGridDoorKey, MiniGridEmpty, MiniGridFourRooms, MiniGridKeyCorridor, MiniGridLavaCrossing, MiniGridPOVFetch, MiniHack, MountainCar, MountainCarContinuous, MujocoInvertedDoublePendulum, MujocoInvertedPendulum, MujocoPusher, MujocoReacher, Parking, Pendulum, Piecewise, PiecewiseZones, PointMazeLarge, PointMazeMedium, PointMazeOpen, PointMazeUMaze, PointmazeGiantNavigate, PointmazeGiantStitch, PointmazeLargeNavigate, PointmazeLargeStitch, PointmazeMediumNavigate, PointmazeMediumStitch, PointmazeTeleportNavigate, PointmazeTeleportStitch, ProcgenChaser, ProcgenHeist, ProcgenMaze, ProcgenMiner, PushT, Racetrack, Roundabout, SimplePointMaze, TwoRoom, TwoRoomTarget.

## PushT

State: 5 floats `agent_x, agent_y, block_x, block_y, block_angle`. Reference: `gym_pusht/PushT-v0`.

### 96x96 (supersample 1)

Official CPU renderer: **491 frames/s** (2,035 us/frame, 0.0136 GB/s) for set_state+render; 495 frames/s (2,021 us/frame) for the draw alone.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 1.792 | 1791.5380 | 558 | 0.0 | 0.22 | 100% | 1.14x |
| 64 | 2.250 | 35.1512 | 28,449 | 0.8 | 0.22 | 100% | 58x |
| 256 | 2.350 | 9.1809 | 108,922 | 3.0 | 0.22 | 100% | 222x |
| 1,024 | 2.691 | 2.6275 | 380,586 | 10.5 | 0.25 | 100% | 775x |
| 4,096 | 3.257 | 0.7952 | 1,257,509 | 34.8 | 0.34 | 100% | 2,559x |
| 16,384 | 5.622 | 0.3431 | 2,914,271 | 80.6 | 0.69 | 100% | 5,931x |
| 65,536 | 31.155 | 0.4754 | 2,103,552 | 58.2 | 2.13 | 100% | 4,281x |

Storage: 20 B of state against 27,648 B of raw frame (96x96) — **1,382x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 1.615 | 0.054 | 1.776 | 18,536 | 0.5 |
| 256 | 2.122 | 0.162 | 2.345 | 1,581,769 | 43.7 |
| 4,096 | 2.138 | 1.009 | 3.271 | 4,060,052 | 112.3 |
| 16,384 | 2.137 | 3.350 | 5.615 | 4,890,308 | 135.2 |

### 224x224 (supersample 1)

Official CPU renderer: **438 frames/s** (2,282 us/frame, 0.0660 GB/s) for set_state+render; 471 frames/s (2,125 us/frame) for the draw alone.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 1.760 | 1759.6680 | 568 | 0.1 | 0.22 | 100% | 1.30x |
| 64 | 2.393 | 37.3911 | 26,744 | 4.0 | 0.23 | 100% | 61x |
| 256 | 2.741 | 10.7087 | 93,382 | 14.1 | 0.26 | 100% | 213x |
| 1,024 | 3.550 | 3.4671 | 288,428 | 43.4 | 0.37 | 100% | 658x |
| 4,096 | 6.741 | 1.6459 | 607,585 | 91.5 | 0.84 | 100% | 1,386x |
| 16,384 | 19.520 | 1.1914 | 839,360 | 126.3 | 2.71 | 100% | 1,915x |
| 65,536 | output would be 9.9 GB, over the 6.0 GB budget | | | | | | |

Storage: 20 B of state against 150,528 B of raw frame (224x224) — **7,526x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 1.625 | 0.056 | 1.760 | 17,969 | 2.7 |
| 256 | 2.123 | 0.487 | 2.729 | 526,137 | 79.2 |
| 4,096 | 2.135 | 4.483 | 6.754 | 913,734 | 137.5 |
| 16,384 | 2.122 | 17.232 | 19.503 | 950,807 | 143.1 |

## Acrobot

State: 2 floats `theta1, theta2`. Reference: `Acrobot-v1`.

### 500x500 (supersample 1)

Official CPU renderer: **278 frames/s** (3,596 us/frame, 0.2086 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 67.968 | 67968.1553 | 15 | 0.0 | 0.00 | 100% | 0.05x |
| 64 | 68.829 | 1075.4507 | 930 | 0.7 | 0.16 | 100% | 3.34x |
| 256 | 71.393 | 278.8788 | 3,586 | 2.7 | 0.64 | 100% | 13x |
| 1,024 | 78.595 | 76.7528 | 13,029 | 9.8 | 2.56 | 100% | 47x |
| 4,096 | 130.174 | 31.7808 | 31,466 | 23.6 | 10.24 | 100% | 113x |
| 16,384 | output would be 12.3 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 49.2 GB, over the 10.0 GB budget | | | | | | |

Storage: 8 B of state against 750,000 B of raw frame (500x500) — **93,750x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

### 96x96

Official CPU renderer unavailable: ValueError: Acrobot renders at its reference's native 500x500 and nothing else (got size=96). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

### 224x224

Official CPU renderer unavailable: ValueError: Acrobot renders at its reference's native 500x500 and nothing else (got size=224). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

## AntmazeGiantNavigate

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-giant-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **351 frames/s** (2,849 us/frame, 0.0528 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.160 | 3159.7214 | 316 | 0.0 | 0.01 | 100% | 0.90x |
| 64 | 3.714 | 58.0242 | 17,234 | 2.6 | 0.05 | 100% | 49x |
| 256 | 5.290 | 20.6623 | 48,397 | 7.3 | 0.20 | 100% | 138x |
| 1,024 | 13.313 | 13.0013 | 76,915 | 11.6 | 0.77 | 725% | 219x |
| 4,096 | 47.013 | 11.4778 | 87,125 | 13.1 | 3.06 | 293% | 248x |
| 16,384 | 185.678 | 11.3329 | 88,239 | 13.3 | 12.22 | 148% | 251x |
| 65,536 | 796.934 | 12.1602 | 82,235 | 12.4 | 48.87 | 114% | 234x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **445 frames/s** (2,246 us/frame, 0.0123 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.173 | 3172.7385 | 315 | 0.0 | 0.00 | 100% | 0.71x |
| 64 | 3.710 | 57.9634 | 17,252 | 0.5 | 0.04 | 100% | 39x |
| 256 | 5.308 | 20.7332 | 48,232 | 1.3 | 0.16 | 100% | 108x |
| 1,024 | 13.148 | 12.8396 | 77,884 | 2.2 | 0.64 | 729% | 175x |
| 4,096 | 46.193 | 11.2776 | 88,671 | 2.5 | 2.55 | 283% | 199x |
| 16,384 | 182.705 | 11.1514 | 89,675 | 2.5 | 10.21 | 149% | 201x |
| 65,536 | 784.392 | 11.9689 | 83,550 | 2.3 | 40.81 | 114% | 188x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeGiantStitch

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-giant-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **400 frames/s** (2,498 us/frame, 0.0603 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.171 | 3170.8051 | 315 | 0.0 | 0.01 | 100% | 0.79x |
| 64 | 3.724 | 58.1868 | 17,186 | 2.6 | 0.05 | 100% | 43x |
| 256 | 5.289 | 20.6601 | 48,403 | 7.3 | 0.20 | 100% | 121x |
| 1,024 | 13.330 | 13.0178 | 76,818 | 11.6 | 0.77 | 720% | 192x |
| 4,096 | 47.050 | 11.4867 | 87,057 | 13.1 | 3.06 | 281% | 217x |
| 16,384 | 185.710 | 11.3349 | 88,223 | 13.3 | 12.22 | 148% | 220x |
| 65,536 | 796.367 | 12.1516 | 82,294 | 12.4 | 48.87 | 113% | 206x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **444 frames/s** (2,252 us/frame, 0.0123 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.177 | 3177.1306 | 315 | 0.0 | 0.00 | 100% | 0.71x |
| 64 | 3.718 | 58.0986 | 17,212 | 0.5 | 0.04 | 100% | 39x |
| 256 | 5.291 | 20.6670 | 48,386 | 1.3 | 0.16 | 100% | 109x |
| 1,024 | 13.158 | 12.8499 | 77,821 | 2.2 | 0.64 | 730% | 175x |
| 4,096 | 46.284 | 11.2999 | 88,496 | 2.4 | 2.55 | 284% | 199x |
| 16,384 | 182.675 | 11.1496 | 89,689 | 2.5 | 10.21 | 159% | 202x |
| 65,536 | 785.308 | 11.9828 | 83,453 | 2.3 | 40.81 | 114% | 188x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeLargeExplore

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-large-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **433 frames/s** (2,310 us/frame, 0.0652 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.160 | 3160.1703 | 316 | 0.0 | 0.01 | 100% | 0.73x |
| 64 | 3.682 | 57.5311 | 17,382 | 2.6 | 0.06 | 100% | 40x |
| 256 | 5.240 | 20.4707 | 48,850 | 7.4 | 0.20 | 100% | 113x |
| 1,024 | 12.927 | 12.6238 | 79,216 | 11.9 | 0.77 | 743% | 183x |
| 4,096 | 45.531 | 11.1160 | 89,960 | 13.5 | 3.06 | 286% | 208x |
| 16,384 | 179.697 | 10.9678 | 91,176 | 13.7 | 12.22 | 150% | 211x |
| 65,536 | 771.516 | 11.7724 | 84,944 | 12.8 | 48.87 | 114% | 196x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **485 frames/s** (2,060 us/frame, 0.0134 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.182 | 3182.2268 | 314 | 0.0 | 0.01 | 100% | 0.65x |
| 64 | 3.705 | 57.8880 | 17,275 | 0.5 | 0.05 | 100% | 36x |
| 256 | 5.186 | 20.2572 | 49,365 | 1.4 | 0.17 | 100% | 102x |
| 1,024 | 12.779 | 12.4790 | 80,135 | 2.2 | 0.65 | 746% | 165x |
| 4,096 | 44.818 | 10.9419 | 91,392 | 2.5 | 2.56 | 290% | 188x |
| 16,384 | 176.932 | 10.7991 | 92,600 | 2.6 | 10.21 | 160% | 191x |
| 65,536 | 759.719 | 11.5924 | 86,263 | 2.4 | 40.82 | 114% | 178x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeLargeNavigate

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-large-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **435 frames/s** (2,299 us/frame, 0.0655 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.163 | 3162.6504 | 316 | 0.0 | 0.01 | 100% | 0.73x |
| 64 | 3.680 | 57.4995 | 17,391 | 2.6 | 0.06 | 100% | 40x |
| 256 | 5.187 | 20.2608 | 49,356 | 7.4 | 0.20 | 100% | 113x |
| 1,024 | 12.964 | 12.6605 | 78,986 | 11.9 | 0.77 | 738% | 182x |
| 4,096 | 45.731 | 11.1648 | 89,567 | 13.5 | 3.06 | 286% | 206x |
| 16,384 | 179.578 | 10.9606 | 91,236 | 13.7 | 12.22 | 150% | 210x |
| 65,536 | 774.419 | 11.8167 | 84,626 | 12.7 | 48.87 | 114% | 195x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **486 frames/s** (2,059 us/frame, 0.0134 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.179 | 3179.1450 | 315 | 0.0 | 0.01 | 100% | 0.65x |
| 64 | 3.688 | 57.6280 | 17,353 | 0.5 | 0.05 | 100% | 36x |
| 256 | 5.234 | 20.4436 | 48,915 | 1.4 | 0.17 | 100% | 101x |
| 1,024 | 12.774 | 12.4749 | 80,161 | 2.2 | 0.65 | 752% | 165x |
| 4,096 | 44.750 | 10.9252 | 91,532 | 2.5 | 2.56 | 300% | 188x |
| 16,384 | 176.615 | 10.7797 | 92,767 | 2.6 | 10.21 | 150% | 191x |
| 65,536 | 760.220 | 11.6000 | 86,207 | 2.4 | 40.82 | 114% | 178x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeLargeStitch

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-large-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **444 frames/s** (2,254 us/frame, 0.0668 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.159 | 3158.6653 | 317 | 0.0 | 0.01 | 100% | 0.71x |
| 64 | 3.692 | 57.6927 | 17,333 | 2.6 | 0.06 | 100% | 39x |
| 256 | 5.252 | 20.5145 | 48,746 | 7.3 | 0.20 | 100% | 110x |
| 1,024 | 12.976 | 12.6717 | 78,916 | 11.9 | 0.77 | 739% | 178x |
| 4,096 | 45.592 | 11.1310 | 89,839 | 13.5 | 3.06 | 287% | 203x |
| 16,384 | 179.539 | 10.9582 | 91,256 | 13.7 | 12.22 | 150% | 206x |
| 65,536 | 772.679 | 11.7902 | 84,817 | 12.8 | 48.87 | 113% | 191x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **486 frames/s** (2,058 us/frame, 0.0134 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.164 | 3164.3119 | 316 | 0.0 | 0.01 | 100% | 0.65x |
| 64 | 3.687 | 57.6054 | 17,359 | 0.5 | 0.05 | 100% | 36x |
| 256 | 5.198 | 20.3033 | 49,253 | 1.4 | 0.17 | 100% | 101x |
| 1,024 | 12.750 | 12.4513 | 80,313 | 2.2 | 0.65 | 750% | 165x |
| 4,096 | 44.946 | 10.9730 | 91,132 | 2.5 | 2.56 | 289% | 188x |
| 16,384 | 177.039 | 10.8056 | 92,544 | 2.6 | 10.21 | 150% | 190x |
| 65,536 | 760.479 | 11.6040 | 86,177 | 2.4 | 40.82 | 114% | 177x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeMediumExplore

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-medium-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **458 frames/s** (2,185 us/frame, 0.0689 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.152 | 3152.3574 | 317 | 0.0 | 0.01 | 100% | 0.69x |
| 64 | 3.665 | 57.2624 | 17,463 | 2.6 | 0.05 | 100% | 38x |
| 256 | 5.126 | 20.0234 | 49,941 | 7.5 | 0.20 | 100% | 109x |
| 1,024 | 12.811 | 12.5106 | 79,932 | 12.0 | 0.77 | 749% | 175x |
| 4,096 | 44.912 | 10.9648 | 91,201 | 13.7 | 3.06 | 290% | 199x |
| 16,384 | 176.644 | 10.7815 | 92,751 | 14.0 | 12.22 | 150% | 203x |
| 65,536 | 761.750 | 11.6234 | 86,034 | 13.0 | 48.87 | 114% | 188x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **508 frames/s** (1,970 us/frame, 0.0140 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.147 | 3146.7257 | 318 | 0.0 | 0.01 | 100% | 0.63x |
| 64 | 3.670 | 57.3376 | 17,441 | 0.5 | 0.04 | 100% | 34x |
| 256 | 5.129 | 20.0363 | 49,909 | 1.4 | 0.16 | 100% | 98x |
| 1,024 | 12.541 | 12.2472 | 81,652 | 2.3 | 0.64 | 763% | 161x |
| 4,096 | 44.001 | 10.7425 | 93,088 | 2.6 | 2.56 | 293% | 183x |
| 16,384 | 173.354 | 10.5807 | 94,512 | 2.6 | 10.21 | 152% | 186x |
| 65,536 | 746.620 | 11.3925 | 87,777 | 2.4 | 40.82 | 114% | 173x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeMediumNavigate

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-medium-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **467 frames/s** (2,142 us/frame, 0.0703 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.163 | 3162.5386 | 316 | 0.0 | 0.01 | 100% | 0.68x |
| 64 | 3.668 | 57.3082 | 17,450 | 2.6 | 0.05 | 100% | 37x |
| 256 | 5.121 | 20.0026 | 49,993 | 7.5 | 0.20 | 100% | 107x |
| 1,024 | 12.831 | 12.5301 | 79,808 | 12.0 | 0.77 | 743% | 171x |
| 4,096 | 44.790 | 10.9352 | 91,448 | 13.8 | 3.06 | 290% | 196x |
| 16,384 | 176.877 | 10.7957 | 92,629 | 13.9 | 12.22 | 150% | 198x |
| 65,536 | 759.169 | 11.5840 | 86,326 | 13.0 | 48.87 | 114% | 185x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **507 frames/s** (1,972 us/frame, 0.0140 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.161 | 3160.6928 | 316 | 0.0 | 0.01 | 100% | 0.62x |
| 64 | 3.665 | 57.2726 | 17,460 | 0.5 | 0.04 | 100% | 34x |
| 256 | 5.141 | 20.0805 | 49,799 | 1.4 | 0.16 | 100% | 98x |
| 1,024 | 12.593 | 12.2977 | 81,316 | 2.2 | 0.64 | 760% | 160x |
| 4,096 | 44.058 | 10.7563 | 92,968 | 2.6 | 2.56 | 293% | 183x |
| 16,384 | 174.770 | 10.6671 | 93,746 | 2.6 | 10.21 | 151% | 185x |
| 65,536 | 756.092 | 11.5370 | 86,677 | 2.4 | 40.82 | 116% | 171x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeMediumStitch

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-medium-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **466 frames/s** (2,148 us/frame, 0.0701 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.165 | 3165.1873 | 316 | 0.0 | 0.01 | 100% | 0.68x |
| 64 | 3.664 | 57.2532 | 17,466 | 2.6 | 0.05 | 100% | 38x |
| 256 | 5.127 | 20.0291 | 49,927 | 7.5 | 0.20 | 100% | 107x |
| 1,024 | 12.760 | 12.4613 | 80,249 | 12.1 | 0.77 | 751% | 172x |
| 4,096 | 44.820 | 10.9424 | 91,388 | 13.8 | 3.06 | 290% | 196x |
| 16,384 | 177.927 | 10.8598 | 92,083 | 13.9 | 12.22 | 150% | 198x |
| 65,536 | 768.765 | 11.7304 | 85,248 | 12.8 | 48.87 | 115% | 183x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **507 frames/s** (1,972 us/frame, 0.0140 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.155 | 3155.1523 | 317 | 0.0 | 0.01 | 100% | 0.62x |
| 64 | 3.660 | 57.1842 | 17,487 | 0.5 | 0.04 | 100% | 34x |
| 256 | 5.131 | 20.0427 | 49,893 | 1.4 | 0.16 | 100% | 98x |
| 1,024 | 12.575 | 12.2801 | 81,432 | 2.3 | 0.64 | 762% | 161x |
| 4,096 | 44.135 | 10.7753 | 92,805 | 2.6 | 2.56 | 313% | 183x |
| 16,384 | 174.169 | 10.6304 | 94,070 | 2.6 | 10.21 | 165% | 185x |
| 65,536 | 757.887 | 11.5644 | 86,472 | 2.4 | 40.82 | 116% | 171x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeTeleportExplore

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-teleport-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **417 frames/s** (2,401 us/frame, 0.0627 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.152 | 3151.7614 | 317 | 0.0 | 0.01 | 100% | 0.76x |
| 64 | 3.681 | 57.5203 | 17,385 | 2.6 | 0.06 | 100% | 42x |
| 256 | 5.178 | 20.2266 | 49,440 | 7.4 | 0.20 | 100% | 119x |
| 1,024 | 13.033 | 12.7279 | 78,567 | 11.8 | 0.77 | 741% | 189x |
| 4,096 | 45.705 | 11.1584 | 89,619 | 13.5 | 3.06 | 287% | 215x |
| 16,384 | 180.836 | 11.0373 | 90,602 | 13.6 | 12.22 | 149% | 218x |
| 65,536 | 774.322 | 11.8152 | 84,637 | 12.7 | 48.87 | 114% | 203x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **456 frames/s** (2,193 us/frame, 0.0126 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.161 | 3160.6164 | 316 | 0.0 | 0.01 | 100% | 0.69x |
| 64 | 3.686 | 57.5865 | 17,365 | 0.5 | 0.05 | 100% | 38x |
| 256 | 5.249 | 20.5027 | 48,774 | 1.3 | 0.17 | 100% | 107x |
| 1,024 | 12.787 | 12.4869 | 80,084 | 2.2 | 0.65 | 746% | 176x |
| 4,096 | 44.952 | 10.9746 | 91,120 | 2.5 | 2.56 | 290% | 200x |
| 16,384 | 177.463 | 10.8315 | 92,323 | 2.6 | 10.21 | 150% | 202x |
| 65,536 | 763.211 | 11.6457 | 85,869 | 2.4 | 40.82 | 114% | 188x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeTeleportNavigate

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-teleport-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **425 frames/s** (2,352 us/frame, 0.0640 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.182 | 3182.0536 | 314 | 0.0 | 0.01 | 100% | 0.74x |
| 64 | 3.694 | 57.7124 | 17,327 | 2.6 | 0.06 | 100% | 41x |
| 256 | 5.199 | 20.3091 | 49,239 | 7.4 | 0.20 | 100% | 116x |
| 1,024 | 13.001 | 12.6965 | 78,762 | 11.9 | 0.77 | 737% | 185x |
| 4,096 | 45.815 | 11.1853 | 89,403 | 13.5 | 3.06 | 285% | 210x |
| 16,384 | 182.313 | 11.1275 | 89,867 | 13.5 | 12.23 | 174% | 211x |
| 65,536 | 783.048 | 11.9484 | 83,693 | 12.6 | 48.87 | 114% | 197x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **457 frames/s** (2,187 us/frame, 0.0126 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.167 | 3167.0779 | 316 | 0.0 | 0.01 | 100% | 0.69x |
| 64 | 3.690 | 57.6555 | 17,344 | 0.5 | 0.05 | 100% | 38x |
| 256 | 5.248 | 20.4985 | 48,784 | 1.3 | 0.17 | 100% | 107x |
| 1,024 | 12.832 | 12.5313 | 79,800 | 2.2 | 0.65 | 758% | 174x |
| 4,096 | 45.099 | 11.0106 | 90,822 | 2.5 | 2.56 | 288% | 199x |
| 16,384 | 177.788 | 10.8513 | 92,154 | 2.5 | 10.21 | 150% | 202x |
| 65,536 | 763.320 | 11.6473 | 85,856 | 2.4 | 40.82 | 114% | 188x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## AntmazeTeleportStitch

State: 15 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, hip_1 ...`. Reference: `ogbench/antmaze-teleport-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **424 frames/s** (2,357 us/frame, 0.0639 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.181 | 3180.9499 | 314 | 0.0 | 0.01 | 100% | 0.74x |
| 64 | 3.693 | 57.7045 | 17,330 | 2.6 | 0.06 | 100% | 41x |
| 256 | 5.210 | 20.3518 | 49,136 | 7.4 | 0.20 | 100% | 116x |
| 1,024 | 13.015 | 12.7098 | 78,679 | 11.8 | 0.77 | 738% | 185x |
| 4,096 | 45.844 | 11.1924 | 89,346 | 13.4 | 3.06 | 286% | 211x |
| 16,384 | 181.031 | 11.0493 | 90,504 | 13.6 | 12.23 | 149% | 213x |
| 65,536 | 777.440 | 11.8628 | 84,297 | 12.7 | 48.87 | 114% | 199x |

Storage: 60 B of state against 150,528 B of raw frame (224x224) — **2,509x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **457 frames/s** (2,189 us/frame, 0.0126 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.184 | 3183.8110 | 314 | 0.0 | 0.01 | 100% | 0.69x |
| 64 | 3.702 | 57.8474 | 17,287 | 0.5 | 0.05 | 100% | 38x |
| 256 | 5.208 | 20.3455 | 49,151 | 1.4 | 0.17 | 100% | 108x |
| 1,024 | 12.812 | 12.5122 | 79,922 | 2.2 | 0.65 | 749% | 175x |
| 4,096 | 45.167 | 11.0272 | 90,685 | 2.5 | 2.56 | 289% | 199x |
| 16,384 | 177.351 | 10.8246 | 92,382 | 2.6 | 10.21 | 150% | 202x |
| 65,536 | 763.567 | 11.6511 | 85,829 | 2.4 | 40.82 | 114% | 188x |

Storage: 60 B of state against 27,648 B of raw frame (96x96) — **461x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## BabyAIGoToLocal

State: 67 floats `cell[0,0], cell[0,1], cell[0,2], cell[0,3], cell[0,4], cell[0,5], cell[0,6], cell[0,7] ...`. Reference: `BabyAI-GoToLocal-v0`.

### 256x256 (supersample 1)

Official CPU renderer: **651 frames/s** (1,535 us/frame, 0.1281 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.695 | 2695.0594 | 371 | 0.1 | 0.03 | 100% | 0.57x |
| 64 | 2.754 | 43.0313 | 23,239 | 4.6 | 0.04 | 100% | 36x |
| 256 | 2.770 | 10.8194 | 92,427 | 18.2 | 0.08 | 100% | 142x |
| 1,024 | 2.905 | 2.8373 | 352,451 | 69.3 | 0.23 | 100% | 541x |
| 4,096 | 3.380 | 0.8252 | 1,211,765 | 238.2 | 0.86 | 100% | 1,860x |
| 16,384 | 5.307 | 0.3239 | 3,087,381 | 607.0 | 3.36 | 100% | 4,739x |
| 65,536 | output would be 12.9 GB, over the 10.0 GB budget | | | | | | |

Storage: 268 B of state against 196,608 B of raw frame (256x256) — **734x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 1)

Official CPU renderer: **1,139 frames/s** (878 us/frame, 0.0315 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.694 | 2694.1970 | 371 | 0.0 | 0.03 | 100% | 0.33x |
| 64 | 2.750 | 42.9729 | 23,270 | 0.6 | 0.03 | 100% | 20x |
| 256 | 2.755 | 10.7636 | 92,906 | 2.6 | 0.04 | 100% | 82x |
| 1,024 | 2.796 | 2.7308 | 366,194 | 10.1 | 0.06 | 100% | 321x |
| 4,096 | 2.978 | 0.7272 | 1,375,207 | 38.0 | 0.17 | 100% | 1,207x |
| 16,384 | 3.714 | 0.2267 | 4,411,056 | 122.0 | 0.59 | 100% | 3,871x |
| 65,536 | 6.845 | 0.1044 | 9,574,837 | 264.7 | 2.29 | 100% | 8,403x |

Storage: 268 B of state against 27,648 B of raw frame (96x96) — **103x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 1)

Official CPU renderer: **745 frames/s** (1,342 us/frame, 0.1122 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.692 | 2692.4182 | 371 | 0.1 | 0.04 | 100% | 0.50x |
| 64 | 2.755 | 43.0421 | 23,233 | 3.5 | 0.05 | 100% | 31x |
| 256 | 2.769 | 10.8149 | 92,465 | 13.9 | 0.08 | 100% | 124x |
| 1,024 | 2.886 | 2.8187 | 354,775 | 53.4 | 0.20 | 100% | 476x |
| 4,096 | 3.303 | 0.8065 | 1,239,945 | 186.6 | 0.68 | 100% | 1,664x |
| 16,384 | 5.000 | 0.3052 | 3,276,621 | 493.2 | 2.62 | 100% | 4,396x |
| 65,536 | 11.922 | 0.1819 | 5,497,058 | 827.5 | 10.35 | 100% | 7,375x |

Storage: 268 B of state against 150,528 B of raw frame (224x224) — **562x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## CartPole

State: 2 floats `x, theta`. Reference: `CartPole-v1`.

### 600x600 (supersample 1)

Official CPU renderer: **438 frames/s** (2,283 us/frame, 0.4731 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 67.296 | 67295.9732 | 15 | 0.0 | 0.04 | 100% | 0.03x |
| 64 | 68.938 | 1077.1586 | 928 | 0.7 | 0.19 | 100% | 2.12x |
| 256 | 70.601 | 275.7836 | 3,626 | 2.6 | 0.65 | 100% | 8.28x |
| 1,024 | 76.661 | 74.8642 | 13,358 | 9.6 | 2.50 | 100% | 30x |
| 4,096 | 115.227 | 28.1316 | 35,547 | 25.6 | 9.87 | 100% | 81x |
| 16,384 | output would be 11.8 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 47.2 GB, over the 10.0 GB budget | | | | | | |

Storage: 8 B of state against 720,000 B of raw frame (400x600) — **90,000x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

### 96x96

Official CPU renderer unavailable: ValueError: CartPole renders at its reference's native 400x600 and nothing else (got size=96). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

### 224x224

Official CPU renderer unavailable: ValueError: CartPole renders at its reference's native 400x600 and nothing else (got size=224). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

## CraftaxClassic

State: 127 floats `block_r0c0, block_r0c1, block_r0c2, block_r0c3, block_r0c4, block_r0c5, block_r0c6, block_r0c7 ...`. Reference: `Craftax-Classic-Pixels-v1`.

### 16x16 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'jax'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.433 | 432.6124 | 2,312 | 0.1 | 0.04 | 100% | - |
| 64 | 0.428 | 6.6894 | 149,490 | 7.2 | 0.11 | 100% | - |
| 256 | 0.688 | 2.6860 | 372,302 | 18.0 | 0.30 | 100% | - |
| 1,024 | 1.959 | 1.9131 | 522,698 | 25.3 | 1.09 | 100% | - |
| 4,096 | 7.203 | 1.7585 | 568,657 | 27.5 | 4.21 | 100% | - |
| 16,384 | 28.275 | 1.7258 | 579,444 | 28.0 | 16.72 | 100% | - |
| 65,536 | 112.585 | 1.7179 | 582,104 | 28.2 | 66.76 | 100% | - |

Storage: 508 B of state against 48,384 B of raw frame (112x144) — **95x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 1)

Official CPU renderer unavailable: ValueError: craftax draws at 7 px (the agent observation) or 16 px per tile; there is no official frame to compare a 96 px render against

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.426 | 425.6405 | 2,349 | 4.1 | 0.08 | 100% | - |
| 64 | 4.086 | 63.8480 | 15,662 | 27.3 | 2.39 | 100% | - |
| 256 | 15.861 | 61.9567 | 16,140 | 28.1 | 9.41 | 100% | - |
| 1,024 | 62.996 | 61.5200 | 16,255 | 28.3 | 37.50 | 100% | - |
| 4,096 | OOM | | | | | | |
| 16,384 | output would be 28.5 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 114.2 GB, over the 10.0 GB budget | | | | | | |

Storage: 508 B of state against 1,741,824 B of raw frame (672x864) — **3,429x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

### 224x224 (supersample 1)

Official CPU renderer unavailable: ValueError: craftax draws at 7 px (the agent observation) or 16 px per tile; there is no official frame to compare a 224 px render against

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.576 | 575.5546 | 1,737 | 16.5 | 0.27 | 100% | - |
| 64 | 21.576 | 337.1295 | 2,966 | 28.1 | 12.82 | 100% | - |
| 256 | 85.857 | 335.3777 | 2,982 | 28.3 | 51.06 | 100% | - |
| 1,024 | OOM | | | | | | |
| 4,096 | output would be 38.8 GB, over the 10.0 GB budget | | | | | | |
| 16,384 | output would be 155.4 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 621.5 GB, over the 10.0 GB budget | | | | | | |

Storage: 508 B of state against 9,483,264 B of raw frame (1568x2016) — **18,668x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

## Crafter

State: 80 floats `cell[0,0], cell[0,1], cell[0,2], cell[0,3], cell[0,4], cell[0,5], cell[0,6], cell[1,0] ...`. Reference: `CrafterReward-v1`.

### 64x64 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'crafter'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.963 | 962.8516 | 1,039 | 0.0 | 0.07 | 100% | - |
| 64 | 1.167 | 18.2377 | 54,832 | 0.7 | 0.10 | 100% | - |
| 256 | 1.184 | 4.6231 | 216,305 | 2.7 | 0.21 | 100% | - |
| 1,024 | 2.223 | 2.1707 | 460,691 | 5.7 | 0.59 | 100% | - |
| 4,096 | 6.229 | 1.5208 | 657,557 | 8.1 | 0.81 | 100% | - |
| 16,384 | 23.694 | 1.4462 | 691,470 | 8.5 | 1.28 | 100% | - |
| 65,536 | 92.257 | 1.4077 | 710,362 | 8.7 | 3.19 | 100% | - |

Storage: 320 B of state against 12,288 B of raw frame (64x64) — **38x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'crafter'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.956 | 955.6348 | 1,046 | 0.0 | 0.07 | 100% | - |
| 64 | 1.169 | 18.2641 | 54,752 | 1.5 | 0.14 | 100% | - |
| 256 | 1.326 | 5.1799 | 193,055 | 5.3 | 0.36 | 100% | - |
| 1,024 | 3.545 | 3.4619 | 288,855 | 8.0 | 0.73 | 100% | - |
| 4,096 | 12.200 | 2.9785 | 335,738 | 9.3 | 0.97 | 100% | - |
| 16,384 | 46.831 | 2.8583 | 349,857 | 9.7 | 1.94 | 100% | - |
| 65,536 | 185.675 | 2.8332 | 352,960 | 9.8 | 5.82 | 100% | - |

Storage: 320 B of state against 27,648 B of raw frame (96x96) — **86x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'crafter'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.967 | 966.9196 | 1,034 | 0.2 | 0.07 | 100% | - |
| 64 | 1.588 | 24.8174 | 40,294 | 6.1 | 0.50 | 100% | - |
| 256 | 4.779 | 18.6661 | 53,573 | 8.1 | 0.76 | 100% | - |
| 1,024 | 16.991 | 16.5929 | 60,267 | 9.1 | 1.09 | 100% | - |
| 4,096 | 66.234 | 16.1703 | 61,842 | 9.3 | 2.42 | 100% | - |
| 16,384 | 262.828 | 16.0417 | 62,337 | 9.4 | 7.75 | 100% | - |
| 65,536 | 1049.140 | 16.0086 | 62,466 | 9.4 | 29.04 | 100% | - |

Storage: 320 B of state against 150,528 B of raw frame (224x224) — **470x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## Highway

State: 48 floats `ego_x, ego_y, ego_heading, ego_crashed, v1_x, v1_y, v1_heading, v1_crashed ...`. Reference: `highway-v0`.

### 150x150 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.703 | 3702.5493 | 270 | 0.1 | 0.10 | 100% | - |
| 64 | 3.995 | 62.4172 | 16,021 | 4.3 | 0.12 | 100% | - |
| 256 | 4.742 | 18.5241 | 53,984 | 14.6 | 0.18 | 100% | - |
| 1,024 | 7.731 | 7.5498 | 132,454 | 35.8 | 0.40 | 100% | - |
| 4,096 | 20.614 | 5.0328 | 198,696 | 53.6 | 1.28 | 100% | - |
| 16,384 | 84.981 | 5.1868 | 192,797 | 52.1 | 4.81 | 100% | - |
| 65,536 | output would be 17.7 GB, over the 10.0 GB budget | | | | | | |

Storage: 192 B of state against 270,000 B of raw frame (150x600) — **1,406x** smaller before any compression.

### 96x96 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.685 | 3684.5487 | 271 | 0.0 | 0.10 | 100% | - |
| 64 | 3.831 | 59.8612 | 16,705 | 1.8 | 0.11 | 100% | - |
| 256 | 4.096 | 15.9987 | 62,505 | 6.9 | 0.14 | 100% | - |
| 1,024 | 5.138 | 5.0177 | 199,295 | 22.0 | 0.23 | 100% | - |
| 4,096 | 9.760 | 2.3829 | 419,659 | 46.4 | 0.61 | 100% | - |
| 16,384 | 32.167 | 1.9633 | 509,348 | 56.3 | 2.14 | 100% | - |
| 65,536 | 121.227 | 1.8498 | 540,608 | 59.8 | 8.25 | 100% | - |

Storage: 192 B of state against 110,592 B of raw frame (96x384) — **576x** smaller before any compression.

### 224x224 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.687 | 3686.7131 | 271 | 0.2 | 0.10 | 100% | - |
| 64 | 4.339 | 67.7910 | 14,751 | 8.9 | 0.14 | 100% | - |
| 256 | 6.220 | 24.2988 | 41,154 | 24.8 | 0.26 | 100% | - |
| 1,024 | 13.797 | 13.4741 | 74,216 | 44.7 | 0.74 | 100% | - |
| 4,096 | 46.079 | 11.2498 | 88,890 | 53.5 | 2.66 | 100% | - |
| 16,384 | 210.109 | 12.8240 | 77,979 | 47.0 | 10.33 | 100% | - |
| 65,536 | output would be 39.5 GB, over the 10.0 GB budget | | | | | | |

Storage: 192 B of state against 602,112 B of raw frame (224x896) — **3,136x** smaller before any compression.

## HumanoidmazeGiantNavigate

State: 28 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, abdomen_z ...`. Reference: `ogbench/humanoidmaze-giant-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **386 frames/s** (2,593 us/frame, 0.0581 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.197 | 3197.2416 | 313 | 0.0 | 0.10 | 100% | 0.81x |
| 64 | 3.824 | 59.7427 | 16,738 | 2.5 | 0.15 | 100% | 43x |
| 256 | 5.767 | 22.5265 | 44,392 | 6.7 | 0.29 | 1510% | 115x |
| 1,024 | 14.881 | 14.5324 | 68,812 | 10.4 | 0.86 | 662% | 178x |
| 4,096 | 53.273 | 13.0061 | 76,887 | 11.6 | 3.16 | 259% | 199x |
| 16,384 | 213.092 | 13.0061 | 76,887 | 11.6 | 12.32 | 144% | 199x |
| 65,536 | 932.543 | 14.2295 | 70,277 | 10.6 | 49.00 | 117% | 182x |

Storage: 112 B of state against 150,528 B of raw frame (224x224) — **1,344x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **444 frames/s** (2,251 us/frame, 0.0123 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.193 | 3193.0581 | 313 | 0.0 | 0.10 | 100% | 0.71x |
| 64 | 3.817 | 59.6426 | 16,767 | 0.5 | 0.14 | 100% | 38x |
| 256 | 5.778 | 22.5711 | 44,304 | 1.2 | 0.26 | 1513% | 100x |
| 1,024 | 14.757 | 14.4109 | 69,392 | 1.9 | 0.74 | 662% | 156x |
| 4,096 | 52.594 | 12.8402 | 77,880 | 2.2 | 2.65 | 286% | 175x |
| 16,384 | 207.981 | 12.6941 | 78,776 | 2.2 | 10.31 | 144% | 177x |
| 65,536 | 911.307 | 13.9054 | 71,914 | 2.0 | 40.94 | 117% | 162x |

Storage: 112 B of state against 27,648 B of raw frame (96x96) — **247x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## HumanoidmazeGiantStitch

State: 28 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, abdomen_z ...`. Reference: `ogbench/humanoidmaze-giant-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **398 frames/s** (2,512 us/frame, 0.0599 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.191 | 3191.1219 | 313 | 0.0 | 0.10 | 100% | 0.79x |
| 64 | 3.824 | 59.7542 | 16,735 | 2.5 | 0.15 | 100% | 42x |
| 256 | 5.767 | 22.5260 | 44,393 | 6.7 | 0.29 | 1510% | 112x |
| 1,024 | 14.953 | 14.6027 | 68,480 | 10.3 | 0.87 | 654% | 172x |
| 4,096 | 53.438 | 13.0463 | 76,650 | 11.5 | 3.16 | 258% | 193x |
| 16,384 | 212.529 | 12.9717 | 77,091 | 11.6 | 12.33 | 143% | 194x |
| 65,536 | 938.420 | 14.3191 | 69,837 | 10.5 | 49.00 | 120% | 175x |

Storage: 112 B of state against 150,528 B of raw frame (224x224) — **1,344x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **443 frames/s** (2,255 us/frame, 0.0123 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.194 | 3194.1924 | 313 | 0.0 | 0.10 | 100% | 0.71x |
| 64 | 3.825 | 59.7669 | 16,732 | 0.5 | 0.14 | 100% | 38x |
| 256 | 5.753 | 22.4741 | 44,496 | 1.2 | 0.26 | 1511% | 100x |
| 1,024 | 14.724 | 14.3790 | 69,546 | 1.9 | 0.74 | 665% | 157x |
| 4,096 | 52.597 | 12.8411 | 77,875 | 2.2 | 2.65 | 261% | 176x |
| 16,384 | 209.206 | 12.7689 | 78,315 | 2.2 | 10.31 | 144% | 177x |
| 65,536 | 921.001 | 14.0534 | 71,157 | 2.0 | 40.94 | 118% | 160x |

Storage: 112 B of state against 27,648 B of raw frame (96x96) — **247x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## HumanoidmazeLargeNavigate

State: 28 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, abdomen_z ...`. Reference: `ogbench/humanoidmaze-large-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **359 frames/s** (2,785 us/frame, 0.0540 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.176 | 3176.2673 | 315 | 0.0 | 0.10 | 100% | 0.88x |
| 64 | 3.770 | 58.9105 | 16,975 | 2.6 | 0.15 | 100% | 47x |
| 256 | 5.624 | 21.9676 | 45,521 | 6.9 | 0.29 | 1540% | 127x |
| 1,024 | 14.521 | 14.1807 | 70,519 | 10.6 | 0.87 | 671% | 196x |
| 4,096 | 51.742 | 12.6323 | 79,162 | 11.9 | 3.16 | 277% | 220x |
| 16,384 | 206.266 | 12.5895 | 79,431 | 12.0 | 12.33 | 145% | 221x |
| 65,536 | 899.398 | 13.7237 | 72,867 | 11.0 | 49.00 | 118% | 203x |

Storage: 112 B of state against 150,528 B of raw frame (224x224) — **1,344x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **492 frames/s** (2,034 us/frame, 0.0136 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.158 | 3158.1596 | 317 | 0.0 | 0.10 | 100% | 0.64x |
| 64 | 3.752 | 58.6311 | 17,056 | 0.5 | 0.14 | 100% | 35x |
| 256 | 5.623 | 21.9630 | 45,531 | 1.3 | 0.26 | 1553% | 93x |
| 1,024 | 14.344 | 14.0073 | 71,391 | 2.0 | 0.74 | 679% | 145x |
| 4,096 | 51.058 | 12.4652 | 80,223 | 2.2 | 2.66 | 268% | 163x |
| 16,384 | 202.909 | 12.3846 | 80,745 | 2.2 | 10.31 | 145% | 164x |
| 65,536 | 892.205 | 13.6140 | 73,454 | 2.0 | 40.95 | 119% | 149x |

Storage: 112 B of state against 27,648 B of raw frame (96x96) — **247x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## HumanoidmazeLargeStitch

State: 28 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, abdomen_z ...`. Reference: `ogbench/humanoidmaze-large-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **446 frames/s** (2,244 us/frame, 0.0671 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.179 | 3179.3807 | 315 | 0.0 | 0.11 | 100% | 0.71x |
| 64 | 3.758 | 58.7158 | 17,031 | 2.6 | 0.15 | 100% | 38x |
| 256 | 5.604 | 21.8900 | 45,683 | 6.9 | 0.30 | 1560% | 103x |
| 1,024 | 14.501 | 14.1612 | 70,616 | 10.6 | 0.87 | 674% | 158x |
| 4,096 | 51.895 | 12.6697 | 78,928 | 11.9 | 3.16 | 265% | 177x |
| 16,384 | 207.838 | 12.6854 | 78,831 | 11.9 | 12.33 | 145% | 177x |
| 65,536 | 904.844 | 13.8068 | 72,428 | 10.9 | 49.00 | 117% | 163x |

Storage: 112 B of state against 150,528 B of raw frame (224x224) — **1,344x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **491 frames/s** (2,036 us/frame, 0.0136 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.174 | 3173.8989 | 315 | 0.0 | 0.10 | 100% | 0.64x |
| 64 | 3.767 | 58.8590 | 16,990 | 0.5 | 0.14 | 100% | 35x |
| 256 | 5.627 | 21.9806 | 45,495 | 1.3 | 0.26 | 1549% | 93x |
| 1,024 | 14.336 | 14.0002 | 71,428 | 2.0 | 0.74 | 681% | 145x |
| 4,096 | 51.208 | 12.5020 | 79,987 | 2.2 | 2.66 | 267% | 163x |
| 16,384 | 204.671 | 12.4921 | 80,050 | 2.2 | 10.31 | 146% | 163x |
| 65,536 | 886.662 | 13.5294 | 73,913 | 2.0 | 40.95 | 118% | 150x |

Storage: 112 B of state against 27,648 B of raw frame (96x96) — **247x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## HumanoidmazeMediumNavigate

State: 28 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, abdomen_z ...`. Reference: `ogbench/humanoidmaze-medium-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **452 frames/s** (2,211 us/frame, 0.0681 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.174 | 3173.6353 | 315 | 0.0 | 0.10 | 100% | 0.70x |
| 64 | 3.759 | 58.7394 | 17,024 | 2.6 | 0.15 | 100% | 38x |
| 256 | 5.579 | 21.7945 | 45,883 | 6.9 | 0.29 | 1566% | 101x |
| 1,024 | 14.332 | 13.9965 | 71,447 | 10.8 | 0.87 | 682% | 158x |
| 4,096 | 51.117 | 12.4797 | 80,130 | 12.1 | 3.16 | 266% | 177x |
| 16,384 | 203.624 | 12.4283 | 80,462 | 12.1 | 12.33 | 145% | 178x |
| 65,536 | 887.100 | 13.5361 | 73,877 | 11.1 | 49.00 | 118% | 163x |

Storage: 112 B of state against 150,528 B of raw frame (224x224) — **1,344x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **498 frames/s** (2,009 us/frame, 0.0138 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.168 | 3168.1452 | 316 | 0.0 | 0.10 | 100% | 0.63x |
| 64 | 3.758 | 58.7219 | 17,029 | 0.5 | 0.14 | 100% | 34x |
| 256 | 5.599 | 21.8697 | 45,725 | 1.3 | 0.26 | 1560% | 92x |
| 1,024 | 14.186 | 13.8534 | 72,184 | 2.0 | 0.74 | 686% | 145x |
| 4,096 | 50.346 | 12.2915 | 81,357 | 2.2 | 2.65 | 271% | 163x |
| 16,384 | 200.608 | 12.2441 | 81,672 | 2.3 | 10.31 | 146% | 164x |
| 65,536 | 879.276 | 13.4167 | 74,534 | 2.1 | 40.94 | 118% | 150x |

Storage: 112 B of state against 27,648 B of raw frame (96x96) — **247x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## HumanoidmazeMediumStitch

State: 28 floats `root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z, abdomen_z ...`. Reference: `ogbench/humanoidmaze-medium-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **459 frames/s** (2,178 us/frame, 0.0691 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.173 | 3172.8167 | 315 | 0.0 | 0.10 | 100% | 0.69x |
| 64 | 3.755 | 58.6786 | 17,042 | 2.6 | 0.15 | 100% | 37x |
| 256 | 5.556 | 21.7021 | 46,079 | 6.9 | 0.29 | 1563% | 100x |
| 1,024 | 14.421 | 14.0828 | 71,009 | 10.7 | 0.87 | 677% | 155x |
| 4,096 | 51.314 | 12.5278 | 79,822 | 12.0 | 3.16 | 268% | 174x |
| 16,384 | 203.203 | 12.4025 | 80,629 | 12.1 | 12.33 | 145% | 176x |
| 65,536 | 888.087 | 13.5511 | 73,795 | 11.1 | 49.00 | 118% | 161x |

Storage: 112 B of state against 150,528 B of raw frame (224x224) — **1,344x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **501 frames/s** (1,997 us/frame, 0.0138 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.172 | 3172.3464 | 315 | 0.0 | 0.10 | 100% | 0.63x |
| 64 | 3.765 | 58.8323 | 16,997 | 0.5 | 0.14 | 100% | 34x |
| 256 | 5.577 | 21.7848 | 45,904 | 1.3 | 0.26 | 1561% | 92x |
| 1,024 | 14.196 | 13.8628 | 72,135 | 2.0 | 0.74 | 688% | 144x |
| 4,096 | 50.415 | 12.3084 | 81,245 | 2.2 | 2.65 | 271% | 162x |
| 16,384 | 199.806 | 12.1952 | 82,000 | 2.3 | 10.31 | 146% | 164x |
| 65,536 | 888.625 | 13.5593 | 73,750 | 2.0 | 40.94 | 119% | 147x |

Storage: 112 B of state against 27,648 B of raw frame (96x96) — **247x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## Intersection

State: 40 floats `v0_x, v0_y, v0_heading, v0_crashed, v1_x, v1_y, v1_heading, v1_crashed ...`. Reference: `intersection-v0`.

### 600x600 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.830 | 3830.2913 | 261 | 0.3 | 0.10 | 100% | - |
| 64 | 6.023 | 94.1031 | 10,627 | 11.5 | 0.17 | 100% | - |
| 256 | 12.879 | 50.3072 | 19,878 | 21.5 | 0.39 | 100% | - |
| 1,024 | 40.470 | 39.5213 | 25,303 | 27.3 | 1.26 | 100% | - |
| 4,096 | 174.312 | 42.5567 | 23,498 | 25.4 | 4.72 | 100% | - |
| 16,384 | output would be 17.7 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 70.8 GB, over the 10.0 GB budget | | | | | | |

Storage: 160 B of state against 1,080,000 B of raw frame (600x600) — **6,750x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 3.608 | 0.144 | 3.843 | 6,967 | 7.5 |
| 256 | 3.700 | 9.079 | 12.868 | 28,197 | 30.5 |
| 4,096 | 4.020 | 169.928 | 174.121 | 24,104 | 26.0 |

### 96x96 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.778 | 3777.7387 | 265 | 0.0 | 0.10 | 100% | - |
| 64 | 3.888 | 60.7531 | 16,460 | 0.5 | 0.10 | 100% | - |
| 256 | 4.039 | 15.7780 | 63,379 | 1.8 | 0.11 | 100% | - |
| 1,024 | 4.523 | 4.4170 | 226,398 | 6.3 | 0.15 | 100% | - |
| 4,096 | 6.836 | 1.6689 | 599,199 | 16.6 | 0.30 | 100% | - |
| 16,384 | 17.266 | 1.0538 | 948,922 | 26.2 | 0.89 | 100% | - |
| 65,536 | 62.552 | 0.9545 | 1,047,708 | 29.0 | 3.25 | 100% | - |

Storage: 160 B of state against 27,648 B of raw frame (96x96) — **173x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 3.584 | 0.107 | 3.783 | 9,340 | 0.3 |
| 256 | 3.683 | 0.268 | 4.037 | 956,233 | 26.4 |
| 4,096 | 3.948 | 2.800 | 6.819 | 1,462,974 | 40.4 |
| 16,384 | 5.477 | 11.731 | 17.274 | 1,396,670 | 38.6 |

### 224x224 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.793 | 3792.6855 | 264 | 0.0 | 0.10 | 100% | - |
| 64 | 4.112 | 64.2474 | 15,565 | 2.3 | 0.11 | 100% | - |
| 256 | 4.847 | 18.9322 | 52,820 | 8.0 | 0.15 | 100% | - |
| 1,024 | 7.762 | 7.5797 | 131,931 | 19.9 | 0.28 | 100% | - |
| 4,096 | 21.223 | 5.1813 | 193,002 | 29.1 | 0.83 | 100% | - |
| 16,384 | 81.295 | 4.9618 | 201,538 | 30.3 | 3.02 | 100% | - |
| 65,536 | 318.437 | 4.8590 | 205,805 | 31.0 | 11.78 | 100% | - |

Storage: 160 B of state against 150,528 B of raw frame (224x224) — **941x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 3.585 | 0.116 | 3.775 | 8,629 | 1.3 |
| 256 | 3.692 | 1.069 | 4.859 | 239,434 | 36.0 |
| 4,096 | 3.935 | 17.199 | 21.317 | 238,153 | 35.8 |
| 16,384 | 6.212 | 74.996 | 81.470 | 218,464 | 32.9 |

## Merge

State: 20 floats `ego_x, ego_y, ego_heading, ego_crashed, v1_x, v1_y, v1_heading, v1_crashed ...`. Reference: `merge-v0`.

### 150x150 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 4.819 | 4818.9731 | 208 | 0.1 | 0.10 | 100% | - |
| 64 | 5.138 | 80.2862 | 12,455 | 3.4 | 0.12 | 100% | - |
| 256 | 5.853 | 22.8619 | 43,741 | 11.8 | 0.17 | 100% | - |
| 1,024 | 8.688 | 8.4847 | 117,859 | 31.8 | 0.39 | 100% | - |
| 4,096 | 20.690 | 5.0513 | 197,968 | 53.5 | 1.28 | 100% | - |
| 16,384 | 81.411 | 4.9689 | 201,250 | 54.3 | 4.80 | 100% | - |
| 65,536 | output would be 17.7 GB, over the 10.0 GB budget | | | | | | |

Storage: 80 B of state against 270,000 B of raw frame (150x600) — **3,375x** smaller before any compression.

### 96x96 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 4.812 | 4811.6427 | 208 | 0.0 | 0.10 | 100% | - |
| 64 | 4.982 | 77.8446 | 12,846 | 1.4 | 0.11 | 100% | - |
| 256 | 5.255 | 20.5292 | 48,711 | 5.4 | 0.13 | 100% | - |
| 1,024 | 6.276 | 6.1289 | 163,161 | 18.0 | 0.23 | 100% | - |
| 4,096 | 10.491 | 2.5613 | 390,428 | 43.2 | 0.61 | 100% | - |
| 16,384 | 31.567 | 1.9267 | 519,016 | 57.4 | 2.14 | 100% | - |
| 65,536 | 115.192 | 1.7577 | 568,931 | 62.9 | 8.26 | 100% | - |

Storage: 80 B of state against 110,592 B of raw frame (96x384) — **1,382x** smaller before any compression.

### 224x224 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 4.818 | 4817.8751 | 208 | 0.1 | 0.10 | 100% | - |
| 64 | 5.428 | 84.8137 | 11,791 | 7.1 | 0.14 | 100% | - |
| 256 | 7.085 | 27.6772 | 36,131 | 21.8 | 0.26 | 100% | - |
| 1,024 | 13.755 | 13.4326 | 74,446 | 44.8 | 0.74 | 100% | - |
| 4,096 | 41.826 | 10.2115 | 97,929 | 59.0 | 2.65 | 100% | - |
| 16,384 | 185.529 | 11.3238 | 88,310 | 53.2 | 10.29 | 100% | - |
| 65,536 | output would be 39.5 GB, over the 10.0 GB budget | | | | | | |

Storage: 80 B of state against 602,112 B of raw frame (224x896) — **7,526x** smaller before any compression.

## MiniGridDoorKey

State: 67 floats `cell[0,0], cell[0,1], cell[0,2], cell[0,3], cell[0,4], cell[0,5], cell[0,6], cell[0,7] ...`. Reference: `MiniGrid-DoorKey-8x8-v0`.

### 256x256 (supersample 1)

Official CPU renderer: **982 frames/s** (1,018 us/frame, 0.1931 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.667 | 2667.1700 | 375 | 0.1 | 0.10 | 100% | 0.38x |
| 64 | 2.729 | 42.6427 | 23,451 | 4.6 | 0.11 | 100% | 24x |
| 256 | 2.761 | 10.7844 | 92,726 | 18.2 | 0.15 | 100% | 94x |
| 1,024 | 2.887 | 2.8192 | 354,715 | 69.7 | 0.31 | 100% | 361x |
| 4,096 | 3.354 | 0.8189 | 1,221,083 | 240.1 | 0.94 | 100% | 1,243x |
| 16,384 | 5.289 | 0.3228 | 3,097,761 | 609.0 | 3.43 | 100% | 3,153x |
| 65,536 | output would be 12.9 GB, over the 10.0 GB budget | | | | | | |

Storage: 268 B of state against 196,608 B of raw frame (256x256) — **734x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 1)

Official CPU renderer: **1,326 frames/s** (754 us/frame, 0.0367 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.678 | 2678.0069 | 373 | 0.0 | 0.10 | 100% | 0.28x |
| 64 | 2.732 | 42.6870 | 23,426 | 0.6 | 0.10 | 100% | 18x |
| 256 | 2.744 | 10.7181 | 93,300 | 2.6 | 0.11 | 100% | 70x |
| 1,024 | 2.797 | 2.7311 | 366,147 | 10.1 | 0.14 | 100% | 276x |
| 4,096 | 2.963 | 0.7235 | 1,382,196 | 38.2 | 0.24 | 100% | 1,042x |
| 16,384 | 3.708 | 0.2263 | 4,418,208 | 122.2 | 0.67 | 100% | 3,331x |
| 65,536 | 6.852 | 0.1045 | 9,564,852 | 264.4 | 2.36 | 100% | 7,212x |

Storage: 268 B of state against 27,648 B of raw frame (96x96) — **103x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 1)

Official CPU renderer: **1,053 frames/s** (950 us/frame, 0.1585 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.686 | 2686.2333 | 372 | 0.1 | 0.10 | 100% | 0.35x |
| 64 | 2.723 | 42.5431 | 23,506 | 3.5 | 0.11 | 100% | 22x |
| 256 | 2.747 | 10.7318 | 93,181 | 14.0 | 0.14 | 100% | 88x |
| 1,024 | 2.865 | 2.7980 | 357,399 | 53.8 | 0.26 | 100% | 339x |
| 4,096 | 3.290 | 0.8033 | 1,244,913 | 187.4 | 0.75 | 100% | 1,182x |
| 16,384 | 4.975 | 0.3037 | 3,293,227 | 495.7 | 2.68 | 100% | 3,128x |
| 65,536 | 11.915 | 0.1818 | 5,500,494 | 828.0 | 10.42 | 100% | 5,224x |

Storage: 268 B of state against 150,528 B of raw frame (224x224) — **562x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## MiniGridEmpty

State: 67 floats `cell[0,0], cell[0,1], cell[0,2], cell[0,3], cell[0,4], cell[0,5], cell[0,6], cell[0,7] ...`. Reference: `MiniGrid-Empty-8x8-v0`.

### 256x256 (supersample 1)

Official CPU renderer: **1,044 frames/s** (958 us/frame, 0.2053 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.426 | 2426.3766 | 412 | 0.1 | 0.10 | 100% | 0.39x |
| 64 | 2.475 | 38.6683 | 25,861 | 5.1 | 0.11 | 100% | 25x |
| 256 | 2.500 | 9.7664 | 102,392 | 20.1 | 0.15 | 100% | 98x |
| 1,024 | 2.624 | 2.5626 | 390,228 | 76.7 | 0.31 | 100% | 374x |
| 4,096 | 3.109 | 0.7589 | 1,317,638 | 259.1 | 0.94 | 100% | 1,262x |
| 16,384 | 5.032 | 0.3071 | 3,256,137 | 640.2 | 3.43 | 100% | 3,119x |
| 65,536 | output would be 12.9 GB, over the 10.0 GB budget | | | | | | |

Storage: 268 B of state against 196,608 B of raw frame (256x256) — **734x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 1)

Official CPU renderer: **1,227 frames/s** (815 us/frame, 0.0339 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.420 | 2419.7539 | 413 | 0.0 | 0.10 | 100% | 0.34x |
| 64 | 2.482 | 38.7825 | 25,785 | 0.7 | 0.10 | 100% | 21x |
| 256 | 2.469 | 9.6426 | 103,706 | 2.9 | 0.11 | 100% | 85x |
| 1,024 | 2.514 | 2.4554 | 407,260 | 11.3 | 0.14 | 100% | 332x |
| 4,096 | 2.696 | 0.6582 | 1,519,327 | 42.0 | 0.24 | 100% | 1,238x |
| 16,384 | 3.441 | 0.2101 | 4,760,753 | 131.6 | 0.67 | 100% | 3,880x |
| 65,536 | 6.596 | 0.1007 | 9,935,170 | 274.7 | 2.36 | 100% | 8,098x |

Storage: 268 B of state against 27,648 B of raw frame (96x96) — **103x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 1)

Official CPU renderer: **1,116 frames/s** (896 us/frame, 0.1680 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.415 | 2414.7844 | 414 | 0.1 | 0.10 | 100% | 0.37x |
| 64 | 2.465 | 38.5226 | 25,959 | 3.9 | 0.11 | 100% | 23x |
| 256 | 2.482 | 9.6967 | 103,128 | 15.5 | 0.14 | 100% | 92x |
| 1,024 | 2.592 | 2.5311 | 395,090 | 59.5 | 0.26 | 100% | 354x |
| 4,096 | 3.024 | 0.7382 | 1,354,583 | 203.9 | 0.75 | 100% | 1,214x |
| 16,384 | 4.724 | 0.2884 | 3,467,914 | 522.0 | 2.68 | 100% | 3,107x |
| 65,536 | 11.668 | 0.1780 | 5,616,774 | 845.5 | 10.42 | 100% | 5,032x |

Storage: 268 B of state against 150,528 B of raw frame (224x224) — **562x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## MiniGridFourRooms

State: 364 floats `cell[0,0], cell[0,1], cell[0,2], cell[0,3], cell[0,4], cell[0,5], cell[0,6], cell[0,7] ...`. Reference: `MiniGrid-FourRooms-v0`.

### 608x608 (supersample 1)

Official CPU renderer: **205 frames/s** (4,882 us/frame, 0.2272 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.710 | 2710.4756 | 369 | 0.4 | 0.10 | 100% | 1.80x |
| 64 | 2.789 | 43.5822 | 22,945 | 25.4 | 0.17 | 100% | 112x |
| 256 | 2.932 | 11.4527 | 87,315 | 96.8 | 0.39 | 100% | 426x |
| 1,024 | 3.498 | 3.4161 | 292,733 | 324.6 | 1.27 | 100% | 1,429x |
| 4,096 | 5.754 | 1.4048 | 711,840 | 789.4 | 4.76 | 100% | 3,475x |
| 16,384 | output would be 18.2 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 72.7 GB, over the 10.0 GB budget | | | | | | |

Storage: 1456 B of state against 1,108,992 B of raw frame (608x608) — **762x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

### 96x96

Official CPU renderer unavailable: ValueError: MiniGridFourRooms: size 96 is not a whole tile size for a 19x19 grid; use a multiple of 19 that is at least 57

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 1456 B of state against 27,648 B of raw frame (96x96) — **19x** smaller before any compression.

### 224x224

Official CPU renderer unavailable: ValueError: MiniGridFourRooms: size 224 is not a whole tile size for a 19x19 grid; use a multiple of 19 that is at least 57

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 1456 B of state against 150,528 B of raw frame (224x224) — **103x** smaller before any compression.

## MiniGridKeyCorridor

State: 103 floats `cell[0,0], cell[0,1], cell[0,2], cell[0,3], cell[0,4], cell[0,5], cell[0,6], cell[0,7] ...`. Reference: `MiniGrid-KeyCorridorS4R3-v0`.

### 320x320 (supersample 1)

Official CPU renderer: **304 frames/s** (3,289 us/frame, 0.0934 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.702 | 2701.5181 | 370 | 0.1 | 0.10 | 100% | 1.22x |
| 64 | 2.749 | 42.9517 | 23,282 | 7.2 | 0.12 | 100% | 77x |
| 256 | 2.785 | 10.8802 | 91,910 | 28.2 | 0.18 | 100% | 302x |
| 1,024 | 2.919 | 2.8510 | 350,751 | 107.8 | 0.42 | 100% | 1,154x |
| 4,096 | 3.522 | 0.8600 | 1,162,844 | 357.2 | 1.40 | 100% | 3,825x |
| 16,384 | 5.945 | 0.3629 | 2,755,913 | 846.6 | 5.29 | 100% | 9,065x |
| 65,536 | output would be 20.1 GB, over the 10.0 GB budget | | | | | | |

Storage: 412 B of state against 307,200 B of raw frame (320x320) — **746x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96

Official CPU renderer unavailable: ValueError: MiniGridKeyCorridor: size 96 is not a whole tile size for a 10x10 grid; use a multiple of 10 that is at least 30

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 412 B of state against 27,648 B of raw frame (96x96) — **67x** smaller before any compression.

### 224x224

Official CPU renderer unavailable: ValueError: MiniGridKeyCorridor: size 224 is not a whole tile size for a 10x10 grid; use a multiple of 10 that is at least 30

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 412 B of state against 150,528 B of raw frame (224x224) — **365x** smaller before any compression.

## MiniGridLavaCrossing

State: 84 floats `cell[0,0], cell[0,1], cell[0,2], cell[0,3], cell[0,4], cell[0,5], cell[0,6], cell[0,7] ...`. Reference: `MiniGrid-LavaCrossingS9N2-v0`.

### 288x288 (supersample 1)

Official CPU renderer: **910 frames/s** (1,099 us/frame, 0.2265 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.695 | 2695.1367 | 371 | 0.1 | 0.10 | 99% | 0.41x |
| 64 | 2.732 | 42.6924 | 23,423 | 5.8 | 0.12 | 100% | 26x |
| 256 | 2.757 | 10.7713 | 92,839 | 23.1 | 0.17 | 100% | 102x |
| 1,024 | 2.893 | 2.8252 | 353,956 | 88.1 | 0.36 | 100% | 389x |
| 4,096 | 3.445 | 0.8411 | 1,188,878 | 295.8 | 1.15 | 100% | 1,306x |
| 16,384 | 5.643 | 0.3444 | 2,903,588 | 722.5 | 4.31 | 100% | 3,190x |
| 65,536 | output would be 16.3 GB, over the 10.0 GB budget | | | | | | |

Storage: 336 B of state against 248,832 B of raw frame (288x288) — **741x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96

Official CPU renderer unavailable: ValueError: MiniGridLavaCrossing: size 96 is not a whole tile size for a 9x9 grid; use a multiple of 9 that is at least 27

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 336 B of state against 27,648 B of raw frame (96x96) — **82x** smaller before any compression.

### 224x224

Official CPU renderer unavailable: ValueError: MiniGridLavaCrossing: size 224 is not a whole tile size for a 9x9 grid; use a multiple of 9 that is at least 27

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 336 B of state against 150,528 B of raw frame (224x224) — **448x** smaller before any compression.

## MiniGridPOVFetch

State: 49 floats `view[0,0], view[0,1], view[0,2], view[0,3], view[0,4], view[0,5], view[0,6], view[1,0] ...`. Reference: `MiniGrid-Fetch-8x8-N3-v0`.

### 224x224 (supersample 1)

Official CPU renderer: **1,724 frames/s** (580 us/frame, 0.2594 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.329 | 329.0493 | 3,039 | 0.5 | 0.10 | 100% | 1.76x |
| 64 | 0.375 | 5.8579 | 170,708 | 25.7 | 0.11 | 100% | 99x |
| 256 | 0.394 | 1.5402 | 649,262 | 97.7 | 0.14 | 100% | 377x |
| 1,024 | 0.499 | 0.4876 | 2,050,796 | 308.7 | 0.26 | 100% | 1,190x |
| 4,096 | 0.925 | 0.2260 | 4,425,725 | 666.2 | 0.73 | 100% | 2,568x |
| 16,384 | 2.592 | 0.1582 | 6,319,913 | 951.3 | 2.62 | 100% | 3,667x |
| 65,536 | 9.351 | 0.1427 | 7,008,330 | 1054.9 | 10.18 | 100% | 4,066x |

Storage: 196 B of state against 150,528 B of raw frame (224x224) — **768x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96

Official CPU renderer unavailable: ValueError: MiniGridPOVFetch: size 96 is not a whole tile size for a 7x7 grid; use a multiple of 7 that is at least 21

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 196 B of state against 27,648 B of raw frame (96x96) — **141x** smaller before any compression.

## MiniHack

State: 1659 floats `glyph_r00c00, glyph_r00c01, glyph_r00c02, glyph_r00c03, glyph_r00c04, glyph_r00c05, glyph_r00c06, glyph_r00c07 ...`. Reference: `minihack.tiles.glyph_mapper.GlyphMapper`.

### 16x16 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'minihack'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.151 | 151.3716 | 6,606 | 8.4 | 0.10 | 100% | - |
| 64 | 0.423 | 6.6095 | 151,298 | 192.8 | 0.27 | 100% | - |
| 256 | 1.198 | 4.6798 | 213,686 | 272.3 | 0.77 | 100% | - |
| 1,024 | 4.295 | 4.1947 | 238,395 | 303.7 | 2.75 | 100% | - |
| 4,096 | 16.753 | 4.0902 | 244,488 | 311.5 | 10.70 | 100% | - |
| 16,384 | output would be 20.9 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 83.5 GB, over the 10.0 GB budget | | | | | | |

Storage: 6636 B of state against 1,274,112 B of raw frame (336x1264) — **192x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

### 96x96 (supersample 1)

Official CPU renderer unavailable: ValueError: the official MiniHack renderer has exactly one tile size (16 px); there is no reference to compare a 96 px render against

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.434 | 434.2105 | 2,303 | 105.6 | 0.22 | 100% | - |
| 64 | 8.265 | 129.1338 | 7,744 | 355.2 | 6.01 | 100% | - |
| 256 | output would be 11.7 GB, over the 10.0 GB budget | | | | | | |
| 1,024 | output would be 47.0 GB, over the 10.0 GB budget | | | | | | |
| 4,096 | output would be 187.9 GB, over the 10.0 GB budget | | | | | | |
| 16,384 | output would be 751.5 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 3006.0 GB, over the 10.0 GB budget | | | | | | |

Storage: 6636 B of state against 45,868,032 B of raw frame (2016x7584) — **6,912x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

### 224x224 (supersample 1)

Official CPU renderer unavailable: ValueError: the official MiniHack renderer has exactly one tile size (16 px); there is no reference to compare a 224 px render against

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 1.543 | 1543.1885 | 648 | 161.8 | 0.76 | 100% | - |
| 64 | output would be 16.0 GB, over the 10.0 GB budget | | | | | | |
| 256 | output would be 63.9 GB, over the 10.0 GB budget | | | | | | |
| 1,024 | output would be 255.7 GB, over the 10.0 GB budget | | | | | | |
| 4,096 | output would be 1022.9 GB, over the 10.0 GB budget | | | | | | |
| 16,384 | output would be 4091.5 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 16366.0 GB, over the 10.0 GB budget | | | | | | |

Storage: 6636 B of state against 249,725,952 B of raw frame (4704x17696) — **37,632x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

## MountainCar

State: 1 floats `position`. Reference: `MountainCar-v0`.

### 600x600 (supersample 1)

Official CPU renderer: **424 frames/s** (2,359 us/frame, 0.4578 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 60.743 | 60743.2360 | 16 | 0.0 | 0.10 | 100% | 0.04x |
| 64 | 61.519 | 961.2331 | 1,040 | 0.7 | 0.25 | 100% | 2.45x |
| 256 | 63.790 | 249.1805 | 4,013 | 2.9 | 0.72 | 100% | 9.47x |
| 1,024 | 71.343 | 69.6710 | 14,353 | 10.3 | 2.56 | 100% | 34x |
| 4,096 | 131.900 | 32.2021 | 31,054 | 22.4 | 9.93 | 100% | 73x |
| 16,384 | output would be 11.8 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 47.2 GB, over the 10.0 GB budget | | | | | | |

Storage: 4 B of state against 720,000 B of raw frame (400x600) — **180,000x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

### 96x96

Official CPU renderer unavailable: ValueError: MountainCar renders at its reference's native 400x600 and nothing else (got size=96). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 4 B of state against 27,648 B of raw frame (96x96) — **6,912x** smaller before any compression.

### 224x224

Official CPU renderer unavailable: ValueError: MountainCar renders at its reference's native 400x600 and nothing else (got size=224). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 4 B of state against 150,528 B of raw frame (224x224) — **37,632x** smaller before any compression.

## MountainCarContinuous

State: 1 floats `position`. Reference: `MountainCarContinuous-v0`.

### 600x600 (supersample 1)

Official CPU renderer: **428 frames/s** (2,337 us/frame, 0.4620 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 59.950 | 59950.4514 | 17 | 0.0 | 0.10 | 100% | 0.04x |
| 64 | 61.039 | 953.7336 | 1,049 | 0.8 | 0.25 | 100% | 2.45x |
| 256 | 63.447 | 247.8392 | 4,035 | 2.9 | 0.72 | 100% | 9.43x |
| 1,024 | 71.174 | 69.5055 | 14,387 | 10.4 | 2.56 | 100% | 34x |
| 4,096 | 131.580 | 32.1240 | 31,129 | 22.4 | 9.93 | 100% | 73x |
| 16,384 | output would be 11.8 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 47.2 GB, over the 10.0 GB budget | | | | | | |

Storage: 4 B of state against 720,000 B of raw frame (400x600) — **180,000x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

### 96x96

Official CPU renderer unavailable: ValueError: MountainCarContinuous renders at its reference's native 400x600 and nothing else (got size=96). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 4 B of state against 27,648 B of raw frame (96x96) — **6,912x** smaller before any compression.

### 224x224

Official CPU renderer unavailable: ValueError: MountainCarContinuous renders at its reference's native 400x600 and nothing else (got size=224). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 4 B of state against 150,528 B of raw frame (224x224) — **37,632x** smaller before any compression.

## MujocoInvertedDoublePendulum

State: 3 floats `slider, hinge, hinge2`. Reference: `gymnasium/InvertedDoublePendulum-v5`.

### 240x240 (supersample 2)

Official CPU renderer: **2,485 frames/s** (402 us/frame, 0.4294 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.704 | 2704.0783 | 370 | 0.1 | 0.14 | 100% | 0.15x |
| 64 | 6.710 | 104.8425 | 9,538 | 1.6 | 2.17 | 100% | 3.84x |
| 256 | 25.493 | 99.5827 | 10,042 | 1.7 | 2.47 | 100% | 4.04x |
| 1,024 | 99.079 | 96.7567 | 10,335 | 1.8 | 2.60 | 100% | 4.16x |
| 4,096 | 390.955 | 95.4481 | 10,477 | 1.8 | 3.13 | 100% | 4.22x |
| 16,384 | 1567.239 | 95.6567 | 10,454 | 1.8 | 5.25 | 100% | 4.21x |
| 65,536 | output would be 11.3 GB, over the 10.0 GB budget | | | | | | |

Storage: 12 B of state against 172,800 B of raw frame (240x240) — **14,400x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **5,177 frames/s** (193 us/frame, 0.1431 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.686 | 2685.6083 | 372 | 0.0 | 0.11 | 100% | 0.07x |
| 64 | 3.527 | 55.1056 | 18,147 | 0.5 | 0.45 | 100% | 3.51x |
| 256 | 7.608 | 29.7176 | 33,650 | 0.9 | 1.44 | 100% | 6.50x |
| 1,024 | 27.554 | 26.9085 | 37,163 | 1.0 | 2.50 | 100% | 7.18x |
| 4,096 | 106.950 | 26.1108 | 38,298 | 1.1 | 2.58 | 100% | 7.40x |
| 16,384 | 424.428 | 25.9050 | 38,603 | 1.1 | 2.92 | 100% | 7.46x |
| 65,536 | 1703.859 | 25.9988 | 38,463 | 1.1 | 4.28 | 100% | 7.43x |

Storage: 12 B of state against 27,648 B of raw frame (96x96) — **2,304x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 2)

Official CPU renderer: **2,391 frames/s** (418 us/frame, 0.3600 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.695 | 2695.0883 | 371 | 0.1 | 0.14 | 100% | 0.16x |
| 64 | 6.220 | 97.1830 | 10,290 | 1.5 | 1.93 | 100% | 4.30x |
| 256 | 23.754 | 92.7879 | 10,777 | 1.6 | 2.47 | 100% | 4.51x |
| 1,024 | 87.861 | 85.8017 | 11,655 | 1.8 | 2.59 | 100% | 4.87x |
| 4,096 | 347.703 | 84.8883 | 11,780 | 1.8 | 3.05 | 100% | 4.93x |
| 16,384 | 1388.811 | 84.7663 | 11,797 | 1.8 | 4.90 | 100% | 4.93x |
| 65,536 | 5571.152 | 85.0090 | 11,763 | 1.8 | 12.31 | 100% | 4.92x |

Storage: 12 B of state against 150,528 B of raw frame (224x224) — **12,544x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## MujocoInvertedPendulum

State: 2 floats `slider, hinge`. Reference: `gymnasium/InvertedPendulum-v5`.

### 240x240 (supersample 2)

Official CPU renderer: **2,466 frames/s** (405 us/frame, 0.4262 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.708 | 2708.4639 | 369 | 0.1 | 0.16 | 100% | 0.15x |
| 64 | 6.712 | 104.8786 | 9,535 | 1.6 | 2.36 | 100% | 3.87x |
| 256 | 25.499 | 99.6047 | 10,040 | 1.7 | 2.68 | 100% | 4.07x |
| 1,024 | 99.182 | 96.8573 | 10,324 | 1.8 | 2.81 | 100% | 4.19x |
| 4,096 | 391.964 | 95.6943 | 10,450 | 1.8 | 3.45 | 100% | 4.24x |
| 16,384 | 1568.523 | 95.7350 | 10,445 | 1.8 | 5.57 | 100% | 4.24x |
| 65,536 | output would be 11.3 GB, over the 10.0 GB budget | | | | | | |

Storage: 8 B of state against 172,800 B of raw frame (240x240) — **21,600x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **5,948 frames/s** (168 us/frame, 0.1645 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.697 | 2697.4538 | 371 | 0.0 | 0.12 | 100% | 0.06x |
| 64 | 3.235 | 50.5496 | 19,783 | 0.5 | 0.49 | 100% | 3.33x |
| 256 | 6.650 | 25.9766 | 38,496 | 1.1 | 1.60 | 100% | 6.47x |
| 1,024 | 23.935 | 23.3742 | 42,782 | 1.2 | 2.78 | 100% | 7.19x |
| 4,096 | 92.647 | 22.6190 | 44,211 | 1.2 | 2.86 | 100% | 7.43x |
| 16,384 | 365.932 | 22.3347 | 44,773 | 1.2 | 3.20 | 100% | 7.53x |
| 65,536 | 1472.102 | 22.4625 | 44,519 | 1.2 | 4.56 | 100% | 7.48x |

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 2)

Official CPU renderer: **2,810 frames/s** (356 us/frame, 0.4230 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.705 | 2704.7629 | 370 | 0.1 | 0.15 | 100% | 0.13x |
| 64 | 6.151 | 96.1113 | 10,405 | 1.6 | 2.09 | 100% | 3.70x |
| 256 | 23.590 | 92.1471 | 10,852 | 1.6 | 2.68 | 100% | 3.86x |
| 1,024 | 87.495 | 85.4440 | 11,704 | 1.8 | 2.79 | 100% | 4.17x |
| 4,096 | 346.108 | 84.4991 | 11,834 | 1.8 | 3.36 | 100% | 4.21x |
| 16,384 | 1381.795 | 84.3381 | 11,857 | 1.8 | 5.21 | 100% | 4.22x |
| 65,536 | 5540.928 | 84.5478 | 11,828 | 1.8 | 12.61 | 100% | 4.21x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## MujocoPusher

State: 11 floats `r_shoulder_pan_joint, r_shoulder_lift_joint, r_upper_arm_roll_joint, r_elbow_flex_joint, r_forearm_roll_joint, r_wrist_flex_joint, r_wrist_roll_joint, obj_slidey ...`. Reference: `gymnasium/Pusher-v5`.

### 240x240 (supersample 2)

Official CPU renderer: **466 frames/s** (2,148 us/frame, 0.0805 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 4.139 | 4138.5777 | 242 | 0.0 | 0.17 | 100% | 0.52x |
| 64 | 12.301 | 192.2024 | 5,203 | 0.9 | 2.76 | 100% | 11x |
| 256 | 47.797 | 186.7063 | 5,356 | 0.9 | 3.13 | 100% | 12x |
| 1,024 | 187.633 | 183.2356 | 5,457 | 0.9 | 3.26 | 100% | 12x |
| 4,096 | 743.170 | 181.4380 | 5,512 | 1.0 | 3.80 | 100% | 12x |
| 16,384 | 2976.707 | 181.6837 | 5,504 | 1.0 | 5.92 | 100% | 12x |
| 65,536 | output would be 11.3 GB, over the 10.0 GB budget | | | | | | |

Storage: 44 B of state against 172,800 B of raw frame (240x240) — **3,927x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **527 frames/s** (1,899 us/frame, 0.0146 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 4.128 | 4127.8014 | 242 | 0.0 | 0.13 | 100% | 0.46x |
| 64 | 4.767 | 74.4833 | 13,426 | 0.4 | 0.56 | 100% | 25x |
| 256 | 10.734 | 41.9316 | 23,848 | 0.7 | 1.81 | 859% | 45x |
| 1,024 | 39.442 | 38.5179 | 25,962 | 0.7 | 3.14 | 515% | 49x |
| 4,096 | 154.090 | 37.6195 | 26,582 | 0.7 | 3.23 | 578% | 50x |
| 16,384 | 608.638 | 37.1483 | 26,919 | 0.7 | 3.57 | 585% | 51x |
| 65,536 | 2434.097 | 37.1414 | 26,924 | 0.7 | 4.93 | 584% | 51x |

Storage: 44 B of state against 27,648 B of raw frame (96x96) — **628x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 2)

Official CPU renderer: **469 frames/s** (2,133 us/frame, 0.0706 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 4.128 | 4127.7632 | 242 | 0.0 | 0.17 | 100% | 0.52x |
| 64 | 11.127 | 173.8536 | 5,752 | 0.9 | 2.45 | 100% | 12x |
| 256 | 43.722 | 170.7872 | 5,855 | 0.9 | 3.14 | 100% | 12x |
| 1,024 | 164.708 | 160.8480 | 6,217 | 0.9 | 3.26 | 100% | 13x |
| 4,096 | 653.651 | 159.5828 | 6,266 | 0.9 | 3.72 | 100% | 13x |
| 16,384 | 2612.319 | 159.4433 | 6,272 | 0.9 | 5.57 | 100% | 13x |
| 65,536 | 10466.658 | 159.7085 | 6,261 | 0.9 | 12.98 | 100% | 13x |

Storage: 44 B of state against 150,528 B of raw frame (224x224) — **3,421x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## MujocoReacher

State: 4 floats `joint0, joint1, target_x, target_y`. Reference: `gymnasium/Reacher-v5`.

### 240x240 (supersample 2)

Official CPU renderer: **2,274 frames/s** (440 us/frame, 0.3929 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.712 | 2712.1622 | 369 | 0.1 | 0.17 | 100% | 0.16x |
| 64 | 6.603 | 103.1738 | 9,692 | 1.7 | 2.29 | 100% | 4.26x |
| 256 | 24.989 | 97.6117 | 10,245 | 1.8 | 2.60 | 100% | 4.51x |
| 1,024 | 96.965 | 94.6919 | 10,561 | 1.8 | 2.73 | 100% | 4.64x |
| 4,096 | 382.265 | 93.3264 | 10,715 | 1.9 | 3.26 | 100% | 4.71x |
| 16,384 | 1531.676 | 93.4861 | 10,697 | 1.8 | 5.38 | 100% | 4.70x |
| 65,536 | output would be 11.3 GB, over the 10.0 GB budget | | | | | | |

Storage: 16 B of state against 172,800 B of raw frame (240x240) — **10,800x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **5,096 frames/s** (196 us/frame, 0.1409 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.695 | 2694.9830 | 371 | 0.0 | 0.14 | 100% | 0.07x |
| 64 | 3.250 | 50.7835 | 19,691 | 0.5 | 0.49 | 100% | 3.86x |
| 256 | 6.539 | 25.5427 | 39,150 | 1.1 | 1.52 | 100% | 7.68x |
| 1,024 | 23.284 | 22.7386 | 43,978 | 1.2 | 2.60 | 100% | 8.63x |
| 4,096 | 89.908 | 21.9502 | 45,558 | 1.3 | 2.69 | 100% | 8.94x |
| 16,384 | 355.372 | 21.6902 | 46,104 | 1.3 | 3.03 | 100% | 9.05x |
| 65,536 | 1427.802 | 21.7865 | 45,900 | 1.3 | 4.39 | 100% | 9.01x |

Storage: 16 B of state against 27,648 B of raw frame (96x96) — **1,728x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 2)

Official CPU renderer: **2,156 frames/s** (464 us/frame, 0.3245 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.703 | 2702.7465 | 370 | 0.1 | 0.18 | 100% | 0.17x |
| 64 | 6.074 | 94.9103 | 10,536 | 1.6 | 2.03 | 100% | 4.89x |
| 256 | 23.143 | 90.4035 | 11,062 | 1.7 | 2.61 | 100% | 5.13x |
| 1,024 | 85.515 | 83.5111 | 11,974 | 1.8 | 2.72 | 100% | 5.55x |
| 4,096 | 338.347 | 82.6042 | 12,106 | 1.8 | 3.18 | 100% | 5.62x |
| 16,384 | 1351.028 | 82.4602 | 12,127 | 1.8 | 5.03 | 100% | 5.63x |
| 65,536 | 5418.608 | 82.6814 | 12,095 | 1.8 | 12.44 | 100% | 5.61x |

Storage: 16 B of state against 150,528 B of raw frame (224x224) — **9,408x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## Parking

State: 7 floats `ego_x, ego_y, ego_heading, ego_steering, landmark0_x, landmark0_y, landmark0_heading`. Reference: `parking-v0`.

### 300x300 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 8.290 | 8290.0338 | 121 | 0.1 | 0.15 | 100% | - |
| 64 | 9.179 | 143.4291 | 6,972 | 3.8 | 0.18 | 100% | - |
| 256 | 11.617 | 45.3790 | 22,037 | 11.9 | 0.29 | 100% | - |
| 1,024 | 21.481 | 20.9772 | 47,671 | 25.7 | 0.73 | 100% | - |
| 4,096 | 65.862 | 16.0795 | 62,191 | 33.6 | 2.49 | 100% | - |
| 16,384 | 278.999 | 17.0287 | 58,724 | 31.7 | 9.52 | 100% | - |
| 65,536 | output would be 35.4 GB, over the 10.0 GB budget | | | | | | |

Storage: 28 B of state against 540,000 B of raw frame (300x600) — **19,286x** smaller before any compression.

### 96x96 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 8.199 | 8199.2522 | 122 | 0.0 | 0.15 | 100% | - |
| 64 | 8.486 | 132.5983 | 7,542 | 0.4 | 0.15 | 100% | - |
| 256 | 8.754 | 34.1949 | 29,244 | 1.6 | 0.17 | 100% | - |
| 1,024 | 9.862 | 9.6307 | 103,835 | 5.7 | 0.24 | 100% | - |
| 4,096 | 14.628 | 3.5712 | 280,020 | 15.5 | 0.50 | 100% | - |
| 16,384 | 38.365 | 2.3416 | 427,050 | 23.6 | 1.58 | 100% | - |
| 65,536 | 132.835 | 2.0269 | 493,364 | 27.3 | 5.87 | 100% | - |

Storage: 28 B of state against 55,296 B of raw frame (96x192) — **1,975x** smaller before any compression.

### 224x224 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 8.201 | 8200.9062 | 122 | 0.0 | 0.15 | 100% | - |
| 64 | 8.793 | 137.3847 | 7,279 | 2.2 | 0.17 | 100% | - |
| 256 | 10.216 | 39.9048 | 25,060 | 7.5 | 0.23 | 100% | - |
| 1,024 | 15.808 | 15.4375 | 64,777 | 19.5 | 0.49 | 100% | - |
| 4,096 | 40.450 | 9.8755 | 101,261 | 30.5 | 1.51 | 100% | - |
| 16,384 | 158.483 | 9.6730 | 103,380 | 31.1 | 5.60 | 100% | - |
| 65,536 | output would be 19.7 GB, over the 10.0 GB budget | | | | | | |

Storage: 28 B of state against 301,056 B of raw frame (224x448) — **10,752x** smaller before any compression.

## Pendulum

State: 2 floats `theta, torque`. Reference: `Pendulum-v1`.

### 500x500 (supersample 1)

Official CPU renderer: **294 frames/s** (3,405 us/frame, 0.2203 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 35.795 | 35795.0134 | 28 | 0.0 | 0.16 | 100% | 0.10x |
| 64 | 36.462 | 569.7177 | 1,755 | 1.3 | 0.32 | 100% | 5.98x |
| 256 | 38.874 | 151.8506 | 6,585 | 4.9 | 0.80 | 100% | 22x |
| 1,024 | 51.944 | 50.7263 | 19,714 | 14.8 | 2.72 | 100% | 67x |
| 4,096 | 106.672 | 26.0430 | 38,398 | 28.8 | 10.40 | 100% | 131x |
| 16,384 | output would be 12.3 GB, over the 10.0 GB budget | | | | | | |
| 65,536 | output would be 49.2 GB, over the 10.0 GB budget | | | | | | |

Storage: 8 B of state against 750,000 B of raw frame (500x500) — **93,750x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render(); over the output budget.

### 96x96

Official CPU renderer unavailable: ValueError: Pendulum renders at its reference's native 500x500 and nothing else (got size=96). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

### 224x224

Official CPU renderer unavailable: ValueError: Pendulum renders at its reference's native 500x500 and nothing else (got size=224). The official renderer hardcodes every primitive in pixels, so a resized frame has no reference to be exact against; resample the output if you want one.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

## Piecewise

State: 2 floats `agent_x, agent_y`. Reference: `swm/Piecewise-v0`.

### 224x224 (supersample 1)

Official CPU renderer: **853 frames/s** (1,172 us/frame, 0.1284 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.405 | 404.9456 | 2,469 | 0.4 | 0.16 | 100% | 2.90x |
| 64 | 0.497 | 7.7594 | 128,875 | 19.4 | 0.36 | 100% | 151x |
| 256 | 1.412 | 5.5167 | 181,268 | 27.3 | 0.96 | 100% | 213x |
| 1,024 | 5.497 | 5.3685 | 186,270 | 28.0 | 1.08 | 100% | 218x |
| 4,096 | 21.842 | 5.3324 | 187,532 | 28.2 | 1.54 | 100% | 220x |
| 16,384 | 87.248 | 5.3252 | 187,786 | 28.3 | 3.39 | 100% | 220x |
| 65,536 | 348.809 | 5.3224 | 187,885 | 28.3 | 10.79 | 100% | 220x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96

Official CPU renderer unavailable: ValueError: Piecewise renders at 224px only, and asking for 96 is not a resolution change. Every length in the reference is in output pixels — a 4 px border, a door half-extent in pixels, a Gaussian sigma in pixels, a wall at column 112 — with no world-unit description underneath, so there is nothing to re-sample. Resize the frames afterwards if a different size is wanted; that is what stable_worldmodel's own pipeline does.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

## PiecewiseZones

State: 2 floats `agent_x, agent_y`. Reference: `swm/Piecewise-v0`.

### 224x224 (supersample 1)

Official CPU renderer: **538 frames/s** (1,858 us/frame, 0.0810 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.590 | 590.1726 | 1,694 | 0.3 | 0.16 | 100% | 3.15x |
| 64 | 0.694 | 10.8393 | 92,257 | 13.9 | 0.36 | 100% | 171x |
| 256 | 1.868 | 7.2963 | 137,056 | 20.6 | 0.96 | 100% | 255x |
| 1,024 | 7.322 | 7.1505 | 139,849 | 21.1 | 1.08 | 100% | 260x |
| 4,096 | 29.130 | 7.1119 | 140,610 | 21.2 | 1.54 | 100% | 261x |
| 16,384 | 116.396 | 7.1043 | 140,761 | 21.2 | 3.39 | 100% | 262x |
| 65,536 | 465.305 | 7.1000 | 140,845 | 21.2 | 10.79 | 100% | 262x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96

Official CPU renderer unavailable: ValueError: PiecewiseZones renders at 224px only, and asking for 96 is not a resolution change. Every length in the reference is in output pixels — a 4 px border, a door half-extent in pixels, a Gaussian sigma in pixels, a wall at column 112 — with no world-unit description underneath, so there is nothing to re-sample. Resize the frames afterwards if a different size is wanted; that is what stable_worldmodel's own pipeline does.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

## PointMazeLarge

State: 4 floats `ball_x, ball_y, goal_x, goal_y`. Reference: `PointMaze_Large-v3`.

### 224x224 (supersample 2)

Official CPU renderer: **1,719 frames/s** (582 us/frame, 0.2587 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.507 | 5507.3593 | 182 | 0.0 | 0.15 | 100% | 0.11x |
| 64 | 5.682 | 88.7760 | 11,264 | 1.7 | 0.25 | 100% | 6.55x |
| 256 | 6.181 | 24.1463 | 41,414 | 6.2 | 0.58 | 100% | 24x |
| 1,024 | 15.239 | 14.8814 | 67,198 | 10.1 | 1.83 | 100% | 39x |
| 4,096 | 42.918 | 10.4779 | 95,439 | 14.4 | 6.86 | 100% | 56x |
| 16,384 | 159.516 | 9.7361 | 102,711 | 15.5 | 26.98 | 100% | 60x |
| 65,536 | 640.054 | 9.7664 | 102,391 | 15.4 | 107.47 | 100% | 60x |

Storage: 16 B of state against 150,528 B of raw frame (224x224) — **9,408x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **3,113 frames/s** (321 us/frame, 0.0861 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.500 | 5499.9720 | 182 | 0.0 | 0.15 | 100% | 0.06x |
| 64 | 5.658 | 88.4078 | 11,311 | 0.3 | 0.24 | 100% | 3.63x |
| 256 | 5.930 | 23.1637 | 43,171 | 1.2 | 0.51 | 100% | 14x |
| 1,024 | 13.854 | 13.5290 | 73,916 | 2.0 | 1.57 | 100% | 24x |
| 4,096 | 37.239 | 9.0915 | 109,993 | 3.0 | 5.85 | 100% | 35x |
| 16,384 | 135.177 | 8.2506 | 121,204 | 3.4 | 22.95 | 100% | 39x |
| 65,536 | 548.616 | 8.3712 | 119,457 | 3.3 | 91.37 | 100% | 38x |

Storage: 16 B of state against 27,648 B of raw frame (96x96) — **1,728x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointMazeMedium

State: 4 floats `ball_x, ball_y, goal_x, goal_y`. Reference: `PointMaze_Medium-v3`.

### 224x224 (supersample 2)

Official CPU renderer: **581 frames/s** (1,720 us/frame, 0.0875 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.493 | 5492.7710 | 182 | 0.0 | 0.15 | 100% | 0.31x |
| 64 | 5.648 | 88.2486 | 11,332 | 1.7 | 0.26 | 100% | 19x |
| 256 | 6.210 | 24.2578 | 41,224 | 6.2 | 0.58 | 100% | 71x |
| 1,024 | 15.218 | 14.8617 | 67,287 | 10.1 | 1.83 | 100% | 116x |
| 4,096 | 43.004 | 10.4990 | 95,247 | 14.3 | 6.86 | 100% | 164x |
| 16,384 | 158.640 | 9.6826 | 103,278 | 15.5 | 26.98 | 100% | 178x |
| 65,536 | 641.877 | 9.7943 | 102,101 | 15.4 | 107.48 | 100% | 176x |

Storage: 16 B of state against 150,528 B of raw frame (224x224) — **9,408x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **3,574 frames/s** (280 us/frame, 0.0988 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.480 | 5479.5379 | 182 | 0.0 | 0.15 | 100% | 0.05x |
| 64 | 5.633 | 88.0147 | 11,362 | 0.3 | 0.24 | 100% | 3.18x |
| 256 | 5.904 | 23.0616 | 43,362 | 1.2 | 0.52 | 100% | 12x |
| 1,024 | 13.845 | 13.5204 | 73,962 | 2.0 | 1.58 | 100% | 21x |
| 4,096 | 37.232 | 9.0899 | 110,012 | 3.0 | 5.85 | 100% | 31x |
| 16,384 | 136.091 | 8.3063 | 120,390 | 3.3 | 22.96 | 100% | 34x |
| 65,536 | 548.248 | 8.3656 | 119,537 | 3.3 | 91.37 | 100% | 33x |

Storage: 16 B of state against 27,648 B of raw frame (96x96) — **1,728x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointMazeOpen

State: 4 floats `ball_x, ball_y, goal_x, goal_y`. Reference: `PointMaze_Open-v3`.

### 224x224 (supersample 2)

Official CPU renderer: **1,293 frames/s** (773 us/frame, 0.1947 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.501 | 5500.8205 | 182 | 0.0 | 0.15 | 100% | 0.14x |
| 64 | 5.674 | 88.6499 | 11,280 | 1.7 | 0.25 | 100% | 8.72x |
| 256 | 6.178 | 24.1322 | 41,438 | 6.2 | 0.58 | 100% | 32x |
| 1,024 | 15.225 | 14.8679 | 67,259 | 10.1 | 1.83 | 100% | 52x |
| 4,096 | 42.985 | 10.4945 | 95,288 | 14.3 | 6.86 | 100% | 74x |
| 16,384 | 158.808 | 9.6929 | 103,169 | 15.5 | 26.98 | 100% | 80x |
| 65,536 | 641.798 | 9.7931 | 102,113 | 15.4 | 107.47 | 100% | 79x |

Storage: 16 B of state against 150,528 B of raw frame (224x224) — **9,408x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **4,019 frames/s** (249 us/frame, 0.1111 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.486 | 5486.0758 | 182 | 0.0 | 0.15 | 100% | 0.05x |
| 64 | 5.636 | 88.0676 | 11,355 | 0.3 | 0.24 | 100% | 2.83x |
| 256 | 5.912 | 23.0947 | 43,300 | 1.2 | 0.51 | 100% | 11x |
| 1,024 | 13.837 | 13.5132 | 74,002 | 2.0 | 1.58 | 100% | 18x |
| 4,096 | 37.218 | 9.0864 | 110,054 | 3.0 | 5.85 | 100% | 27x |
| 16,384 | 135.319 | 8.2592 | 121,077 | 3.3 | 22.95 | 100% | 30x |
| 65,536 | 548.093 | 8.3632 | 119,571 | 3.3 | 91.37 | 100% | 30x |

Storage: 16 B of state against 27,648 B of raw frame (96x96) — **1,728x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointMazeUMaze

State: 4 floats `ball_x, ball_y, goal_x, goal_y`. Reference: `PointMaze_UMaze-v3`.

### 224x224 (supersample 2)

Official CPU renderer: **1,232 frames/s** (812 us/frame, 0.1855 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.500 | 5499.7373 | 182 | 0.0 | 0.15 | 100% | 0.15x |
| 64 | 5.682 | 88.7803 | 11,264 | 1.7 | 0.25 | 100% | 9.14x |
| 256 | 6.182 | 24.1469 | 41,413 | 6.2 | 0.58 | 100% | 34x |
| 1,024 | 15.237 | 14.8797 | 67,206 | 10.1 | 1.83 | 100% | 55x |
| 4,096 | 43.008 | 10.5001 | 95,237 | 14.3 | 6.86 | 100% | 77x |
| 16,384 | 159.863 | 9.7573 | 102,487 | 15.4 | 26.98 | 100% | 83x |
| 65,536 | 643.106 | 9.8130 | 101,905 | 15.3 | 107.47 | 100% | 83x |

Storage: 16 B of state against 150,528 B of raw frame (224x224) — **9,408x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **4,055 frames/s** (247 us/frame, 0.1121 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.503 | 5503.2857 | 182 | 0.0 | 0.15 | 100% | 0.04x |
| 64 | 5.679 | 88.7386 | 11,269 | 0.3 | 0.24 | 100% | 2.78x |
| 256 | 5.943 | 23.2139 | 43,078 | 1.2 | 0.51 | 100% | 11x |
| 1,024 | 13.881 | 13.5561 | 73,767 | 2.0 | 1.58 | 100% | 18x |
| 4,096 | 37.209 | 9.0843 | 110,081 | 3.0 | 5.85 | 100% | 27x |
| 16,384 | 135.170 | 8.2501 | 121,210 | 3.4 | 22.95 | 100% | 30x |
| 65,536 | 548.547 | 8.3702 | 119,472 | 3.3 | 91.37 | 100% | 29x |

Storage: 16 B of state against 27,648 B of raw frame (96x96) — **1,728x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointmazeGiantNavigate

State: 2 floats `x, y`. Reference: `ogbench/pointmaze-giant-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **364 frames/s** (2,750 us/frame, 0.0547 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.146 | 3146.4202 | 318 | 0.0 | 0.15 | 100% | 0.87x |
| 64 | 3.246 | 50.7129 | 19,719 | 3.0 | 0.20 | 100% | 54x |
| 256 | 3.503 | 13.6841 | 73,078 | 11.0 | 0.34 | 100% | 201x |
| 1,024 | 5.608 | 5.4763 | 182,604 | 27.5 | 0.92 | 100% | 502x |
| 4,096 | 17.001 | 4.1506 | 240,927 | 36.3 | 3.20 | 100% | 663x |
| 16,384 | 66.341 | 4.0491 | 246,967 | 37.2 | 12.35 | 100% | 679x |
| 65,536 | 281.898 | 4.3014 | 232,481 | 35.0 | 48.95 | 100% | 639x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **475 frames/s** (2,104 us/frame, 0.0131 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.142 | 3142.2134 | 318 | 0.0 | 0.15 | 100% | 0.67x |
| 64 | 3.231 | 50.4785 | 19,810 | 0.5 | 0.19 | 100% | 42x |
| 256 | 3.476 | 13.5783 | 73,647 | 2.0 | 0.31 | 100% | 155x |
| 1,024 | 5.404 | 5.2777 | 189,476 | 5.2 | 0.79 | 100% | 399x |
| 4,096 | 16.160 | 3.9453 | 253,466 | 7.0 | 2.70 | 100% | 533x |
| 16,384 | 63.339 | 3.8659 | 258,670 | 7.2 | 10.34 | 100% | 544x |
| 65,536 | 269.722 | 4.1156 | 242,976 | 6.7 | 40.90 | 100% | 511x |

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointmazeGiantStitch

State: 2 floats `x, y`. Reference: `ogbench/pointmaze-giant-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **374 frames/s** (2,671 us/frame, 0.0563 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.166 | 3165.6297 | 316 | 0.0 | 0.15 | 100% | 0.84x |
| 64 | 3.256 | 50.8804 | 19,654 | 3.0 | 0.20 | 100% | 53x |
| 256 | 3.501 | 13.6753 | 73,125 | 11.0 | 0.35 | 100% | 195x |
| 1,024 | 5.604 | 5.4726 | 182,728 | 27.5 | 0.92 | 100% | 488x |
| 4,096 | 16.957 | 4.1399 | 241,551 | 36.4 | 3.20 | 100% | 645x |
| 16,384 | 66.656 | 4.0684 | 245,798 | 37.0 | 12.35 | 100% | 657x |
| 65,536 | 281.931 | 4.3019 | 232,454 | 35.0 | 48.95 | 100% | 621x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **475 frames/s** (2,105 us/frame, 0.0131 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.160 | 3159.5519 | 317 | 0.0 | 0.15 | 100% | 0.67x |
| 64 | 3.242 | 50.6524 | 19,742 | 0.5 | 0.19 | 100% | 42x |
| 256 | 3.486 | 13.6185 | 73,430 | 2.0 | 0.31 | 100% | 155x |
| 1,024 | 5.426 | 5.2983 | 188,738 | 5.2 | 0.79 | 100% | 397x |
| 4,096 | 16.192 | 3.9530 | 252,969 | 7.0 | 2.70 | 100% | 533x |
| 16,384 | 63.209 | 3.8580 | 259,204 | 7.2 | 10.34 | 100% | 546x |
| 65,536 | 269.676 | 4.1149 | 243,017 | 6.7 | 40.90 | 100% | 512x |

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointmazeLargeNavigate

State: 2 floats `x, y`. Reference: `ogbench/pointmaze-large-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **444 frames/s** (2,252 us/frame, 0.0668 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.146 | 3146.2330 | 318 | 0.0 | 0.15 | 100% | 0.72x |
| 64 | 3.245 | 50.7052 | 19,722 | 3.0 | 0.20 | 100% | 44x |
| 256 | 3.500 | 13.6720 | 73,142 | 11.0 | 0.34 | 100% | 165x |
| 1,024 | 5.615 | 5.4838 | 182,355 | 27.4 | 0.91 | 100% | 411x |
| 4,096 | 16.934 | 4.1343 | 241,878 | 36.4 | 3.20 | 100% | 545x |
| 16,384 | 66.317 | 4.0477 | 247,056 | 37.2 | 12.35 | 100% | 556x |
| 65,536 | 282.608 | 4.3123 | 231,897 | 34.9 | 48.95 | 100% | 522x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **501 frames/s** (1,997 us/frame, 0.0138 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.175 | 3174.7753 | 315 | 0.0 | 0.15 | 100% | 0.63x |
| 64 | 3.265 | 51.0210 | 19,600 | 0.5 | 0.19 | 100% | 39x |
| 256 | 3.494 | 13.6487 | 73,267 | 2.0 | 0.31 | 100% | 146x |
| 1,024 | 5.427 | 5.3002 | 188,671 | 5.2 | 0.79 | 100% | 377x |
| 4,096 | 16.178 | 3.9497 | 253,184 | 7.0 | 2.70 | 100% | 506x |
| 16,384 | 63.249 | 3.8604 | 259,039 | 7.2 | 10.34 | 100% | 517x |
| 65,536 | 270.933 | 4.1341 | 241,890 | 6.7 | 40.90 | 100% | 483x |

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointmazeLargeStitch

State: 2 floats `x, y`. Reference: `ogbench/pointmaze-large-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **406 frames/s** (2,465 us/frame, 0.0611 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.144 | 3143.5397 | 318 | 0.0 | 0.15 | 100% | 0.78x |
| 64 | 3.243 | 50.6641 | 19,738 | 3.0 | 0.20 | 100% | 49x |
| 256 | 3.492 | 13.6393 | 73,318 | 11.0 | 0.34 | 100% | 181x |
| 1,024 | 5.601 | 5.4694 | 182,836 | 27.5 | 0.92 | 100% | 451x |
| 4,096 | 16.921 | 4.1311 | 242,068 | 36.4 | 3.20 | 100% | 597x |
| 16,384 | 66.287 | 4.0459 | 247,167 | 37.2 | 12.35 | 100% | 609x |
| 65,536 | 282.529 | 4.3111 | 231,962 | 34.9 | 48.95 | 100% | 572x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **504 frames/s** (1,982 us/frame, 0.0139 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.139 | 3139.4297 | 319 | 0.0 | 0.15 | 100% | 0.63x |
| 64 | 3.237 | 50.5832 | 19,769 | 0.5 | 0.19 | 100% | 39x |
| 256 | 3.476 | 13.5799 | 73,638 | 2.0 | 0.31 | 100% | 146x |
| 1,024 | 5.411 | 5.2840 | 189,251 | 5.2 | 0.79 | 100% | 375x |
| 4,096 | 16.161 | 3.9455 | 253,456 | 7.0 | 2.70 | 100% | 502x |
| 16,384 | 63.210 | 3.8581 | 259,198 | 7.2 | 10.34 | 100% | 514x |
| 65,536 | 269.699 | 4.1153 | 242,997 | 6.7 | 40.90 | 100% | 482x |

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointmazeMediumNavigate

State: 2 floats `x, y`. Reference: `ogbench/pointmaze-medium-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **425 frames/s** (2,353 us/frame, 0.0640 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.136 | 3136.3545 | 319 | 0.0 | 0.16 | 100% | 0.75x |
| 64 | 3.232 | 50.5040 | 19,800 | 3.0 | 0.20 | 100% | 47x |
| 256 | 3.488 | 13.6237 | 73,401 | 11.0 | 0.35 | 100% | 173x |
| 1,024 | 5.609 | 5.4780 | 182,550 | 27.5 | 0.92 | 100% | 430x |
| 4,096 | 16.960 | 4.1407 | 241,503 | 36.4 | 3.20 | 100% | 568x |
| 16,384 | 66.339 | 4.0490 | 246,973 | 37.2 | 12.35 | 100% | 581x |
| 65,536 | 281.926 | 4.3018 | 232,459 | 35.0 | 48.95 | 100% | 547x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **540 frames/s** (1,851 us/frame, 0.0149 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.136 | 3135.7985 | 319 | 0.0 | 0.15 | 100% | 0.59x |
| 64 | 3.231 | 50.4827 | 19,809 | 0.5 | 0.19 | 100% | 37x |
| 256 | 3.473 | 13.5660 | 73,713 | 2.0 | 0.31 | 100% | 136x |
| 1,024 | 5.399 | 5.2724 | 189,667 | 5.2 | 0.79 | 100% | 351x |
| 4,096 | 16.179 | 3.9500 | 253,167 | 7.0 | 2.70 | 100% | 469x |
| 16,384 | 63.349 | 3.8665 | 258,629 | 7.2 | 10.34 | 100% | 479x |
| 65,536 | 269.583 | 4.1135 | 243,101 | 6.7 | 40.90 | 100% | 450x |

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointmazeMediumStitch

State: 2 floats `x, y`. Reference: `ogbench/pointmaze-medium-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **428 frames/s** (2,335 us/frame, 0.0645 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.150 | 3150.0002 | 317 | 0.0 | 0.16 | 100% | 0.74x |
| 64 | 3.246 | 50.7143 | 19,718 | 3.0 | 0.20 | 100% | 46x |
| 256 | 3.490 | 13.6341 | 73,345 | 11.0 | 0.35 | 100% | 171x |
| 1,024 | 5.608 | 5.4770 | 182,581 | 27.5 | 0.92 | 100% | 426x |
| 4,096 | 17.035 | 4.1590 | 240,444 | 36.2 | 3.20 | 100% | 561x |
| 16,384 | 66.499 | 4.0588 | 246,379 | 37.1 | 12.35 | 100% | 575x |
| 65,536 | 284.631 | 4.3431 | 230,249 | 34.7 | 48.95 | 100% | 538x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **539 frames/s** (1,857 us/frame, 0.0149 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.147 | 3146.7723 | 318 | 0.0 | 0.15 | 100% | 0.59x |
| 64 | 3.238 | 50.5979 | 19,764 | 0.5 | 0.19 | 100% | 37x |
| 256 | 3.472 | 13.5614 | 73,739 | 2.0 | 0.31 | 100% | 137x |
| 1,024 | 5.399 | 5.2728 | 189,651 | 5.2 | 0.79 | 100% | 352x |
| 4,096 | 16.177 | 3.9495 | 253,197 | 7.0 | 2.70 | 100% | 470x |
| 16,384 | 63.358 | 3.8671 | 258,594 | 7.1 | 10.34 | 100% | 480x |
| 65,536 | 269.695 | 4.1152 | 243,001 | 6.7 | 40.90 | 100% | 451x |

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointmazeTeleportNavigate

State: 2 floats `x, y`. Reference: `ogbench/pointmaze-teleport-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **403 frames/s** (2,484 us/frame, 0.0606 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.150 | 3149.9667 | 317 | 0.0 | 0.16 | 100% | 0.79x |
| 64 | 3.252 | 50.8172 | 19,678 | 3.0 | 0.20 | 100% | 49x |
| 256 | 3.488 | 13.6260 | 73,389 | 11.0 | 0.35 | 100% | 182x |
| 1,024 | 5.595 | 5.4642 | 183,011 | 27.5 | 0.92 | 100% | 455x |
| 4,096 | 16.907 | 4.1276 | 242,273 | 36.5 | 3.20 | 100% | 602x |
| 16,384 | 66.293 | 4.0462 | 247,144 | 37.2 | 12.35 | 100% | 614x |
| 65,536 | 282.325 | 4.3079 | 232,130 | 34.9 | 48.95 | 100% | 577x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **508 frames/s** (1,969 us/frame, 0.0140 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.126 | 3125.5074 | 320 | 0.0 | 0.15 | 100% | 0.63x |
| 64 | 3.245 | 50.6993 | 19,724 | 0.5 | 0.19 | 100% | 39x |
| 256 | 3.469 | 13.5508 | 73,797 | 2.0 | 0.31 | 100% | 145x |
| 1,024 | 5.404 | 5.2771 | 189,499 | 5.2 | 0.79 | 100% | 373x |
| 4,096 | 16.173 | 3.9485 | 253,258 | 7.0 | 2.70 | 100% | 499x |
| 16,384 | 63.316 | 3.8645 | 258,767 | 7.2 | 10.34 | 100% | 509x |
| 65,536 | 269.202 | 4.1077 | 243,445 | 6.7 | 40.90 | 100% | 479x |

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## PointmazeTeleportStitch

State: 2 floats `x, y`. Reference: `ogbench/pointmaze-teleport-v0`.

### 224x224 (supersample 2)

Official CPU renderer: **403 frames/s** (2,483 us/frame, 0.0606 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.139 | 3139.4567 | 319 | 0.0 | 0.16 | 100% | 0.79x |
| 64 | 3.240 | 50.6193 | 19,755 | 3.0 | 0.20 | 100% | 49x |
| 256 | 3.481 | 13.5963 | 73,549 | 11.1 | 0.35 | 100% | 183x |
| 1,024 | 5.601 | 5.4694 | 182,834 | 27.5 | 0.92 | 100% | 454x |
| 4,096 | 16.925 | 4.1321 | 242,009 | 36.4 | 3.20 | 100% | 601x |
| 16,384 | 66.495 | 4.0585 | 246,394 | 37.1 | 12.35 | 100% | 612x |
| 65,536 | 281.930 | 4.3019 | 232,455 | 35.0 | 48.95 | 100% | 577x |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 2)

Official CPU renderer: **508 frames/s** (1,967 us/frame, 0.0141 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.135 | 3134.6204 | 319 | 0.0 | 0.15 | 100% | 0.63x |
| 64 | 3.235 | 50.5517 | 19,782 | 0.5 | 0.19 | 100% | 39x |
| 256 | 3.461 | 13.5210 | 73,959 | 2.0 | 0.31 | 100% | 145x |
| 1,024 | 5.397 | 5.2701 | 189,748 | 5.2 | 0.79 | 100% | 373x |
| 4,096 | 16.167 | 3.9469 | 253,361 | 7.0 | 2.70 | 100% | 498x |
| 16,384 | 63.279 | 3.8623 | 258,915 | 7.2 | 10.34 | 100% | 509x |
| 65,536 | 270.886 | 4.1334 | 241,932 | 6.7 | 40.90 | 100% | 476x |

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## ProcgenHeist

State: 196 floats `cell_0_0, cell_0_1, cell_0_2, cell_0_3, cell_0_4, cell_0_5, cell_0_6, cell_0_7 ...`. Reference: `procgen-heist-v0`.

### 64x64 (supersample 1)

Official CPU renderer: **958 frames/s** (1,044 us/frame, 0.0118 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 10.118 | 10117.6330 | 99 | 0.0 | 0.00 | 100% | 0.10x |
| 64 | 10.275 | 160.5410 | 6,229 | 0.1 | 0.01 | 100% | 6.50x |
| 256 | 10.333 | 40.3624 | 24,776 | 0.3 | 0.04 | 100% | 26x |
| 1,024 | 10.434 | 10.1899 | 98,137 | 1.2 | 0.14 | 100% | 102x |
| 4,096 | 12.533 | 3.0598 | 326,819 | 4.0 | 0.57 | 100% | 341x |
| 16,384 | 26.793 | 1.6353 | 611,498 | 7.5 | 2.29 | 100% | 638x |
| 65,536 | 95.133 | 1.4516 | 688,890 | 8.5 | 9.14 | 100% | 719x |

Storage: 784 B of state against 12,288 B of raw frame (64x64) — **16x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 1)

Official CPU renderer unavailable: ValueError: ProcgenHeist: procgen's observation is 64x64 only; asked for 96

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 10.167 | 10166.7710 | 98 | 0.0 | 0.01 | 100% | - |
| 64 | 10.286 | 160.7193 | 6,222 | 0.2 | 0.03 | 100% | - |
| 256 | 10.378 | 40.5396 | 24,667 | 0.7 | 0.09 | 100% | - |
| 1,024 | 11.213 | 10.9504 | 91,321 | 2.5 | 0.32 | 100% | - |
| 4,096 | 16.503 | 4.0290 | 248,199 | 6.9 | 1.27 | 100% | - |
| 16,384 | 54.006 | 3.2963 | 303,374 | 8.4 | 5.07 | 100% | - |
| 65,536 | 205.102 | 3.1296 | 319,528 | 8.8 | 20.27 | 100% | - |

Storage: 784 B of state against 27,648 B of raw frame (96x96) — **35x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 1)

Official CPU renderer unavailable: ValueError: ProcgenHeist: procgen's observation is 64x64 only; asked for 224

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 10.182 | 10181.5810 | 98 | 0.0 | 0.02 | 100% | - |
| 64 | 10.292 | 160.8127 | 6,218 | 0.9 | 0.12 | 100% | - |
| 256 | 11.627 | 45.4173 | 22,018 | 3.3 | 0.44 | 100% | - |
| 1,024 | 20.704 | 20.2185 | 49,460 | 7.4 | 1.72 | 100% | - |
| 4,096 | 70.781 | 17.2805 | 57,869 | 8.7 | 6.83 | 100% | - |
| 16,384 | 272.326 | 16.6214 | 60,163 | 9.1 | 27.28 | 100% | - |
| 65,536 | output would be 9.9 GB, over the 6.0 GB budget | | | | | | |

Storage: 784 B of state against 150,528 B of raw frame (224x224) — **192x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## ProcgenMaze

State: 629 floats `cell_0_0, cell_0_1, cell_0_2, cell_0_3, cell_0_4, cell_0_5, cell_0_6, cell_0_7 ...`. Reference: `procgen-maze-v0`.

### 64x64 (supersample 1)

Official CPU renderer: **925 frames/s** (1,081 us/frame, 0.0114 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 1.984 | 1983.5540 | 504 | 0.0 | 0.02 | 100% | 0.55x |
| 64 | 2.036 | 31.8081 | 31,439 | 0.4 | 0.03 | 100% | 34x |
| 256 | 2.034 | 7.9457 | 125,854 | 1.5 | 0.04 | 100% | 136x |
| 1,024 | 2.083 | 2.0339 | 491,674 | 6.0 | 0.11 | 100% | 532x |
| 4,096 | 2.484 | 0.6064 | 1,648,955 | 20.3 | 0.37 | 100% | 1,783x |
| 16,384 | 5.453 | 0.3329 | 3,004,341 | 36.9 | 1.43 | 100% | 3,248x |
| 65,536 | 18.665 | 0.2848 | 3,511,215 | 43.1 | 5.64 | 100% | 3,796x |

Storage: 2516 B of state against 12,288 B of raw frame (64x64) — **5x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 1)

Official CPU renderer unavailable: ValueError: ProcgenMaze: procgen's observation is 64x64 only; asked for 96

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 1.987 | 1986.8610 | 503 | 0.0 | 0.02 | 100% | - |
| 64 | 2.040 | 31.8723 | 31,375 | 0.9 | 0.03 | 100% | - |
| 256 | 2.051 | 8.0127 | 124,801 | 3.5 | 0.07 | 100% | - |
| 1,024 | 2.229 | 2.1771 | 459,320 | 12.7 | 0.21 | 100% | - |
| 4,096 | 3.559 | 0.8688 | 1,151,005 | 31.8 | 0.77 | 100% | - |
| 16,384 | 10.825 | 0.6607 | 1,513,583 | 41.8 | 3.02 | 100% | - |
| 65,536 | 40.301 | 0.6149 | 1,626,160 | 45.0 | 12.02 | 100% | - |

Storage: 2516 B of state against 27,648 B of raw frame (96x96) — **11x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 1)

Official CPU renderer unavailable: ValueError: ProcgenMaze: procgen's observation is 64x64 only; asked for 224

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 1.997 | 1997.1030 | 501 | 0.1 | 0.03 | 100% | - |
| 64 | 2.059 | 32.1648 | 31,090 | 4.7 | 0.09 | 100% | - |
| 256 | 2.347 | 9.1672 | 109,084 | 16.4 | 0.28 | 100% | - |
| 1,024 | 4.440 | 4.3361 | 230,620 | 34.7 | 1.02 | 100% | - |
| 4,096 | 14.353 | 3.5041 | 285,382 | 43.0 | 3.97 | 100% | - |
| 16,384 | 54.108 | 3.3025 | 302,802 | 45.6 | 15.79 | 100% | - |
| 65,536 | output would be 9.9 GB, over the 6.0 GB budget | | | | | | |

Storage: 2516 B of state against 150,528 B of raw frame (224x224) — **60x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## ProcgenMiner

State: 407 floats `cell_0_0, cell_0_1, cell_0_2, cell_0_3, cell_0_4, cell_0_5, cell_0_6, cell_0_7 ...`. Reference: `procgen-miner-v0`.

### 64x64 (supersample 1)

Official CPU renderer: **985 frames/s** (1,016 us/frame, 0.0121 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.888 | 2887.8191 | 346 | 0.0 | 0.05 | 100% | 0.35x |
| 64 | 2.945 | 46.0157 | 21,732 | 0.3 | 0.06 | 100% | 22x |
| 256 | 2.960 | 11.5637 | 86,478 | 1.1 | 0.08 | 100% | 88x |
| 1,024 | 2.972 | 2.9024 | 344,544 | 4.2 | 0.14 | 100% | 350x |
| 4,096 | 3.398 | 0.8295 | 1,205,588 | 14.8 | 0.39 | 100% | 1,225x |
| 16,384 | 6.861 | 0.4188 | 2,387,852 | 29.3 | 1.42 | 100% | 2,425x |
| 65,536 | 23.427 | 0.3575 | 2,797,458 | 34.4 | 5.50 | 100% | 2,841x |

Storage: 1628 B of state against 12,288 B of raw frame (64x64) — **8x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 1)

Official CPU renderer unavailable: ValueError: ProcgenMiner: procgen's observation is 64x64 only; asked for 96

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.881 | 2881.3679 | 347 | 0.0 | 0.09 | 100% | - |
| 64 | 2.936 | 45.8815 | 21,795 | 0.6 | 0.10 | 100% | - |
| 256 | 2.960 | 11.5621 | 86,489 | 2.4 | 0.14 | 100% | - |
| 1,024 | 3.138 | 3.0640 | 326,374 | 9.0 | 0.28 | 100% | - |
| 4,096 | 4.675 | 1.1414 | 876,106 | 24.2 | 0.83 | 100% | - |
| 16,384 | 13.651 | 0.8332 | 1,200,240 | 33.2 | 3.05 | 100% | - |
| 65,536 | 50.822 | 0.7755 | 1,289,511 | 35.7 | 11.92 | 100% | - |

Storage: 1628 B of state against 27,648 B of raw frame (96x96) — **17x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 224x224 (supersample 1)

Official CPU renderer unavailable: ValueError: ProcgenMiner: procgen's observation is 64x64 only; asked for 224

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 2.896 | 2895.8850 | 345 | 0.1 | 0.18 | 100% | - |
| 64 | 2.970 | 46.4096 | 21,547 | 3.2 | 0.24 | 100% | - |
| 256 | 3.275 | 12.7939 | 78,162 | 11.8 | 0.43 | 100% | - |
| 1,024 | 5.636 | 5.5043 | 181,675 | 27.3 | 1.17 | 100% | - |
| 4,096 | 18.062 | 4.4097 | 226,775 | 34.1 | 4.11 | 100% | - |
| 16,384 | 68.275 | 4.1671 | 239,972 | 36.1 | 15.90 | 100% | - |
| 65,536 | output would be 9.9 GB, over the 6.0 GB budget | | | | | | |

Storage: 1628 B of state against 150,528 B of raw frame (224x224) — **92x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

## Racetrack

State: 9 floats `ego_x, ego_y, ego_heading, ego_crashed, v1_x, v1_y, v1_heading, v1_crashed ...`. Reference: `racetrack-v0`.

### 600x600 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.894 | 5893.9620 | 170 | 0.2 | 0.22 | 100% | - |
| 64 | 8.020 | 125.3132 | 7,980 | 8.6 | 0.29 | 100% | - |
| 256 | 14.601 | 57.0363 | 17,533 | 18.9 | 0.51 | 100% | - |
| 1,024 | 40.976 | 40.0154 | 24,990 | 27.0 | 1.37 | 100% | - |
| 4,096 | 167.466 | 40.8854 | 24,459 | 26.4 | 4.84 | 100% | - |
| 16,384 | output would be 17.7 GB, over the 6.0 GB budget | | | | | | |
| 65,536 | output would be 70.8 GB, over the 6.0 GB budget | | | | | | |

Storage: 36 B of state against 1,080,000 B of raw frame (600x600) — **30,000x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 5.657 | 0.139 | 5.890 | 7,216 | 7.8 |
| 256 | 5.819 | 8.684 | 14.598 | 29,480 | 31.8 |
| 4,096 | 6.007 | 161.937 | 168.827 | 25,294 | 27.3 |

### 96x96 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.875 | 5874.9029 | 170 | 0.0 | 0.22 | 100% | - |
| 64 | 6.037 | 94.3295 | 10,601 | 0.3 | 0.22 | 100% | - |
| 256 | 6.221 | 24.2999 | 41,152 | 1.1 | 0.23 | 100% | - |
| 1,024 | 6.831 | 6.6710 | 149,901 | 4.1 | 0.28 | 100% | - |
| 4,096 | 9.580 | 2.3389 | 427,556 | 11.8 | 0.46 | 100% | - |
| 16,384 | 22.829 | 1.3934 | 717,690 | 19.8 | 1.18 | 100% | - |
| 65,536 | 76.963 | 1.1744 | 851,528 | 23.5 | 4.06 | 100% | - |

Storage: 36 B of state against 27,648 B of raw frame (96x96) — **768x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 5.647 | 0.117 | 5.873 | 8,557 | 0.2 |
| 256 | 5.825 | 0.308 | 6.206 | 830,613 | 23.0 |
| 4,096 | 5.845 | 3.632 | 9.565 | 1,127,661 | 31.2 |
| 16,384 | 7.653 | 15.151 | 22.793 | 1,081,405 | 29.9 |

### 224x224 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 5.864 | 5863.7259 | 171 | 0.0 | 0.22 | 100% | - |
| 64 | 6.286 | 98.2138 | 10,182 | 1.5 | 0.23 | 100% | - |
| 256 | 7.293 | 28.4895 | 35,101 | 5.3 | 0.27 | 100% | - |
| 1,024 | 11.227 | 10.9638 | 91,209 | 13.7 | 0.42 | 100% | - |
| 4,096 | 31.022 | 7.5738 | 132,035 | 19.9 | 1.02 | 100% | - |
| 16,384 | 113.084 | 6.9021 | 144,884 | 21.8 | 3.42 | 100% | - |
| 65,536 | output would be 9.9 GB, over the 6.0 GB budget | | | | | | |

Storage: 36 B of state against 150,528 B of raw frame (224x224) — **4,181x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 5.626 | 0.131 | 5.852 | 7,606 | 1.1 |
| 256 | 5.808 | 1.401 | 7.294 | 182,695 | 27.5 |
| 4,096 | 5.932 | 24.994 | 31.014 | 163,879 | 24.7 |
| 16,384 | 8.843 | 104.253 | 113.110 | 157,156 | 23.7 |

## Roundabout

State: 20 floats `ego_x, ego_y, ego_heading, ego_crashed, v1_x, v1_y, v1_heading, v1_crashed ...`. Reference: `roundabout-v0`.

### 600x600 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.613 | 3613.0780 | 277 | 0.3 | 0.22 | 100% | - |
| 64 | 7.046 | 110.0958 | 9,083 | 9.8 | 0.29 | 100% | - |
| 256 | 17.851 | 69.7321 | 14,341 | 15.5 | 0.51 | 100% | - |
| 1,024 | 61.452 | 60.0115 | 16,663 | 18.0 | 1.41 | 100% | - |
| 4,096 | 308.227 | 75.2506 | 13,289 | 14.4 | 4.98 | 100% | - |
| 16,384 | output would be 17.7 GB, over the 6.0 GB budget | | | | | | |
| 65,536 | output would be 70.8 GB, over the 6.0 GB budget | | | | | | |

Storage: 80 B of state against 1,080,000 B of raw frame (600x600) — **13,500x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 3.341 | 0.179 | 3.601 | 5,580 | 6.0 |
| 256 | 3.439 | 14.318 | 17.846 | 17,880 | 19.3 |
| 4,096 | 4.307 | 303.388 | 307.379 | 13,501 | 14.6 |

### 96x96 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.565 | 3564.6040 | 281 | 0.0 | 0.22 | 100% | - |
| 64 | 3.668 | 57.3122 | 17,448 | 0.5 | 0.22 | 100% | - |
| 256 | 3.905 | 15.2532 | 65,560 | 1.8 | 0.24 | 100% | - |
| 1,024 | 4.692 | 4.5821 | 218,240 | 6.0 | 0.29 | 100% | - |
| 4,096 | 8.926 | 2.1791 | 458,899 | 12.7 | 0.52 | 100% | - |
| 16,384 | 27.909 | 1.7034 | 587,045 | 16.2 | 1.44 | 100% | - |
| 65,536 | 104.176 | 1.5896 | 629,090 | 17.4 | 5.11 | 100% | - |

Storage: 80 B of state against 27,648 B of raw frame (96x96) — **346x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 3.336 | 0.131 | 3.542 | 7,615 | 0.2 |
| 256 | 3.433 | 0.377 | 3.896 | 678,662 | 18.8 |
| 4,096 | 3.648 | 5.187 | 8.900 | 789,594 | 21.8 |
| 16,384 | 7.075 | 20.814 | 27.894 | 787,160 | 21.8 |

### 224x224 (supersample 1)

Official CPU renderer unavailable: ModuleNotFoundError: No module named 'highway_env'

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.566 | 3565.5471 | 280 | 0.0 | 0.22 | 100% | - |
| 64 | 4.005 | 62.5708 | 15,982 | 2.4 | 0.23 | 100% | - |
| 256 | 5.217 | 20.3786 | 49,071 | 7.4 | 0.27 | 100% | - |
| 1,024 | 10.015 | 9.7800 | 102,249 | 15.4 | 0.43 | 100% | - |
| 4,096 | 35.643 | 8.7018 | 114,918 | 17.3 | 1.07 | 100% | - |
| 16,384 | 136.289 | 8.3184 | 120,215 | 18.1 | 3.62 | 100% | - |
| 65,536 | output would be 9.9 GB, over the 6.0 GB budget | | | | | | |

Storage: 80 B of state against 150,528 B of raw frame (224x224) — **1,882x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 3.331 | 0.146 | 3.559 | 6,847 | 1.0 |
| 256 | 3.437 | 1.685 | 5.211 | 151,972 | 22.9 |
| 4,096 | 3.833 | 31.745 | 35.637 | 129,029 | 19.4 |
| 16,384 | 7.914 | 128.348 | 136.299 | 127,653 | 19.2 |

## SimplePointMaze

State: 2 floats `agent_x, agent_y`. Reference: `swm/SimplePointMaze-v0`.

### 500x500 (supersample 1)

Official CPU renderer: **54 frames/s** (18,585 us/frame, 0.0404 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 3.300 | 3299.9470 | 303 | 0.2 | 0.22 | 100% | 5.63x |
| 64 | 5.247 | 81.9866 | 12,197 | 9.1 | 0.27 | 100% | 227x |
| 256 | 7.396 | 28.8915 | 34,612 | 26.0 | 0.41 | 100% | 643x |
| 1,024 | 17.408 | 17.0004 | 58,822 | 44.1 | 0.99 | 100% | 1,093x |
| 4,096 | 57.502 | 14.0385 | 71,233 | 53.4 | 3.30 | 100% | 1,324x |
| 16,384 | output would be 12.3 GB, over the 6.0 GB budget | | | | | | |
| 65,536 | output would be 49.2 GB, over the 6.0 GB budget | | | | | | |

Storage: 8 B of state against 750,000 B of raw frame (500x500) — **93,750x** smaller before any compression.

Where the time goes:

| batch | scene() ms | rasterize ms | render() ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 3.154 | 0.073 | 3.281 | 13,742 | 10.3 |
| 256 | 3.690 | 3.560 | 7.400 | 71,914 | 53.9 |
| 4,096 | 3.695 | 53.629 | 57.497 | 76,377 | 57.3 |

### 96x96

Official CPU renderer unavailable: ValueError: SimplePointMaze renders at 500px only: the reference is a 5x5 inch matplotlib figure at 100 dpi and its axes spine is a 0.8 pt line, i.e. a constant in OUTPUT PIXELS with no world-unit description. Asked for 96.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

### 224x224

Official CPU renderer unavailable: ValueError: SimplePointMaze renders at 500px only: the reference is a 5x5 inch matplotlib figure at 100 dpi and its axes spine is a 0.8 pt line, i.e. a constant in OUTPUT PIXELS with no world-unit description. Asked for 224.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

## TwoRoom

State: 2 floats `agent_x, agent_y`. Reference: `swm/TwoRoom-v1`.

### 224x224 (supersample 1)

Official CPU renderer: **720 frames/s** (1,388 us/frame, 0.1084 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 0.862 | 862.3861 | 1,160 | 0.2 | 0.22 | 100% | 1.61x |
| 64 | 0.974 | 15.2115 | 65,740 | 9.9 | 0.42 | 100% | 91x |
| 256 | 2.251 | 8.7934 | 113,722 | 17.1 | 1.02 | 100% | 158x |
| 1,024 | 8.826 | 8.6192 | 116,020 | 17.5 | 1.13 | 100% | 161x |
| 4,096 | 35.100 | 8.5694 | 116,695 | 17.6 | 1.60 | 100% | 162x |
| 16,384 | 140.161 | 8.5547 | 116,894 | 17.6 | 3.45 | 100% | 162x |
| 65,536 | output would be 9.9 GB, over the 6.0 GB budget | | | | | | |

Storage: 8 B of state against 150,528 B of raw frame (224x224) — **18,816x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96

Official CPU renderer unavailable: ValueError: TwoRoom renders at 224px only, and asking for 96 is not a resolution change. Every length in the reference is in output pixels — a 4 px border, a door half-extent in pixels, a Gaussian sigma in pixels, a wall at column 112 — with no world-unit description underneath, so there is nothing to re-sample. Resize the frames afterwards if a different size is wanted; that is what stable_worldmodel's own pipeline does.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 8 B of state against 27,648 B of raw frame (96x96) — **3,456x** smaller before any compression.

## TwoRoomTarget

State: 4 floats `agent_x, agent_y, target_x, target_y`. Reference: `swm/TwoRoom-v1`.

### 224x224 (supersample 1)

Official CPU renderer: **521 frames/s** (1,920 us/frame, 0.0784 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 1.115 | 1114.7320 | 897 | 0.1 | 0.22 | 100% | 1.72x |
| 64 | 1.302 | 20.3397 | 49,165 | 7.4 | 0.42 | 99% | 94x |
| 256 | 3.402 | 13.2879 | 75,256 | 11.3 | 1.02 | 100% | 144x |
| 1,024 | 13.431 | 13.1161 | 76,242 | 11.5 | 1.14 | 100% | 146x |
| 4,096 | 53.484 | 13.0575 | 76,584 | 11.5 | 1.60 | 100% | 147x |
| 16,384 | 213.714 | 13.0440 | 76,663 | 11.5 | 3.45 | 100% | 147x |
| 65,536 | output would be 9.9 GB, over the 6.0 GB budget | | | | | | |

Storage: 16 B of state against 150,528 B of raw frame (224x224) — **9,408x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96

Official CPU renderer unavailable: ValueError: TwoRoomTarget renders at 224px only, and asking for 96 is not a resolution change. Every length in the reference is in output pixels — a 4 px border, a door half-extent in pixels, a Gaussian sigma in pixels, a wall at column 112 — with no world-unit description underneath, so there is nothing to re-sample. Resize the frames afterwards if a different size is wanted; that is what stable_worldmodel's own pipeline does.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|

Storage: 16 B of state against 27,648 B of raw frame (96x96) — **1,728x** smaller before any compression.

## ProcgenChaser

State: 199 floats `cell_0_0, cell_0_1, cell_0_2, cell_0_3, cell_0_4, cell_0_5, cell_0_6, cell_0_7 ...`. Reference: `procgen-chaser-v0`.

### 64x64 (supersample 1)

Official CPU renderer: **878 frames/s** (1,139 us/frame, 0.0108 GB/s) for set_state+render.

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 11.056 | 11056.2332 | 90 | 0.0 | 0.16 | 100% | 0.10x |
| 64 | 11.153 | 174.2636 | 5,738 | 0.1 | 0.16 | 100% | 6.53x |
| 256 | 11.257 | 43.9715 | 22,742 | 0.3 | 0.18 | 100% | 26x |
| 1,024 | 11.337 | 11.0710 | 90,326 | 1.1 | 0.24 | 100% | 103x |
| 4,096 | 11.571 | 2.8249 | 353,996 | 4.3 | 0.49 | 100% | 403x |
| 16,384 | 19.472 | 1.1885 | 841,416 | 10.3 | 1.48 | 100% | 958x |
| 65,536 | 67.179 | 1.0251 | 975,546 | 12.0 | 5.45 | 100% | 1,111x |

Storage: 796 B of state against 12,288 B of raw frame (64x64) — **15x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().

### 96x96 (supersample 1)

Official CPU renderer unavailable: ValueError: ProcgenChaser: procgen's observation is 64x64 only; asked for 96

| batch | ms / batch | us / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 11.165 | 11165.0787 | 90 | 0.0 | 0.16 | 100% | - |
| 64 | 11.292 | 176.4391 | 5,668 | 0.2 | 0.17 | 100% | - |
| 256 | 11.331 | 44.2627 | 22,592 | 0.6 | 0.20 | 100% | - |
| 1,024 | 11.565 | 11.2942 | 88,541 | 2.4 | 0.34 | 100% | - |
| 4,096 | 12.388 | 3.0244 | 330,640 | 9.1 | 0.89 | 100% | - |
| 16,384 | 38.703 | 2.3623 | 423,322 | 11.7 | 3.08 | 100% | - |
| 65,536 | 144.793 | 2.2094 | 452,619 | 12.5 | 11.84 | 100% | - |

Storage: 796 B of state against 27,648 B of raw frame (96x96) — **35x** smaller before any compression.

No scene/rasterize split: no Scene2D; this env overrides render().
