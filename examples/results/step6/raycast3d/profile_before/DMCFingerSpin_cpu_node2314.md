# Profile note — DMCFingerSpin on cpu

- workstream: `step6/raycast3d`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 84px  ·  state_dim: 3
- states: `examples/results/step4/states/DMCFingerSpin.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **48.59 ms** median (min 43.93, max 48.99, sd 2.12) = **21 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 17780 | 18090 |
| host->device | 0 | - | 2029 | 2180 |
| allocation | 0 | - | 1927 | 1824 |
| kernel/raster | 0 | - | 20993 | 4045 |
| synchronization | 0 | - | 4 | 4 |
| output conversion | 0 | - | 42 | 1 |

26144 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 1729 | 0 | 7069 |
| `aten::select` | 3625 | 0 | 5076 |
| `aten::cat` | 165 | 0 | 4376 |
| `aten::add` | 1146 | 0 | 3112 |
| `aten::gt` | 390 | 0 | 2617 |
| `aten::where` | 245 | 0 | 2336 |
| `aten::as_strided` | 4507 | 0 | 2018 |
| `aten::sub` | 249 | 0 | 1650 |

## batch 16

render **283.49 ms** median (min 281.22, max 293.60, sd 3.41) = **56 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 74014 | 18249 |
| host->device | 0 | - | 2871 | 2182 |
| allocation | 0 | - | 2401 | 1825 |
| kernel/raster | 0 | - | 206524 | 4045 |
| synchronization | 0 | - | 5 | 4 |
| output conversion | 0 | - | 205 | 1 |

26306 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 1729 | 0 | 63341 |
| `aten::cat` | 165 | 0 | 60177 |
| `aten::add` | 1146 | 0 | 29929 |
| `aten::where` | 245 | 0 | 21354 |
| `aten::gt` | 390 | 0 | 17478 |
| `aten::sqrt` | 98 | 0 | 15361 |
| `aten::sub` | 249 | 0 | 15110 |
| `aten::lt` | 183 | 0 | 7775 |

## batch 64

render **1208.10 ms** median (min 1190.84, max 1233.85, sd 13.06) = **53 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 281695 | 18404 |
| host->device | 0 | - | 4893 | 2182 |
| allocation | 0 | - | 3386 | 1825 |
| kernel/raster | 0 | - | 959065 | 4045 |
| synchronization | 0 | - | 5 | 4 |
| output conversion | 0 | - | 803 | 1 |

26461 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 1729 | 0 | 297348 |
| `aten::cat` | 165 | 0 | 241199 |
| `aten::add` | 1146 | 0 | 175726 |
| `aten::where` | 245 | 0 | 88663 |
| `aten::sub` | 249 | 0 | 75570 |
| `aten::gt` | 390 | 0 | 67003 |
| `aten::sqrt` | 98 | 0 | 61180 |
| `aten::div` | 115 | 0 | 38002 |

## batch 256

render **5420.73 ms** median (min 5194.06, max 5615.19, sd 155.34) = **47 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1232845 | 18405 |
| host->device | 0 | - | 11613 | 2182 |
| allocation | 0 | - | 5935 | 1825 |
| kernel/raster | 0 | - | 4692873 | 4045 |
| synchronization | 0 | - | 6 | 4 |
| output conversion | 0 | - | 3599 | 1 |

26462 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 1729 | 0 | 1603090 |
| `aten::cat` | 165 | 0 | 987085 |
| `aten::add` | 1146 | 0 | 954743 |
| `aten::where` | 245 | 0 | 397299 |
| `aten::sub` | 249 | 0 | 371528 |
| `aten::gt` | 390 | 0 | 268420 |
| `aten::sqrt` | 98 | 0 | 244094 |
| `aten::div` | 115 | 0 | 190984 |

## largest bottleneck

_to be completed by the workstream from the table above._

## structural hypothesis

_..._

## proposed intervention

_..._

## correctness/fidelity risk

_..._

## affected Step 4 configurations to replay

_from `examples/results/step6/plan.json`, field `step4_replay_configs`._
