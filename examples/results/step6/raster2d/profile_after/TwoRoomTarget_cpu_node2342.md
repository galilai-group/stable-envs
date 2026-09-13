# Profile note — TwoRoomTarget on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2342.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 4
- states: `examples/results/step4/states/TwoRoomTarget.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **6.16 ms** median (min 5.81, max 6.55, sd 0.22) = **162 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1414 | 328 |
| host->device | 0 | - | 996 | 88 |
| allocation | 0 | - | 96 | 39 |
| kernel/raster | 0 | - | 5030 | 49 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 2442 |
| `aten::where` | 6 | 0 | 1386 |
| `aten::copy_` | 29 | 0 | 882 |
| `aten::add` | 8 | 0 | 606 |
| `aten::sub` | 10 | 0 | 308 |
| `aten::ge` | 5 | 0 | 236 |
| `aten::le` | 5 | 0 | 220 |
| `aten::bitwise_and` | 10 | 0 | 210 |

## batch 16

render **85.85 ms** median (min 83.15, max 103.15, sd 6.81) = **186 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 14919 | 328 |
| host->device | 0 | - | 13230 | 88 |
| allocation | 0 | - | 141 | 39 |
| kernel/raster | 0 | - | 54284 | 49 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 21576 |
| `aten::where` | 6 | 0 | 18391 |
| `aten::copy_` | 29 | 0 | 13004 |
| `aten::add` | 8 | 0 | 7312 |
| `aten::sub` | 10 | 0 | 3462 |
| `aten::bitwise_and` | 10 | 0 | 3097 |
| `aten::ge` | 5 | 0 | 2960 |
| `aten::exp` | 2 | 0 | 2921 |

## batch 64

render **406.45 ms** median (min 373.96, max 421.50, sd 16.50) = **157 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 62047 | 328 |
| host->device | 0 | - | 56162 | 88 |
| allocation | 0 | - | 216 | 39 |
| kernel/raster | 0 | - | 293616 | 49 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 137172 |
| `aten::where` | 6 | 0 | 74522 |
| `aten::copy_` | 29 | 0 | 55870 |
| `aten::add` | 8 | 0 | 53326 |
| `aten::bitwise_and` | 10 | 0 | 14097 |
| `aten::sub` | 10 | 0 | 13552 |
| `aten::exp` | 2 | 0 | 12545 |
| `aten::ge` | 5 | 0 | 12239 |

## batch 256

render **2720.59 ms** median (min 2592.95, max 2765.36, sd 50.43) = **94 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 401915 | 328 |
| host->device | 0 | - | 437471 | 88 |
| allocation | 0 | - | 350 | 39 |
| kernel/raster | 0 | - | 1743882 | 49 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 739400 |
| `aten::copy_` | 29 | 0 | 437126 |
| `aten::where` | 6 | 0 | 370654 |
| `aten::add` | 8 | 0 | 329422 |
| `aten::exp` | 2 | 0 | 153761 |
| `aten::sub` | 10 | 0 | 141194 |
| `aten::div` | 5 | 0 | 134902 |
| `aten::bitwise_and` | 10 | 0 | 64955 |

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
