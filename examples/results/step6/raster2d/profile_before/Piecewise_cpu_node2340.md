# Profile note — Piecewise on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2340.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/Piecewise.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **2.54 ms** median (min 2.51, max 2.73, sd 0.06) = **394 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 455 | 110 |
| host->device | 0 | - | 505 | 39 |
| allocation | 0 | - | 39 | 17 |
| kernel/raster | 0 | - | 2913 | 19 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 1395 |
| `aten::where` | 2 | 0 | 640 |
| `aten::copy_` | 13 | 0 | 445 |
| `aten::add` | 2 | 0 | 421 |
| `aten::sub` | 3 | 0 | 257 |
| `aten::exp` | 1 | 0 | 171 |
| `aten::div` | 2 | 0 | 153 |
| `aten::_to_copy` | 11 | 0 | 47 |

## batch 16

render **35.07 ms** median (min 34.28, max 54.22, sd 5.96) = **456 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1802 | 110 |
| host->device | 0 | - | 7437 | 39 |
| allocation | 0 | - | 47 | 17 |
| kernel/raster | 0 | - | 27779 | 19 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 3 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 10385 |
| `aten::where` | 2 | 0 | 8131 |
| `aten::copy_` | 13 | 0 | 7366 |
| `aten::add` | 2 | 0 | 6467 |
| `aten::sub` | 3 | 0 | 1488 |
| `aten::exp` | 1 | 0 | 1031 |
| `aten::div` | 2 | 0 | 975 |
| `aten::clamp` | 1 | 0 | 315 |

## batch 64

render **205.01 ms** median (min 177.27, max 229.19, sd 14.46) = **312 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 18448 | 110 |
| host->device | 0 | - | 48367 | 39 |
| allocation | 0 | - | 116 | 17 |
| kernel/raster | 0 | - | 144190 | 19 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 71634 |
| `aten::copy_` | 13 | 0 | 48229 |
| `aten::where` | 2 | 0 | 32552 |
| `aten::add` | 2 | 0 | 26718 |
| `aten::exp` | 1 | 0 | 14908 |
| `aten::sub` | 3 | 0 | 6341 |
| `aten::div` | 2 | 0 | 4809 |
| `aten::clamp` | 1 | 0 | 2107 |

## batch 256

render **1022.50 ms** median (min 973.82, max 1054.64, sd 29.92) = **250 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 83660 | 110 |
| host->device | 0 | - | 205048 | 39 |
| allocation | 0 | - | 166 | 17 |
| kernel/raster | 0 | - | 713389 | 19 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 310641 |
| `aten::copy_` | 13 | 0 | 204896 |
| `aten::where` | 2 | 0 | 157373 |
| `aten::add` | 2 | 0 | 118427 |
| `aten::sub` | 3 | 0 | 59324 |
| `aten::exp` | 1 | 0 | 59210 |
| `aten::div` | 2 | 0 | 52773 |
| `aten::neg` | 1 | 0 | 19538 |

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
