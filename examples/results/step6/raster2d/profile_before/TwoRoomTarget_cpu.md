# Profile note — TwoRoomTarget on cpu

- workstream: `step6/raster2d`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 4
- states: `examples/results/step4/states/TwoRoomTarget.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **5.84 ms** median (min 5.78, max 5.89, sd 0.03) = **171 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1717 | 328 |
| host->device | 0 | - | 1188 | 88 |
| allocation | 0 | - | 106 | 39 |
| kernel/raster | 0 | - | 5336 | 49 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 2238 |
| `aten::where` | 6 | 0 | 1551 |
| `aten::copy_` | 29 | 0 | 1081 |
| `aten::add` | 8 | 0 | 916 |
| `aten::exp` | 2 | 0 | 341 |
| `aten::sub` | 10 | 0 | 303 |
| `aten::div` | 5 | 0 | 258 |
| `aten::ge` | 5 | 0 | 253 |

## batch 16

render **91.24 ms** median (min 90.14, max 93.18, sd 0.95) = **175 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 14311 | 328 |
| host->device | 0 | - | 11985 | 88 |
| allocation | 0 | - | 160 | 39 |
| kernel/raster | 0 | - | 55802 | 49 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 23120 |
| `aten::where` | 6 | 0 | 17831 |
| `aten::copy_` | 29 | 0 | 11810 |
| `aten::add` | 8 | 0 | 8767 |
| `aten::exp` | 2 | 0 | 3785 |
| `aten::sub` | 10 | 0 | 3027 |
| `aten::bitwise_and` | 10 | 0 | 2869 |
| `aten::ge` | 5 | 0 | 2815 |

## batch 64

render **442.82 ms** median (min 393.47, max 481.22, sd 25.48) = **145 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 78827 | 328 |
| host->device | 0 | - | 81235 | 88 |
| allocation | 0 | - | 252 | 39 |
| kernel/raster | 0 | - | 295362 | 49 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 11 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 134509 |
| `aten::copy_` | 29 | 0 | 80949 |
| `aten::where` | 6 | 0 | 71757 |
| `aten::add` | 8 | 0 | 62641 |
| `aten::exp` | 2 | 0 | 33357 |
| `aten::bitwise_and` | 10 | 0 | 12606 |
| `aten::sub` | 10 | 0 | 12426 |
| `aten::ge` | 5 | 0 | 11634 |

## batch 256

render **2408.70 ms** median (min 2351.04, max 2526.32, sd 51.08) = **106 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 344777 | 328 |
| host->device | 0 | - | 386906 | 88 |
| allocation | 0 | - | 300 | 39 |
| kernel/raster | 0 | - | 1456582 | 49 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 4 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 616259 |
| `aten::copy_` | 29 | 0 | 386594 |
| `aten::where` | 6 | 0 | 325630 |
| `aten::add` | 8 | 0 | 232233 |
| `aten::sub` | 10 | 0 | 143669 |
| `aten::exp` | 2 | 0 | 117913 |
| `aten::div` | 5 | 0 | 105402 |
| `aten::bitwise_and` | 10 | 0 | 60234 |

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
