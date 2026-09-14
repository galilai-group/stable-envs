# Profile note — TwoRoomTarget on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2340.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 4
- states: `examples/results/step4/states/TwoRoomTarget.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **5.69 ms** median (min 5.37, max 6.18, sd 0.25) = **176 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1373 | 328 |
| host->device | 0 | - | 851 | 88 |
| allocation | 0 | - | 73 | 39 |
| kernel/raster | 0 | - | 4182 | 49 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 2021 |
| `aten::where` | 6 | 0 | 1154 |
| `aten::copy_` | 29 | 0 | 765 |
| `aten::add` | 8 | 0 | 519 |
| `aten::exp` | 2 | 0 | 292 |
| `aten::sub` | 10 | 0 | 229 |
| `aten::div` | 5 | 0 | 202 |
| `aten::ge` | 5 | 0 | 191 |

## batch 16

render **90.77 ms** median (min 74.66, max 104.27, sd 11.02) = **176 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 12686 | 328 |
| host->device | 0 | - | 11343 | 88 |
| allocation | 0 | - | 102 | 39 |
| kernel/raster | 0 | - | 53416 | 49 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 20055 |
| `aten::where` | 6 | 0 | 17609 |
| `aten::copy_` | 29 | 0 | 11191 |
| `aten::add` | 8 | 0 | 9726 |
| `aten::sub` | 10 | 0 | 3012 |
| `aten::bitwise_and` | 10 | 0 | 2845 |
| `aten::ge` | 5 | 0 | 2761 |
| `aten::le` | 5 | 0 | 2707 |

## batch 64

render **436.60 ms** median (min 375.88, max 487.42, sd 32.51) = **147 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 70430 | 328 |
| host->device | 0 | - | 68195 | 88 |
| allocation | 0 | - | 231 | 39 |
| kernel/raster | 0 | - | 284143 | 49 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 126398 |
| `aten::where` | 6 | 0 | 76435 |
| `aten::copy_` | 29 | 0 | 67906 |
| `aten::add` | 8 | 0 | 51112 |
| `aten::exp` | 2 | 0 | 24651 |
| `aten::div` | 5 | 0 | 13862 |
| `aten::bitwise_and` | 10 | 0 | 12718 |
| `aten::sub` | 10 | 0 | 12186 |

## batch 256

render **2415.98 ms** median (min 2323.10, max 2563.47, sd 75.25) = **106 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 332133 | 328 |
| host->device | 0 | - | 390271 | 88 |
| allocation | 0 | - | 337 | 39 |
| kernel/raster | 0 | - | 1456884 | 49 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

506 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 13 | 0 | 623532 |
| `aten::copy_` | 29 | 0 | 389952 |
| `aten::where` | 6 | 0 | 328581 |
| `aten::add` | 8 | 0 | 237372 |
| `aten::sub` | 10 | 0 | 126959 |
| `aten::exp` | 2 | 0 | 118865 |
| `aten::div` | 5 | 0 | 107819 |
| `aten::bitwise_and` | 10 | 0 | 51421 |

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
