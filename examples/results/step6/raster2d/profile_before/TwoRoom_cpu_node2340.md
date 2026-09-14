# Profile note — TwoRoom on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2340.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/TwoRoom.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **4.45 ms** median (min 3.89, max 5.89, sd 0.62) = **225 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1346 | 266 |
| host->device | 0 | - | 678 | 63 |
| allocation | 0 | - | 62 | 30 |
| kernel/raster | 0 | - | 3406 | 32 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 1406 |
| `aten::where` | 5 | 0 | 1228 |
| `aten::copy_` | 21 | 0 | 610 |
| `aten::add` | 6 | 0 | 475 |
| `aten::exp` | 1 | 0 | 256 |
| `aten::ge` | 5 | 0 | 219 |
| `aten::bitwise_and` | 10 | 0 | 217 |
| `aten::le` | 5 | 0 | 199 |

## batch 16

render **54.54 ms** median (min 51.39, max 75.62, sd 8.80) = **293 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 11005 | 266 |
| host->device | 0 | - | 7398 | 63 |
| allocation | 0 | - | 90 | 30 |
| kernel/raster | 0 | - | 43723 | 32 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 17720 |
| `aten::where` | 5 | 0 | 16907 |
| `aten::copy_` | 21 | 0 | 7290 |
| `aten::add` | 6 | 0 | 6145 |
| `aten::bitwise_and` | 10 | 0 | 2829 |
| `aten::ge` | 5 | 0 | 2777 |
| `aten::le` | 5 | 0 | 2710 |
| `aten::sub` | 7 | 0 | 1497 |

## batch 64

render **285.05 ms** median (min 247.29, max 306.77, sd 17.05) = **225 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 57180 | 266 |
| host->device | 0 | - | 39367 | 63 |
| allocation | 0 | - | 151 | 30 |
| kernel/raster | 0 | - | 196794 | 32 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 80614 |
| `aten::where` | 5 | 0 | 68172 |
| `aten::copy_` | 21 | 0 | 39180 |
| `aten::add` | 6 | 0 | 30494 |
| `aten::exp` | 1 | 0 | 14883 |
| `aten::bitwise_and` | 10 | 0 | 12607 |
| `aten::le` | 5 | 0 | 11595 |
| `aten::ge` | 5 | 0 | 11576 |

## batch 256

render **1504.03 ms** median (min 1450.99, max 1575.69, sd 38.32) = **170 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 247565 | 266 |
| host->device | 0 | - | 245943 | 63 |
| allocation | 0 | - | 221 | 30 |
| kernel/raster | 0 | - | 926273 | 32 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 5 | 0 | 333479 |
| `aten::mul` | 7 | 0 | 314927 |
| `aten::copy_` | 21 | 0 | 245749 |
| `aten::add` | 6 | 0 | 116834 |
| `aten::sub` | 7 | 0 | 81379 |
| `aten::exp` | 1 | 0 | 57432 |
| `aten::div` | 3 | 0 | 54004 |
| `aten::bitwise_and` | 10 | 0 | 53274 |

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
