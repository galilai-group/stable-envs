# Profile note — TwoRoom on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2342.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/TwoRoom.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **4.70 ms** median (min 4.31, max 4.79, sd 0.14) = **213 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1446 | 266 |
| host->device | 0 | - | 820 | 63 |
| allocation | 0 | - | 69 | 30 |
| kernel/raster | 0 | - | 3738 | 32 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 1580 |
| `aten::where` | 5 | 0 | 1242 |
| `aten::copy_` | 21 | 0 | 751 |
| `aten::add` | 6 | 0 | 595 |
| `aten::exp` | 1 | 0 | 336 |
| `aten::ge` | 5 | 0 | 223 |
| `aten::bitwise_and` | 10 | 0 | 218 |
| `aten::le` | 5 | 0 | 205 |

## batch 16

render **66.76 ms** median (min 56.17, max 69.87, sd 3.78) = **240 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 12160 | 266 |
| host->device | 0 | - | 11385 | 63 |
| allocation | 0 | - | 156 | 30 |
| kernel/raster | 0 | - | 40949 | 32 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 5 | 0 | 17449 |
| `aten::mul` | 7 | 0 | 15252 |
| `aten::copy_` | 21 | 0 | 11252 |
| `aten::add` | 6 | 0 | 4996 |
| `aten::bitwise_and` | 10 | 0 | 3118 |
| `aten::ge` | 5 | 0 | 3103 |
| `aten::le` | 5 | 0 | 3020 |
| `aten::sub` | 7 | 0 | 1660 |

## batch 64

render **257.99 ms** median (min 243.21, max 277.07, sd 11.44) = **248 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 50824 | 266 |
| host->device | 0 | - | 53426 | 63 |
| allocation | 0 | - | 159 | 30 |
| kernel/raster | 0 | - | 177813 | 32 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 5 | 0 | 70745 |
| `aten::mul` | 7 | 0 | 69898 |
| `aten::copy_` | 21 | 0 | 53216 |
| `aten::add` | 6 | 0 | 22678 |
| `aten::bitwise_and` | 10 | 0 | 14050 |
| `aten::ge` | 5 | 0 | 11887 |
| `aten::le` | 5 | 0 | 11807 |
| `aten::sub` | 7 | 0 | 6852 |

## batch 256

render **1499.56 ms** median (min 1424.31, max 1601.76, sd 50.23) = **171 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 244553 | 266 |
| host->device | 0 | - | 237695 | 63 |
| allocation | 0 | - | 270 | 30 |
| kernel/raster | 0 | - | 889063 | 32 |
| synchronization | 0 | - | 4 | 1 |
| output conversion | 0 | - | 3 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 5 | 0 | 326633 |
| `aten::mul` | 7 | 0 | 312695 |
| `aten::copy_` | 21 | 0 | 237478 |
| `aten::add` | 6 | 0 | 155531 |
| `aten::bitwise_and` | 10 | 0 | 62827 |
| `aten::exp` | 1 | 0 | 48855 |
| `aten::sub` | 7 | 0 | 47687 |
| `aten::ge` | 5 | 0 | 47078 |

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
