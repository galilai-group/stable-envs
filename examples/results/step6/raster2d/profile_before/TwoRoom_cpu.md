# Profile note — TwoRoom on cpu

- workstream: `step6/raster2d`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/TwoRoom.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **4.38 ms** median (min 3.98, max 5.75, sd 0.56) = **228 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 5635 | 266 |
| host->device | 0 | - | 616 | 63 |
| allocation | 0 | - | 69 | 30 |
| kernel/raster | 0 | - | 2833 | 32 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::clone` | 1 | 0 | 2340 |
| `aten::abs` | 2 | 0 | 2037 |
| `aten::where` | 5 | 0 | 1362 |
| `aten::mul` | 7 | 0 | 948 |
| `aten::copy_` | 21 | 0 | 543 |
| `aten::bitwise_and` | 10 | 0 | 238 |
| `aten::ge` | 5 | 0 | 229 |
| `aten::le` | 5 | 0 | 221 |

## batch 16

render **59.62 ms** median (min 52.02, max 66.22, sd 3.46) = **268 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 10965 | 266 |
| host->device | 0 | - | 7350 | 63 |
| allocation | 0 | - | 73 | 30 |
| kernel/raster | 0 | - | 33155 | 32 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 5 | 0 | 17061 |
| `aten::mul` | 7 | 0 | 10169 |
| `aten::copy_` | 21 | 0 | 7257 |
| `aten::add` | 6 | 0 | 2923 |
| `aten::bitwise_and` | 10 | 0 | 2835 |
| `aten::ge` | 5 | 0 | 2801 |
| `aten::le` | 5 | 0 | 2675 |
| `aten::sub` | 7 | 0 | 1443 |

## batch 64

render **301.64 ms** median (min 268.95, max 339.40, sd 21.02) = **212 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 61615 | 266 |
| host->device | 0 | - | 33017 | 63 |
| allocation | 0 | - | 121 | 30 |
| kernel/raster | 0 | - | 196768 | 32 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 3 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 81720 |
| `aten::where` | 5 | 0 | 68545 |
| `aten::add` | 6 | 0 | 32966 |
| `aten::copy_` | 21 | 0 | 32830 |
| `aten::exp` | 1 | 0 | 19345 |
| `aten::bitwise_and` | 10 | 0 | 12991 |
| `aten::ge` | 5 | 0 | 11579 |
| `aten::le` | 5 | 0 | 11457 |

## batch 256

render **1577.74 ms** median (min 1055.23, max 1716.63, sd 235.62) = **162 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 265533 | 266 |
| host->device | 0 | - | 241947 | 63 |
| allocation | 0 | - | 199 | 30 |
| kernel/raster | 0 | - | 1009799 | 32 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

393 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 353525 |
| `aten::where` | 5 | 0 | 336337 |
| `aten::copy_` | 21 | 0 | 241731 |
| `aten::add` | 6 | 0 | 148626 |
| `aten::sub` | 7 | 0 | 87536 |
| `aten::bitwise_and` | 10 | 0 | 69713 |
| `aten::exp` | 1 | 0 | 58417 |
| `aten::div` | 3 | 0 | 58054 |

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
