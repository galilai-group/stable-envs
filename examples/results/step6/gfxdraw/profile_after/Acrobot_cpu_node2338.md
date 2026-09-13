# Profile note — Acrobot on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node2338.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Acrobot.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **21.29 ms** median (min 18.78, max 21.92, sd 1.10) = **47 fps**  ·  set_state 0.08 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 10203 | 8762 |
| host->device | 0 | - | 1559 | 1013 |
| allocation | 0 | - | 1122 | 1282 |
| kernel/raster | 0 | - | 5482 | 1682 |
| synchronization | 0 | - | 3 | 1 |

12740 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 670 | 0 | 2178 |
| `aten::where` | 484 | 0 | 1490 |
| `aten::mul` | 249 | 0 | 1027 |
| `aten::add` | 389 | 0 | 817 |
| `aten::index` | 126 | 0 | 801 |
| `aten::copy_` | 292 | 0 | 770 |
| `aten::ge` | 96 | 0 | 612 |
| `aten::__rshift__` | 150 | 0 | 596 |

## batch 16

render **57.93 ms** median (min 51.83, max 67.52, sd 4.63) = **276 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 36095 | 8854 |
| host->device | 0 | - | 13547 | 1059 |
| allocation | 0 | - | 1496 | 1374 |
| kernel/raster | 0 | - | 18342 | 1682 |
| synchronization | 0 | - | 3 | 1 |

12970 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 338 | 0 | 12642 |
| `aten::mul` | 249 | 0 | 8457 |
| `aten::bitwise_xor` | 2 | 0 | 7022 |
| `aten::bitwise_and` | 670 | 0 | 4930 |
| `aten::flip` | 3 | 0 | 4692 |
| `aten::ge` | 96 | 0 | 3318 |
| `aten::index` | 126 | 0 | 3165 |
| `aten::le` | 20 | 0 | 3055 |

## batch 64

render **255.45 ms** median (min 252.03, max 268.23, sd 6.21) = **251 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 134084 | 8854 |
| host->device | 0 | - | 51637 | 1059 |
| allocation | 0 | - | 1796 | 1374 |
| kernel/raster | 0 | - | 61906 | 1682 |
| synchronization | 0 | - | 3 | 1 |

12970 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 338 | 0 | 50603 |
| `aten::mul` | 249 | 0 | 31681 |
| `aten::bitwise_xor` | 2 | 0 | 25681 |
| `aten::fill_` | 58 | 0 | 20894 |
| `aten::flip` | 3 | 0 | 19152 |
| `aten::bitwise_and` | 670 | 0 | 14224 |
| `aten::le` | 20 | 0 | 14044 |
| `aten::ge` | 96 | 0 | 12477 |

## batch 256

render **1063.11 ms** median (min 1034.46, max 1113.63, sd 27.69) = **241 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 554731 | 8854 |
| host->device | 0 | - | 200547 | 1059 |
| allocation | 0 | - | 1953 | 1374 |
| kernel/raster | 0 | - | 230346 | 1682 |
| synchronization | 0 | - | 4 | 1 |

12970 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 338 | 0 | 199430 |
| `aten::mul` | 249 | 0 | 135746 |
| `aten::bitwise_xor` | 2 | 0 | 111288 |
| `aten::flip` | 3 | 0 | 93064 |
| `aten::fill_` | 58 | 0 | 82492 |
| `aten::ge` | 96 | 0 | 69790 |
| `aten::le` | 20 | 0 | 64947 |
| `aten::bitwise_xor_` | 2 | 0 | 43316 |

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
