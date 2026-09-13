# Profile note — CraftaxClassic on cpu

- workstream: `step6/tilemap`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 16px  ·  state_dim: 127
- states: `examples/results/step4/states/CraftaxClassic.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **0.82 ms** median (min 0.77, max 0.85, sd 0.03) = **1220 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 259 | 116 |
| host->device | 0 | - | 170 | 51 |
| allocation | 0 | - | 36 | 21 |
| kernel/raster | 0 | - | 541 | 20 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 41 | 7 |

216 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8 | 0 | 268 |
| `aten::index` | 5 | 0 | 197 |
| `aten::copy_` | 18 | 0 | 125 |
| `aten::abs` | 2 | 0 | 36 |
| `aten::sub` | 2 | 0 | 36 |
| `aten::view` | 15 | 0 | 34 |
| `aten::_to_copy` | 14 | 0 | 32 |
| `aten::slice` | 9 | 0 | 26 |

## batch 16

render **7.98 ms** median (min 7.70, max 8.66, sd 0.32) = **2006 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 253 | 116 |
| host->device | 0 | - | 1630 | 51 |
| allocation | 0 | - | 39 | 21 |
| kernel/raster | 0 | - | 5953 | 20 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 184 | 7 |

216 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8 | 0 | 2499 |
| `aten::index` | 5 | 0 | 2336 |
| `aten::copy_` | 18 | 0 | 1577 |
| `aten::add` | 2 | 0 | 588 |
| `aten::sub` | 2 | 0 | 385 |
| `aten::round_` | 1 | 0 | 159 |
| `aten::clamp_` | 1 | 0 | 136 |
| `aten::_to_copy` | 14 | 0 | 32 |

## batch 64

render **62.61 ms** median (min 33.80, max 69.23, sd 11.11) = **1022 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 387 | 116 |
| host->device | 0 | - | 7441 | 51 |
| allocation | 0 | - | 56 | 21 |
| kernel/raster | 0 | - | 29150 | 20 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 901 | 7 |

216 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8 | 0 | 11260 |
| `aten::index` | 5 | 0 | 8513 |
| `aten::copy_` | 18 | 0 | 7386 |
| `aten::add` | 2 | 0 | 6819 |
| `aten::sub` | 2 | 0 | 1923 |
| `aten::round_` | 1 | 0 | 864 |
| `aten::clamp_` | 1 | 0 | 623 |
| `aten::remainder` | 3 | 0 | 52 |

## batch 256

render **330.80 ms** median (min 308.49, max 381.93, sd 19.81) = **774 fps**  ·  set_state 0.13 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 655 | 116 |
| host->device | 0 | - | 87355 | 51 |
| allocation | 0 | - | 138 | 21 |
| kernel/raster | 0 | - | 278389 | 20 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 4723 | 7 |

216 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8 | 0 | 120939 |
| `aten::index` | 5 | 0 | 92130 |
| `aten::copy_` | 18 | 0 | 87264 |
| `aten::add` | 2 | 0 | 35052 |
| `aten::sub` | 2 | 0 | 25659 |
| `aten::round_` | 1 | 0 | 4653 |
| `aten::clamp_` | 1 | 0 | 4583 |
| `aten::remainder` | 3 | 0 | 102 |

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
