# Profile note — PiecewiseZones on cpu

- workstream: `step6/raster2d`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/PiecewiseZones.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **4.81 ms** median (min 4.78, max 5.00, sd 0.06) = **208 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 656 | 170 |
| host->device | 0 | - | 789 | 51 |
| allocation | 0 | - | 60 | 25 |
| kernel/raster | 0 | - | 4834 | 23 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 2740 |
| `aten::mul` | 7 | 0 | 1392 |
| `aten::copy_` | 17 | 0 | 733 |
| `aten::add` | 2 | 0 | 420 |
| `aten::eq` | 5 | 0 | 189 |
| `aten::sub` | 3 | 0 | 139 |
| `aten::exp` | 1 | 0 | 120 |
| `aten::div` | 2 | 0 | 111 |

## batch 16

render **64.11 ms** median (min 61.99, max 69.41, sd 2.96) = **250 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 2173 | 170 |
| host->device | 0 | - | 7262 | 51 |
| allocation | 0 | - | 61 | 25 |
| kernel/raster | 0 | - | 53747 | 23 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 37525 |
| `aten::mul` | 7 | 0 | 10321 |
| `aten::copy_` | 17 | 0 | 7188 |
| `aten::add` | 2 | 0 | 2809 |
| `aten::sub` | 3 | 0 | 1578 |
| `aten::exp` | 1 | 0 | 1159 |
| `aten::div` | 2 | 0 | 1075 |
| `aten::clamp` | 1 | 0 | 423 |

## batch 64

render **308.90 ms** median (min 299.23, max 336.87, sd 11.88) = **207 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 8213 | 170 |
| host->device | 0 | - | 51838 | 51 |
| allocation | 0 | - | 136 | 25 |
| kernel/raster | 0 | - | 257344 | 23 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 149114 |
| `aten::mul` | 7 | 0 | 70036 |
| `aten::copy_` | 17 | 0 | 51701 |
| `aten::add` | 2 | 0 | 25477 |
| `aten::sub` | 3 | 0 | 6251 |
| `aten::div` | 2 | 0 | 4523 |
| `aten::exp` | 1 | 0 | 4493 |
| `aten::neg` | 1 | 0 | 1981 |

## batch 256

render **1603.68 ms** median (min 1550.58, max 1623.37, sd 26.04) = **160 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 83978 | 170 |
| host->device | 0 | - | 219821 | 51 |
| allocation | 0 | - | 159 | 25 |
| kernel/raster | 0 | - | 1213183 | 23 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 635927 |
| `aten::mul` | 7 | 0 | 305455 |
| `aten::copy_` | 17 | 0 | 219654 |
| `aten::add` | 2 | 0 | 115729 |
| `aten::sub` | 3 | 0 | 80624 |
| `aten::exp` | 1 | 0 | 57037 |
| `aten::div` | 2 | 0 | 51132 |
| `aten::clamp` | 1 | 0 | 24283 |

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
