# Profile note — Piecewise on cpu

- workstream: `step6/raster2d`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/Piecewise.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **2.51 ms** median (min 2.50, max 2.53, sd 0.01) = **398 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 428 | 110 |
| host->device | 0 | - | 699 | 39 |
| allocation | 0 | - | 44 | 17 |
| kernel/raster | 0 | - | 2626 | 19 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 1303 |
| `aten::copy_` | 13 | 0 | 656 |
| `aten::where` | 2 | 0 | 605 |
| `aten::add` | 2 | 0 | 388 |
| `aten::exp` | 1 | 0 | 154 |
| `aten::sub` | 3 | 0 | 149 |
| `aten::div` | 2 | 0 | 144 |
| `aten::abs` | 2 | 0 | 33 |

## batch 16

render **33.32 ms** median (min 32.59, max 40.03, sd 3.01) = **480 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1782 | 110 |
| host->device | 0 | - | 7085 | 39 |
| allocation | 0 | - | 39 | 17 |
| kernel/raster | 0 | - | 23843 | 19 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 10129 |
| `aten::where` | 2 | 0 | 8100 |
| `aten::copy_` | 13 | 0 | 7028 |
| `aten::add` | 2 | 0 | 2672 |
| `aten::sub` | 3 | 0 | 1469 |
| `aten::exp` | 1 | 0 | 1069 |
| `aten::div` | 2 | 0 | 1044 |
| `aten::clamp` | 1 | 0 | 414 |

## batch 64

render **206.29 ms** median (min 183.40, max 221.67, sd 11.27) = **310 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 18330 | 110 |
| host->device | 0 | - | 38220 | 39 |
| allocation | 0 | - | 83 | 17 |
| kernel/raster | 0 | - | 159321 | 19 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 73229 |
| `aten::copy_` | 13 | 0 | 38086 |
| `aten::where` | 2 | 0 | 36425 |
| `aten::add` | 2 | 0 | 30033 |
| `aten::exp` | 1 | 0 | 14794 |
| `aten::div` | 2 | 0 | 12044 |
| `aten::sub` | 3 | 0 | 5742 |
| `aten::neg` | 1 | 0 | 2182 |

## batch 256

render **1204.34 ms** median (min 1193.83, max 1216.08, sd 6.29) = **213 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 86571 | 110 |
| host->device | 0 | - | 261405 | 39 |
| allocation | 0 | - | 179 | 17 |
| kernel/raster | 0 | - | 752818 | 19 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 311691 |
| `aten::copy_` | 13 | 0 | 261255 |
| `aten::where` | 2 | 0 | 163638 |
| `aten::add` | 2 | 0 | 117174 |
| `aten::sub` | 3 | 0 | 82610 |
| `aten::exp` | 1 | 0 | 59433 |
| `aten::div` | 2 | 0 | 52298 |
| `aten::clamp` | 1 | 0 | 25377 |

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
