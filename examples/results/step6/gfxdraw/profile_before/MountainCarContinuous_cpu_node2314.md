# Profile note — MountainCarContinuous on cpu

- workstream: `step6/gfxdraw`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCarContinuous.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **47.67 ms** median (min 47.54, max 47.88, sd 0.12) = **21 fps**  ·  set_state 0.17 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 20144 | 14816 |
| host->device | 0 | - | 4226 | 2051 |
| allocation | 0 | - | 3064 | 3136 |
| kernel/raster | 0 | - | 21188 | 3182 |
| synchronization | 0 | - | 2 | 1 |

23186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 13922 |
| `aten::bitwise_and` | 1222 | 0 | 5100 |
| `aten::copy_` | 643 | 0 | 2446 |
| `aten::abs` | 24 | 0 | 2440 |
| `aten::index` | 114 | 0 | 2257 |
| `aten::add` | 745 | 0 | 1648 |
| `aten::_to_copy` | 635 | 0 | 1279 |
| `aten::empty_strided` | 988 | 0 | 1168 |

## batch 16

render **238.55 ms** median (min 236.85, max 243.52, sd 2.18) = **67 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 46236 | 14876 |
| host->device | 0 | - | 25579 | 2081 |
| allocation | 0 | - | 2659 | 3200 |
| kernel/raster | 0 | - | 155961 | 3182 |
| synchronization | 0 | - | 3 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 140486 |
| `aten::copy_` | 674 | 0 | 24045 |
| `aten::bitwise_and` | 1222 | 0 | 11602 |
| `aten::index` | 114 | 0 | 9839 |
| `aten::le` | 22 | 0 | 9461 |
| `aten::fill_` | 194 | 0 | 6799 |
| `aten::ge` | 87 | 0 | 5108 |
| `aten::flip` | 1 | 0 | 2384 |

## batch 64

render **982.56 ms** median (min 969.25, max 1004.20, sd 9.31) = **65 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 227705 | 14876 |
| host->device | 0 | - | 96109 | 2081 |
| allocation | 0 | - | 2723 | 3200 |
| kernel/raster | 0 | - | 651038 | 3182 |
| synchronization | 0 | - | 2 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 585970 |
| `aten::copy_` | 674 | 0 | 94550 |
| `aten::bitwise_and` | 1222 | 0 | 57690 |
| `aten::le` | 22 | 0 | 55274 |
| `aten::index` | 114 | 0 | 55129 |
| `aten::fill_` | 194 | 0 | 44300 |
| `aten::ge` | 87 | 0 | 25240 |
| `aten::flip` | 1 | 0 | 21237 |

## batch 256

render **4352.71 ms** median (min 4091.22, max 4362.08, sd 82.97) = **59 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 998769 | 14876 |
| host->device | 0 | - | 401386 | 2081 |
| allocation | 0 | - | 3122 | 3200 |
| kernel/raster | 0 | - | 2824381 | 3182 |
| synchronization | 0 | - | 3 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 2555584 |
| `aten::copy_` | 674 | 0 | 399686 |
| `aten::le` | 22 | 0 | 266260 |
| `aten::bitwise_and` | 1222 | 0 | 261031 |
| `aten::index` | 114 | 0 | 242256 |
| `aten::fill_` | 194 | 0 | 177846 |
| `aten::ge` | 87 | 0 | 133819 |
| `aten::flip` | 1 | 0 | 110241 |

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
