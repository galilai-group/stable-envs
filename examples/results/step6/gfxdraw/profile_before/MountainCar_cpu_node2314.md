# Profile note — MountainCar on cpu

- workstream: `step6/gfxdraw`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCar.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **47.11 ms** median (min 46.14, max 47.63, sd 0.38) = **21 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 14304 | 14816 |
| host->device | 0 | - | 3313 | 2051 |
| allocation | 0 | - | 2347 | 3136 |
| kernel/raster | 0 | - | 17314 | 3182 |
| synchronization | 0 | - | 5 | 1 |

23186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 12020 |
| `aten::bitwise_and` | 1222 | 0 | 4256 |
| `aten::copy_` | 643 | 0 | 1941 |
| `aten::index` | 114 | 0 | 1424 |
| `aten::add` | 745 | 0 | 1262 |
| `aten::_to_copy` | 635 | 0 | 967 |
| `aten::empty_strided` | 988 | 0 | 873 |
| `aten::bitwise_not` | 400 | 0 | 828 |

## batch 16

render **236.43 ms** median (min 232.79, max 238.92, sd 1.91) = **68 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 45917 | 14876 |
| host->device | 0 | - | 25331 | 2081 |
| allocation | 0 | - | 2649 | 3200 |
| kernel/raster | 0 | - | 155665 | 3182 |
| synchronization | 0 | - | 3 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 140395 |
| `aten::copy_` | 674 | 0 | 23830 |
| `aten::bitwise_and` | 1222 | 0 | 11597 |
| `aten::index` | 114 | 0 | 9670 |
| `aten::le` | 22 | 0 | 9347 |
| `aten::fill_` | 194 | 0 | 6749 |
| `aten::ge` | 87 | 0 | 5151 |
| `aten::flip` | 1 | 0 | 2208 |

## batch 64

render **955.61 ms** median (min 909.33, max 975.76, sd 25.93) = **67 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 184013 | 14876 |
| host->device | 0 | - | 98544 | 2081 |
| allocation | 0 | - | 2758 | 3200 |
| kernel/raster | 0 | - | 650778 | 3182 |
| synchronization | 0 | - | 3 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 603072 |
| `aten::copy_` | 674 | 0 | 96976 |
| `aten::bitwise_and` | 1222 | 0 | 48010 |
| `aten::le` | 22 | 0 | 46137 |
| `aten::index` | 114 | 0 | 36952 |
| `aten::fill_` | 194 | 0 | 27190 |
| `aten::flip` | 1 | 0 | 21961 |
| `aten::ge` | 87 | 0 | 21795 |

## batch 256

render **4089.70 ms** median (min 3970.24, max 4123.21, sd 54.84) = **63 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 902271 | 14876 |
| host->device | 0 | - | 379710 | 2081 |
| allocation | 0 | - | 3102 | 3200 |
| kernel/raster | 0 | - | 2704998 | 3182 |
| synchronization | 0 | - | 3 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 2462332 |
| `aten::copy_` | 674 | 0 | 378031 |
| `aten::le` | 22 | 0 | 243257 |
| `aten::bitwise_and` | 1222 | 0 | 222541 |
| `aten::index` | 114 | 0 | 215653 |
| `aten::fill_` | 194 | 0 | 178139 |
| `aten::ge` | 87 | 0 | 122758 |
| `aten::flip` | 1 | 0 | 86079 |

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
