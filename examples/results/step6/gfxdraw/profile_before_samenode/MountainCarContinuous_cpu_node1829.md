# Profile note — MountainCarContinuous on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCarContinuous.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/baseline/stable_envs/__init__.py`

## batch 1

render **42.25 ms** median (min 42.05, max 42.59, sd 0.14) = **24 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 14847 | 14816 |
| host->device | 0 | - | 3399 | 2051 |
| allocation | 0 | - | 2517 | 3136 |
| kernel/raster | 0 | - | 17909 | 3182 |
| synchronization | 0 | - | 2 | 1 |

23186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 12346 |
| `aten::bitwise_and` | 1222 | 0 | 4336 |
| `aten::copy_` | 643 | 0 | 1955 |
| `aten::index` | 114 | 0 | 1451 |
| `aten::add` | 745 | 0 | 1379 |
| `aten::_to_copy` | 635 | 0 | 1026 |
| `aten::empty_strided` | 988 | 0 | 930 |
| `aten::bitwise_not` | 400 | 0 | 841 |

## batch 16

render **246.83 ms** median (min 234.68, max 249.96, sd 5.78) = **65 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 46072 | 14876 |
| host->device | 0 | - | 25784 | 2081 |
| allocation | 0 | - | 2681 | 3200 |
| kernel/raster | 0 | - | 156215 | 3182 |
| synchronization | 0 | - | 19 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 140710 |
| `aten::copy_` | 674 | 0 | 24283 |
| `aten::bitwise_and` | 1222 | 0 | 11355 |
| `aten::index` | 114 | 0 | 9850 |
| `aten::le` | 22 | 0 | 9471 |
| `aten::fill_` | 194 | 0 | 6809 |
| `aten::ge` | 87 | 0 | 5141 |
| `aten::flip` | 1 | 0 | 2325 |

## batch 64

render **943.65 ms** median (min 921.24, max 949.14, sd 9.17) = **68 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 177328 | 14876 |
| host->device | 0 | - | 97289 | 2081 |
| allocation | 0 | - | 2762 | 3200 |
| kernel/raster | 0 | - | 640822 | 3182 |
| synchronization | 0 | - | 2 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 585242 |
| `aten::copy_` | 674 | 0 | 95732 |
| `aten::bitwise_and` | 1222 | 0 | 47801 |
| `aten::index` | 114 | 0 | 45655 |
| `aten::le` | 22 | 0 | 40694 |
| `aten::fill_` | 194 | 0 | 33702 |
| `aten::ge` | 87 | 0 | 22456 |
| `aten::flip` | 1 | 0 | 13876 |

## batch 256

render **3684.11 ms** median (min 3604.30, max 3706.47, sd 33.38) = **69 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 707919 | 14876 |
| host->device | 0 | - | 390536 | 2081 |
| allocation | 0 | - | 3316 | 3200 |
| kernel/raster | 0 | - | 2525158 | 3182 |
| synchronization | 0 | - | 3 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 2316091 |
| `aten::copy_` | 674 | 0 | 388841 |
| `aten::le` | 22 | 0 | 189095 |
| `aten::index` | 114 | 0 | 181053 |
| `aten::bitwise_and` | 1222 | 0 | 177688 |
| `aten::fill_` | 194 | 0 | 139715 |
| `aten::ge` | 87 | 0 | 96224 |
| `aten::flip` | 1 | 0 | 53832 |

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
