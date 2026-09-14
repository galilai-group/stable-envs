# Profile note — MountainCar on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCar.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/baseline/stable_envs/__init__.py`

## batch 1

render **46.00 ms** median (min 38.03, max 46.19, sd 3.26) = **22 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 14116 | 14816 |
| host->device | 0 | - | 3241 | 2051 |
| allocation | 0 | - | 2376 | 3136 |
| kernel/raster | 0 | - | 17006 | 3182 |
| synchronization | 0 | - | 2 | 1 |

23186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 11742 |
| `aten::bitwise_and` | 1222 | 0 | 4061 |
| `aten::copy_` | 643 | 0 | 1876 |
| `aten::index` | 114 | 0 | 1389 |
| `aten::add` | 745 | 0 | 1298 |
| `aten::_to_copy` | 635 | 0 | 974 |
| `aten::empty_strided` | 988 | 0 | 894 |
| `aten::bitwise_not` | 400 | 0 | 815 |

## batch 16

render **233.22 ms** median (min 232.50, max 236.08, sd 0.98) = **69 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 46124 | 14876 |
| host->device | 0 | - | 24821 | 2081 |
| allocation | 0 | - | 2460 | 3200 |
| kernel/raster | 0 | - | 156150 | 3182 |
| synchronization | 0 | - | 3 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 141256 |
| `aten::copy_` | 674 | 0 | 23411 |
| `aten::bitwise_and` | 1222 | 0 | 11084 |
| `aten::index` | 114 | 0 | 9672 |
| `aten::le` | 22 | 0 | 9304 |
| `aten::fill_` | 194 | 0 | 6723 |
| `aten::ge` | 87 | 0 | 4979 |
| `aten::flip` | 1 | 0 | 3877 |

## batch 64

render **907.71 ms** median (min 903.32, max 913.55, sd 3.17) = **71 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 171034 | 14876 |
| host->device | 0 | - | 95839 | 2081 |
| allocation | 0 | - | 2551 | 3200 |
| kernel/raster | 0 | - | 631341 | 3182 |
| synchronization | 0 | - | 2 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 577210 |
| `aten::copy_` | 674 | 0 | 94370 |
| `aten::bitwise_and` | 1222 | 0 | 47269 |
| `aten::index` | 114 | 0 | 44750 |
| `aten::le` | 22 | 0 | 40290 |
| `aten::fill_` | 194 | 0 | 32911 |
| `aten::ge` | 87 | 0 | 18935 |
| `aten::flip` | 1 | 0 | 13821 |

## batch 256

render **3661.56 ms** median (min 3636.63, max 3676.36, sd 13.05) = **70 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 712954 | 14876 |
| host->device | 0 | - | 382287 | 2081 |
| allocation | 0 | - | 3347 | 3200 |
| kernel/raster | 0 | - | 2566405 | 3182 |
| synchronization | 0 | - | 3 | 1 |

23340 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1225 | 0 | 2359758 |
| `aten::copy_` | 674 | 0 | 380403 |
| `aten::le` | 22 | 0 | 193996 |
| `aten::index` | 114 | 0 | 178755 |
| `aten::bitwise_and` | 1222 | 0 | 178046 |
| `aten::fill_` | 194 | 0 | 136758 |
| `aten::ge` | 87 | 0 | 97939 |
| `aten::flip` | 1 | 0 | 54769 |

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
