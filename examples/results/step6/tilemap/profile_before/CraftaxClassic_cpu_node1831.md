# Profile note — CraftaxClassic on cpu

- workstream: `step6/tilemap`
- baseline commit: ``
- host: node1831.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- tree measured: `/oscar/scratch/sboughan/se6/baseline/stable_envs/__init__.py`  ·  load average 38.2
- frame: 16px  ·  state_dim: 127
- states: `examples/results/step4/states/CraftaxClassic.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **0.80 ms** median (min 0.78, max 0.84, sd 0.01) = **1244 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 4383 | 116 |
| host->device | 0 | - | 217 | 51 |
| allocation | 0 | - | 60 | 21 |
| kernel/raster | 0 | - | 595 | 20 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 38 | 7 |

216 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::clone` | 3 | 0 | 2118 |
| `aten::abs` | 2 | 0 | 1980 |
| `aten::index` | 5 | 0 | 218 |
| `aten::mul` | 8 | 0 | 184 |
| `aten::copy_` | 18 | 0 | 159 |
| `aten::add` | 2 | 0 | 110 |
| `aten::slice` | 9 | 0 | 45 |
| `aten::view` | 15 | 0 | 42 |

## batch 16

render **8.46 ms** median (min 8.03, max 9.46, sd 0.33) = **1892 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 308 | 116 |
| host->device | 0 | - | 2507 | 51 |
| allocation | 0 | - | 59 | 21 |
| kernel/raster | 0 | - | 8053 | 20 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 191 | 7 |

216 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8 | 0 | 3663 |
| `aten::index` | 5 | 0 | 2489 |
| `aten::copy_` | 18 | 0 | 2444 |
| `aten::add` | 2 | 0 | 1313 |
| `aten::sub` | 2 | 0 | 431 |
| `aten::round_` | 1 | 0 | 153 |
| `aten::clamp_` | 1 | 0 | 144 |
| `aten::_to_copy` | 14 | 0 | 46 |

## batch 64

render **43.75 ms** median (min 36.61, max 48.48, sd 3.63) = **1463 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 432 | 116 |
| host->device | 0 | - | 8212 | 51 |
| allocation | 0 | - | 65 | 21 |
| kernel/raster | 0 | - | 26172 | 20 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 1098 | 7 |

216 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8 | 0 | 11481 |
| `aten::index` | 5 | 0 | 8865 |
| `aten::copy_` | 18 | 0 | 8146 |
| `aten::add` | 2 | 0 | 2856 |
| `aten::sub` | 2 | 0 | 2008 |
| `aten::round_` | 1 | 0 | 1056 |
| `aten::clamp_` | 1 | 0 | 948 |
| `aten::remainder` | 3 | 0 | 55 |

## batch 256

render **234.14 ms** median (min 232.78, max 239.21, sd 1.71) = **1093 fps**  ·  set_state 0.13 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 797 | 116 |
| host->device | 0 | - | 52313 | 51 |
| allocation | 0 | - | 155 | 21 |
| kernel/raster | 0 | - | 173225 | 20 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 4652 | 7 |

216 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8 | 0 | 75128 |
| `aten::index` | 5 | 0 | 59863 |
| `aten::copy_` | 18 | 0 | 52214 |
| `aten::add` | 2 | 0 | 18574 |
| `aten::sub` | 2 | 0 | 15094 |
| `aten::round_` | 1 | 0 | 4565 |
| `aten::clamp_` | 1 | 0 | 4525 |
| `aten::remainder` | 3 | 0 | 113 |

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
