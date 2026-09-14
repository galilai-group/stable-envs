# Profile note — CraftaxClassic on cpu

- workstream: `step6/tilemap`
- baseline commit: ``
- host: node1831.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- tree measured: `/oscar/scratch/sboughan/se6/tilemap/stable_envs/__init__.py`  ·  load average 37.8
- frame: 16px  ·  state_dim: 127
- states: `examples/results/step4/states/CraftaxClassic.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **0.32 ms** median (min 0.31, max 0.35, sd 0.01) = **3078 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 293 | 106 |
| host->device | 0 | - | 66 | 40 |
| allocation | 0 | - | 46 | 25 |
| kernel/raster | 0 | - | 73 | 10 |
| synchronization | 0 | - | 4 | 2 |
| output conversion | 0 | - | 12 | 3 |

186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::abs` | 2 | 0 | 45 |
| `aten::view` | 12 | 0 | 34 |
| `aten::remainder` | 3 | 0 | 32 |
| `aten::_to_copy` | 12 | 0 | 30 |
| `aten::copy_` | 13 | 0 | 25 |
| `aten::index_select` | 1 | 0 | 23 |
| `aten::empty_strided` | 14 | 0 | 21 |
| `aten::ne` | 1 | 0 | 20 |

## batch 16

render **0.47 ms** median (min 0.46, max 0.48, sd 0.00) = **34031 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 189 | 106 |
| host->device | 0 | - | 49 | 40 |
| allocation | 0 | - | 38 | 25 |
| kernel/raster | 0 | - | 164 | 10 |
| synchronization | 0 | - | 3 | 2 |
| output conversion | 0 | - | 11 | 3 |

186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::index_select` | 1 | 0 | 105 |
| `aten::remainder` | 3 | 0 | 30 |
| `aten::add` | 3 | 0 | 30 |
| `aten::view` | 10 | 0 | 22 |
| `aten::_to_copy` | 12 | 0 | 21 |
| `aten::copy_` | 13 | 0 | 20 |
| `aten::empty_strided` | 14 | 0 | 18 |
| `aten::abs` | 2 | 0 | 16 |

## batch 64

render **0.80 ms** median (min 0.79, max 1.47, sd 0.17) = **79653 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 233 | 106 |
| host->device | 0 | - | 54 | 40 |
| allocation | 0 | - | 34 | 25 |
| kernel/raster | 0 | - | 397 | 10 |
| synchronization | 0 | - | 3 | 2 |
| output conversion | 0 | - | 13 | 3 |

186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::index_select` | 1 | 0 | 267 |
| `aten::add` | 3 | 0 | 92 |
| `aten::remainder` | 3 | 0 | 50 |
| `aten::copy_` | 13 | 0 | 24 |
| `aten::view` | 10 | 0 | 22 |
| `aten::mul` | 4 | 0 | 22 |
| `aten::_to_copy` | 12 | 0 | 21 |
| `aten::slice` | 5 | 0 | 20 |

## batch 256

render **2.43 ms** median (min 2.32, max 2.85, sd 0.14) = **105297 fps**  ·  set_state 0.13 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 387 | 106 |
| host->device | 0 | - | 83 | 40 |
| allocation | 0 | - | 38 | 25 |
| kernel/raster | 0 | - | 1775 | 10 |
| synchronization | 0 | - | 3 | 2 |
| output conversion | 0 | - | 35 | 3 |

186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::index_select` | 1 | 0 | 1344 |
| `aten::add` | 3 | 0 | 351 |
| `aten::remainder` | 3 | 0 | 115 |
| `aten::mul` | 4 | 0 | 50 |
| `aten::copy_` | 13 | 0 | 50 |
| `aten::eq` | 2 | 0 | 38 |
| `aten::round` | 3 | 0 | 35 |
| `aten::ne` | 1 | 0 | 34 |

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
