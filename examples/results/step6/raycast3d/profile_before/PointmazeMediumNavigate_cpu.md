# Profile note — PointmazeMediumNavigate on cpu

- workstream: `step6/raycast3d`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/PointmazeMediumNavigate.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **3.76 ms** median (min 3.75, max 3.77, sd 0.01) = **266 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1165 | 803 |
| host->device | 0 | - | 700 | 151 |
| allocation | 0 | - | 136 | 90 |
| kernel/raster | 0 | - | 1263 | 177 |
| synchronization | 0 | - | 7 | 6 |
| output conversion | 0 | - | 5 | 1 |

1228 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 45 | 0 | 595 |
| `aten::mul` | 62 | 0 | 335 |
| `aten::sum` | 4 | 0 | 297 |
| `aten::index` | 8 | 0 | 257 |
| `aten::add` | 43 | 0 | 221 |
| `aten::where` | 14 | 0 | 146 |
| `aten::select` | 90 | 0 | 126 |
| `aten::div` | 9 | 0 | 91 |

## batch 16

render **17.90 ms** median (min 17.46, max 18.87, sd 0.38) = **894 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 5205 | 823 |
| host->device | 0 | - | 725 | 152 |
| allocation | 0 | - | 120 | 93 |
| kernel/raster | 0 | - | 10685 | 177 |
| synchronization | 0 | - | 6 | 6 |
| output conversion | 0 | - | 15 | 2 |

1253 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 62 | 0 | 3056 |
| `aten::sum` | 4 | 0 | 2996 |
| `aten::index` | 8 | 0 | 2788 |
| `aten::add` | 43 | 0 | 1898 |
| `aten::where` | 14 | 0 | 1433 |
| `aten::pow` | 2 | 0 | 628 |
| `aten::div` | 9 | 0 | 623 |
| `aten::copy_` | 46 | 0 | 612 |

## batch 64

render **70.84 ms** median (min 65.79, max 80.40, sd 4.75) = **903 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 19176 | 823 |
| host->device | 0 | - | 1744 | 152 |
| allocation | 0 | - | 159 | 93 |
| kernel/raster | 0 | - | 44992 | 177 |
| synchronization | 0 | - | 7 | 6 |
| output conversion | 0 | - | 102 | 2 |

1253 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 62 | 0 | 12849 |
| `aten::sum` | 4 | 0 | 11948 |
| `aten::index` | 8 | 0 | 10956 |
| `aten::add` | 43 | 0 | 8986 |
| `aten::where` | 14 | 0 | 5313 |
| `aten::div` | 9 | 0 | 2649 |
| `aten::pow` | 2 | 0 | 2160 |
| `aten::copy_` | 46 | 0 | 1618 |

## batch 256

render **287.24 ms** median (min 267.77, max 327.92, sd 17.27) = **891 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 73876 | 823 |
| host->device | 0 | - | 5431 | 152 |
| allocation | 0 | - | 182 | 93 |
| kernel/raster | 0 | - | 193213 | 177 |
| synchronization | 0 | - | 8 | 6 |
| output conversion | 0 | - | 411 | 2 |

1253 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 62 | 0 | 61105 |
| `aten::sum` | 4 | 0 | 47658 |
| `aten::index` | 8 | 0 | 43746 |
| `aten::add` | 43 | 0 | 39799 |
| `aten::where` | 14 | 0 | 21697 |
| `aten::div` | 9 | 0 | 9603 |
| `aten::pow` | 2 | 0 | 7612 |
| `aten::clamp` | 12 | 0 | 7144 |

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
