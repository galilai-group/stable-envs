# Profile note — PointmazeLargeNavigate on cpu

- workstream: `step6/raycast3d`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/PointmazeLargeNavigate.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **3.90 ms** median (min 3.83, max 4.04, sd 0.07) = **256 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1175 | 803 |
| host->device | 0 | - | 788 | 151 |
| allocation | 0 | - | 135 | 90 |
| kernel/raster | 0 | - | 1253 | 177 |
| synchronization | 0 | - | 7 | 6 |
| output conversion | 0 | - | 5 | 1 |

1228 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 45 | 0 | 684 |
| `aten::mul` | 62 | 0 | 326 |
| `aten::sum` | 4 | 0 | 289 |
| `aten::index` | 8 | 0 | 275 |
| `aten::add` | 43 | 0 | 213 |
| `aten::where` | 14 | 0 | 142 |
| `aten::select` | 90 | 0 | 136 |
| `aten::as_strided` | 160 | 0 | 80 |

## batch 16

render **19.30 ms** median (min 19.03, max 20.20, sd 0.32) = **829 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 5576 | 823 |
| host->device | 0 | - | 878 | 152 |
| allocation | 0 | - | 134 | 93 |
| kernel/raster | 0 | - | 11727 | 177 |
| synchronization | 0 | - | 7 | 6 |
| output conversion | 0 | - | 29 | 2 |

1253 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 62 | 0 | 3265 |
| `aten::sum` | 4 | 0 | 3018 |
| `aten::index` | 8 | 0 | 3005 |
| `aten::add` | 43 | 0 | 2251 |
| `aten::where` | 14 | 0 | 1424 |
| `aten::copy_` | 46 | 0 | 747 |
| `aten::div` | 9 | 0 | 714 |
| `aten::pow` | 2 | 0 | 584 |

## batch 64

render **72.18 ms** median (min 70.31, max 86.57, sd 5.59) = **887 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 20084 | 823 |
| host->device | 0 | - | 1949 | 152 |
| allocation | 0 | - | 197 | 93 |
| kernel/raster | 0 | - | 53100 | 177 |
| synchronization | 0 | - | 7 | 6 |
| output conversion | 0 | - | 73 | 2 |

1253 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 62 | 0 | 15478 |
| `aten::add` | 43 | 0 | 12496 |
| `aten::sum` | 4 | 0 | 12082 |
| `aten::index` | 8 | 0 | 11320 |
| `aten::where` | 14 | 0 | 5330 |
| `aten::div` | 9 | 0 | 3912 |
| `aten::pow` | 2 | 0 | 2288 |
| `aten::sub` | 14 | 0 | 1924 |

## batch 256

render **327.77 ms** median (min 315.86, max 367.00, sd 15.59) = **781 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 78473 | 823 |
| host->device | 0 | - | 5600 | 152 |
| allocation | 0 | - | 219 | 93 |
| kernel/raster | 0 | - | 233738 | 177 |
| synchronization | 0 | - | 7 | 6 |
| output conversion | 0 | - | 459 | 2 |

1253 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 62 | 0 | 71515 |
| `aten::add` | 43 | 0 | 58646 |
| `aten::sum` | 4 | 0 | 47915 |
| `aten::index` | 8 | 0 | 44347 |
| `aten::where` | 14 | 0 | 21437 |
| `aten::div` | 9 | 0 | 19256 |
| `aten::pow` | 2 | 0 | 8186 |
| `aten::sub` | 14 | 0 | 7736 |

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
