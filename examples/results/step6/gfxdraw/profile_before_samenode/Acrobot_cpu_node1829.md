# Profile note — Acrobot on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Acrobot.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/baseline/stable_envs/__init__.py`

## batch 1

render **44.37 ms** median (min 44.33, max 44.83, sd 0.15) = **23 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 17120 | 16797 |
| host->device | 0 | - | 3926 | 2312 |
| allocation | 0 | - | 2764 | 3575 |
| kernel/raster | 0 | - | 18874 | 3620 |
| synchronization | 0 | - | 2 | 1 |

26305 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1396 | 0 | 13251 |
| `aten::bitwise_and` | 1392 | 0 | 4985 |
| `aten::copy_` | 725 | 0 | 2100 |
| `aten::add` | 855 | 0 | 1639 |
| `aten::_to_copy` | 719 | 0 | 1226 |
| `aten::empty_strided` | 1124 | 0 | 1109 |
| `aten::__and__` | 1392 | 0 | 1034 |
| `aten::bitwise_not` | 457 | 0 | 970 |

## batch 16

render **246.70 ms** median (min 244.14, max 249.86, sd 1.97) = **65 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 52210 | 16861 |
| host->device | 0 | - | 25749 | 2348 |
| allocation | 0 | - | 3051 | 3647 |
| kernel/raster | 0 | - | 161654 | 3620 |
| synchronization | 0 | - | 3 | 1 |

26477 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1396 | 0 | 149933 |
| `aten::copy_` | 761 | 0 | 23952 |
| `aten::bitwise_and` | 1392 | 0 | 13343 |
| `aten::le` | 24 | 0 | 10029 |
| `aten::fill_` | 220 | 0 | 7097 |
| `aten::ge` | 98 | 0 | 5519 |
| `aten::index` | 128 | 0 | 3397 |
| `aten::flip` | 1 | 0 | 2500 |

## batch 64

render **923.33 ms** median (min 918.09, max 933.30, sd 5.29) = **69 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 191292 | 16861 |
| host->device | 0 | - | 96528 | 2348 |
| allocation | 0 | - | 3420 | 3647 |
| kernel/raster | 0 | - | 646327 | 3620 |
| synchronization | 0 | - | 2 | 1 |

26477 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1396 | 0 | 615896 |
| `aten::copy_` | 761 | 0 | 94601 |
| `aten::bitwise_and` | 1392 | 0 | 50132 |
| `aten::le` | 24 | 0 | 43154 |
| `aten::fill_` | 220 | 0 | 36213 |
| `aten::ge` | 98 | 0 | 20737 |
| `aten::flip` | 1 | 0 | 14284 |
| `aten::index` | 128 | 0 | 10629 |

## batch 256

render **3692.02 ms** median (min 3659.84, max 3707.98, sd 15.11) = **69 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 763745 | 16861 |
| host->device | 0 | - | 369971 | 2348 |
| allocation | 0 | - | 3917 | 3647 |
| kernel/raster | 0 | - | 2510205 | 3620 |
| synchronization | 0 | - | 4 | 1 |

26477 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1396 | 0 | 2409332 |
| `aten::copy_` | 761 | 0 | 367891 |
| `aten::le` | 24 | 0 | 197493 |
| `aten::bitwise_and` | 1392 | 0 | 191853 |
| `aten::fill_` | 220 | 0 | 141897 |
| `aten::ge` | 98 | 0 | 101292 |
| `aten::flip` | 1 | 0 | 55907 |
| `aten::index` | 128 | 0 | 39704 |

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
