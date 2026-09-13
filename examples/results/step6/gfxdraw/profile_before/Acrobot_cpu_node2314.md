# Profile note — Acrobot on cpu

- workstream: `step6/gfxdraw`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Acrobot.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **49.77 ms** median (min 49.01, max 50.15, sd 0.33) = **20 fps**  ·  set_state 0.14 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 16696 | 16797 |
| host->device | 0 | - | 3838 | 2312 |
| allocation | 0 | - | 2701 | 3575 |
| kernel/raster | 0 | - | 18632 | 3620 |
| synchronization | 0 | - | 3 | 1 |

26305 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1396 | 0 | 13088 |
| `aten::bitwise_and` | 1392 | 0 | 4834 |
| `aten::copy_` | 725 | 0 | 1998 |
| `aten::add` | 855 | 0 | 1569 |
| `aten::_to_copy` | 719 | 0 | 1222 |
| `aten::empty_strided` | 1124 | 0 | 1073 |
| `aten::__and__` | 1392 | 0 | 984 |
| `aten::bitwise_not` | 457 | 0 | 970 |

## batch 16

render **249.68 ms** median (min 247.13, max 251.08, sd 1.56) = **64 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 56339 | 16861 |
| host->device | 0 | - | 25184 | 2348 |
| allocation | 0 | - | 2995 | 3647 |
| kernel/raster | 0 | - | 160885 | 3620 |
| synchronization | 0 | - | 3 | 1 |

26477 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1396 | 0 | 149406 |
| `aten::copy_` | 761 | 0 | 23431 |
| `aten::bitwise_and` | 1392 | 0 | 14938 |
| `aten::le` | 24 | 0 | 11052 |
| `aten::fill_` | 220 | 0 | 9146 |
| `aten::ge` | 98 | 0 | 5462 |
| `aten::index` | 128 | 0 | 3250 |
| `aten::amin` | 2 | 0 | 2427 |

## batch 64

render **975.95 ms** median (min 952.37, max 988.66, sd 11.46) = **66 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 207772 | 16861 |
| host->device | 0 | - | 96227 | 2348 |
| allocation | 0 | - | 3231 | 3647 |
| kernel/raster | 0 | - | 663915 | 3620 |
| synchronization | 0 | - | 2 | 1 |

26477 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1396 | 0 | 634434 |
| `aten::copy_` | 761 | 0 | 94343 |
| `aten::bitwise_and` | 1392 | 0 | 52326 |
| `aten::fill_` | 220 | 0 | 46974 |
| `aten::le` | 24 | 0 | 39246 |
| `aten::flip` | 1 | 0 | 22751 |
| `aten::ge` | 98 | 0 | 20413 |
| `aten::index` | 128 | 0 | 11077 |

## batch 256

render **4105.84 ms** median (min 4082.12, max 4128.78, sd 15.05) = **62 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 959425 | 16861 |
| host->device | 0 | - | 369239 | 2348 |
| allocation | 0 | - | 3642 | 3647 |
| kernel/raster | 0 | - | 2679562 | 3620 |
| synchronization | 0 | - | 21 | 1 |

26477 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1396 | 0 | 2579828 |
| `aten::copy_` | 761 | 0 | 367310 |
| `aten::le` | 24 | 0 | 253876 |
| `aten::bitwise_and` | 1392 | 0 | 238896 |
| `aten::fill_` | 220 | 0 | 171249 |
| `aten::ge` | 98 | 0 | 130882 |
| `aten::flip` | 1 | 0 | 90458 |
| `aten::index` | 128 | 0 | 39376 |

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
