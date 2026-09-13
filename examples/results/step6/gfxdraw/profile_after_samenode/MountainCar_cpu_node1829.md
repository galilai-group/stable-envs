# Profile note — MountainCar on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCar.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `/oscar/scratch/sboughan/se6/gfxdraw/stable_envs/__init__.py`

## batch 1

render **17.13 ms** median (min 16.93, max 17.57, sd 0.17) = **58 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 8539 | 8166 |
| host->device | 0 | - | 1570 | 968 |
| allocation | 0 | - | 1002 | 1223 |
| kernel/raster | 0 | - | 4931 | 1569 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 6 | 1 |

11928 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 621 | 0 | 1794 |
| `aten::where` | 465 | 0 | 1323 |
| `aten::copy_` | 282 | 0 | 853 |
| `aten::mul` | 228 | 0 | 846 |
| `aten::add` | 358 | 0 | 641 |
| `aten::index` | 111 | 0 | 603 |
| `aten::__rshift__` | 133 | 0 | 551 |
| `aten::_to_copy` | 275 | 0 | 525 |

## batch 16

render **46.03 ms** median (min 42.26, max 49.44, sd 1.88) = **348 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 26446 | 8253 |
| host->device | 0 | - | 11121 | 1008 |
| allocation | 0 | - | 1391 | 1307 |
| kernel/raster | 0 | - | 11054 | 1569 |
| synchronization | 0 | - | 4 | 1 |
| output conversion | 0 | - | 6 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 10321 |
| `aten::mul` | 228 | 0 | 4633 |
| `aten::le` | 18 | 0 | 3532 |
| `aten::bitwise_xor` | 2 | 0 | 3459 |
| `aten::ge` | 86 | 0 | 3416 |
| `aten::bitwise_and` | 621 | 0 | 3367 |
| `aten::flip` | 3 | 0 | 3052 |
| `aten::where` | 465 | 0 | 1692 |

## batch 64

render **157.54 ms** median (min 154.04, max 158.81, sd 1.60) = **406 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 80734 | 8253 |
| host->device | 0 | - | 40731 | 1008 |
| allocation | 0 | - | 1557 | 1307 |
| kernel/raster | 0 | - | 30537 | 1569 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 7 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 39825 |
| `aten::mul` | 228 | 0 | 17523 |
| `aten::bitwise_xor` | 2 | 0 | 12822 |
| `aten::flip` | 3 | 0 | 12400 |
| `aten::le` | 18 | 0 | 11247 |
| `aten::fill_` | 58 | 0 | 9926 |
| `aten::ge` | 86 | 0 | 9197 |
| `aten::bitwise_and` | 621 | 0 | 7938 |

## batch 256

render **591.76 ms** median (min 585.44, max 600.07, sd 4.61) = **433 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 306920 | 8253 |
| host->device | 0 | - | 164980 | 1008 |
| allocation | 0 | - | 1848 | 1307 |
| kernel/raster | 0 | - | 105188 | 1569 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 7 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 163988 |
| `aten::mul` | 228 | 0 | 66468 |
| `aten::flip` | 3 | 0 | 52697 |
| `aten::bitwise_xor` | 2 | 0 | 49491 |
| `aten::ge` | 86 | 0 | 43878 |
| `aten::le` | 18 | 0 | 43602 |
| `aten::fill_` | 58 | 0 | 39131 |
| `aten::bitwise_xor_` | 2 | 0 | 26494 |

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
