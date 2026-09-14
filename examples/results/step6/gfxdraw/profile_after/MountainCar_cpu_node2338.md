# Profile note — MountainCar on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node2338.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCar.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **18.84 ms** median (min 17.92, max 19.79, sd 0.59) = **53 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 9104 | 8166 |
| host->device | 0 | - | 1642 | 968 |
| allocation | 0 | - | 1047 | 1223 |
| kernel/raster | 0 | - | 5209 | 1569 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 7 | 1 |

11928 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 621 | 0 | 1953 |
| `aten::where` | 465 | 0 | 1379 |
| `aten::mul` | 228 | 0 | 903 |
| `aten::copy_` | 282 | 0 | 880 |
| `aten::add` | 358 | 0 | 718 |
| `aten::index` | 111 | 0 | 634 |
| `aten::__rshift__` | 133 | 0 | 559 |
| `aten::_to_copy` | 275 | 0 | 548 |

## batch 16

render **58.06 ms** median (min 56.38, max 58.48, sd 0.63) = **276 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 24133 | 8253 |
| host->device | 0 | - | 13591 | 1008 |
| allocation | 0 | - | 1282 | 1307 |
| kernel/raster | 0 | - | 12840 | 1569 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 5 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 12830 |
| `aten::mul` | 228 | 0 | 6456 |
| `aten::bitwise_xor` | 2 | 0 | 3829 |
| `aten::flip` | 3 | 0 | 3014 |
| `aten::bitwise_and` | 621 | 0 | 2834 |
| `aten::ge` | 86 | 0 | 2516 |
| `aten::le` | 18 | 0 | 2167 |
| `aten::where` | 465 | 0 | 1664 |

## batch 64

render **216.54 ms** median (min 215.53, max 219.28, sd 1.16) = **296 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 109326 | 8253 |
| host->device | 0 | - | 54868 | 1008 |
| allocation | 0 | - | 1460 | 1307 |
| kernel/raster | 0 | - | 39477 | 1569 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 6 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 54011 |
| `aten::mul` | 228 | 0 | 26049 |
| `aten::bitwise_xor` | 2 | 0 | 20252 |
| `aten::fill_` | 58 | 0 | 19411 |
| `aten::flip` | 3 | 0 | 18398 |
| `aten::le` | 18 | 0 | 13791 |
| `aten::bitwise_and` | 621 | 0 | 10542 |
| `aten::ge` | 86 | 0 | 9036 |

## batch 256

render **900.15 ms** median (min 889.86, max 931.28, sd 12.32) = **284 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 452882 | 8253 |
| host->device | 0 | - | 216869 | 1008 |
| allocation | 0 | - | 1909 | 1307 |
| kernel/raster | 0 | - | 143479 | 1569 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 7 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 215753 |
| `aten::mul` | 228 | 0 | 101170 |
| `aten::flip` | 3 | 0 | 87992 |
| `aten::fill_` | 58 | 0 | 80640 |
| `aten::bitwise_xor` | 2 | 0 | 79911 |
| `aten::ge` | 86 | 0 | 58525 |
| `aten::le` | 18 | 0 | 57595 |
| `aten::bitwise_and` | 621 | 0 | 34814 |

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
