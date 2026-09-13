# Profile note — CartPole on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 2
- states: `examples/results/step4/states/CartPole.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `/oscar/scratch/sboughan/se6/gfxdraw/stable_envs/__init__.py`

## batch 1

render **16.19 ms** median (min 15.86, max 16.91, sd 0.28) = **62 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 8334 | 8519 |
| host->device | 0 | - | 1449 | 986 |
| allocation | 0 | - | 984 | 1266 |
| kernel/raster | 0 | - | 4504 | 1632 |
| synchronization | 0 | - | 2 | 1 |

12404 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 659 | 0 | 1921 |
| `aten::where` | 479 | 0 | 1266 |
| `aten::mul` | 238 | 0 | 758 |
| `aten::add` | 372 | 0 | 709 |
| `aten::copy_` | 285 | 0 | 700 |
| `aten::index` | 123 | 0 | 671 |
| `aten::__rshift__` | 147 | 0 | 536 |
| `aten::_to_copy` | 280 | 0 | 535 |

## batch 16

render **41.29 ms** median (min 38.85, max 46.96, sd 2.18) = **387 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 19040 | 8605 |
| host->device | 0 | - | 7727 | 1029 |
| allocation | 0 | - | 1338 | 1352 |
| kernel/raster | 0 | - | 9447 | 1632 |
| synchronization | 0 | - | 3 | 1 |

12619 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 328 | 0 | 6979 |
| `aten::bitwise_and` | 659 | 0 | 2705 |
| `aten::mul` | 238 | 0 | 2366 |
| `aten::flip` | 3 | 0 | 2253 |
| `aten::index` | 123 | 0 | 2217 |
| `aten::ge` | 92 | 0 | 1991 |
| `aten::le` | 20 | 0 | 1694 |
| `aten::where` | 479 | 0 | 1683 |

## batch 64

render **119.94 ms** median (min 113.64, max 131.42, sd 5.49) = **534 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 52844 | 8605 |
| host->device | 0 | - | 28728 | 1029 |
| allocation | 0 | - | 1452 | 1352 |
| kernel/raster | 0 | - | 27178 | 1632 |
| synchronization | 0 | - | 3 | 1 |

12619 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 328 | 0 | 27905 |
| `aten::mul` | 238 | 0 | 11002 |
| `aten::flip` | 3 | 0 | 8996 |
| `aten::bitwise_xor` | 2 | 0 | 6923 |
| `aten::ge` | 92 | 0 | 6796 |
| `aten::index` | 123 | 0 | 6571 |
| `aten::le` | 20 | 0 | 6224 |
| `aten::bitwise_and` | 659 | 0 | 5169 |

## batch 256

render **469.81 ms** median (min 467.14, max 477.35, sd 2.62) = **545 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 229192 | 8605 |
| host->device | 0 | - | 139305 | 1029 |
| allocation | 0 | - | 1685 | 1352 |
| kernel/raster | 0 | - | 102959 | 1632 |
| synchronization | 0 | - | 4 | 1 |

12619 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 328 | 0 | 138427 |
| `aten::mul` | 238 | 0 | 46964 |
| `aten::flip` | 3 | 0 | 41540 |
| `aten::fill_` | 60 | 0 | 38823 |
| `aten::bitwise_xor` | 2 | 0 | 34732 |
| `aten::ge` | 92 | 0 | 26351 |
| `aten::index` | 123 | 0 | 25811 |
| `aten::le` | 20 | 0 | 24960 |

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
