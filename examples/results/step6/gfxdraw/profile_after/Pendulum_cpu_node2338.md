# Profile note — Pendulum on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node2338.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Pendulum.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **11.20 ms** median (min 11.11, max 11.23, sd 0.04) = **89 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 6256 | 4882 |
| host->device | 0 | - | 1623 | 584 |
| allocation | 0 | - | 745 | 689 |
| kernel/raster | 0 | - | 4094 | 921 |
| synchronization | 0 | - | 5 | 1 |
| output conversion | 0 | - | 4 | 1 |

7078 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 360 | 0 | 1188 |
| `aten::copy_` | 168 | 0 | 1089 |
| `aten::where` | 253 | 0 | 920 |
| `aten::index` | 70 | 0 | 887 |
| `aten::add` | 212 | 0 | 644 |
| `aten::mul` | 142 | 0 | 612 |
| `aten::__rshift__` | 82 | 0 | 413 |
| `aten::_to_copy` | 163 | 0 | 391 |

## batch 16

render **43.03 ms** median (min 42.45, max 46.72, sd 1.37) = **372 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 14328 | 4941 |
| host->device | 0 | - | 10285 | 613 |
| allocation | 0 | - | 841 | 747 |
| kernel/raster | 0 | - | 15251 | 921 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 5 | 1 |

7224 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 197 | 0 | 9781 |
| `aten::index` | 70 | 0 | 7133 |
| `aten::flip` | 3 | 0 | 2527 |
| `aten::mul` | 142 | 0 | 2164 |
| `aten::bitwise_and` | 360 | 0 | 1834 |
| `aten::add` | 212 | 0 | 1619 |
| `aten::fill_` | 33 | 0 | 1572 |
| `aten::where` | 253 | 0 | 1463 |

## batch 64

render **199.42 ms** median (min 187.81, max 220.45, sd 8.61) = **321 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 64533 | 4941 |
| host->device | 0 | - | 69842 | 613 |
| allocation | 0 | - | 1378 | 747 |
| kernel/raster | 0 | - | 77115 | 921 |
| synchronization | 0 | - | 13 | 1 |
| output conversion | 0 | - | 6 | 1 |

7224 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 197 | 0 | 69064 |
| `aten::index` | 70 | 0 | 26556 |
| `aten::fill_` | 33 | 0 | 23158 |
| `aten::mul` | 142 | 0 | 15273 |
| `aten::add` | 212 | 0 | 11691 |
| `aten::sub` | 74 | 0 | 11673 |
| `aten::flip` | 3 | 0 | 11337 |
| `aten::_index_put_impl_` | 26 | 0 | 5231 |

## batch 256

render **1004.95 ms** median (min 935.70, max 1106.17, sd 45.62) = **255 fps**  ·  set_state 0.08 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 333769 | 4941 |
| host->device | 0 | - | 280776 | 613 |
| allocation | 0 | - | 1722 | 747 |
| kernel/raster | 0 | - | 314853 | 921 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 7 | 1 |

7224 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 197 | 0 | 279783 |
| `aten::index` | 70 | 0 | 112595 |
| `aten::fill_` | 33 | 0 | 102982 |
| `aten::flip` | 3 | 0 | 94456 |
| `aten::mul` | 142 | 0 | 86224 |
| `aten::add` | 212 | 0 | 39045 |
| `aten::sub` | 74 | 0 | 37367 |
| `aten::bitwise_xor` | 1 | 0 | 33613 |

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
