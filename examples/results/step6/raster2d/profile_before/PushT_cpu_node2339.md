# Profile note — PushT on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2339.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 96px  ·  state_dim: 5
- states: `examples/results/step4/states/PushT.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **14.98 ms** median (min 14.38, max 16.55, sd 0.54) = **67 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 13131 | 4116 |
| host->device | 0 | - | 895 | 469 |
| allocation | 0 | - | 933 | 562 |
| kernel/raster | 0 | - | 12378 | 1485 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 42 | 13 |

6646 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::abs` | 2 | 0 | 4562 |
| `aten::where` | 126 | 0 | 4377 |
| `aten::sub` | 450 | 0 | 3570 |
| `aten::mul` | 522 | 0 | 2266 |
| `aten::reshape` | 89 | 0 | 2142 |
| `aten::bitwise_and` | 63 | 0 | 1075 |
| `aten::add` | 186 | 0 | 1049 |
| `aten::select` | 284 | 0 | 712 |

## batch 16

render **124.69 ms** median (min 117.72, max 136.84, sd 5.27) = **128 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 26644 | 4202 |
| host->device | 0 | - | 1641 | 512 |
| allocation | 0 | - | 986 | 648 |
| kernel/raster | 0 | - | 96800 | 1485 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 160 | 56 |

6904 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 126 | 0 | 32404 |
| `aten::sub` | 450 | 0 | 27590 |
| `aten::mul` | 522 | 0 | 19513 |
| `aten::add` | 186 | 0 | 11960 |
| `aten::bitwise_and` | 63 | 0 | 7361 |
| `aten::le` | 45 | 0 | 5188 |
| `aten::fill_` | 103 | 0 | 3351 |
| `aten::maximum` | 36 | 0 | 3134 |

## batch 64

render **559.99 ms** median (min 540.80, max 590.04, sd 14.77) = **114 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 106456 | 4202 |
| host->device | 0 | - | 4523 | 512 |
| allocation | 0 | - | 1617 | 648 |
| kernel/raster | 0 | - | 440398 | 1485 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 624 | 56 |

6904 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 126 | 0 | 128985 |
| `aten::sub` | 450 | 0 | 123558 |
| `aten::mul` | 522 | 0 | 107870 |
| `aten::add` | 186 | 0 | 57521 |
| `aten::bitwise_and` | 63 | 0 | 26759 |
| `aten::le` | 45 | 0 | 20162 |
| `aten::fill_` | 103 | 0 | 16972 |
| `aten::maximum` | 36 | 0 | 16555 |

## batch 256

render **2609.21 ms** median (min 2399.75, max 2697.90, sd 89.37) = **98 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 476246 | 4202 |
| host->device | 0 | - | 15398 | 512 |
| allocation | 0 | - | 2501 | 648 |
| kernel/raster | 0 | - | 2287212 | 1485 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2524 | 56 |

6904 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 126 | 0 | 620780 |
| `aten::sub` | 450 | 0 | 620680 |
| `aten::mul` | 522 | 0 | 584835 |
| `aten::add` | 186 | 0 | 326841 |
| `aten::bitwise_and` | 63 | 0 | 106066 |
| `aten::fill_` | 103 | 0 | 86981 |
| `aten::maximum` | 36 | 0 | 81791 |
| `aten::le` | 45 | 0 | 81092 |

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
