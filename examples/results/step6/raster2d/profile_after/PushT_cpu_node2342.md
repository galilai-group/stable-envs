# Profile note — PushT on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2342.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 96px  ·  state_dim: 5
- states: `examples/results/step4/states/PushT.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **6.60 ms** median (min 6.49, max 8.82, sd 0.73) = **151 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 2409 | 2559 |
| host->device | 0 | - | 489 | 390 |
| allocation | 0 | - | 289 | 228 |
| kernel/raster | 0 | - | 1659 | 625 |
| synchronization | 0 | - | 4 | 1 |
| output conversion | 0 | - | 76 | 29 |

3832 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::sub` | 180 | 0 | 429 |
| `aten::mul` | 207 | 0 | 406 |
| `aten::select` | 185 | 0 | 281 |
| `aten::view` | 226 | 0 | 266 |
| `aten::where` | 37 | 0 | 242 |
| `aten::as_strided` | 440 | 0 | 222 |
| `aten::copy_` | 100 | 0 | 193 |
| `aten::add` | 83 | 0 | 179 |

## batch 16

render **31.19 ms** median (min 29.66, max 32.03, sd 0.72) = **513 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 5856 | 2637 |
| host->device | 0 | - | 1389 | 433 |
| allocation | 0 | - | 390 | 314 |
| kernel/raster | 0 | - | 21373 | 625 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 121 | 72 |

4082 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 37 | 0 | 8081 |
| `aten::sub` | 180 | 0 | 5524 |
| `aten::mul` | 207 | 0 | 4489 |
| `aten::add` | 83 | 0 | 2062 |
| `aten::copy_` | 143 | 0 | 1077 |
| `aten::bitwise_and` | 17 | 0 | 892 |
| `aten::le` | 12 | 0 | 703 |
| `aten::maximum` | 19 | 0 | 674 |

## batch 64

render **135.57 ms** median (min 121.08, max 147.34, sd 6.35) = **472 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 18729 | 2633 |
| host->device | 0 | - | 5217 | 433 |
| allocation | 0 | - | 509 | 314 |
| kernel/raster | 0 | - | 108940 | 625 |
| synchronization | 0 | - | 4 | 1 |
| output conversion | 0 | - | 170 | 72 |

4078 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 37 | 0 | 32099 |
| `aten::mul` | 207 | 0 | 29905 |
| `aten::sub` | 180 | 0 | 29010 |
| `aten::add` | 83 | 0 | 13175 |
| `aten::copy_` | 143 | 0 | 4787 |
| `aten::maximum` | 19 | 0 | 4278 |
| `aten::minimum` | 10 | 0 | 3505 |
| `aten::bitwise_and` | 17 | 0 | 3277 |

## batch 256

render **563.60 ms** median (min 543.77, max 580.86, sd 10.87) = **454 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 71220 | 2633 |
| host->device | 0 | - | 19876 | 433 |
| allocation | 0 | - | 629 | 314 |
| kernel/raster | 0 | - | 459954 | 625 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 339 | 72 |

4078 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 37 | 0 | 126495 |
| `aten::mul` | 207 | 0 | 124126 |
| `aten::sub` | 180 | 0 | 117724 |
| `aten::add` | 83 | 0 | 62058 |
| `aten::maximum` | 19 | 0 | 19853 |
| `aten::copy_` | 143 | 0 | 19353 |
| `aten::minimum` | 10 | 0 | 17156 |
| `aten::div` | 16 | 0 | 15240 |

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
