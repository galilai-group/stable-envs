# Profile note — CartPole on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node2338.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 2
- states: `examples/results/step4/states/CartPole.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **18.31 ms** median (min 17.82, max 18.97, sd 0.29) = **55 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 9665 | 8519 |
| host->device | 0 | - | 1650 | 986 |
| allocation | 0 | - | 1136 | 1266 |
| kernel/raster | 0 | - | 5218 | 1632 |
| synchronization | 0 | - | 7 | 1 |

12404 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 659 | 0 | 2150 |
| `aten::where` | 479 | 0 | 1469 |
| `aten::mul` | 238 | 0 | 855 |
| `aten::add` | 372 | 0 | 811 |
| `aten::index` | 123 | 0 | 794 |
| `aten::copy_` | 285 | 0 | 785 |
| `aten::_to_copy` | 280 | 0 | 640 |
| `aten::__rshift__` | 147 | 0 | 610 |

## batch 16

render **48.98 ms** median (min 42.70, max 72.37, sd 10.32) = **327 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 20232 | 8605 |
| host->device | 0 | - | 8528 | 1029 |
| allocation | 0 | - | 1419 | 1352 |
| kernel/raster | 0 | - | 10264 | 1632 |
| synchronization | 0 | - | 3 | 1 |

12619 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 328 | 0 | 7685 |
| `aten::bitwise_and` | 659 | 0 | 2914 |
| `aten::mul` | 238 | 0 | 2462 |
| `aten::index` | 123 | 0 | 2382 |
| `aten::ge` | 92 | 0 | 2123 |
| `aten::flip` | 3 | 0 | 1955 |
| `aten::where` | 479 | 0 | 1911 |
| `aten::bitwise_xor` | 2 | 0 | 1820 |

## batch 64

render **176.48 ms** median (min 174.40, max 177.35, sd 0.75) = **363 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 74209 | 8605 |
| host->device | 0 | - | 29409 | 1029 |
| allocation | 0 | - | 1590 | 1352 |
| kernel/raster | 0 | - | 34037 | 1632 |
| synchronization | 0 | - | 3 | 1 |

12619 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 328 | 0 | 28519 |
| `aten::fill_` | 60 | 0 | 19161 |
| `aten::mul` | 238 | 0 | 17433 |
| `aten::bitwise_xor` | 2 | 0 | 13238 |
| `aten::flip` | 3 | 0 | 8216 |
| `aten::ge` | 92 | 0 | 7040 |
| `aten::index` | 123 | 0 | 6728 |
| `aten::le` | 20 | 0 | 6438 |

## batch 256

render **817.25 ms** median (min 785.54, max 894.06, sd 40.24) = **313 fps**  ·  set_state 0.08 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 382695 | 8605 |
| host->device | 0 | - | 211595 | 1029 |
| allocation | 0 | - | 2432 | 1352 |
| kernel/raster | 0 | - | 144239 | 1632 |
| synchronization | 0 | - | 4 | 1 |

12619 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 328 | 0 | 210175 |
| `aten::flip` | 3 | 0 | 98967 |
| `aten::fill_` | 60 | 0 | 79563 |
| `aten::mul` | 238 | 0 | 75738 |
| `aten::bitwise_xor` | 2 | 0 | 53239 |
| `aten::le` | 20 | 0 | 40391 |
| `aten::index` | 123 | 0 | 30706 |
| `aten::bitwise_and` | 659 | 0 | 30191 |

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
