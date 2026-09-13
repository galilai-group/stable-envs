# Profile note — MountainCarContinuous on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node2338.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCarContinuous.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **19.03 ms** median (min 17.00, max 19.16, sd 0.79) = **53 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 10618 | 8166 |
| host->device | 0 | - | 2152 | 968 |
| allocation | 0 | - | 1203 | 1223 |
| kernel/raster | 0 | - | 5887 | 1569 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 8 | 1 |

11928 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 621 | 0 | 2121 |
| `aten::where` | 465 | 0 | 1479 |
| `aten::copy_` | 282 | 0 | 1283 |
| `aten::mul` | 228 | 0 | 1043 |
| `aten::add` | 358 | 0 | 801 |
| `aten::index` | 111 | 0 | 776 |
| `aten::_to_copy` | 275 | 0 | 648 |
| `aten::__rshift__` | 133 | 0 | 605 |

## batch 16

render **65.79 ms** median (min 43.03, max 82.86, sd 10.96) = **243 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 39328 | 8253 |
| host->device | 0 | - | 27886 | 1008 |
| allocation | 0 | - | 1612 | 1307 |
| kernel/raster | 0 | - | 15930 | 1569 |
| synchronization | 0 | - | 7 | 1 |
| output conversion | 0 | - | 10 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 26968 |
| `aten::mul` | 228 | 0 | 7675 |
| `aten::flip` | 3 | 0 | 7057 |
| `aten::bitwise_xor` | 2 | 0 | 5222 |
| `aten::ge` | 86 | 0 | 4349 |
| `aten::fill_` | 58 | 0 | 4186 |
| `aten::bitwise_and` | 621 | 0 | 4155 |
| `aten::le` | 18 | 0 | 3687 |

## batch 64

render **257.45 ms** median (min 222.06, max 301.96, sd 27.92) = **249 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 118132 | 8253 |
| host->device | 0 | - | 56093 | 1008 |
| allocation | 0 | - | 1714 | 1307 |
| kernel/raster | 0 | - | 41389 | 1569 |
| synchronization | 0 | - | 6 | 1 |
| output conversion | 0 | - | 8 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 55112 |
| `aten::mul` | 228 | 0 | 26958 |
| `aten::flip` | 3 | 0 | 22681 |
| `aten::fill_` | 58 | 0 | 22377 |
| `aten::bitwise_xor` | 2 | 0 | 20317 |
| `aten::le` | 18 | 0 | 14010 |
| `aten::bitwise_and` | 621 | 0 | 10445 |
| `aten::ge` | 86 | 0 | 9306 |

## batch 256

render **905.55 ms** median (min 846.97, max 939.98, sd 41.37) = **283 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 484800 | 8253 |
| host->device | 0 | - | 228477 | 1008 |
| allocation | 0 | - | 1920 | 1307 |
| kernel/raster | 0 | - | 150671 | 1569 |
| synchronization | 0 | - | 4 | 1 |
| output conversion | 0 | - | 9 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 227369 |
| `aten::mul` | 228 | 0 | 107407 |
| `aten::flip` | 3 | 0 | 90726 |
| `aten::fill_` | 58 | 0 | 86759 |
| `aten::bitwise_xor` | 2 | 0 | 85168 |
| `aten::ge` | 86 | 0 | 63635 |
| `aten::le` | 18 | 0 | 62814 |
| `aten::bitwise_and` | 621 | 0 | 38292 |

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
