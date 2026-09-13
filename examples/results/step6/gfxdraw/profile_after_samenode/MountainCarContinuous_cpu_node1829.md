# Profile note — MountainCarContinuous on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCarContinuous.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/gfxdraw/stable_envs/__init__.py`

## batch 1

render **16.93 ms** median (min 16.81, max 17.17, sd 0.12) = **59 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 8691 | 8166 |
| host->device | 0 | - | 1584 | 968 |
| allocation | 0 | - | 1039 | 1223 |
| kernel/raster | 0 | - | 5059 | 1569 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 6 | 1 |

11928 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 621 | 0 | 1869 |
| `aten::where` | 465 | 0 | 1342 |
| `aten::mul` | 228 | 0 | 863 |
| `aten::copy_` | 282 | 0 | 862 |
| `aten::add` | 358 | 0 | 703 |
| `aten::index` | 111 | 0 | 624 |
| `aten::__rshift__` | 133 | 0 | 537 |
| `aten::_to_copy` | 275 | 0 | 524 |

## batch 16

render **51.13 ms** median (min 50.42, max 51.43, sd 0.34) = **313 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 22998 | 8253 |
| host->device | 0 | - | 11765 | 1008 |
| allocation | 0 | - | 1384 | 1307 |
| kernel/raster | 0 | - | 11466 | 1569 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 6 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 10986 |
| `aten::mul` | 228 | 0 | 4821 |
| `aten::bitwise_xor` | 2 | 0 | 3017 |
| `aten::bitwise_and` | 621 | 0 | 2878 |
| `aten::ge` | 86 | 0 | 2554 |
| `aten::flip` | 3 | 0 | 2477 |
| `aten::le` | 18 | 0 | 2229 |
| `aten::where` | 465 | 0 | 1704 |

## batch 64

render **153.00 ms** median (min 151.84, max 155.14, sd 0.89) = **418 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 76282 | 8253 |
| host->device | 0 | - | 39950 | 1008 |
| allocation | 0 | - | 1417 | 1307 |
| kernel/raster | 0 | - | 28957 | 1569 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 7 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 39147 |
| `aten::mul` | 228 | 0 | 16703 |
| `aten::bitwise_xor` | 2 | 0 | 12547 |
| `aten::le` | 18 | 0 | 10906 |
| `aten::flip` | 3 | 0 | 10341 |
| `aten::fill_` | 58 | 0 | 9774 |
| `aten::ge` | 86 | 0 | 8894 |
| `aten::bitwise_and` | 621 | 0 | 7269 |

## batch 256

render **591.30 ms** median (min 581.12, max 595.05, sd 3.84) = **433 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 312913 | 8253 |
| host->device | 0 | - | 166086 | 1008 |
| allocation | 0 | - | 1828 | 1307 |
| kernel/raster | 0 | - | 105554 | 1569 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 7 | 1 |

12139 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 323 | 0 | 165110 |
| `aten::mul` | 228 | 0 | 66410 |
| `aten::flip` | 3 | 0 | 54053 |
| `aten::bitwise_xor` | 2 | 0 | 50987 |
| `aten::ge` | 86 | 0 | 44728 |
| `aten::le` | 18 | 0 | 44369 |
| `aten::fill_` | 58 | 0 | 39989 |
| `aten::bitwise_xor_` | 2 | 0 | 26802 |

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
