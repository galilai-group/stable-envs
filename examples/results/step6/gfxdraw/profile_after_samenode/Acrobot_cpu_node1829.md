# Profile note — Acrobot on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Acrobot.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/gfxdraw/stable_envs/__init__.py`

## batch 1

render **18.24 ms** median (min 18.03, max 18.39, sd 0.10) = **55 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 9617 | 8762 |
| host->device | 0 | - | 1473 | 1013 |
| allocation | 0 | - | 1077 | 1282 |
| kernel/raster | 0 | - | 5328 | 1682 |
| synchronization | 0 | - | 2 | 1 |

12740 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 670 | 0 | 2099 |
| `aten::where` | 484 | 0 | 1396 |
| `aten::mul` | 249 | 0 | 1046 |
| `aten::add` | 389 | 0 | 802 |
| `aten::index` | 126 | 0 | 790 |
| `aten::copy_` | 292 | 0 | 755 |
| `aten::__rshift__` | 150 | 0 | 590 |
| `aten::ge` | 96 | 0 | 580 |

## batch 16

render **52.68 ms** median (min 51.07, max 61.33, sd 3.26) = **304 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 25860 | 8854 |
| host->device | 0 | - | 8462 | 1059 |
| allocation | 0 | - | 1346 | 1374 |
| kernel/raster | 0 | - | 12256 | 1682 |
| synchronization | 0 | - | 3 | 1 |

12970 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 338 | 0 | 7664 |
| `aten::mul` | 249 | 0 | 3561 |
| `aten::bitwise_and` | 670 | 0 | 3286 |
| `aten::ge` | 96 | 0 | 3187 |
| `aten::bitwise_xor` | 2 | 0 | 2742 |
| `aten::le` | 20 | 0 | 2725 |
| `aten::index` | 126 | 0 | 2674 |
| `aten::bitwise_xor_` | 2 | 0 | 2294 |

## batch 64

render **179.59 ms** median (min 159.92, max 187.48, sd 9.66) = **356 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 89103 | 8854 |
| host->device | 0 | - | 36850 | 1059 |
| allocation | 0 | - | 1574 | 1374 |
| kernel/raster | 0 | - | 45965 | 1682 |
| synchronization | 0 | - | 3 | 1 |

12970 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 338 | 0 | 35972 |
| `aten::mul` | 249 | 0 | 21762 |
| `aten::bitwise_xor` | 2 | 0 | 17007 |
| `aten::le` | 20 | 0 | 12138 |
| `aten::ge` | 96 | 0 | 11306 |
| `aten::flip` | 3 | 0 | 10838 |
| `aten::index` | 126 | 0 | 9375 |
| `aten::bitwise_and` | 670 | 0 | 9049 |

## batch 256

render **706.92 ms** median (min 703.83, max 720.17, sd 5.75) = **362 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 376387 | 8854 |
| host->device | 0 | - | 147477 | 1059 |
| allocation | 0 | - | 1788 | 1374 |
| kernel/raster | 0 | - | 168998 | 1682 |
| synchronization | 0 | - | 3 | 1 |

12970 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 338 | 0 | 146473 |
| `aten::mul` | 249 | 0 | 82991 |
| `aten::bitwise_xor` | 2 | 0 | 60159 |
| `aten::flip` | 3 | 0 | 55752 |
| `aten::ge` | 96 | 0 | 53269 |
| `aten::le` | 20 | 0 | 51512 |
| `aten::fill_` | 58 | 0 | 40643 |
| `aten::bitwise_xor_` | 2 | 0 | 38791 |

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
