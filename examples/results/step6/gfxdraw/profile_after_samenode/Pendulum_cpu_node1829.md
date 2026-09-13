# Profile note — Pendulum on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Pendulum.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/gfxdraw/stable_envs/__init__.py`

## batch 1

render **11.70 ms** median (min 11.53, max 11.92, sd 0.12) = **85 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 5204 | 4882 |
| host->device | 0 | - | 1341 | 584 |
| allocation | 0 | - | 630 | 689 |
| kernel/raster | 0 | - | 3529 | 921 |
| synchronization | 0 | - | 4 | 1 |
| output conversion | 0 | - | 3 | 1 |

7078 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::bitwise_and` | 360 | 0 | 1084 |
| `aten::copy_` | 168 | 0 | 889 |
| `aten::where` | 253 | 0 | 818 |
| `aten::index` | 70 | 0 | 776 |
| `aten::add` | 212 | 0 | 545 |
| `aten::mul` | 142 | 0 | 533 |
| `aten::__rshift__` | 82 | 0 | 386 |
| `aten::view` | 290 | 0 | 327 |

## batch 16

render **38.97 ms** median (min 37.49, max 44.06, sd 2.23) = **411 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 13253 | 4941 |
| host->device | 0 | - | 9127 | 613 |
| allocation | 0 | - | 807 | 747 |
| kernel/raster | 0 | - | 14145 | 921 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 14 | 1 |

7224 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 197 | 0 | 8659 |
| `aten::index` | 70 | 0 | 6464 |
| `aten::flip` | 3 | 0 | 2321 |
| `aten::mul` | 142 | 0 | 1992 |
| `aten::bitwise_and` | 360 | 0 | 1831 |
| `aten::add` | 212 | 0 | 1526 |
| `aten::fill_` | 33 | 0 | 1387 |
| `aten::where` | 253 | 0 | 1381 |

## batch 64

render **136.13 ms** median (min 130.82, max 137.58, sd 2.33) = **470 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 46441 | 4941 |
| host->device | 0 | - | 41752 | 613 |
| allocation | 0 | - | 1310 | 747 |
| kernel/raster | 0 | - | 48702 | 921 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 4 | 1 |

7224 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 197 | 0 | 41235 |
| `aten::index` | 70 | 0 | 23717 |
| `aten::flip` | 3 | 0 | 12474 |
| `aten::fill_` | 33 | 0 | 10307 |
| `aten::mul` | 142 | 0 | 8517 |
| `aten::add` | 212 | 0 | 4762 |
| `aten::sub` | 74 | 0 | 4144 |
| `aten::bitwise_and` | 360 | 0 | 3417 |

## batch 256

render **590.24 ms** median (min 583.89, max 597.55, sd 4.24) = **434 fps**  ·  set_state 0.08 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 186848 | 4941 |
| host->device | 0 | - | 175347 | 613 |
| allocation | 0 | - | 1129 | 747 |
| kernel/raster | 0 | - | 218441 | 921 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 5 | 1 |

7224 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 197 | 0 | 174693 |
| `aten::index` | 70 | 0 | 97129 |
| `aten::flip` | 3 | 0 | 55074 |
| `aten::fill_` | 33 | 0 | 42057 |
| `aten::mul` | 142 | 0 | 39348 |
| `aten::add` | 212 | 0 | 26978 |
| `aten::sub` | 74 | 0 | 25165 |
| `aten::bitwise_xor` | 1 | 0 | 16741 |

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
