# Profile note — Pendulum on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Pendulum.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/baseline/stable_envs/__init__.py`

## batch 1

render **30.48 ms** median (min 30.20, max 31.15, sd 0.25) = **33 fps**  ·  set_state 0.08 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 9778 | 8829 |
| host->device | 0 | - | 2625 | 1230 |
| allocation | 0 | - | 1468 | 1847 |
| kernel/raster | 0 | - | 15216 | 1878 |
| synchronization | 0 | - | 5 | 1 |

13785 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 709 | 0 | 11656 |
| `aten::bitwise_and` | 715 | 0 | 2505 |
| `aten::copy_` | 385 | 0 | 1725 |
| `aten::index` | 73 | 0 | 912 |
| `aten::add` | 437 | 0 | 824 |
| `aten::le` | 16 | 0 | 821 |
| `aten::_to_copy` | 379 | 0 | 649 |
| `aten::__and__` | 715 | 0 | 617 |

## batch 16

render **240.21 ms** median (min 232.12, max 242.48, sd 3.35) = **67 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 43155 | 8870 |
| host->device | 0 | - | 26019 | 1252 |
| allocation | 0 | - | 1733 | 1891 |
| kernel/raster | 0 | - | 164852 | 1878 |
| synchronization | 0 | - | 3 | 1 |

13892 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 709 | 0 | 151434 |
| `aten::copy_` | 407 | 0 | 24964 |
| `aten::le` | 16 | 0 | 10359 |
| `aten::fill_` | 118 | 0 | 9522 |
| `aten::bitwise_and` | 715 | 0 | 8880 |
| `aten::index` | 73 | 0 | 6834 |
| `aten::ge` | 55 | 0 | 3927 |
| `aten::flip` | 1 | 0 | 2371 |

## batch 64

render **915.73 ms** median (min 910.73, max 926.97, sd 5.79) = **70 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 146632 | 8870 |
| host->device | 0 | - | 99691 | 1252 |
| allocation | 0 | - | 1951 | 1891 |
| kernel/raster | 0 | - | 663095 | 1878 |
| synchronization | 0 | - | 4 | 1 |

13892 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 709 | 0 | 615774 |
| `aten::copy_` | 407 | 0 | 98542 |
| `aten::le` | 16 | 0 | 39899 |
| `aten::fill_` | 118 | 0 | 36792 |
| `aten::bitwise_and` | 715 | 0 | 27184 |
| `aten::index` | 73 | 0 | 25064 |
| `aten::flip` | 1 | 0 | 14743 |
| `aten::ge` | 55 | 0 | 10547 |

## batch 256

render **3659.61 ms** median (min 3609.17, max 3677.73, sd 23.59) = **70 fps**  ·  set_state 0.08 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 609399 | 8870 |
| host->device | 0 | - | 398663 | 1252 |
| allocation | 0 | - | 2256 | 1891 |
| kernel/raster | 0 | - | 2648408 | 1878 |
| synchronization | 0 | - | 3 | 1 |

13892 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 709 | 0 | 2464862 |
| `aten::copy_` | 407 | 0 | 397485 |
| `aten::le` | 16 | 0 | 202914 |
| `aten::fill_` | 118 | 0 | 144854 |
| `aten::bitwise_and` | 715 | 0 | 101116 |
| `aten::index` | 73 | 0 | 97160 |
| `aten::flip` | 1 | 0 | 56919 |
| `aten::ge` | 55 | 0 | 51929 |

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
