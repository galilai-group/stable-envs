# Profile note — Pendulum on cpu

- workstream: `step6/gfxdraw`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Pendulum.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **33.59 ms** median (min 33.49, max 33.77, sd 0.09) = **30 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 9439 | 8829 |
| host->device | 0 | - | 2512 | 1230 |
| allocation | 0 | - | 1432 | 1847 |
| kernel/raster | 0 | - | 15072 | 1878 |
| synchronization | 0 | - | 5 | 1 |

13785 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 709 | 0 | 11524 |
| `aten::bitwise_and` | 715 | 0 | 2388 |
| `aten::copy_` | 385 | 0 | 1633 |
| `aten::index` | 73 | 0 | 894 |
| `aten::add` | 437 | 0 | 840 |
| `aten::le` | 16 | 0 | 807 |
| `aten::_to_copy` | 379 | 0 | 652 |
| `aten::__and__` | 715 | 0 | 594 |

## batch 16

render **224.84 ms** median (min 223.40, max 231.50, sd 2.68) = **71 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 36200 | 8870 |
| host->device | 0 | - | 24999 | 1252 |
| allocation | 0 | - | 1558 | 1891 |
| kernel/raster | 0 | - | 157596 | 1878 |
| synchronization | 0 | - | 3 | 1 |

13892 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 709 | 0 | 144728 |
| `aten::copy_` | 407 | 0 | 24029 |
| `aten::le` | 16 | 0 | 9944 |
| `aten::fill_` | 118 | 0 | 7106 |
| `aten::bitwise_and` | 715 | 0 | 6702 |
| `aten::index` | 73 | 0 | 6619 |
| `aten::ge` | 55 | 0 | 2748 |
| `aten::flip` | 1 | 0 | 2182 |

## batch 64

render **980.67 ms** median (min 979.52, max 982.55, sd 0.73) = **65 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 186279 | 8870 |
| host->device | 0 | - | 97244 | 1252 |
| allocation | 0 | - | 1754 | 1891 |
| kernel/raster | 0 | - | 692007 | 1878 |
| synchronization | 0 | - | 3 | 1 |

13892 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 709 | 0 | 643389 |
| `aten::copy_` | 407 | 0 | 96194 |
| `aten::le` | 16 | 0 | 57550 |
| `aten::fill_` | 118 | 0 | 46655 |
| `aten::bitwise_and` | 715 | 0 | 32169 |
| `aten::index` | 73 | 0 | 24694 |
| `aten::flip` | 1 | 0 | 21777 |
| `aten::ge` | 55 | 0 | 10226 |

## batch 256

render **4347.51 ms** median (min 4302.76, max 4402.66, sd 29.26) = **59 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 840133 | 8870 |
| host->device | 0 | - | 440697 | 1252 |
| allocation | 0 | - | 2191 | 1891 |
| kernel/raster | 0 | - | 2903520 | 1878 |
| synchronization | 0 | - | 3 | 1 |

13892 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 709 | 0 | 2674268 |
| `aten::copy_` | 407 | 0 | 439541 |
| `aten::le` | 16 | 0 | 261601 |
| `aten::fill_` | 118 | 0 | 193315 |
| `aten::bitwise_and` | 715 | 0 | 146352 |
| `aten::flip` | 1 | 0 | 110653 |
| `aten::index` | 73 | 0 | 102580 |
| `aten::ge` | 55 | 0 | 66038 |

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
