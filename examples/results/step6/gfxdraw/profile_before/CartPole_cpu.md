# Profile note — CartPole on cpu

- workstream: `step6/gfxdraw`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 2
- states: `examples/results/step4/states/CartPole.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **44.48 ms** median (min 39.13, max 44.61, sd 2.45) = **22 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 16091 | 16589 |
| host->device | 0 | - | 3705 | 2286 |
| allocation | 0 | - | 2762 | 3552 |
| kernel/raster | 0 | - | 16194 | 3576 |
| synchronization | 0 | - | 2 | 1 |

26004 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1391 | 0 | 10795 |
| `aten::bitwise_and` | 1384 | 0 | 4814 |
| `aten::copy_` | 717 | 0 | 1840 |
| `aten::add` | 842 | 0 | 1616 |
| `aten::_to_copy` | 712 | 0 | 1386 |
| `aten::empty_strided` | 1116 | 0 | 1095 |
| `aten::__and__` | 1384 | 0 | 1095 |
| `aten::bitwise_not` | 457 | 0 | 954 |

## batch 16

render **201.08 ms** median (min 195.42, max 206.15, sd 3.32) = **80 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 46937 | 16649 |
| host->device | 0 | - | 22556 | 2320 |
| allocation | 0 | - | 3078 | 3620 |
| kernel/raster | 0 | - | 119121 | 3576 |
| synchronization | 0 | - | 3 | 1 |

26166 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1391 | 0 | 109002 |
| `aten::copy_` | 751 | 0 | 20834 |
| `aten::bitwise_and` | 1384 | 0 | 12767 |
| `aten::le` | 22 | 0 | 7148 |
| `aten::fill_` | 220 | 0 | 6734 |
| `aten::ge` | 96 | 0 | 5159 |
| `aten::index` | 124 | 0 | 2668 |
| `aten::flip` | 1 | 0 | 2463 |

## batch 64

render **771.12 ms** median (min 700.84, max 792.58, sd 29.41) = **83 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 185371 | 16649 |
| host->device | 0 | - | 81880 | 2320 |
| allocation | 0 | - | 3105 | 3620 |
| kernel/raster | 0 | - | 491341 | 3576 |
| synchronization | 0 | - | 2 | 1 |

26166 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1391 | 0 | 469800 |
| `aten::copy_` | 751 | 0 | 80109 |
| `aten::bitwise_and` | 1384 | 0 | 52412 |
| `aten::fill_` | 220 | 0 | 43210 |
| `aten::le` | 22 | 0 | 39792 |
| `aten::ge` | 96 | 0 | 19296 |
| `aten::flip` | 1 | 0 | 9319 |
| `aten::amin` | 2 | 0 | 7665 |

## batch 256

render **3201.77 ms** median (min 3180.31, max 3221.10, sd 11.55) = **80 fps**  ·  set_state 0.07 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 857305 | 16649 |
| host->device | 0 | - | 323803 | 2320 |
| allocation | 0 | - | 3677 | 3620 |
| kernel/raster | 0 | - | 1935034 | 3576 |
| synchronization | 0 | - | 3 | 1 |

26166 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1391 | 0 | 1859655 |
| `aten::copy_` | 751 | 0 | 321853 |
| `aten::bitwise_and` | 1384 | 0 | 225465 |
| `aten::le` | 22 | 0 | 181916 |
| `aten::fill_` | 220 | 0 | 178180 |
| `aten::ge` | 96 | 0 | 122837 |
| `aten::flip` | 1 | 0 | 89380 |
| `aten::amin` | 2 | 0 | 30729 |

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
