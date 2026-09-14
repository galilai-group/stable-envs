# Profile note — CartPole on cpu

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: node1829.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 600px  ·  state_dim: 2
- states: `examples/results/step4/states/CartPole.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/baseline/stable_envs/__init__.py`

## batch 1

render **39.29 ms** median (min 38.87, max 40.02, sd 0.31) = **25 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 16234 | 16589 |
| host->device | 0 | - | 3684 | 2286 |
| allocation | 0 | - | 2706 | 3552 |
| kernel/raster | 0 | - | 15647 | 3576 |
| synchronization | 0 | - | 2 | 1 |

26004 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1391 | 0 | 10296 |
| `aten::bitwise_and` | 1384 | 0 | 4960 |
| `aten::copy_` | 717 | 0 | 1853 |
| `aten::add` | 842 | 0 | 1602 |
| `aten::_to_copy` | 712 | 0 | 1319 |
| `aten::empty_strided` | 1116 | 0 | 1047 |
| `aten::__and__` | 1384 | 0 | 1014 |
| `aten::bitwise_not` | 457 | 0 | 955 |

## batch 16

render **199.02 ms** median (min 198.55, max 202.09, sd 0.98) = **80 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 49581 | 16649 |
| host->device | 0 | - | 22306 | 2320 |
| allocation | 0 | - | 2879 | 3620 |
| kernel/raster | 0 | - | 118617 | 3576 |
| synchronization | 0 | - | 3 | 1 |

26166 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1391 | 0 | 109008 |
| `aten::copy_` | 751 | 0 | 20660 |
| `aten::bitwise_and` | 1384 | 0 | 14232 |
| `aten::fill_` | 220 | 0 | 8713 |
| `aten::le` | 22 | 0 | 7118 |
| `aten::ge` | 96 | 0 | 5069 |
| `aten::index` | 124 | 0 | 2560 |
| `aten::flip` | 1 | 0 | 2340 |

## batch 64

render **724.01 ms** median (min 723.38, max 729.18, sd 1.81) = **88 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 168182 | 16649 |
| host->device | 0 | - | 82681 | 2320 |
| allocation | 0 | - | 3002 | 3620 |
| kernel/raster | 0 | - | 458791 | 3576 |
| synchronization | 0 | - | 2 | 1 |

26166 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1391 | 0 | 436502 |
| `aten::copy_` | 751 | 0 | 80963 |
| `aten::bitwise_and` | 1384 | 0 | 49576 |
| `aten::fill_` | 220 | 0 | 33815 |
| `aten::le` | 22 | 0 | 27883 |
| `aten::ge` | 96 | 0 | 21843 |
| `aten::flip` | 1 | 0 | 13594 |
| `aten::index` | 124 | 0 | 7931 |

## batch 256

render **2818.10 ms** median (min 2812.52, max 2826.66, sd 4.28) = **91 fps**  ·  set_state 0.06 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 675501 | 16649 |
| host->device | 0 | - | 328393 | 2320 |
| allocation | 0 | - | 3673 | 3620 |
| kernel/raster | 0 | - | 1811416 | 3576 |
| synchronization | 0 | - | 3 | 1 |

26166 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 1391 | 0 | 1735129 |
| `aten::copy_` | 751 | 0 | 326477 |
| `aten::bitwise_and` | 1384 | 0 | 181533 |
| `aten::le` | 22 | 0 | 143374 |
| `aten::fill_` | 220 | 0 | 138698 |
| `aten::ge` | 96 | 0 | 98218 |
| `aten::flip` | 1 | 0 | 53620 |
| `aten::amin` | 2 | 0 | 30742 |

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
