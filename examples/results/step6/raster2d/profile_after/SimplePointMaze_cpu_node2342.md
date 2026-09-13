# Profile note — SimplePointMaze on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2342.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/SimplePointMaze.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **10.97 ms** median (min 10.67, max 11.65, sd 0.31) = **91 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 3617 | 3983 |
| host->device | 0 | - | 1714 | 596 |
| allocation | 0 | - | 426 | 328 |
| kernel/raster | 0 | - | 3991 | 905 |
| synchronization | 0 | - | 4 | 2 |
| output conversion | 0 | - | 84 | 25 |

5839 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 162 | 0 | 1315 |
| `aten::where` | 41 | 0 | 1197 |
| `aten::mul` | 307 | 0 | 894 |
| `aten::sub` | 213 | 0 | 880 |
| `aten::select` | 399 | 0 | 591 |
| `aten::add` | 127 | 0 | 476 |
| `aten::view` | 363 | 0 | 406 |
| `aten::as_strided` | 768 | 0 | 330 |

## batch 16

render **134.64 ms** median (min 132.09, max 137.31, sd 1.58) = **119 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 29284 | 4250 |
| host->device | 0 | - | 19371 | 644 |
| allocation | 0 | - | 600 | 424 |
| kernel/raster | 0 | - | 78229 | 905 |
| synchronization | 0 | - | 3 | 2 |
| output conversion | 0 | - | 208 | 73 |

6298 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 41 | 0 | 52800 |
| `aten::copy_` | 210 | 0 | 18853 |
| `aten::sub` | 213 | 0 | 12400 |
| `aten::pow` | 3 | 0 | 11458 |
| `aten::add` | 127 | 0 | 6321 |
| `aten::mul` | 307 | 0 | 5969 |
| `aten::bitwise_and` | 14 | 0 | 5924 |
| `aten::le` | 12 | 0 | 5119 |

## batch 64

render **535.95 ms** median (min 518.95, max 597.59, sd 20.92) = **119 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 95019 | 4250 |
| host->device | 0 | - | 80682 | 644 |
| allocation | 0 | - | 712 | 424 |
| kernel/raster | 0 | - | 344885 | 905 |
| synchronization | 0 | - | 4 | 2 |
| output conversion | 0 | - | 314 | 73 |

6298 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 41 | 0 | 211978 |
| `aten::copy_` | 210 | 0 | 79927 |
| `aten::sub` | 213 | 0 | 58746 |
| `aten::add` | 127 | 0 | 42231 |
| `aten::pow` | 3 | 0 | 40070 |
| `aten::mul` | 307 | 0 | 31065 |
| `aten::bitwise_and` | 14 | 0 | 19541 |
| `aten::le` | 12 | 0 | 15457 |

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
