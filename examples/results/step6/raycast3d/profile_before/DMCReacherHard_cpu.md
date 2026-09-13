# Profile note — DMCReacherHard on cpu

- workstream: `step6/raycast3d`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 84px  ·  state_dim: 4
- states: `examples/results/step4/states/DMCReacherHard.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **22.18 ms** median (min 22.08, max 22.38, sd 0.08) = **45 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 7608 | 6972 |
| host->device | 0 | - | 845 | 905 |
| allocation | 0 | - | 744 | 679 |
| kernel/raster | 0 | - | 10348 | 1671 |
| synchronization | 0 | - | 4 | 4 |
| output conversion | 0 | - | 10 | 1 |

10232 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 694 | 0 | 3377 |
| `aten::select` | 1295 | 0 | 1927 |
| `aten::cat` | 53 | 0 | 1671 |
| `aten::add` | 453 | 0 | 1535 |
| `aten::gt` | 144 | 0 | 1046 |
| `aten::where` | 101 | 0 | 973 |
| `aten::sub` | 125 | 0 | 956 |
| `aten::sqrt` | 52 | 0 | 889 |

## batch 16

render **167.54 ms** median (min 164.98, max 175.83, sd 3.61) = **96 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 40666 | 7046 |
| host->device | 0 | - | 1357 | 905 |
| allocation | 0 | - | 961 | 680 |
| kernel/raster | 0 | - | 130552 | 1671 |
| synchronization | 0 | - | 5 | 4 |
| output conversion | 0 | - | 197 | 1 |

10307 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 694 | 0 | 39165 |
| `aten::cat` | 53 | 0 | 27540 |
| `aten::add` | 453 | 0 | 20257 |
| `aten::where` | 101 | 0 | 12457 |
| `aten::sub` | 125 | 0 | 12388 |
| `aten::sqrt` | 52 | 0 | 10568 |
| `aten::gt` | 144 | 0 | 9167 |
| `aten::index` | 28 | 0 | 6774 |

## batch 64

render **773.75 ms** median (min 751.70, max 783.99, sd 9.98) = **83 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 159526 | 7067 |
| host->device | 0 | - | 2780 | 905 |
| allocation | 0 | - | 1377 | 680 |
| kernel/raster | 0 | - | 598216 | 1671 |
| synchronization | 0 | - | 4 | 4 |
| output conversion | 0 | - | 812 | 1 |

10328 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 694 | 0 | 178278 |
| `aten::cat` | 53 | 0 | 115314 |
| `aten::add` | 453 | 0 | 106010 |
| `aten::sub` | 125 | 0 | 59440 |
| `aten::where` | 101 | 0 | 52150 |
| `aten::sqrt` | 52 | 0 | 43458 |
| `aten::gt` | 144 | 0 | 35642 |
| `aten::index` | 28 | 0 | 34044 |

## batch 256

render **3415.85 ms** median (min 3318.10, max 3510.97, sd 53.52) = **75 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 690507 | 7067 |
| host->device | 0 | - | 7739 | 905 |
| allocation | 0 | - | 2237 | 680 |
| kernel/raster | 0 | - | 2697921 | 1671 |
| synchronization | 0 | - | 5 | 4 |
| output conversion | 0 | - | 3343 | 1 |

10328 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 694 | 0 | 918897 |
| `aten::add` | 453 | 0 | 521699 |
| `aten::cat` | 53 | 0 | 430731 |
| `aten::sub` | 125 | 0 | 246930 |
| `aten::where` | 101 | 0 | 219418 |
| `aten::sqrt` | 52 | 0 | 182619 |
| `aten::gt` | 144 | 0 | 147567 |
| `aten::index` | 28 | 0 | 138968 |

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
