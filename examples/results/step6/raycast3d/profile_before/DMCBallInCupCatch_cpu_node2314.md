# Profile note — DMCBallInCupCatch on cpu

- workstream: `step6/raycast3d`
- baseline commit: `d148f1a582d9`
- host: node2314.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 84px  ·  state_dim: 4
- states: `examples/results/step4/states/DMCBallInCupCatch.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **178.86 ms** median (min 178.16, max 179.80, sd 0.46) = **6 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 76990 | 85440 |
| host->device | 0 | - | 10340 | 10669 |
| allocation | 0 | - | 8595 | 8223 |
| kernel/raster | 0 | - | 76361 | 19511 |
| synchronization | 0 | - | 7 | 4 |
| output conversion | 0 | - | 43 | 1 |

123848 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8392 | 0 | 28019 |
| `aten::select` | 17417 | 0 | 24980 |
| `aten::add` | 5669 | 0 | 12950 |
| `aten::cat` | 797 | 0 | 12639 |
| `aten::gt` | 1956 | 0 | 10747 |
| `aten::as_strided` | 21302 | 0 | 9557 |
| `aten::sub` | 1377 | 0 | 7136 |
| `aten::where` | 1043 | 0 | 6289 |

## batch 16

render **729.55 ms** median (min 713.27, max 744.32, sd 9.07) = **22 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 224219 | 85455 |
| host->device | 0 | - | 12389 | 10669 |
| allocation | 0 | - | 10890 | 8224 |
| kernel/raster | 0 | - | 485859 | 19511 |
| synchronization | 0 | - | 6 | 4 |
| output conversion | 0 | - | 637 | 1 |

123864 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8392 | 0 | 160471 |
| `aten::cat` | 797 | 0 | 143406 |
| `aten::add` | 5669 | 0 | 62437 |
| `aten::sqrt` | 550 | 0 | 50456 |
| `aten::gt` | 1956 | 0 | 48715 |
| `aten::sub` | 1377 | 0 | 40844 |
| `aten::where` | 1043 | 0 | 40378 |
| `aten::select` | 17417 | 0 | 26919 |

## batch 64

render **3451.99 ms** median (min 3428.22, max 3525.00, sd 28.49) = **19 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 841241 | 86518 |
| host->device | 0 | - | 16862 | 10669 |
| allocation | 0 | - | 13341 | 8224 |
| kernel/raster | 0 | - | 2575481 | 19511 |
| synchronization | 0 | - | 7 | 4 |
| output conversion | 0 | - | 1946 | 1 |

124927 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8392 | 0 | 831738 |
| `aten::cat` | 797 | 0 | 714664 |
| `aten::add` | 5669 | 0 | 431441 |
| `aten::sub` | 1377 | 0 | 226699 |
| `aten::sqrt` | 550 | 0 | 221950 |
| `aten::gt` | 1956 | 0 | 205306 |
| `aten::where` | 1043 | 0 | 200462 |
| `aten::div` | 423 | 0 | 98947 |

## batch 256

render **14241.89 ms** median (min 13927.79, max 14605.46, sd 211.55) = **18 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 3321222 | 86519 |
| host->device | 0 | - | 28443 | 10669 |
| allocation | 0 | - | 20940 | 8224 |
| kernel/raster | 0 | - | 11295232 | 19511 |
| synchronization | 0 | - | 6 | 4 |
| output conversion | 0 | - | 6298 | 1 |

124928 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 8392 | 0 | 3891310 |
| `aten::cat` | 797 | 0 | 2685093 |
| `aten::add` | 5669 | 0 | 2193436 |
| `aten::sub` | 1377 | 0 | 1022742 |
| `aten::sqrt` | 550 | 0 | 860874 |
| `aten::where` | 1043 | 0 | 806594 |
| `aten::gt` | 1956 | 0 | 783376 |
| `aten::div` | 423 | 0 | 391602 |

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
