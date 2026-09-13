# Profile note — SimplePointMaze on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2340.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/SimplePointMaze.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **298.57 ms** median (min 286.89, max 309.25, sd 6.71) = **3 fps**  ·  set_state 0.12 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 76982 | 6390 |
| host->device | 0 | - | 3467 | 723 |
| allocation | 0 | - | 2025 | 815 |
| kernel/raster | 0 | - | 286567 | 2255 |
| synchronization | 0 | - | 4 | 2 |
| output conversion | 0 | - | 250 | 3 |

10188 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 182 | 0 | 96091 |
| `aten::sub` | 651 | 0 | 79894 |
| `aten::mul` | 764 | 0 | 62215 |
| `aten::add` | 282 | 0 | 34410 |
| `aten::bitwise_and` | 91 | 0 | 21427 |
| `aten::le` | 65 | 0 | 14479 |
| `aten::fill_` | 152 | 0 | 9931 |
| `aten::maximum` | 52 | 0 | 9531 |

## batch 16

render **6560.07 ms** median (min 6183.71, max 7144.52, sd 312.15) = **2 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1173354 | 6666 |
| host->device | 0 | - | 25831 | 771 |
| allocation | 0 | - | 4451 | 911 |
| kernel/raster | 0 | - | 6059149 | 2255 |
| synchronization | 0 | - | 3 | 2 |
| output conversion | 0 | - | 4020 | 51 |

10656 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 182 | 0 | 1813587 |
| `aten::sub` | 651 | 0 | 1547564 |
| `aten::mul` | 764 | 0 | 1493719 |
| `aten::add` | 282 | 0 | 843359 |
| `aten::bitwise_and` | 91 | 0 | 253855 |
| `aten::fill_` | 152 | 0 | 219789 |
| `aten::maximum` | 52 | 0 | 217299 |
| `aten::minimum` | 52 | 0 | 197375 |

## batch 64

render **72409.95 ms** median (min 70838.84, max 73392.82, sd 799.59) = **1 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 9665917 | 6666 |
| host->device | 0 | - | 165079 | 771 |
| allocation | 0 | - | 7200 | 911 |
| kernel/raster | 0 | - | 54736794 | 2255 |
| synchronization | 0 | - | 6 | 2 |
| output conversion | 0 | - | 17179 | 51 |

10656 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::sub` | 651 | 0 | 16507652 |
| `aten::mul` | 764 | 0 | 15626118 |
| `aten::where` | 182 | 0 | 11793759 |
| `aten::add` | 282 | 0 | 6838853 |
| `aten::fill_` | 152 | 0 | 3304658 |
| `aten::clamp` | 65 | 0 | 2008115 |
| `aten::div` | 65 | 0 | 1942873 |
| `aten::maximum` | 52 | 0 | 1809354 |

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
