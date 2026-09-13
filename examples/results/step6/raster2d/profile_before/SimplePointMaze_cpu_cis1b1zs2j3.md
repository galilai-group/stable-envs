# Profile note — SimplePointMaze on cpu

- workstream: `step6/raster2d`
- baseline commit: `3e35ef9f9b1b`
- host: cis1b1zs2j3.ad.brown.edu  ·  gpu: none  ·  cpu: arm  ·  threads: 1
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/SimplePointMaze.npz`
- torch 2.10.0  ·  triton absent

## batch 1

render **107.50 ms** median (min 107.29, max 107.63, sd 0.12) = **9 fps**  ·  set_state 0.01 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 21518 | 6449 |
| host->device | 0 | - | 776 | 723 |
| allocation | 0 | - | 230 | 815 |
| kernel/raster | 0 | - | 82421 | 2255 |
| synchronization | 0 | - | 2 | 2 |
| output conversion | 0 | - | 138 | 3 |

10247 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 182 | 0 | 30037 |
| `aten::sub` | 651 | 0 | 28923 |
| `aten::mul` | 764 | 0 | 12601 |
| `aten::bitwise_and` | 91 | 0 | 7799 |
| `aten::add` | 282 | 0 | 7303 |
| `aten::le` | 65 | 0 | 6193 |
| `aten::maximum` | 52 | 0 | 1939 |
| `aten::minimum` | 52 | 0 | 1934 |

## batch 64

render **6685.44 ms** median (min 6588.04, max 6990.53, sd 146.61) = **10 fps**  ·  set_state 0.01 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1279253 | 6687 |
| host->device | 0 | - | 35854 | 771 |
| allocation | 0 | - | 1270 | 911 |
| kernel/raster | 0 | - | 5210674 | 2255 |
| synchronization | 0 | - | 4 | 2 |
| output conversion | 0 | - | 8458 | 51 |

10677 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 182 | 0 | 1894346 |
| `aten::sub` | 651 | 0 | 1835251 |
| `aten::mul` | 764 | 0 | 798974 |
| `aten::bitwise_and` | 91 | 0 | 485179 |
| `aten::add` | 282 | 0 | 470907 |
| `aten::le` | 65 | 0 | 387939 |
| `aten::minimum` | 52 | 0 | 122612 |
| `aten::maximum` | 52 | 0 | 122488 |

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
