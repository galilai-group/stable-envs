# Profile note — PushT on cpu

- workstream: `step6/raster2d`
- baseline commit: `3e35ef9f9b1b`
- host: cis1b1zs2j3.ad.brown.edu  ·  gpu: none  ·  cpu: arm  ·  threads: 1
- frame: 96px  ·  state_dim: 5
- states: `examples/results/step4/states/PushT.npz`
- torch 2.10.0  ·  triton absent

## batch 1

render **4.96 ms** median (min 4.94, max 4.97, sd 0.01) = **202 fps**  ·  set_state 0.01 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 1942 | 4152 |
| host->device | 0 | - | 264 | 469 |
| allocation | 0 | - | 138 | 562 |
| kernel/raster | 0 | - | 3040 | 1485 |
| synchronization | 0 | - | 1 | 1 |
| output conversion | 0 | - | 22 | 13 |

6682 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::sub` | 450 | 0 | 989 |
| `aten::where` | 126 | 0 | 897 |
| `aten::mul` | 522 | 0 | 609 |
| `aten::add` | 186 | 0 | 289 |
| `aten::bitwise_and` | 63 | 0 | 252 |
| `aten::le` | 45 | 0 | 193 |
| `aten::_to_copy` | 104 | 0 | 157 |
| `aten::view` | 389 | 0 | 146 |

## batch 64

render **172.51 ms** median (min 170.55, max 176.45, sd 1.94) = **371 fps**  ·  set_state 0.01 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 34469 | 4243 |
| host->device | 0 | - | 1560 | 512 |
| allocation | 0 | - | 243 | 648 |
| kernel/raster | 0 | - | 132679 | 1485 |
| synchronization | 0 | - | 1 | 1 |
| output conversion | 0 | - | 329 | 56 |

6945 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 126 | 0 | 49118 |
| `aten::sub` | 450 | 0 | 45439 |
| `aten::mul` | 522 | 0 | 20524 |
| `aten::bitwise_and` | 63 | 0 | 12943 |
| `aten::add` | 186 | 0 | 11906 |
| `aten::le` | 45 | 0 | 9991 |
| `aten::maximum` | 36 | 0 | 3222 |
| `aten::minimum` | 36 | 0 | 3182 |

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
