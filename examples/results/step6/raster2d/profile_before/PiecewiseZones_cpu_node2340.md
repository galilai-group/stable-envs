# Profile note — PiecewiseZones on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2340.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/PiecewiseZones.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **5.13 ms** median (min 4.99, max 5.27, sd 0.09) = **195 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 783 | 170 |
| host->device | 0 | - | 814 | 51 |
| allocation | 0 | - | 62 | 25 |
| kernel/raster | 0 | - | 4899 | 23 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 2697 |
| `aten::mul` | 7 | 0 | 1435 |
| `aten::copy_` | 17 | 0 | 754 |
| `aten::add` | 2 | 0 | 441 |
| `aten::eq` | 5 | 0 | 226 |
| `aten::exp` | 1 | 0 | 193 |
| `aten::sub` | 3 | 0 | 144 |
| `aten::div` | 2 | 0 | 137 |

## batch 16

render **78.17 ms** median (min 67.02, max 83.80, sd 4.43) = **205 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 2368 | 170 |
| host->device | 0 | - | 7606 | 51 |
| allocation | 0 | - | 101 | 25 |
| kernel/raster | 0 | - | 53715 | 23 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 3 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 37261 |
| `aten::mul` | 7 | 0 | 10424 |
| `aten::copy_` | 17 | 0 | 7496 |
| `aten::add` | 2 | 0 | 3035 |
| `aten::sub` | 3 | 0 | 1547 |
| `aten::exp` | 1 | 0 | 1162 |
| `aten::div` | 2 | 0 | 1085 |
| `aten::clamp` | 1 | 0 | 346 |

## batch 64

render **309.52 ms** median (min 289.36, max 355.61, sd 17.74) = **207 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 19662 | 170 |
| host->device | 0 | - | 47365 | 51 |
| allocation | 0 | - | 139 | 25 |
| kernel/raster | 0 | - | 259197 | 23 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 149032 |
| `aten::mul` | 7 | 0 | 70625 |
| `aten::copy_` | 17 | 0 | 47161 |
| `aten::add` | 2 | 0 | 25995 |
| `aten::exp` | 1 | 0 | 15479 |
| `aten::sub` | 3 | 0 | 6413 |
| `aten::div` | 2 | 0 | 4987 |
| `aten::neg` | 1 | 0 | 2178 |

## batch 256

render **1603.61 ms** median (min 1586.13, max 1674.40, sd 31.35) = **160 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 88873 | 170 |
| host->device | 0 | - | 225454 | 51 |
| allocation | 0 | - | 166 | 25 |
| kernel/raster | 0 | - | 1240836 | 23 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 2 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 647804 |
| `aten::mul` | 7 | 0 | 311589 |
| `aten::copy_` | 17 | 0 | 225268 |
| `aten::add` | 2 | 0 | 119836 |
| `aten::sub` | 3 | 0 | 82417 |
| `aten::exp` | 1 | 0 | 59733 |
| `aten::div` | 2 | 0 | 53156 |
| `aten::clamp` | 1 | 0 | 26004 |

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
