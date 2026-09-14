# Profile note — Piecewise on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2342.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/Piecewise.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **2.68 ms** median (min 2.55, max 2.93, sd 0.13) = **373 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 510 | 110 |
| host->device | 0 | - | 538 | 39 |
| allocation | 0 | - | 42 | 17 |
| kernel/raster | 0 | - | 3025 | 19 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 1453 |
| `aten::where` | 2 | 0 | 633 |
| `aten::copy_` | 13 | 0 | 484 |
| `aten::add` | 2 | 0 | 421 |
| `aten::sub` | 3 | 0 | 301 |
| `aten::exp` | 1 | 0 | 210 |
| `aten::div` | 2 | 0 | 175 |
| `aten::_to_copy` | 11 | 0 | 40 |

## batch 16

render **36.39 ms** median (min 35.18, max 42.86, sd 2.18) = **440 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 2217 | 110 |
| host->device | 0 | - | 7829 | 39 |
| allocation | 0 | - | 59 | 17 |
| kernel/raster | 0 | - | 29664 | 19 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 12669 |
| `aten::where` | 2 | 0 | 8319 |
| `aten::copy_` | 13 | 0 | 7729 |
| `aten::add` | 2 | 0 | 5630 |
| `aten::sub` | 3 | 0 | 1543 |
| `aten::exp` | 1 | 0 | 1218 |
| `aten::div` | 2 | 0 | 1193 |
| `aten::neg` | 1 | 0 | 472 |

## batch 64

render **182.75 ms** median (min 170.77, max 199.00, sd 10.01) = **350 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 9301 | 110 |
| host->device | 0 | - | 40223 | 39 |
| allocation | 0 | - | 111 | 17 |
| kernel/raster | 0 | - | 138605 | 19 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 64856 |
| `aten::copy_` | 13 | 0 | 40080 |
| `aten::where` | 2 | 0 | 34024 |
| `aten::add` | 2 | 0 | 25433 |
| `aten::sub` | 3 | 0 | 6676 |
| `aten::div` | 2 | 0 | 5318 |
| `aten::exp` | 1 | 0 | 5310 |
| `aten::neg` | 1 | 0 | 2303 |

## batch 256

render **1256.46 ms** median (min 1206.30, max 1290.42, sd 22.72) = **204 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 85464 | 110 |
| host->device | 0 | - | 263509 | 39 |
| allocation | 0 | - | 203 | 17 |
| kernel/raster | 0 | - | 845023 | 19 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

187 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::mul` | 7 | 0 | 353908 |
| `aten::copy_` | 13 | 0 | 263358 |
| `aten::where` | 2 | 0 | 173536 |
| `aten::add` | 2 | 0 | 157146 |
| `aten::sub` | 3 | 0 | 76140 |
| `aten::div` | 2 | 0 | 67090 |
| `aten::exp` | 1 | 0 | 60226 |
| `aten::neg` | 1 | 0 | 18168 |

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
