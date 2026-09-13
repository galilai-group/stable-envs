# Profile note — PiecewiseZones on cpu

- workstream: `step6/raster2d`
- baseline commit: ``
- host: node2342.oscar.ccv.brown.edu  ·  gpu: none  ·  cpu: x86_64  ·  threads: 1
- frame: 224px  ·  state_dim: 2
- states: `examples/results/step4/states/PiecewiseZones.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **5.82 ms** median (min 5.68, max 5.91, sd 0.07) = **172 fps**  ·  set_state 0.05 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 865 | 170 |
| host->device | 0 | - | 718 | 51 |
| allocation | 0 | - | 72 | 25 |
| kernel/raster | 0 | - | 4755 | 23 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 2645 |
| `aten::mul` | 7 | 0 | 1353 |
| `aten::copy_` | 17 | 0 | 653 |
| `aten::add` | 2 | 0 | 459 |
| `aten::exp` | 1 | 0 | 261 |
| `aten::eq` | 5 | 0 | 231 |
| `aten::sub` | 3 | 0 | 155 |
| `aten::div` | 2 | 0 | 100 |

## batch 16

render **69.90 ms** median (min 65.83, max 83.86, sd 5.53) = **229 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 2752 | 170 |
| host->device | 0 | - | 8249 | 51 |
| allocation | 0 | - | 100 | 25 |
| kernel/raster | 0 | - | 59556 | 23 |
| synchronization | 0 | - | 2 | 1 |
| output conversion | 0 | - | 2 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 38445 |
| `aten::mul` | 7 | 0 | 12600 |
| `aten::copy_` | 17 | 0 | 8122 |
| `aten::add` | 2 | 0 | 5282 |
| `aten::sub` | 3 | 0 | 1612 |
| `aten::exp` | 1 | 0 | 1345 |
| `aten::div` | 2 | 0 | 1209 |
| `aten::clamp` | 1 | 0 | 391 |

## batch 64

render **305.19 ms** median (min 282.80, max 321.61, sd 10.06) = **210 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 14165 | 170 |
| host->device | 0 | - | 31656 | 51 |
| allocation | 0 | - | 120 | 25 |
| kernel/raster | 0 | - | 238902 | 23 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 149885 |
| `aten::mul` | 7 | 0 | 55601 |
| `aten::copy_` | 17 | 0 | 31480 |
| `aten::add` | 2 | 0 | 19607 |
| `aten::exp` | 1 | 0 | 9914 |
| `aten::sub` | 3 | 0 | 6679 |
| `aten::div` | 2 | 0 | 4951 |
| `aten::neg` | 1 | 0 | 2160 |

## batch 256

render **1740.47 ms** median (min 1713.07, max 1769.41, sd 15.69) = **147 fps**  ·  set_state 0.04 ms

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 0 | - | 87517 | 170 |
| host->device | 0 | - | 269786 | 51 |
| allocation | 0 | - | 219 | 25 |
| kernel/raster | 0 | - | 1331936 | 23 |
| synchronization | 0 | - | 3 | 1 |
| output conversion | 0 | - | 3 | 1 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::where` | 6 | 0 | 656484 |
| `aten::mul` | 7 | 0 | 360026 |
| `aten::copy_` | 17 | 0 | 269580 |
| `aten::add` | 2 | 0 | 166711 |
| `aten::sub` | 3 | 0 | 65398 |
| `aten::exp` | 1 | 0 | 63223 |
| `aten::div` | 2 | 0 | 56446 |
| `aten::clamp` | 1 | 0 | 26837 |

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
