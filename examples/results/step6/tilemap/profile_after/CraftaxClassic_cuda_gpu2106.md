# Profile note — CraftaxClassic on cuda

- workstream: `step6/tilemap`
- baseline commit: ``
- host: gpu2106.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- tree measured: `/oscar/scratch/sboughan/se6/tilemap/stable_envs/__init__.py`  ·  load average 28.8
- frame: 16px  ·  state_dim: 127
- states: `examples/results/step4/states/CraftaxClassic.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **0.60 ms** median (min 0.59, max 0.62, sd 0.01) = **1672 fps**  ·  set_state 0.12 ms  ·  peak VRAM 3 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 56 | 59.4 | 117 | 51 |
| other | 19 | 20.1 | 1665 | 142 |
| kernel/raster | 15 | 15.9 | 190 | 11 |
| output conversion | 4 | 4.5 | 44 | 3 |
| allocation | 0 | 0.0 | 117 | 19 |
| synchronization | 0 | 0.0 | 26 | 6 |

232 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Unrecognized` | 1 | 0 | 871 |
| `cudaLaunchKernel` | 27 | 0 | 241 |
| `aten::empty_strided` | 8 | 0 | 68 |
| `cudaMemcpyAsync` | 3 | 0 | 55 |
| `aten::index_select` | 1 | 0 | 54 |
| `aten::abs` | 2 | 0 | 53 |
| `aten::view` | 13 | 0 | 46 |
| `aten::empty` | 6 | 0 | 32 |

## batch 16

render **0.76 ms** median (min 0.74, max 0.91, sd 0.04) = **21058 fps**  ·  set_state 0.14 ms  ·  peak VRAM 4 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 77 | 59.1 | 83 | 51 |
| kernel/raster | 27 | 20.8 | 292 | 11 |
| other | 21 | 15.9 | 1393 | 142 |
| output conversion | 6 | 4.2 | 33 | 3 |
| allocation | 0 | 0.0 | 74 | 19 |
| synchronization | 0 | 0.0 | 17 | 6 |

232 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Unrecognized` | 1 | 0 | 869 |
| `cudaLaunchKernel` | 28 | 0 | 202 |
| `aten::empty_strided` | 8 | 0 | 41 |
| `cudaMemcpyAsync` | 2 | 0 | 28 |
| `aten::view` | 11 | 0 | 25 |
| `aten::to` | 8 | 0 | 24 |
| `aten::empty` | 6 | 0 | 23 |
| `aten::_to_copy` | 6 | 0 | 20 |

## batch 64

render **0.78 ms** median (min 0.76, max 0.79, sd 0.01) = **82227 fps**  ·  set_state 0.16 ms  ·  peak VRAM 6 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 115 | 56.9 | 96 | 51 |
| kernel/raster | 61 | 30.2 | 106 | 11 |
| other | 21 | 10.3 | 1410 | 142 |
| output conversion | 5 | 2.6 | 33 | 3 |
| allocation | 0 | 0.0 | 76 | 19 |
| synchronization | 0 | 0.0 | 18 | 6 |

232 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Unrecognized` | 1 | 0 | 853 |
| `cudaLaunchKernel` | 28 | 0 | 201 |
| `aten::gather` | 1 | 47 | 12 |
| `void at::native::vectorized_gather_kernel<16, int>(char*, char*, int*, int, long, long, long, long, bool)` | 1 | 47 | 0 |
| `aten::empty_strided` | 8 | 0 | 42 |
| `aten::abs` | 2 | 0 | 42 |
| `aten::to` | 8 | 0 | 36 |
| `aten::view` | 11 | 0 | 32 |

## batch 256

render **0.97 ms** median (min 0.97, max 0.99, sd 0.01) = **263400 fps**  ·  set_state 0.23 ms  ·  peak VRAM 16 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 267 | 54.1 | 91 | 51 |
| kernel/raster | 198 | 40.0 | 124 | 11 |
| other | 23 | 4.6 | 1849 | 142 |
| output conversion | 6 | 1.3 | 38 | 3 |
| allocation | 0 | 0.0 | 81 | 19 |
| synchronization | 0 | 0.0 | 141 | 6 |

232 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Unrecognized` | 1 | 0 | 885 |
| `cudaMemcpyAsync` | 2 | 0 | 341 |
| `cudaLaunchKernel` | 28 | 0 | 224 |
| `aten::gather` | 1 | 183 | 15 |
| `void at::native::vectorized_gather_kernel<16, int>(char*, char*, int*, int, long, long, long, long, bool)` | 1 | 183 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 123 |
| `aten::empty_strided` | 8 | 0 | 44 |
| `aten::ne` | 1 | 0 | 35 |

## batch 1024

render **1.62 ms** median (min 1.60, max 1.64, sd 0.01) = **632899 fps**  ·  set_state 0.33 ms  ·  peak VRAM 58 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 863 | 52.6 | 113 | 52 |
| kernel/raster | 746 | 45.5 | 143 | 11 |
| other | 25 | 1.5 | 1277 | 146 |
| output conversion | 8 | 0.5 | 38 | 3 |
| allocation | 0 | 0.0 | 82 | 20 |
| synchronization | 0 | 0.0 | 704 | 6 |

238 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::gather` | 1 | 726 | 15 |
| `void at::native::vectorized_gather_kernel<16, int>(char*, char*, int*, int, long, long, long, long, bool)` | 1 | 726 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 666 |
| `cudaMemcpyAsync` | 2 | 0 | 356 |
| `Unrecognized` | 1 | 0 | 268 |
| `cudaLaunchKernel` | 28 | 0 | 218 |
| `aten::to` | 8 | 0 | 47 |
| `aten::empty_strided` | 8 | 0 | 44 |

## batch 4096

render **4.30 ms** median (min 4.23, max 4.61, sd 0.11) = **953009 fps**  ·  set_state 0.80 ms  ·  peak VRAM 221 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 3300 | 52.4 | 136 | 52 |
| kernel/raster | 2951 | 46.9 | 184 | 11 |
| other | 30 | 0.5 | 2705 | 146 |
| output conversion | 12 | 0.2 | 48 | 3 |
| allocation | 0 | 0.0 | 104 | 20 |
| synchronization | 0 | 0.0 | 2892 | 6 |

238 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::gather` | 1 | 2900 | 19 |
| `void at::native::vectorized_gather_kernel<16, int>(char*, char*, int*, int, long, long, long, long, bool)` | 1 | 2900 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 2828 |
| `Unrecognized` | 1 | 0 | 957 |
| `cudaMemcpyAsync` | 2 | 0 | 508 |
| `cudaLaunchKernel` | 28 | 0 | 240 |
| `aten::copy_` | 8 | 153 | 60 |
| `Memcpy HtoD (Pageable -> Device)` | 1 | 138 | 0 |

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
