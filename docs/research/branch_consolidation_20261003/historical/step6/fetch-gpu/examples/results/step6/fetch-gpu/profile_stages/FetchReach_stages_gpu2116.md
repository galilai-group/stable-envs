# Stage profile — FetchReach on cuda (NVIDIA GeForce RTX 3090)

- workstream: `step6/fetch-gpu`  ·  baseline ``
- host gpu2116.oscar.ccv.brown.edu  ·  slurm 6343470  ·  82 SMs  ·  25.3 GB
- frame 480px  ·  MJW_CHUNK 256  ·  mujoco 3.11.0 / mujoco-warp 3.11.0 / warp 1.17.0
- model: {'dynamic_state_bytes_per_chunk': 72, 'static_appearance_bytes_per_chunk': 600, 'nq': 15, 'nmat': 35, 'nlight': 1, 'ngeom': 24, 'nbody': 33}

## headline

| B | render ms | fps | us/frame | set_state ms | chunks | kernels | peak VRAM MiB |
|---|---|---|---|---|---|---|---|
| 1 | 1.824 | 548 | 1823.7 | 0.143 | 1 | 19 | 4 |
| 16 | 5.047 | 3170 | 315.4 | 0.167 | 1 | 26 | 40 |
| 64 | 13.543 | 4726 | 211.6 | 0.167 | 1 | 26 | 156 |
| 256 | 49.749 | 5146 | 194.3 | 0.178 | 1 | 26 | 621 |
| 1024 | 195.747 | 5231 | 191.2 | 0.231 | 4 | 98 | 1355 |
| 4096 | 786.311 | 5209 | 192.0 | 0.292 | 16 | 385 | 3379 |

## wall partition (ms, median; sums to the whole render)

| stage | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| `warp_sync` | 0.347 | 3.319 | 11.149 | 44.488 | 177.961 | 716.805 |
| `torch_sync` | 0.023 | 0.025 | 0.025 | 0.025 | 8.606 | 43.162 |
| `gap:enter->plate` | 0.318 | 0.367 | 0.407 | 0.670 | 2.005 | 8.687 |
| `upload` | 0.348 | 0.416 | 0.408 | 0.419 | 1.615 | 6.399 |
| `gap:warp_sync->context` | 0.000 | 0.000 | 0.000 | 0.000 | 0.761 | 3.744 |
| `camera_batch` | 0.353 | 0.391 | 0.415 | 0.514 | 0.995 | 3.378 |
| `tail_drain(after return)` | 0.023 | 0.068 | 0.679 | 3.224 | 3.220 | 3.234 |
| `gap:warp_sync->return` | 0.217 | 0.260 | 0.262 | 0.270 | 0.275 | 0.274 |
| `graph_launch` | 0.015 | 0.021 | 0.017 | 0.017 | 0.066 | 0.261 |
| `gap:plate->camera_batch` | 0.080 | 0.095 | 0.095 | 0.097 | 0.100 | 0.105 |
| `gap:context->upload` | 0.009 | 0.011 | 0.011 | 0.011 | 0.028 | 0.098 |
| `context` | 0.004 | 0.004 | 0.004 | 0.004 | 0.016 | 0.066 |
| `plate` | 0.057 | 0.055 | 0.056 | 0.056 | 0.057 | 0.058 |
| `gap:upload->torch_sync` | 0.002 | 0.002 | 0.002 | 0.002 | 0.007 | 0.028 |
| `gap:torch_sync->graph_launch` | 0.001 | 0.001 | 0.001 | 0.001 | 0.004 | 0.015 |
| `gap:graph_launch->warp_sync` | 0.001 | 0.001 | 0.001 | 0.001 | 0.003 | 0.011 |
| `gap:camera_batch->context` | 0.005 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |
| **total** | 1.823 | 5.050 | 13.543 | 49.810 | 195.740 | 786.384 |

## device partition (us, one profiled render)

| bucket | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| kernel_other | 6 (3) | 40 (3) | 138 (3) | 535 (3) | 2140 (12) | 8561 (48) |
| mjwarp_raytrace_megakernel | 287 (1) | 3251 (1) | 11040 (1) | 44194 (1) | 176687 (4) | 711856 (16) |
| mjwarp_scene_update | 53 (7) | 61 (7) | 58 (7) | 63 (7) | 251 (28) | 1006 (112) |
| torch_composite_output | 35 (8) | 241 (15) | 786 (15) | 3065 (15) | 12266 (54) | 49285 (209) |

## memcpy per render (count, bytes)

| direction | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| Memcpy DtoD (Device -> Device) | 10 / 1382548B | 3 / 11750592B | 3 / 44928768B | 3 / 177641472B | 9 / 708492288B | 33 / 2831895552B |
| Memcpy DtoH (Device -> Pageable) | 2 / 52B | 2 / 832B | 2 / 3328B | 2 / 13312B | 2 / 53248B | 1 / 49152B |
| Memcpy HtoD (Pageable -> Device) | 4 / 60B | 4 / 600B | 4 / 2328B | 4 / 9240B | 7 / 36960B | 19 / 147840B |
| Memset (Device) | 0 / 0B | 0 / 0B | 1 / 10236B | 1 / 40956B | 4 / 163824B | 16 / 655296B |
| Memset (Unknown) | 1 / 156B | 1 / 2556B | 0 / 0B | 0 / 0B | 0 / 0B | 0 / 0B |

## host<->device sync points per render

| call | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| `Tensor.cpu` | 2 | 2 | 2 | 2 | 2 | 2 |
| `Tensor.numpy` | 2 | 2 | 2 | 2 | 2 | 2 |
| `_context` | 1 | 1 | 1 | 1 | 4 | 16 |
| `_upload` | 1 | 1 | 1 | 1 | 4 | 16 |
| `torch.cuda.synchronize` | 1 | 1 | 1 | 1 | 4 | 16 |
| `warp.capture_launch` | 1 | 1 | 1 | 1 | 4 | 16 |
| `warp.synchronize` | 1 | 1 | 1 | 1 | 4 | 16 |

## resolution sweep — the same code path at several frame sizes

| size | fps(B=1) | fps(B=256) | fps(B=4096) | fps(max)/fps(1) |
|---|---|---|---|---|
| 84px | 531 | 46200 | 52425 | 98.6x |
| 120px | 540 | 31476 | 34219 | 63.4x |
| 240px | 540 | 13355 | 14067 | 26.1x |
| 480px | 513 | 5106 | 5209 | 10.2x |

## largest bottleneck

_filled in findings.md_

