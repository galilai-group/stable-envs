# Stage profile — FetchPickAndPlace on cuda (NVIDIA GeForce RTX 3090)

- workstream: `step6/fetch-gpu`  ·  baseline ``
- host gpu2116.oscar.ccv.brown.edu  ·  slurm 6343470  ·  82 SMs  ·  25.3 GB
- frame 480px  ·  MJW_CHUNK 256  ·  mujoco 3.11.0 / mujoco-warp 3.11.0 / warp 1.17.0
- model: {'dynamic_state_bytes_per_chunk': 100, 'static_appearance_bytes_per_chunk': 616, 'nq': 22, 'nmat': 36, 'nlight': 1, 'ngeom': 25, 'nbody': 34}

## headline

| B | render ms | fps | us/frame | set_state ms | chunks | kernels | peak VRAM MiB |
|---|---|---|---|---|---|---|---|
| 1 | 1.859 | 538 | 1859.1 | 0.142 | 1 | 19 | 4 |
| 16 | 5.063 | 3160 | 316.4 | 0.167 | 1 | 27 | 40 |
| 64 | 13.670 | 4682 | 213.6 | 0.168 | 1 | 27 | 156 |
| 256 | 50.313 | 5088 | 196.5 | 0.183 | 1 | 27 | 621 |
| 1024 | 197.732 | 5179 | 193.1 | 0.254 | 4 | 102 | 1355 |
| 4096 | 794.187 | 5157 | 193.9 | 0.321 | 16 | 402 | 3380 |

## wall partition (ms, median; sums to the whole render)

| stage | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| `warp_sync` | 0.349 | 3.317 | 11.257 | 44.969 | 181.093 | 724.595 |
| `torch_sync` | 0.023 | 0.025 | 0.025 | 0.025 | 8.553 | 42.777 |
| `gap:enter->plate` | 0.322 | 0.362 | 0.409 | 0.680 | 2.037 | 8.856 |
| `upload` | 0.386 | 0.448 | 0.446 | 0.456 | 1.765 | 6.900 |
| `gap:warp_sync->context` | 0.000 | 0.000 | 0.000 | 0.000 | 0.754 | 3.736 |
| `camera_batch` | 0.346 | 0.393 | 0.414 | 0.511 | 0.988 | 3.328 |
| `tail_drain(after return)` | 0.023 | 0.067 | 0.679 | 3.228 | 3.229 | 3.238 |
| `gap:warp_sync->return` | 0.216 | 0.263 | 0.264 | 0.268 | 0.278 | 0.272 |
| `graph_launch` | 0.015 | 0.021 | 0.017 | 0.017 | 0.066 | 0.264 |
| `gap:plate->camera_batch` | 0.080 | 0.094 | 0.095 | 0.095 | 0.098 | 0.103 |
| `gap:context->upload` | 0.010 | 0.011 | 0.011 | 0.011 | 0.028 | 0.097 |
| `context` | 0.004 | 0.003 | 0.004 | 0.003 | 0.016 | 0.063 |
| `plate` | 0.057 | 0.056 | 0.057 | 0.056 | 0.058 | 0.059 |
| `gap:upload->torch_sync` | 0.002 | 0.002 | 0.002 | 0.002 | 0.007 | 0.028 |
| `gap:torch_sync->graph_launch` | 0.001 | 0.001 | 0.001 | 0.001 | 0.004 | 0.014 |
| `gap:graph_launch->warp_sync` | 0.001 | 0.001 | 0.001 | 0.001 | 0.003 | 0.011 |
| `gap:camera_batch->context` | 0.005 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |
| **total** | 1.843 | 5.073 | 13.689 | 50.349 | 198.992 | 794.390 |

## device partition (us, one profiled render)

| bucket | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| kernel_other | 6 (3) | 40 (3) | 138 (3) | 535 (3) | 2140 (12) | 8561 (48) |
| mjwarp_raytrace_megakernel | 294 (1) | 3233 (1) | 11236 (1) | 44622 (1) | 179882 (4) | 719657 (16) |
| mjwarp_scene_update | 55 (7) | 65 (7) | 59 (7) | 63 (7) | 255 (28) | 1017 (112) |
| torch_composite_output | 34 (8) | 242 (16) | 791 (16) | 3069 (16) | 12319 (58) | 49342 (226) |

## memcpy per render (count, bytes)

| direction | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| Memcpy DtoD (Device -> Device) | 11 / 1382588B | 3 / 11750592B | 3 / 44928768B | 3 / 177641472B | 9 / 708492288B | 33 / 2831895552B |
| Memcpy DtoH (Device -> Pageable) | 2 / 52B | 2 / 832B | 2 / 3328B | 2 / 13312B | 2 / 53248B | 2 / 212992B |
| Memcpy HtoD (Pageable -> Device) | 4 / 60B | 4 / 600B | 4 / 2328B | 4 / 9240B | 7 / 36960B | 19 / 147840B |
| Memset (Device) | 0 / 0B | 0 / 0B | 1 / 10748B | 1 / 43004B | 4 / 172016B | 16 / 688064B |
| Memset (Unknown) | 1 / 164B | 1 / 2684B | 0 / 0B | 0 / 0B | 0 / 0B | 0 / 0B |

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

## largest bottleneck

_filled in findings.md_

