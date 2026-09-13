# Profile note — DMCBallInCupCatch on cuda

- workstream: `step6/raycast3d`
- baseline commit: `d148f1a582d9`
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 8
- frame: 84px  ·  state_dim: 4
- states: `examples/results/step4/states/DMCBallInCupCatch.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **6.29 ms** median (min 6.27, max 13.41, sd 2.50) = **159 fps**  ·  set_state 0.11 ms  ·  peak VRAM 1 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 660 | 44.4 | 452 | 448 |
| kernel/raster | 465 | 31.3 | 4019 | 263 |
| other | 360 | 24.2 | 7804 | 1453 |
| output conversion | 1 | 0.1 | 20 | 3 |
| allocation | 0 | 0.0 | 569 | 74 |
| synchronization | 0 | 0.0 | 66 | 17 |

2258 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 338 | 0 | 3471 |
| `Unrecognized` | 1 | 0 | 1275 |
| `aten::select` | 113 | 0 | 401 |
| `aten::empty` | 36 | 0 | 288 |
| `aten::as_strided` | 230 | 0 | 245 |
| `aten::empty_strided` | 24 | 0 | 226 |
| `_tri_cast` | 4 | 212 | 0 |
| `cudaMemcpyAsync` | 12 | 0 | 203 |

## batch 16

render **12.01 ms** median (min 11.91, max 153.41, sd 42.42) = **1333 fps**  ·  set_state 0.13 ms  ·  peak VRAM 8 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 769 | 44.7 | 270 | 447 |
| kernel/raster | 550 | 32.0 | 2392 | 264 |
| other | 398 | 23.2 | 4031 | 1451 |
| output conversion | 2 | 0.1 | 11 | 3 |
| allocation | 0 | 0.0 | 324 | 75 |
| synchronization | 0 | 0.0 | 49 | 17 |

2257 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 339 | 0 | 2016 |
| `Unrecognized` | 1 | 0 | 283 |
| `aten::select` | 113 | 0 | 255 |
| `_tri_cast` | 4 | 236 | 0 |
| `aten::empty` | 36 | 0 | 177 |
| `aten::mul` | 100 | 171 | 845 |
| `aten::as_strided` | 230 | 0 | 146 |
| `cudaMemcpyAsync` | 11 | 0 | 118 |

## batch 64

render **26.15 ms** median (min 26.03, max 26.51, sd 0.14) = **2448 fps**  ·  set_state 0.13 ms  ·  peak VRAM 31 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 956 | 41.7 | 285 | 447 |
| other | 665 | 29.0 | 4845 | 1451 |
| kernel/raster | 661 | 28.8 | 2426 | 264 |
| output conversion | 12 | 0.5 | 11 | 3 |
| allocation | 0 | 0.0 | 318 | 75 |
| synchronization | 0 | 0.0 | 183 | 17 |

2257 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 339 | 0 | 2024 |
| `Unrecognized` | 1 | 0 | 1075 |
| `_tri_cast` | 4 | 478 | 0 |
| `aten::select` | 113 | 0 | 249 |
| `aten::mul` | 100 | 200 | 852 |
| `cudaStreamSynchronize` | 11 | 0 | 173 |
| `aten::empty` | 36 | 0 | 168 |
| `aten::as_strided` | 230 | 0 | 145 |

## batch 256

render **83.69 ms** median (min 83.21, max 95.89, sd 4.74) = **3059 fps**  ·  set_state 0.14 ms  ·  peak VRAM 121 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 1948 | 38.1 | 422 | 447 |
| other | 1753 | 34.3 | 5001 | 1453 |
| kernel/raster | 1357 | 26.5 | 2447 | 264 |
| output conversion | 53 | 1.0 | 13 | 3 |
| allocation | 0 | 0.0 | 336 | 75 |
| synchronization | 0 | 0.0 | 997 | 17 |

2259 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 339 | 0 | 2103 |
| `_tri_cast` | 4 | 1468 | 0 |
| `cudaStreamSynchronize` | 11 | 0 | 892 |
| `Unrecognized` | 1 | 0 | 747 |
| `cudaMemcpyAsync` | 11 | 0 | 460 |
| `aten::mul` | 100 | 430 | 839 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 56 | 308 | 0 |
| `aten::select` | 113 | 0 | 252 |

## batch 1024

render **316.71 ms** median (min 313.00, max 364.95, sd 21.65) = **3233 fps**  ·  set_state 0.15 ms  ·  peak VRAM 475 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 6221 | 36.4 | 5350 | 1457 |
| host->device | 6213 | 36.3 | 746 | 447 |
| kernel/raster | 4468 | 26.1 | 2485 | 264 |
| output conversion | 207 | 1.2 | 12 | 3 |
| allocation | 0 | 0.0 | 342 | 75 |
| synchronization | 0 | 0.0 | 6266 | 17 |

2263 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `_tri_cast` | 4 | 5528 | 0 |
| `cudaStreamSynchronize` | 11 | 0 | 5271 |
| `cudaLaunchKernel` | 339 | 0 | 2184 |
| `aten::mul` | 100 | 1479 | 860 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 56 | 1053 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 989 |
| `aten::add` | 53 | 948 | 431 |
| `cudaMemcpyAsync` | 11 | 0 | 833 |

## batch 4096

render **1248.98 ms** median (min 1243.40, max 1258.55, sd 3.76) = **3279 fps**  ·  set_state 0.18 ms  ·  peak VRAM 1133 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 24336 | 36.5 | 12733 | 2849 |
| host->device | 24182 | 36.2 | 1633 | 875 |
| kernel/raster | 17390 | 26.1 | 5174 | 524 |
| output conversion | 823 | 1.2 | 25 | 6 |
| allocation | 0 | 0.0 | 718 | 144 |
| synchronization | 0 | 0.0 | 29049 | 28 |

4426 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaStreamSynchronize` | 19 | 0 | 27226 |
| `_tri_cast` | 8 | 21821 | 0 |
| `aten::mul` | 199 | 5797 | 1776 |
| `cudaLaunchKernel` | 677 | 0 | 4333 |
| `cudaMemcpyAsync` | 19 | 0 | 4184 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 112 | 4087 | 0 |
| `aten::add` | 106 | 3806 | 882 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 80 | 3194 | 0 |

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
