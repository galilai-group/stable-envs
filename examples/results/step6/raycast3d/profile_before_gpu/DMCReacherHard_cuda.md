# Profile note — DMCReacherHard on cuda

- workstream: `step6/raycast3d`
- baseline commit: `d148f1a582d9`
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 8
- frame: 84px  ·  state_dim: 4
- states: `examples/results/step4/states/DMCReacherHard.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **6.97 ms** median (min 6.82, max 11.82, sd 1.57) = **144 fps**  ·  set_state 0.24 ms  ·  peak VRAM 2 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 767 | 50.6 | 359 | 514 |
| kernel/raster | 567 | 37.4 | 4475 | 321 |
| other | 180 | 11.9 | 13023 | 1691 |
| output conversion | 1 | 0.1 | 19 | 3 |
| allocation | 0 | 0.0 | 445 | 74 |
| synchronization | 0 | 0.0 | 59 | 17 |

2620 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 397 | 0 | 3510 |
| `aten::set_` | 2 | 0 | 2655 |
| `Unrecognized` | 1 | 0 | 1999 |
| `aten::select` | 159 | 0 | 529 |
| `aten::as_strided` | 280 | 0 | 264 |
| `aten::empty` | 34 | 0 | 223 |
| `aten::mul` | 126 | 199 | 1679 |
| `aten::empty_strided` | 26 | 0 | 174 |

## batch 16

render **8.97 ms** median (min 8.81, max 9.07, sd 0.08) = **1783 fps**  ·  set_state 0.13 ms  ·  peak VRAM 11 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 901 | 50.0 | 257 | 513 |
| kernel/raster | 684 | 38.0 | 2958 | 324 |
| other | 214 | 11.9 | 5154 | 1695 |
| output conversion | 2 | 0.1 | 11 | 3 |
| allocation | 0 | 0.0 | 312 | 75 |
| synchronization | 0 | 0.0 | 47 | 17 |

2627 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 399 | 0 | 2374 |
| `Unrecognized` | 1 | 0 | 852 |
| `aten::select` | 159 | 0 | 341 |
| `aten::mul` | 126 | 227 | 1110 |
| `aten::as_strided` | 280 | 0 | 183 |
| `aten::empty` | 34 | 0 | 163 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 78 | 155 | 0 |
| `aten::index` | 28 | 125 | 360 |

## batch 64

render **9.45 ms** median (min 9.36, max 10.35, sd 0.30) = **6769 fps**  ·  set_state 0.13 ms  ·  peak VRAM 50 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 1452 | 49.3 | 252 | 513 |
| kernel/raster | 1147 | 38.9 | 2881 | 324 |
| other | 334 | 11.3 | 4514 | 1695 |
| output conversion | 12 | 0.4 | 10 | 3 |
| allocation | 0 | 0.0 | 310 | 75 |
| synchronization | 0 | 0.0 | 48 | 17 |

2627 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 399 | 0 | 2345 |
| `aten::mul` | 126 | 395 | 1078 |
| `aten::select` | 159 | 0 | 330 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 78 | 286 | 0 |
| `Unrecognized` | 1 | 0 | 282 |
| `aten::add` | 69 | 222 | 551 |
| `aten::as_strided` | 280 | 0 | 180 |
| `aten::index` | 28 | 179 | 351 |

## batch 256

render **12.88 ms** median (min 12.82, max 12.89, sd 0.02) = **19882 fps**  ·  set_state 0.14 ms  ·  peak VRAM 186 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 4442 | 49.2 | 268 | 513 |
| kernel/raster | 3729 | 41.3 | 2945 | 324 |
| other | 804 | 8.9 | 5523 | 1695 |
| output conversion | 53 | 0.6 | 11 | 3 |
| allocation | 0 | 0.0 | 307 | 75 |
| synchronization | 0 | 0.0 | 356 | 17 |

2627 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 399 | 0 | 2419 |
| `aten::mul` | 126 | 1460 | 1068 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 78 | 1095 | 0 |
| `Unrecognized` | 1 | 0 | 829 |
| `aten::add` | 69 | 819 | 546 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 54 | 671 | 0 |
| `aten::index` | 28 | 453 | 378 |
| `cudaMemcpyAsync` | 11 | 0 | 418 |

## batch 1024

render **42.39 ms** median (min 41.53, max 47.61, sd 2.19) = **24155 fps**  ·  set_state 0.15 ms  ·  peak VRAM 731 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 16328 | 49.2 | 13345 | 513 |
| kernel/raster | 13977 | 42.1 | 2935 | 324 |
| other | 2665 | 8.0 | 5476 | 1697 |
| output conversion | 206 | 0.6 | 11 | 3 |
| allocation | 0 | 0.0 | 324 | 75 |
| synchronization | 0 | 0.0 | 9003 | 17 |

2629 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaStreamSynchronize` | 11 | 0 | 5829 |
| `aten::mul` | 126 | 5454 | 1076 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 78 | 3950 | 0 |
| `aten::add` | 69 | 3345 | 553 |
| `cudaDeviceSynchronize` | 2 | 0 | 3168 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 54 | 2812 | 0 |
| `cudaLaunchKernel` | 399 | 0 | 2432 |
| `aten::index` | 28 | 1465 | 367 |

## batch 4096

render **162.88 ms** median (min 151.80, max 165.62, sd 3.82) = **25147 fps**  ·  set_state 0.18 ms  ·  peak VRAM 1732 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 63272 | 49.2 | 21535 | 1008 |
| kernel/raster | 54316 | 42.2 | 6020 | 645 |
| other | 10244 | 8.0 | 20597 | 3323 |
| output conversion | 822 | 0.6 | 23 | 6 |
| allocation | 0 | 0.0 | 628 | 144 |
| synchronization | 0 | 0.0 | 39019 | 28 |

5154 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaStreamSynchronize` | 19 | 0 | 31825 |
| `aten::mul` | 251 | 21142 | 2249 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 156 | 15274 | 0 |
| `aten::add` | 138 | 13180 | 1154 |
| `cudaMemcpyAsync` | 19 | 0 | 11570 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 108 | 11131 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 7185 |
| `aten::index` | 56 | 5559 | 732 |

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
