# Profile note — DMCFingerSpin on cuda

- workstream: `step6/raycast3d`
- baseline commit: `d148f1a582d9`
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 8
- frame: 84px  ·  state_dim: 3
- states: `examples/results/step4/states/DMCFingerSpin.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **5.09 ms** median (min 5.06, max 5.13, sd 0.02) = **196 fps**  ·  set_state 0.11 ms  ·  peak VRAM 1 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 500 | 49.7 | 281 | 338 |
| kernel/raster | 326 | 32.3 | 2153 | 253 |
| other | 180 | 17.8 | 4365 | 1220 |
| output conversion | 1 | 0.1 | 14 | 2 |
| allocation | 0 | 0.0 | 326 | 71 |
| synchronization | 0 | 0.0 | 53 | 17 |

1901 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 232 | 0 | 1516 |
| `Unrecognized` | 1 | 0 | 860 |
| `aten::select` | 113 | 0 | 252 |
| `aten::empty` | 34 | 0 | 161 |
| `aten::as_strided` | 217 | 0 | 139 |
| `cudaMemcpyAsync` | 12 | 0 | 135 |
| `aten::empty_strided` | 24 | 0 | 129 |
| `aten::mul` | 97 | 100 | 724 |

## batch 16

render **7.45 ms** median (min 7.40, max 7.49, sd 0.03) = **2148 fps**  ·  set_state 0.13 ms  ·  peak VRAM 9 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 754 | 49.3 | 251 | 432 |
| kernel/raster | 537 | 35.1 | 2372 | 254 |
| other | 236 | 15.4 | 4533 | 1406 |
| output conversion | 2 | 0.2 | 11 | 3 |
| allocation | 0 | 0.0 | 308 | 74 |
| synchronization | 0 | 0.0 | 48 | 17 |

2186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 327 | 0 | 1962 |
| `Unrecognized` | 1 | 0 | 865 |
| `aten::select` | 113 | 0 | 249 |
| `aten::mul` | 97 | 174 | 819 |
| `aten::empty` | 35 | 0 | 163 |
| `aten::as_strided` | 221 | 0 | 140 |
| `aten::index` | 27 | 117 | 429 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 56 | 111 | 0 |

## batch 64

render **8.03 ms** median (min 8.01, max 8.06, sd 0.02) = **7975 fps**  ·  set_state 0.13 ms  ·  peak VRAM 38 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 1038 | 48.1 | 260 | 432 |
| kernel/raster | 750 | 34.8 | 2288 | 254 |
| other | 357 | 16.5 | 3877 | 1406 |
| output conversion | 11 | 0.5 | 11 | 3 |
| allocation | 0 | 0.0 | 306 | 74 |
| synchronization | 0 | 0.0 | 50 | 17 |

2186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 327 | 0 | 1895 |
| `Unrecognized` | 1 | 0 | 276 |
| `aten::select` | 113 | 0 | 276 |
| `aten::mul` | 97 | 259 | 825 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 56 | 178 | 0 |
| `aten::empty` | 35 | 0 | 162 |
| `_tri_cast` | 4 | 151 | 0 |
| `aten::index` | 27 | 148 | 357 |

## batch 256

render **10.89 ms** median (min 10.87, max 11.04, sd 0.05) = **23515 fps**  ·  set_state 0.13 ms  ·  peak VRAM 149 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 2674 | 47.4 | 264 | 432 |
| kernel/raster | 2043 | 36.2 | 2353 | 254 |
| other | 870 | 15.4 | 4938 | 1406 |
| output conversion | 54 | 1.0 | 12 | 3 |
| allocation | 0 | 0.0 | 315 | 74 |
| synchronization | 0 | 0.0 | 333 | 17 |

2186 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 327 | 0 | 1962 |
| `Unrecognized` | 1 | 0 | 931 |
| `aten::mul` | 97 | 803 | 853 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 56 | 579 | 0 |
| `_tri_cast` | 4 | 493 | 0 |
| `aten::add` | 50 | 457 | 406 |
| `cudaMemcpyAsync` | 11 | 0 | 413 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 40 | 410 | 0 |

## batch 1024

render **39.69 ms** median (min 28.18, max 40.09, sd 4.40) = **25800 fps**  ·  set_state 0.14 ms  ·  peak VRAM 596 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 9016 | 46.9 | 439 | 432 |
| kernel/raster | 7102 | 37.0 | 2420 | 254 |
| other | 2894 | 15.1 | 5386 | 1410 |
| output conversion | 207 | 1.1 | 31 | 3 |
| allocation | 0 | 0.0 | 351 | 74 |
| synchronization | 0 | 0.0 | 5603 | 17 |

2190 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaStreamSynchronize` | 11 | 0 | 4496 |
| `aten::mul` | 97 | 2879 | 891 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 56 | 2062 | 0 |
| `cudaLaunchKernel` | 327 | 0 | 2007 |
| `_tri_cast` | 4 | 1858 | 0 |
| `aten::add` | 50 | 1750 | 408 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 40 | 1607 | 0 |
| `Unrecognized` | 1 | 0 | 1122 |

## batch 4096

render **102.75 ms** median (min 102.37, max 105.85, sd 1.00) = **39865 fps**  ·  set_state 0.18 ms  ·  peak VRAM 1418 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 34551 | 47.0 | 755 | 846 |
| kernel/raster | 27415 | 37.3 | 4692 | 505 |
| other | 10714 | 14.6 | 11527 | 2755 |
| output conversion | 823 | 1.1 | 23 | 6 |
| allocation | 0 | 0.0 | 622 | 142 |
| synchronization | 0 | 0.0 | 26422 | 28 |

4282 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaStreamSynchronize` | 19 | 0 | 24387 |
| `aten::mul` | 193 | 11201 | 1667 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 112 | 7992 | 0 |
| `_tri_cast` | 8 | 6970 | 0 |
| `aten::add` | 100 | 6954 | 825 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 80 | 6393 | 0 |
| `cudaLaunchKernel` | 653 | 0 | 4028 |
| `cudaMemcpyAsync` | 19 | 0 | 3665 |

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
