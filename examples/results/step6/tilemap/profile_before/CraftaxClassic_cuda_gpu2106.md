# Profile note — CraftaxClassic on cuda

- workstream: `step6/tilemap`
- baseline commit: ``
- host: gpu2106.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- tree measured: `/oscar/scratch/sboughan/se6/baseline/stable_envs/__init__.py`  ·  load average 28.4
- frame: 16px  ·  state_dim: 127
- states: `examples/results/step4/states/CraftaxClassic.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **0.73 ms** median (min 0.72, max 0.76, sd 0.01) = **1366 fps**  ·  set_state 0.11 ms  ·  peak VRAM 3 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 91 | 65.0 | 191 | 69 |
| kernel/raster | 35 | 24.9 | 318 | 20 |
| other | 8 | 6.1 | 6271 | 156 |
| output conversion | 6 | 4.1 | 80 | 7 |
| allocation | 0 | 0.0 | 110 | 15 |
| synchronization | 0 | 0.0 | 32 | 4 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::clone` | 3 | 0 | 2368 |
| `aten::abs` | 2 | 0 | 2120 |
| `Unrecognized` | 1 | 0 | 973 |
| `cudaLaunchKernel` | 36 | 0 | 338 |
| `aten::empty_strided` | 8 | 0 | 75 |
| `cudaMemcpyAsync` | 1 | 0 | 72 |
| `aten::view` | 15 | 0 | 52 |
| `aten::rsub` | 2 | 0 | 48 |

## batch 16

render **0.92 ms** median (min 0.91, max 0.93, sd 0.00) = **17379 fps**  ·  set_state 0.14 ms  ·  peak VRAM 17 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 168 | 65.0 | 119 | 69 |
| kernel/raster | 71 | 27.4 | 195 | 20 |
| other | 10 | 4.0 | 1455 | 156 |
| output conversion | 9 | 3.6 | 49 | 7 |
| allocation | 0 | 0.0 | 65 | 15 |
| synchronization | 0 | 0.0 | 10 | 4 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Unrecognized` | 1 | 0 | 953 |
| `cudaLaunchKernel` | 36 | 0 | 238 |
| `aten::empty_strided` | 8 | 0 | 44 |
| `aten::copy_` | 12 | 39 | 70 |
| `aten::mul` | 8 | 34 | 66 |
| `aten::reshape` | 12 | 0 | 28 |
| `aten::slice` | 9 | 0 | 27 |
| `aten::view` | 13 | 0 | 27 |

## batch 64

render **1.07 ms** median (min 1.06, max 1.08, sd 0.01) = **59897 fps**  ·  set_state 0.16 ms  ·  peak VRAM 62 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 638 | 63.4 | 117 | 69 |
| kernel/raster | 321 | 32.0 | 210 | 20 |
| output conversion | 36 | 3.6 | 48 | 7 |
| other | 10 | 1.0 | 1517 | 156 |
| allocation | 0 | 0.0 | 65 | 15 |
| synchronization | 0 | 0.0 | 82 | 4 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Unrecognized` | 1 | 0 | 980 |
| `cudaLaunchKernel` | 36 | 0 | 241 |
| `aten::mul` | 8 | 160 | 81 |
| `aten::copy_` | 12 | 135 | 71 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1})` | 4 | 97 | 0 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>, false>(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)` | 3 | 92 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 75 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 4 | 68 | 0 |

## batch 256

render **2.25 ms** median (min 2.23, max 2.37, sd 0.03) = **113696 fps**  ·  set_state 0.23 ms  ·  peak VRAM 243 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 2265 | 62.8 | 120 | 69 |
| kernel/raster | 1205 | 33.4 | 203 | 20 |
| output conversion | 125 | 3.5 | 51 | 7 |
| other | 12 | 0.3 | 1172 | 156 |
| allocation | 0 | 0.0 | 66 | 15 |
| synchronization | 0 | 0.0 | 1139 | 4 |

271 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 1127 |
| `aten::mul` | 8 | 604 | 71 |
| `aten::copy_` | 12 | 462 | 71 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1})` | 4 | 360 | 0 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>, false>(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)` | 3 | 358 | 0 |
| `cudaMemcpyAsync` | 1 | 0 | 330 |
| `cudaLaunchKernel` | 36 | 0 | 267 |
| `Unrecognized` | 1 | 0 | 255 |

## batch 1024

render **7.44 ms** median (min 7.42, max 7.62, sd 0.04) = **137605 fps**  ·  set_state 0.32 ms  ·  peak VRAM 954 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 8706 | 62.6 | 127 | 70 |
| kernel/raster | 4714 | 33.9 | 208 | 20 |
| output conversion | 476 | 3.4 | 50 | 7 |
| other | 11 | 0.1 | 1596 | 160 |
| allocation | 0 | 0.0 | 67 | 16 |
| synchronization | 0 | 0.0 | 6108 | 4 |

277 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 6076 |
| `aten::mul` | 8 | 2360 | 73 |
| `aten::copy_` | 13 | 1753 | 75 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1})` | 4 | 1409 | 0 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>, false>(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)` | 3 | 1407 | 0 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 4 | 953 | 0 |
| `aten::add` | 2 | 706 | 17 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 2 | 706 | 0 |

## batch 4096

render **28.51 ms** median (min 28.43, max 28.92, sd 0.12) = **143683 fps**  ·  set_state 0.86 ms  ·  peak VRAM 3790 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 34634 | 62.5 | 142 | 70 |
| kernel/raster | 18870 | 34.1 | 259 | 20 |
| output conversion | 1888 | 3.4 | 58 | 7 |
| other | 13 | 0.0 | 1888 | 160 |
| allocation | 0 | 0.0 | 74 | 16 |
| synchronization | 0 | 0.0 | 26727 | 4 |

277 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 26670 |
| `aten::mul` | 8 | 9439 | 112 |
| `aten::copy_` | 13 | 6931 | 80 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>, false>(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)` | 3 | 5634 | 0 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1} const&)::{lambda(int)#1})` | 4 | 5598 | 0 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 4 | 3805 | 0 |
| `aten::add` | 2 | 2841 | 18 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 2 | 2841 | 0 |

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
