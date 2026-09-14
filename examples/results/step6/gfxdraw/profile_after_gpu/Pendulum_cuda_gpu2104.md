# Profile note — Pendulum on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Pendulum.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **25.86 ms** median (min 24.91, max 26.62, sd 0.47) = **39 fps**  ·  set_state 0.16 ms  ·  peak VRAM 14 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 3589 | 54.3 | 1624 | 2537 |
| other | 1557 | 23.6 | 25224 | 7127 |
| kernel/raster | 1463 | 22.1 | 8119 | 917 |
| allocation | 0 | 0.0 | 2802 | 726 |
| synchronization | 0 | 0.0 | 17 | 4 |
| output conversion | 0 | 0.0 | 12 | 9 |

11320 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 2001 | 0 | 11900 |
| `aten::empty` | 404 | 0 | 1588 |
| `aten::empty_strided` | 202 | 0 | 1041 |
| `Unrecognized` | 1 | 0 | 957 |
| `aten::resize_` | 230 | 0 | 687 |
| `aten::bitwise_and` | 360 | 557 | 3371 |
| `aten::_to_copy` | 146 | 0 | 512 |
| `aten::view` | 289 | 0 | 446 |

## batch 16

render **33.35 ms** median (min 32.85, max 33.74, sd 0.25) = **480 fps**  ·  set_state 0.20 ms  ·  peak VRAM 40 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 4417 | 56.4 | 1715 | 2595 |
| other | 1713 | 21.9 | 24703 | 7214 |
| kernel/raster | 1695 | 21.7 | 7709 | 917 |
| allocation | 0 | 0.0 | 2964 | 784 |
| synchronization | 0 | 0.0 | 12 | 4 |
| output conversion | 0 | 0.0 | 11 | 9 |

11523 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 2030 | 0 | 11679 |
| `aten::empty` | 433 | 0 | 1752 |
| `aten::empty_strided` | 202 | 0 | 1002 |
| `Unrecognized` | 1 | 0 | 996 |
| `aten::resize_` | 230 | 0 | 703 |
| `aten::bitwise_and` | 360 | 592 | 3312 |
| `aten::copy_` | 188 | 514 | 1048 |
| `aten::_to_copy` | 146 | 0 | 462 |

## batch 64

render **34.12 ms** median (min 33.53, max 39.34, sd 1.61) = **1875 fps**  ·  set_state 0.19 ms  ·  peak VRAM 120 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 6004 | 60.1 | 1752 | 2595 |
| other | 2012 | 20.1 | 24170 | 7214 |
| kernel/raster | 1980 | 19.8 | 7786 | 917 |
| allocation | 0 | 0.0 | 3020 | 784 |
| synchronization | 0 | 0.0 | 422 | 4 |
| output conversion | 0 | 0.0 | 10 | 9 |

11523 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 2030 | 0 | 11781 |
| `aten::empty` | 433 | 0 | 1768 |
| `aten::empty_strided` | 202 | 0 | 1047 |
| `aten::copy_` | 188 | 1017 | 1074 |
| `aten::resize_` | 230 | 0 | 708 |
| `aten::bitwise_and` | 360 | 608 | 3316 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 8 | 529 | 0 |
| `aten::where` | 254 | 493 | 1957 |

## batch 256

render **35.81 ms** median (min 35.59, max 36.22, sd 0.22) = **7149 fps**  ·  set_state 0.20 ms  ·  peak VRAM 442 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 11853 | 66.1 | 1735 | 2595 |
| other | 3083 | 17.2 | 24891 | 7214 |
| kernel/raster | 2996 | 16.7 | 7878 | 917 |
| allocation | 0 | 0.0 | 3236 | 784 |
| synchronization | 0 | 0.0 | 2094 | 4 |
| output conversion | 0 | 0.0 | 11 | 9 |

11523 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 2030 | 0 | 12000 |
| `aten::copy_` | 188 | 2898 | 1062 |
| `cudaDeviceSynchronize` | 2 | 0 | 2090 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 8 | 2081 | 0 |
| `aten::empty` | 433 | 0 | 1960 |
| `aten::empty_strided` | 202 | 0 | 1072 |
| `aten::flip` | 3 | 882 | 24 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1}>(long, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1})` | 3 | 882 | 0 |

## batch 1024

render **45.93 ms** median (min 45.51, max 46.34, sd 0.25) = **22293 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1721 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 33161 | 70.8 | 1742 | 2595 |
| other | 6918 | 14.8 | 25017 | 7214 |
| kernel/raster | 6761 | 14.4 | 7821 | 917 |
| allocation | 0 | 0.0 | 2997 | 784 |
| synchronization | 0 | 0.0 | 11514 | 4 |
| output conversion | 0 | 0.0 | 12 | 9 |

11523 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 2030 | 0 | 11614 |
| `cudaDeviceSynchronize` | 2 | 0 | 11510 |
| `aten::copy_` | 188 | 9751 | 1062 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 8 | 7978 | 0 |
| `aten::flip` | 3 | 3154 | 25 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1}>(long, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1})` | 3 | 3154 | 0 |
| `aten::where` | 254 | 2025 | 1956 |
| `aten::empty` | 433 | 0 | 1750 |

## batch 4096

render **93.67 ms** median (min 93.26, max 94.05, sd 0.22) = **43729 fps**  ·  set_state 0.22 ms  ·  peak VRAM 6849 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 124219 | 71.6 | 1752 | 2602 |
| other | 24991 | 14.4 | 24724 | 7221 |
| kernel/raster | 24328 | 14.0 | 7855 | 917 |
| allocation | 0 | 0.0 | 3026 | 784 |
| synchronization | 0 | 0.0 | 57938 | 4 |
| output conversion | 0 | 0.0 | 11 | 9 |

11537 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 57932 |
| `aten::copy_` | 188 | 37460 | 1082 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 13 | 31444 | 0 |
| `aten::flip` | 3 | 12916 | 24 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1}>(long, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1})` | 3 | 12916 | 0 |
| `cudaLaunchKernel` | 2037 | 0 | 11497 |
| `aten::where` | 254 | 7324 | 1970 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int)#1})` | 2 | 6029 | 0 |

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
