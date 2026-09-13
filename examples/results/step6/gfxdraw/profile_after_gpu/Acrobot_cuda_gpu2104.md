# Profile note — Acrobot on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Acrobot.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **44.90 ms** median (min 44.57, max 45.75, sd 0.35) = **22 fps**  ·  set_state 0.17 ms  ·  peak VRAM 2 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 6495 | 54.0 | 2756 | 4610 |
| other | 2879 | 23.9 | 43976 | 12885 |
| kernel/raster | 2653 | 22.1 | 14398 | 1674 |
| allocation | 0 | 0.0 | 5102 | 1342 |
| synchronization | 0 | 0.0 | 19 | 4 |
| output conversion | 0 | 0.0 | 10 | 8 |

20523 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3671 | 0 | 20852 |
| `aten::empty` | 745 | 0 | 2881 |
| `aten::empty_strided` | 370 | 0 | 1846 |
| `aten::resize_` | 438 | 0 | 1308 |
| `aten::bitwise_and` | 670 | 1034 | 6241 |
| `Unrecognized` | 1 | 0 | 944 |
| `aten::_to_copy` | 263 | 0 | 845 |
| `aten::__and__` | 670 | 0 | 728 |

## batch 16

render **60.61 ms** median (min 60.08, max 64.67, sd 1.25) = **264 fps**  ·  set_state 0.19 ms  ·  peak VRAM 28 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 7670 | 55.4 | 2963 | 4702 |
| other | 3121 | 22.5 | 43522 | 13023 |
| kernel/raster | 3064 | 22.1 | 13858 | 1674 |
| allocation | 0 | 0.0 | 5363 | 1434 |
| synchronization | 0 | 0.0 | 13 | 4 |
| output conversion | 0 | 0.0 | 8 | 8 |

20845 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3717 | 0 | 20991 |
| `aten::empty` | 791 | 0 | 3124 |
| `aten::empty_strided` | 370 | 0 | 1848 |
| `aten::resize_` | 438 | 0 | 1327 |
| `aten::bitwise_and` | 670 | 1100 | 6169 |
| `Unrecognized` | 1 | 0 | 867 |
| `aten::where` | 486 | 819 | 3698 |
| `aten::_to_copy` | 263 | 0 | 806 |

## batch 64

render **61.52 ms** median (min 60.71, max 61.66, sd 0.26) = **1040 fps**  ·  set_state 0.19 ms  ·  peak VRAM 108 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 9413 | 57.3 | 3016 | 4702 |
| other | 3552 | 21.6 | 45622 | 13023 |
| kernel/raster | 3453 | 21.0 | 14106 | 1674 |
| allocation | 0 | 0.0 | 5459 | 1434 |
| synchronization | 0 | 0.0 | 389 | 4 |
| output conversion | 0 | 0.0 | 8 | 8 |

20845 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3717 | 0 | 22876 |
| `aten::empty` | 791 | 0 | 3162 |
| `aten::empty_strided` | 370 | 0 | 1897 |
| `aten::resize_` | 438 | 0 | 1356 |
| `aten::copy_` | 322 | 1225 | 1815 |
| `aten::bitwise_and` | 670 | 1165 | 6289 |
| `aten::where` | 486 | 1143 | 3725 |
| `aten::_to_copy` | 263 | 0 | 840 |

## batch 256

render **64.00 ms** median (min 63.40, max 64.56, sd 0.29) = **4000 fps**  ·  set_state 0.20 ms  ·  peak VRAM 430 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 15698 | 61.2 | 3047 | 4702 |
| other | 5019 | 19.6 | 45595 | 13023 |
| kernel/raster | 4941 | 19.3 | 14188 | 1674 |
| allocation | 0 | 0.0 | 5495 | 1434 |
| synchronization | 0 | 0.0 | 2091 | 4 |
| output conversion | 0 | 0.0 | 9 | 8 |

20845 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3717 | 0 | 21460 |
| `aten::empty` | 791 | 0 | 3231 |
| `aten::copy_` | 322 | 2891 | 1813 |
| `aten::where` | 486 | 2364 | 3702 |
| `cudaDeviceSynchronize` | 2 | 0 | 2086 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 7 | 2023 | 0 |
| `aten::empty_strided` | 370 | 0 | 1888 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 2 | 1531 | 0 |

## batch 1024

render **75.75 ms** median (min 75.34, max 76.72, sd 0.47) = **13518 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1710 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 37877 | 64.7 | 2990 | 4702 |
| other | 10336 | 17.7 | 43786 | 13023 |
| kernel/raster | 10310 | 17.6 | 14068 | 1674 |
| allocation | 0 | 0.0 | 5450 | 1434 |
| synchronization | 0 | 0.0 | 13477 | 4 |
| output conversion | 0 | 0.0 | 9 | 8 |

20845 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3717 | 0 | 20732 |
| `cudaDeviceSynchronize` | 2 | 0 | 13473 |
| `aten::copy_` | 322 | 8636 | 1814 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 7 | 7789 | 0 |
| `aten::where` | 486 | 6745 | 3676 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 2 | 5741 | 0 |
| `aten::empty` | 791 | 0 | 3176 |
| `aten::flip` | 3 | 3152 | 26 |

## batch 4096

render **123.54 ms** median (min 123.15, max 124.17, sd 0.28) = **33156 fps**  ·  set_state 0.22 ms  ·  peak VRAM 6837 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 137692 | 65.4 | 3025 | 4709 |
| other | 36595 | 17.4 | 44006 | 13030 |
| kernel/raster | 36193 | 17.2 | 14137 | 1674 |
| allocation | 0 | 0.0 | 5499 | 1434 |
| synchronization | 0 | 0.0 | 62021 | 4 |
| output conversion | 0 | 0.0 | 9 | 8 |

20859 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 61987 |
| `aten::copy_` | 322 | 32473 | 1833 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 12 | 30907 | 0 |
| `aten::where` | 486 | 25331 | 3743 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 22805 | 0 |
| `cudaLaunchKernel` | 3724 | 0 | 21211 |
| `aten::flip` | 3 | 13178 | 25 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1}>(long, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1})` | 3 | 13178 | 0 |

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
