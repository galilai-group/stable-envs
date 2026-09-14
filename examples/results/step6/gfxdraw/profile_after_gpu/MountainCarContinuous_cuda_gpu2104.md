# Profile note — MountainCarContinuous on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCarContinuous.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **41.49 ms** median (min 41.20, max 42.08, sd 0.29) = **24 fps**  ·  set_state 0.17 ms  ·  peak VRAM 2 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 6046 | 54.4 | 2558 | 4330 |
| other | 2661 | 23.9 | 41425 | 12039 |
| kernel/raster | 2412 | 21.7 | 13204 | 1561 |
| allocation | 0 | 0.0 | 4787 | 1278 |
| synchronization | 0 | 0.0 | 72 | 6 |
| output conversion | 0 | 0.0 | 15 | 9 |

19223 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3436 | 0 | 19227 |
| `aten::empty` | 699 | 0 | 2695 |
| `aten::empty_strided` | 356 | 0 | 1770 |
| `aten::resize_` | 424 | 0 | 1289 |
| `Unrecognized` | 1 | 0 | 958 |
| `aten::bitwise_and` | 621 | 946 | 5767 |
| `aten::__and__` | 621 | 0 | 839 |
| `aten::_to_copy` | 251 | 0 | 773 |

## batch 16

render **56.19 ms** median (min 55.54, max 60.08, sd 1.25) = **285 fps**  ·  set_state 0.19 ms  ·  peak VRAM 26 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 7262 | 56.0 | 2817 | 4411 |
| other | 2901 | 22.4 | 40099 | 12165 |
| kernel/raster | 2809 | 21.7 | 12901 | 1562 |
| allocation | 0 | 0.0 | 5028 | 1362 |
| synchronization | 0 | 0.0 | 57 | 6 |
| output conversion | 0 | 0.0 | 13 | 9 |

19515 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3478 | 0 | 18999 |
| `aten::empty` | 741 | 0 | 2905 |
| `aten::empty_strided` | 356 | 0 | 1754 |
| `aten::resize_` | 424 | 0 | 1299 |
| `aten::bitwise_and` | 621 | 1015 | 5648 |
| `Unrecognized` | 1 | 0 | 868 |
| `aten::copy_` | 307 | 799 | 1727 |
| `aten::_to_copy` | 251 | 0 | 771 |

## batch 64

render **56.83 ms** median (min 56.00, max 59.47, sd 0.95) = **1126 fps**  ·  set_state 0.19 ms  ·  peak VRAM 103 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 9076 | 58.6 | 2842 | 4411 |
| other | 3303 | 21.3 | 40492 | 12165 |
| kernel/raster | 3107 | 20.1 | 12930 | 1562 |
| allocation | 0 | 0.0 | 5080 | 1362 |
| synchronization | 0 | 0.0 | 432 | 6 |
| output conversion | 0 | 0.0 | 13 | 9 |

19515 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3478 | 0 | 19247 |
| `aten::empty` | 741 | 0 | 2920 |
| `aten::empty_strided` | 356 | 0 | 1798 |
| `aten::copy_` | 307 | 1357 | 1728 |
| `aten::resize_` | 424 | 0 | 1338 |
| `aten::bitwise_and` | 621 | 1080 | 5683 |
| `aten::where` | 467 | 1005 | 3511 |
| `aten::_to_copy` | 251 | 0 | 791 |

## batch 256

render **59.76 ms** median (min 59.08, max 61.49, sd 0.61) = **4284 fps**  ·  set_state 0.21 ms  ·  peak VRAM 411 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 15567 | 63.9 | 2893 | 4411 |
| other | 4634 | 19.0 | 41738 | 12165 |
| kernel/raster | 4147 | 17.0 | 13182 | 1562 |
| allocation | 0 | 0.0 | 5121 | 1362 |
| synchronization | 0 | 0.0 | 2026 | 6 |
| output conversion | 0 | 0.0 | 13 | 9 |

19515 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3478 | 0 | 19672 |
| `aten::copy_` | 307 | 3418 | 1754 |
| `aten::empty` | 741 | 0 | 2932 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 9 | 2589 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 1977 |
| `aten::where` | 467 | 1925 | 3576 |
| `aten::empty_strided` | 356 | 0 | 1819 |
| `aten::resize_` | 424 | 0 | 1352 |

## batch 1024

render **69.00 ms** median (min 68.38, max 69.89, sd 0.40) = **14840 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1643 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 37098 | 69.7 | 2857 | 4411 |
| other | 8841 | 16.6 | 43993 | 12165 |
| kernel/raster | 7297 | 13.7 | 13064 | 1562 |
| allocation | 0 | 0.0 | 5105 | 1362 |
| synchronization | 0 | 0.0 | 8191 | 6 |
| output conversion | 0 | 0.0 | 14 | 9 |

19515 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3478 | 0 | 19485 |
| `aten::copy_` | 307 | 10502 | 1762 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 9 | 9724 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 8142 |
| `aten::where` | 467 | 4949 | 3515 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 2 | 4183 | 0 |
| `cudaMemcpyAsync` | 3 | 0 | 3123 |
| `aten::flip` | 3 | 3023 | 26 |

## batch 4096

render **117.08 ms** median (min 116.86, max 117.52, sd 0.20) = **34985 fps**  ·  set_state 0.21 ms  ·  peak VRAM 6564 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 130251 | 71.4 | 2867 | 4419 |
| other | 29591 | 16.2 | 52137 | 12173 |
| kernel/raster | 22685 | 12.4 | 13104 | 1562 |
| allocation | 0 | 0.0 | 9360 | 1363 |
| synchronization | 0 | 0.0 | 33487 | 6 |
| output conversion | 0 | 0.0 | 14 | 9 |

19532 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 307 | 39011 | 1759 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 15 | 38028 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 33438 |
| `cudaLaunchKernel` | 3486 | 0 | 19497 |
| `aten::where` | 467 | 17911 | 3534 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 16559 | 0 |
| `aten::flip` | 3 | 12471 | 26 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1}>(long, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1})` | 3 | 12471 | 0 |

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
