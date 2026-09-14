# Profile note — CartPole on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 600px  ·  state_dim: 2
- states: `examples/results/step4/states/CartPole.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **44.10 ms** median (min 43.68, max 44.80, sd 0.33) = **23 fps**  ·  set_state 0.17 ms  ·  peak VRAM 2 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 6322 | 54.0 | 2676 | 4495 |
| other | 2810 | 24.0 | 44150 | 12547 |
| kernel/raster | 2584 | 22.1 | 14015 | 1624 |
| allocation | 0 | 0.0 | 5005 | 1324 |
| synchronization | 0 | 0.0 | 27 | 4 |
| output conversion | 0 | 0.0 | 10 | 8 |

20002 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3583 | 0 | 21258 |
| `aten::empty` | 734 | 0 | 2854 |
| `aten::empty_strided` | 363 | 0 | 1806 |
| `aten::resize_` | 434 | 0 | 1311 |
| `aten::bitwise_and` | 659 | 1016 | 6122 |
| `Unrecognized` | 1 | 0 | 972 |
| `aten::_to_copy` | 256 | 0 | 825 |
| `aten::__and__` | 659 | 0 | 738 |

## batch 16

render **59.28 ms** median (min 58.70, max 64.86, sd 1.73) = **270 fps**  ·  set_state 0.20 ms  ·  peak VRAM 26 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 7337 | 55.4 | 2958 | 4581 |
| other | 3013 | 22.7 | 43329 | 12676 |
| kernel/raster | 2901 | 21.9 | 13562 | 1624 |
| allocation | 0 | 0.0 | 5262 | 1410 |
| synchronization | 0 | 0.0 | 13 | 4 |
| output conversion | 0 | 0.0 | 9 | 8 |

20303 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3626 | 0 | 20970 |
| `aten::empty` | 777 | 0 | 3034 |
| `aten::empty_strided` | 363 | 0 | 1813 |
| `aten::resize_` | 434 | 0 | 1341 |
| `aten::bitwise_and` | 659 | 1066 | 6096 |
| `Unrecognized` | 1 | 0 | 884 |
| `aten::_to_copy` | 256 | 0 | 834 |
| `aten::where` | 481 | 770 | 3731 |

## batch 64

render **59.73 ms** median (min 59.30, max 60.39, sd 0.31) = **1072 fps**  ·  set_state 0.19 ms  ·  peak VRAM 103 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 8852 | 57.5 | 2930 | 4581 |
| other | 3365 | 21.9 | 42671 | 12676 |
| kernel/raster | 3167 | 20.6 | 13549 | 1624 |
| allocation | 0 | 0.0 | 5273 | 1410 |
| synchronization | 0 | 0.0 | 381 | 4 |
| output conversion | 0 | 0.0 | 8 | 8 |

20303 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3626 | 0 | 20573 |
| `aten::empty` | 777 | 0 | 3041 |
| `aten::empty_strided` | 363 | 0 | 1835 |
| `aten::resize_` | 434 | 0 | 1375 |
| `aten::copy_` | 312 | 1181 | 1761 |
| `aten::bitwise_and` | 659 | 1099 | 6117 |
| `aten::where` | 481 | 961 | 3680 |
| `aten::_to_copy` | 256 | 0 | 812 |

## batch 256

render **61.85 ms** median (min 61.15, max 63.72, sd 0.70) = **4139 fps**  ·  set_state 0.20 ms  ·  peak VRAM 411 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 14104 | 62.1 | 2966 | 4581 |
| other | 4534 | 20.0 | 43768 | 12676 |
| kernel/raster | 4069 | 17.9 | 13932 | 1624 |
| allocation | 0 | 0.0 | 5391 | 1410 |
| synchronization | 0 | 0.0 | 1975 | 4 |
| output conversion | 0 | 0.0 | 8 | 8 |

20303 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3626 | 0 | 20929 |
| `aten::empty` | 777 | 0 | 3093 |
| `aten::copy_` | 312 | 2773 | 1788 |
| `cudaDeviceSynchronize` | 2 | 0 | 1971 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 6 | 1942 | 0 |
| `aten::empty_strided` | 363 | 0 | 1888 |
| `aten::where` | 481 | 1650 | 3756 |
| `aten::resize_` | 434 | 0 | 1400 |

## batch 1024

render **71.17 ms** median (min 70.53, max 72.01, sd 0.42) = **14389 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1643 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 34057 | 66.8 | 2951 | 4581 |
| other | 9430 | 18.5 | 43440 | 12676 |
| kernel/raster | 7492 | 14.7 | 13579 | 1624 |
| allocation | 0 | 0.0 | 5297 | 1410 |
| synchronization | 0 | 0.0 | 11051 | 4 |
| output conversion | 0 | 0.0 | 8 | 8 |

20303 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3626 | 0 | 20715 |
| `cudaDeviceSynchronize` | 2 | 0 | 11046 |
| `aten::copy_` | 312 | 8590 | 1769 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 6 | 7714 | 0 |
| `aten::where` | 481 | 4291 | 3677 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 2 | 3349 | 0 |
| `aten::flip` | 3 | 3311 | 26 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1}>(long, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1})` | 3 | 3311 | 0 |

## batch 4096

render **108.26 ms** median (min 107.80, max 109.20, sd 0.38) = **37836 fps**  ·  set_state 0.22 ms  ·  peak VRAM 6564 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 111966 | 68.9 | 2952 | 4588 |
| other | 28619 | 17.6 | 42913 | 12683 |
| kernel/raster | 21916 | 13.5 | 13595 | 1624 |
| allocation | 0 | 0.0 | 5328 | 1410 |
| synchronization | 0 | 0.0 | 48345 | 4 |
| output conversion | 0 | 0.0 | 8 | 8 |

20317 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 48339 |
| `aten::copy_` | 312 | 30736 | 1776 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 11 | 29497 | 0 |
| `cudaLaunchKernel` | 3633 | 0 | 20520 |
| `aten::where` | 481 | 14294 | 3646 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 12510 | 0 |
| `aten::flip` | 3 | 12233 | 26 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1}>(long, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1})` | 3 | 12233 | 0 |

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
