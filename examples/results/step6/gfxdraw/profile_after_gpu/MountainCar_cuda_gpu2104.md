# Profile note — MountainCar on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCar.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **42.06 ms** median (min 41.29, max 47.27, sd 1.64) = **24 fps**  ·  set_state 0.17 ms  ·  peak VRAM 2 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 6044 | 54.4 | 2527 | 4330 |
| other | 2660 | 23.9 | 41414 | 12039 |
| kernel/raster | 2412 | 21.7 | 13193 | 1561 |
| allocation | 0 | 0.0 | 4976 | 1278 |
| synchronization | 0 | 0.0 | 70 | 6 |
| output conversion | 0 | 0.0 | 15 | 9 |

19223 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3436 | 0 | 19410 |
| `aten::empty` | 699 | 0 | 2782 |
| `aten::empty_strided` | 356 | 0 | 1841 |
| `aten::resize_` | 424 | 0 | 1309 |
| `Unrecognized` | 1 | 0 | 948 |
| `aten::bitwise_and` | 621 | 946 | 5840 |
| `aten::__and__` | 621 | 0 | 900 |
| `aten::_to_copy` | 251 | 0 | 755 |

## batch 16

render **57.04 ms** median (min 56.47, max 63.14, sd 1.86) = **281 fps**  ·  set_state 0.20 ms  ·  peak VRAM 26 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 7261 | 56.0 | 2904 | 4411 |
| other | 2903 | 22.4 | 41939 | 12165 |
| kernel/raster | 2807 | 21.6 | 13229 | 1562 |
| allocation | 0 | 0.0 | 5228 | 1362 |
| synchronization | 0 | 0.0 | 57 | 6 |
| output conversion | 0 | 0.0 | 13 | 9 |

19515 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3478 | 0 | 20139 |
| `aten::empty` | 741 | 0 | 3002 |
| `aten::empty_strided` | 356 | 0 | 1847 |
| `aten::resize_` | 424 | 0 | 1344 |
| `aten::bitwise_and` | 621 | 1016 | 5841 |
| `Unrecognized` | 1 | 0 | 916 |
| `aten::_to_copy` | 251 | 0 | 817 |
| `aten::copy_` | 307 | 799 | 1735 |

## batch 64

render **57.48 ms** median (min 57.20, max 57.99, sd 0.22) = **1113 fps**  ·  set_state 0.20 ms  ·  peak VRAM 103 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 9071 | 58.6 | 2864 | 4411 |
| other | 3301 | 21.3 | 41446 | 12165 |
| kernel/raster | 3105 | 20.1 | 13074 | 1562 |
| allocation | 0 | 0.0 | 5254 | 1362 |
| synchronization | 0 | 0.0 | 426 | 6 |
| output conversion | 0 | 0.0 | 14 | 9 |

19515 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3478 | 0 | 20038 |
| `aten::empty` | 741 | 0 | 3000 |
| `aten::empty_strided` | 356 | 0 | 1879 |
| `aten::resize_` | 424 | 0 | 1358 |
| `aten::copy_` | 307 | 1356 | 1733 |
| `aten::bitwise_and` | 621 | 1080 | 5820 |
| `aten::where` | 467 | 1004 | 3577 |
| `aten::_to_copy` | 251 | 0 | 798 |

## batch 256

render **60.41 ms** median (min 59.86, max 60.83, sd 0.29) = **4238 fps**  ·  set_state 0.20 ms  ·  peak VRAM 411 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 15571 | 63.9 | 2864 | 4411 |
| other | 4634 | 19.0 | 42348 | 12165 |
| kernel/raster | 4144 | 17.0 | 13251 | 1562 |
| allocation | 0 | 0.0 | 5285 | 1362 |
| synchronization | 0 | 0.0 | 2038 | 6 |
| output conversion | 0 | 0.0 | 13 | 9 |

19515 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3478 | 0 | 19654 |
| `aten::copy_` | 307 | 3422 | 1728 |
| `aten::empty` | 741 | 0 | 3026 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 9 | 2592 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 1989 |
| `aten::where` | 467 | 1923 | 3589 |
| `aten::empty_strided` | 356 | 0 | 1886 |
| `cudaMemcpyAsync` | 3 | 0 | 1533 |

## batch 1024

render **70.03 ms** median (min 69.44, max 72.06, sd 0.73) = **14622 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1643 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 37109 | 69.7 | 2839 | 4411 |
| other | 8842 | 16.6 | 45171 | 12165 |
| kernel/raster | 7296 | 13.7 | 13219 | 1562 |
| allocation | 0 | 0.0 | 5314 | 1362 |
| synchronization | 0 | 0.0 | 8186 | 6 |
| output conversion | 0 | 0.0 | 14 | 9 |

19515 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 3478 | 0 | 19919 |
| `aten::copy_` | 307 | 10508 | 1721 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 9 | 9730 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 8137 |
| `aten::where` | 467 | 4950 | 3605 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 2 | 4184 | 0 |
| `cudaMemcpyAsync` | 3 | 0 | 3645 |
| `aten::empty` | 741 | 0 | 3066 |

## batch 4096

render **117.60 ms** median (min 116.96, max 118.53, sd 0.52) = **34830 fps**  ·  set_state 0.21 ms  ·  peak VRAM 6564 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 130106 | 71.4 | 2878 | 4419 |
| other | 29579 | 16.2 | 52877 | 12173 |
| kernel/raster | 22646 | 12.4 | 13280 | 1562 |
| allocation | 0 | 0.0 | 8897 | 1363 |
| synchronization | 0 | 0.0 | 33482 | 6 |
| output conversion | 0 | 0.0 | 13 | 9 |

19532 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::copy_` | 307 | 38965 | 1766 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#1}::operator()() const::{lambda(unsigned char)#1} const&)::{lambda(int)#1})` | 15 | 37985 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 33433 |
| `cudaLaunchKernel` | 3486 | 0 | 19908 |
| `aten::where` | 467 | 17870 | 3615 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 16522 | 0 |
| `aten::flip` | 3 | 12480 | 26 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1}>(long, at::native::flip_kernel_impl<at::native::OpaqueType<1> >(at::TensorIterator&)::{lambda(int)#1})` | 3 | 12480 | 0 |

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
