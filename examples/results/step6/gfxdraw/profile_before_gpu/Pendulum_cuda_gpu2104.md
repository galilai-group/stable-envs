# Profile note — Pendulum on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 500px  ·  state_dim: 2
- states: `/oscar/scratch/sboughan/se6/baseline/examples/results/step4/states/Pendulum.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `/oscar/scratch/sboughan/se6/baseline/stable_envs/__init__.py`

## batch 1

render **47.92 ms** median (min 47.53, max 48.23, sd 0.19) = **21 fps**  ·  set_state 0.17 ms  ·  peak VRAM 14 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 7294 | 54.4 | 3075 | 5337 |
| other | 3274 | 24.4 | 48874 | 13908 |
| kernel/raster | 2834 | 21.1 | 14828 | 1876 |
| allocation | 0 | 0.0 | 6443 | 1843 |
| synchronization | 0 | 0.0 | 27 | 4 |
| output conversion | 0 | 0.0 | 6 | 6 |

22974 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 4259 | 0 | 24417 |
| `aten::empty` | 883 | 0 | 3213 |
| `aten::empty_strided` | 531 | 0 | 2571 |
| `aten::resize_` | 696 | 0 | 2110 |
| `aten::bitwise_and` | 715 | 1092 | 6461 |
| `aten::where` | 709 | 1071 | 5251 |
| `aten::_to_copy` | 326 | 0 | 925 |
| `Unrecognized` | 1 | 0 | 878 |

## batch 16

render **65.06 ms** median (min 64.65, max 69.66, sd 1.40) = **246 fps**  ·  set_state 0.19 ms  ·  peak VRAM 40 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 8499 | 55.5 | 3440 | 5381 |
| other | 3577 | 23.3 | 48584 | 13970 |
| kernel/raster | 3247 | 21.2 | 14585 | 1876 |
| allocation | 0 | 0.0 | 6528 | 1887 |
| synchronization | 0 | 0.0 | 52 | 4 |
| output conversion | 0 | 0.0 | 6 | 6 |

23124 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 4281 | 0 | 24346 |
| `aten::empty` | 905 | 0 | 3297 |
| `aten::empty_strided` | 531 | 0 | 2559 |
| `aten::resize_` | 696 | 0 | 2073 |
| `aten::where` | 709 | 1327 | 5208 |
| `aten::bitwise_and` | 715 | 1145 | 6474 |
| `aten::_to_copy` | 326 | 0 | 955 |
| `Unrecognized` | 1 | 0 | 889 |

## batch 64

render **65.78 ms** median (min 65.33, max 66.72, sd 0.38) = **973 fps**  ·  set_state 0.19 ms  ·  peak VRAM 120 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 11312 | 57.0 | 3271 | 5381 |
| other | 4359 | 22.0 | 47978 | 13970 |
| kernel/raster | 4164 | 21.0 | 14698 | 1876 |
| allocation | 0 | 0.0 | 6614 | 1887 |
| synchronization | 0 | 0.0 | 507 | 4 |
| output conversion | 0 | 0.0 | 5 | 6 |

23124 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 4281 | 0 | 23812 |
| `aten::empty` | 905 | 0 | 3306 |
| `aten::empty_strided` | 531 | 0 | 2610 |
| `aten::resize_` | 696 | 0 | 2114 |
| `aten::where` | 709 | 2085 | 5217 |
| `aten::copy_` | 360 | 1402 | 1913 |
| `aten::bitwise_and` | 715 | 1247 | 6501 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 996 | 0 |

## batch 256

render **71.65 ms** median (min 71.56, max 72.31, sd 0.25) = **3573 fps**  ·  set_state 0.20 ms  ·  peak VRAM 442 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 22877 | 59.5 | 3308 | 5381 |
| kernel/raster | 7912 | 20.6 | 14918 | 1876 |
| other | 7628 | 19.9 | 48564 | 13970 |
| allocation | 0 | 0.0 | 6893 | 1887 |
| synchronization | 0 | 0.0 | 4359 | 4 |
| output conversion | 0 | 0.0 | 6 | 6 |

23124 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 4281 | 0 | 24028 |
| `aten::where` | 709 | 5171 | 5272 |
| `cudaDeviceSynchronize` | 2 | 0 | 4354 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 3945 | 0 |
| `aten::copy_` | 360 | 3676 | 1948 |
| `aten::empty` | 905 | 0 | 3515 |
| `aten::empty_strided` | 531 | 0 | 2698 |
| `aten::resize_` | 696 | 0 | 2157 |

## batch 1024

render **98.51 ms** median (min 97.89, max 99.19, sd 0.37) = **10395 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1722 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 64015 | 61.5 | 3334 | 5381 |
| kernel/raster | 21135 | 20.3 | 14841 | 1876 |
| other | 18886 | 18.2 | 49253 | 13970 |
| allocation | 0 | 0.0 | 6731 | 1887 |
| synchronization | 0 | 0.0 | 30978 | 4 |
| output conversion | 0 | 0.0 | 5 | 6 |

23124 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 30973 |
| `cudaLaunchKernel` | 4281 | 0 | 24229 |
| `aten::where` | 709 | 15886 | 5224 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 14322 | 0 |
| `aten::copy_` | 360 | 12004 | 1908 |
| `Memcpy DtoD (Device -> Device)` | 4 | 7269 | 0 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1})` | 24 | 5200 | 0 |
| `aten::fill_` | 253 | 4728 | 1047 |

## batch 4096

render **223.27 ms** median (min 222.55, max 224.57, sd 0.57) = **18346 fps**  ·  set_state 0.22 ms  ·  peak VRAM 6851 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 242963 | 61.6 | 3260 | 5392 |
| kernel/raster | 80531 | 20.4 | 14785 | 1876 |
| other | 70681 | 17.9 | 49098 | 13981 |
| allocation | 0 | 0.0 | 11729 | 1889 |
| synchronization | 0 | 0.0 | 147232 | 4 |
| output conversion | 0 | 0.0 | 6 | 6 |

23148 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 147226 |
| `aten::where` | 709 | 62874 | 5207 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 8 | 59349 | 0 |
| `aten::copy_` | 360 | 45883 | 1906 |
| `Memcpy DtoD (Device -> Device)` | 4 | 29119 | 0 |
| `cudaLaunchKernel` | 4292 | 0 | 24231 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1})` | 24 | 21561 | 0 |
| `aten::le` | 16 | 18297 | 163 |

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
