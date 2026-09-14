# Profile note — MountainCar on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 600px  ·  state_dim: 1
- states: `examples/results/step4/states/MountainCar.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/baseline/stable_envs/__init__.py`

## batch 1

render **81.10 ms** median (min 80.55, max 86.75, sd 1.72) = **12 fps**  ·  set_state 0.16 ms  ·  peak VRAM 2 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 12196 | 54.4 | 5159 | 9028 |
| other | 5537 | 24.7 | 81423 | 23443 |
| kernel/raster | 4670 | 20.8 | 24926 | 3178 |
| allocation | 0 | 0.0 | 11148 | 3124 |
| synchronization | 0 | 0.0 | 75 | 7 |
| output conversion | 0 | 0.0 | 5 | 4 |

38784 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7222 | 0 | 40431 |
| `aten::empty` | 1490 | 0 | 5533 |
| `aten::empty_strided` | 905 | 0 | 4490 |
| `aten::resize_` | 1195 | 0 | 3630 |
| `aten::bitwise_and` | 1222 | 1857 | 11025 |
| `aten::where` | 1225 | 1832 | 9272 |
| `aten::_to_copy` | 552 | 0 | 1642 |
| `aten::__and__` | 1222 | 0 | 1477 |

## batch 16

render **109.41 ms** median (min 108.05, max 110.23, sd 0.69) = **146 fps**  ·  set_state 0.19 ms  ·  peak VRAM 26 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 13750 | 55.2 | 5336 | 9089 |
| other | 5933 | 23.8 | 79623 | 23532 |
| kernel/raster | 5229 | 21.0 | 24508 | 3179 |
| allocation | 0 | 0.0 | 11285 | 3188 |
| synchronization | 0 | 0.0 | 77 | 7 |
| output conversion | 0 | 0.0 | 4 | 4 |

38999 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7254 | 0 | 39731 |
| `aten::empty` | 1522 | 0 | 5678 |
| `aten::empty_strided` | 905 | 0 | 4440 |
| `aten::resize_` | 1195 | 0 | 3666 |
| `aten::where` | 1225 | 2098 | 9030 |
| `aten::bitwise_and` | 1222 | 1958 | 11016 |
| `aten::_to_copy` | 552 | 0 | 1567 |
| `aten::__and__` | 1222 | 0 | 1422 |

## batch 64

render **110.58 ms** median (min 109.07, max 111.03, sd 0.54) = **579 fps**  ·  set_state 0.20 ms  ·  peak VRAM 103 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 16475 | 56.3 | 5365 | 9088 |
| other | 6787 | 23.2 | 81297 | 23532 |
| kernel/raster | 6015 | 20.5 | 24810 | 3179 |
| allocation | 0 | 0.0 | 11347 | 3188 |
| synchronization | 0 | 0.0 | 607 | 7 |
| output conversion | 0 | 0.0 | 4 | 4 |

38998 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7254 | 0 | 40367 |
| `aten::empty` | 1522 | 0 | 5668 |
| `aten::empty_strided` | 905 | 0 | 4508 |
| `aten::resize_` | 1195 | 0 | 3662 |
| `aten::where` | 1225 | 2814 | 9117 |
| `aten::bitwise_and` | 1222 | 2152 | 11253 |
| `aten::copy_` | 595 | 1854 | 3156 |
| `aten::_to_copy` | 552 | 0 | 1576 |

## batch 256

render **114.83 ms** median (min 113.80, max 117.75, sd 1.01) = **2229 fps**  ·  set_state 0.20 ms  ·  peak VRAM 411 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 28034 | 58.7 | 5391 | 9088 |
| other | 10414 | 21.8 | 82448 | 23532 |
| kernel/raster | 9297 | 19.5 | 24844 | 3179 |
| allocation | 0 | 0.0 | 11481 | 3188 |
| synchronization | 0 | 0.0 | 3000 | 7 |
| output conversion | 0 | 0.0 | 4 | 4 |

38998 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7254 | 0 | 40057 |
| `aten::where` | 1225 | 5771 | 9072 |
| `aten::empty` | 1522 | 0 | 5681 |
| `aten::empty_strided` | 905 | 0 | 4636 |
| `aten::copy_` | 595 | 4180 | 3145 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 3788 | 0 |
| `aten::resize_` | 1195 | 0 | 3729 |
| `aten::bitwise_and` | 1222 | 2958 | 11342 |

## batch 1024

render **132.67 ms** median (min 131.92, max 140.77, sd 2.50) = **7718 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1644 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 66614 | 61.4 | 5396 | 9087 |
| other | 21889 | 20.2 | 88520 | 23532 |
| kernel/raster | 20044 | 18.5 | 25246 | 3179 |
| allocation | 0 | 0.0 | 11438 | 3188 |
| synchronization | 0 | 0.0 | 11545 | 7 |
| output conversion | 0 | 0.0 | 4 | 4 |

38997 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7254 | 0 | 40638 |
| `aten::where` | 1225 | 15704 | 9338 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 13723 | 0 |
| `aten::copy_` | 595 | 12357 | 3146 |
| `cudaDeviceSynchronize` | 2 | 0 | 11493 |
| `Memcpy DtoD (Device -> Device)` | 6 | 8724 | 0 |
| `cudaMemcpyAsync` | 10 | 0 | 6730 |
| `aten::empty` | 1522 | 0 | 5695 |

## batch 4096

render **258.80 ms** median (min 258.25, max 260.34, sd 0.60) = **15827 fps**  ·  set_state 0.21 ms  ·  peak VRAM 6565 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 237205 | 62.0 | 5405 | 9098 |
| other | 75924 | 19.8 | 107157 | 23543 |
| kernel/raster | 69672 | 18.2 | 25084 | 3179 |
| allocation | 0 | 0.0 | 75015 | 3198 |
| synchronization | 0 | 0.0 | 48680 | 7 |
| output conversion | 0 | 0.0 | 4 | 4 |

39029 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Command Buffer Full` | 10 | 0 | 63494 |
| `aten::where` | 1225 | 59533 | 9291 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 8 | 56650 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 48625 |
| `aten::copy_` | 595 | 45821 | 3165 |
| `cudaLaunchKernel` | 7265 | 0 | 40413 |
| `Memcpy DtoD (Device -> Device)` | 6 | 34953 | 0 |
| `cudaMemcpyAsync` | 10 | 0 | 26308 |

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
