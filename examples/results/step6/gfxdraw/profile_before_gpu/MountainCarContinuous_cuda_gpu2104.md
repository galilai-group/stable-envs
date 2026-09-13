# Profile note — MountainCarContinuous on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 600px  ·  state_dim: 1
- states: `/oscar/scratch/sboughan/se6/baseline/examples/results/step4/states/MountainCarContinuous.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `/oscar/scratch/sboughan/se6/baseline/stable_envs/__init__.py`

## batch 1

render **81.51 ms** median (min 80.67, max 87.49, sd 1.87) = **12 fps**  ·  set_state 0.17 ms  ·  peak VRAM 2 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 12193 | 54.4 | 5203 | 9028 |
| other | 5535 | 24.7 | 82497 | 23443 |
| kernel/raster | 4670 | 20.8 | 25345 | 3178 |
| allocation | 0 | 0.0 | 11234 | 3124 |
| synchronization | 0 | 0.0 | 65 | 7 |
| output conversion | 0 | 0.0 | 5 | 4 |

38784 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7222 | 0 | 40777 |
| `aten::empty` | 1490 | 0 | 5562 |
| `aten::empty_strided` | 905 | 0 | 4544 |
| `aten::resize_` | 1195 | 0 | 3654 |
| `aten::bitwise_and` | 1222 | 1857 | 11175 |
| `aten::where` | 1225 | 1831 | 9324 |
| `aten::_to_copy` | 552 | 0 | 1688 |
| `aten::__and__` | 1222 | 0 | 1532 |

## batch 16

render **110.34 ms** median (min 109.74, max 110.70, sd 0.36) = **145 fps**  ·  set_state 0.20 ms  ·  peak VRAM 26 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 13753 | 55.2 | 5455 | 9089 |
| other | 5935 | 23.8 | 81841 | 23532 |
| kernel/raster | 5230 | 21.0 | 25156 | 3179 |
| allocation | 0 | 0.0 | 11309 | 3188 |
| synchronization | 0 | 0.0 | 77 | 7 |
| output conversion | 0 | 0.0 | 13 | 4 |

38999 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7254 | 0 | 41144 |
| `aten::empty` | 1522 | 0 | 5670 |
| `aten::empty_strided` | 905 | 0 | 4472 |
| `aten::resize_` | 1195 | 0 | 3654 |
| `aten::where` | 1225 | 2099 | 9304 |
| `aten::bitwise_and` | 1222 | 1959 | 11222 |
| `aten::_to_copy` | 552 | 0 | 1645 |
| `aten::__and__` | 1222 | 0 | 1419 |

## batch 64

render **111.13 ms** median (min 110.71, max 114.11, sd 0.95) = **576 fps**  ·  set_state 0.20 ms  ·  peak VRAM 103 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 16477 | 56.3 | 5434 | 9088 |
| other | 6788 | 23.2 | 81976 | 23532 |
| kernel/raster | 6016 | 20.5 | 25236 | 3179 |
| allocation | 0 | 0.0 | 11370 | 3188 |
| synchronization | 0 | 0.0 | 593 | 7 |
| output conversion | 0 | 0.0 | 4 | 4 |

38998 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7254 | 0 | 40281 |
| `aten::empty` | 1522 | 0 | 5683 |
| `aten::empty_strided` | 905 | 0 | 4524 |
| `aten::resize_` | 1195 | 0 | 3907 |
| `aten::where` | 1225 | 2814 | 9157 |
| `aten::bitwise_and` | 1222 | 2154 | 11330 |
| `aten::copy_` | 595 | 1854 | 3135 |
| `aten::_to_copy` | 552 | 0 | 1659 |

## batch 256

render **116.19 ms** median (min 115.42, max 117.06, sd 0.50) = **2203 fps**  ·  set_state 0.20 ms  ·  peak VRAM 411 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 28046 | 58.7 | 5667 | 9088 |
| other | 10417 | 21.8 | 85788 | 23532 |
| kernel/raster | 9300 | 19.5 | 25479 | 3179 |
| allocation | 0 | 0.0 | 11681 | 3188 |
| synchronization | 0 | 0.0 | 2998 | 7 |
| output conversion | 0 | 0.0 | 4 | 4 |

38998 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7254 | 0 | 41846 |
| `aten::where` | 1225 | 5773 | 9242 |
| `aten::empty` | 1522 | 0 | 5768 |
| `aten::empty_strided` | 905 | 0 | 4717 |
| `aten::copy_` | 595 | 4183 | 3343 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 3789 | 0 |
| `aten::resize_` | 1195 | 0 | 3777 |
| `aten::bitwise_and` | 1222 | 2958 | 11465 |

## batch 1024

render **133.95 ms** median (min 133.29, max 141.64, sd 2.33) = **7645 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1644 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 66812 | 61.4 | 5486 | 9087 |
| other | 21956 | 20.2 | 90696 | 23532 |
| kernel/raster | 20108 | 18.5 | 25501 | 3179 |
| allocation | 0 | 0.0 | 11843 | 3188 |
| synchronization | 0 | 0.0 | 11560 | 7 |
| output conversion | 0 | 0.0 | 4 | 4 |

38997 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 7254 | 0 | 41272 |
| `aten::where` | 1225 | 15744 | 9283 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 13744 | 0 |
| `aten::copy_` | 595 | 12390 | 3158 |
| `cudaDeviceSynchronize` | 2 | 0 | 11507 |
| `Memcpy DtoD (Device -> Device)` | 6 | 8720 | 0 |
| `cudaMemcpyAsync` | 10 | 0 | 7267 |
| `aten::empty` | 1522 | 0 | 5977 |

## batch 4096

render **259.79 ms** median (min 258.89, max 260.29, sd 0.38) = **15767 fps**  ·  set_state 0.21 ms  ·  peak VRAM 6565 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 236808 | 62.0 | 5569 | 9098 |
| other | 75774 | 19.8 | 109674 | 23543 |
| kernel/raster | 69648 | 18.2 | 25519 | 3179 |
| allocation | 0 | 0.0 | 75007 | 3198 |
| synchronization | 0 | 0.0 | 48434 | 7 |
| output conversion | 0 | 0.0 | 4 | 4 |

39029 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Command Buffer Full` | 10 | 0 | 63368 |
| `aten::where` | 1225 | 59480 | 9305 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 8 | 56593 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 48377 |
| `aten::copy_` | 595 | 45710 | 3226 |
| `cudaLaunchKernel` | 7265 | 0 | 41703 |
| `Memcpy DtoD (Device -> Device)` | 6 | 34974 | 0 |
| `cudaMemcpyAsync` | 10 | 0 | 26506 |

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
