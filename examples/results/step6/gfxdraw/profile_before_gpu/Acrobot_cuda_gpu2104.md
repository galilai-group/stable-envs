# Profile note — Acrobot on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 500px  ·  state_dim: 2
- states: `examples/results/step4/states/Acrobot.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/baseline/stable_envs/__init__.py`

## batch 1

render **89.54 ms** median (min 88.69, max 90.90, sd 0.66) = **11 fps**  ·  set_state 0.17 ms  ·  peak VRAM 2 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 13906 | 54.3 | 5735 | 10246 |
| other | 6316 | 24.7 | 92763 | 26612 |
| kernel/raster | 5397 | 21.1 | 28591 | 3616 |
| allocation | 0 | 0.0 | 13122 | 3557 |
| synchronization | 0 | 0.0 | 31 | 4 |
| output conversion | 0 | 0.0 | 5 | 4 |

44039 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8220 | 0 | 45585 |
| `aten::empty` | 1694 | 0 | 6227 |
| `aten::empty_strided` | 1028 | 0 | 5540 |
| `aten::resize_` | 1362 | 0 | 4126 |
| `aten::bitwise_and` | 1392 | 2122 | 12377 |
| `aten::where` | 1396 | 2090 | 10715 |
| `Unrecognized` | 2 | 0 | 1882 |
| `aten::_to_copy` | 623 | 0 | 1815 |

## batch 16

render **121.99 ms** median (min 121.29, max 124.73, sd 1.04) = **131 fps**  ·  set_state 0.19 ms  ·  peak VRAM 28 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 15328 | 54.9 | 6181 | 10318 |
| other | 6705 | 24.0 | 93010 | 26708 |
| kernel/raster | 5884 | 21.1 | 28284 | 3616 |
| allocation | 0 | 0.0 | 13123 | 3629 |
| synchronization | 0 | 0.0 | 23 | 4 |
| output conversion | 0 | 0.0 | 4 | 4 |

44279 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8256 | 0 | 45984 |
| `aten::empty` | 1730 | 0 | 6505 |
| `aten::empty_strided` | 1028 | 0 | 5150 |
| `aten::resize_` | 1362 | 0 | 4191 |
| `aten::where` | 1396 | 2352 | 10526 |
| `aten::bitwise_and` | 1392 | 2223 | 12510 |
| `aten::_to_copy` | 623 | 0 | 1951 |
| `Unrecognized` | 2 | 0 | 1779 |

## batch 64

render **122.85 ms** median (min 122.24, max 127.59, sd 1.49) = **521 fps**  ·  set_state 0.19 ms  ·  peak VRAM 108 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 18053 | 55.8 | 6136 | 10317 |
| other | 7626 | 23.6 | 96856 | 26711 |
| kernel/raster | 6701 | 20.7 | 28342 | 3616 |
| allocation | 0 | 0.0 | 13037 | 3629 |
| synchronization | 0 | 0.0 | 457 | 4 |
| output conversion | 0 | 0.0 | 4 | 4 |

44281 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8256 | 0 | 45985 |
| `aten::empty` | 1730 | 0 | 6452 |
| `aten::empty_strided` | 1028 | 0 | 5108 |
| `aten::resize_` | 1362 | 0 | 4162 |
| `aten::where` | 1396 | 3119 | 10473 |
| `aten::bitwise_and` | 1392 | 2426 | 12572 |
| `aten::_to_copy` | 623 | 0 | 1940 |
| `aten::copy_` | 669 | 1878 | 3504 |

## batch 256

render **126.82 ms** median (min 126.40, max 127.41, sd 0.32) = **2019 fps**  ·  set_state 0.20 ms  ·  peak VRAM 431 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 29740 | 57.8 | 6080 | 10317 |
| other | 11533 | 22.4 | 92606 | 26708 |
| kernel/raster | 10200 | 19.8 | 28493 | 3616 |
| allocation | 0 | 0.0 | 13250 | 3629 |
| synchronization | 0 | 0.0 | 3423 | 4 |
| output conversion | 0 | 0.0 | 5 | 4 |

44278 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8256 | 0 | 45236 |
| `aten::empty` | 1730 | 0 | 6462 |
| `aten::where` | 1396 | 6248 | 10443 |
| `aten::empty_strided` | 1028 | 0 | 5389 |
| `aten::resize_` | 1362 | 0 | 4262 |
| `aten::copy_` | 669 | 4019 | 3545 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 3945 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 3419 |

## batch 1024

render **141.02 ms** median (min 140.14, max 148.29, sd 2.24) = **7261 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1710 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 68790 | 59.7 | 6270 | 10316 |
| other | 24336 | 21.1 | 94383 | 26708 |
| kernel/raster | 22040 | 19.1 | 28790 | 3616 |
| allocation | 0 | 0.0 | 13302 | 3629 |
| synchronization | 0 | 0.0 | 17081 | 4 |
| output conversion | 0 | 0.0 | 4 | 4 |

44277 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8256 | 0 | 46137 |
| `cudaDeviceSynchronize` | 2 | 0 | 17076 |
| `aten::where` | 1396 | 16844 | 10611 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 14260 | 0 |
| `aten::copy_` | 669 | 11221 | 3577 |
| `Memcpy DtoD (Device -> Device)` | 4 | 7261 | 0 |
| `aten::empty` | 1730 | 0 | 6500 |
| `aten::bitwise_and` | 1392 | 5995 | 12825 |

## batch 4096

render **260.42 ms** median (min 259.96, max 261.01, sd 0.33) = **15728 fps**  ·  set_state 0.22 ms  ·  peak VRAM 6839 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 247500 | 60.1 | 6055 | 10326 |
| other | 85848 | 20.8 | 92609 | 26719 |
| kernel/raster | 78591 | 19.1 | 28405 | 3616 |
| allocation | 0 | 0.0 | 60462 | 3638 |
| synchronization | 0 | 0.0 | 77343 | 4 |
| output conversion | 0 | 0.0 | 4 | 4 |

44307 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 77336 |
| `aten::where` | 1396 | 63941 | 10381 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 8 | 58967 | 0 |
| `Command Buffer Full` | 9 | 0 | 47464 |
| `cudaLaunchKernel` | 8266 | 0 | 45285 |
| `aten::copy_` | 669 | 41545 | 3523 |
| `Memcpy DtoD (Device -> Device)` | 4 | 29115 | 0 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1})` | 40 | 24026 | 0 |

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
