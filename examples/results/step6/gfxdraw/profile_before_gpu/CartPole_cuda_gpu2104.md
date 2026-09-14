# Profile note — CartPole on cuda

- workstream: `step6/gfxdraw`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 600px  ·  state_dim: 2
- states: `examples/results/step4/states/CartPole.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se6/baseline/stable_envs/__init__.py`

## batch 1

render **90.74 ms** median (min 90.12, max 94.08, sd 1.09) = **11 fps**  ·  set_state 0.17 ms  ·  peak VRAM 2 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 13798 | 54.2 | 5896 | 10138 |
| other | 6286 | 24.7 | 91870 | 26307 |
| kernel/raster | 5362 | 21.1 | 28269 | 3572 |
| allocation | 0 | 0.0 | 12491 | 3533 |
| synchronization | 0 | 0.0 | 19 | 4 |
| output conversion | 0 | 0.0 | 5 | 4 |

43558 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8139 | 0 | 44899 |
| `aten::empty` | 1681 | 0 | 6170 |
| `aten::empty_strided` | 1020 | 0 | 4858 |
| `aten::resize_` | 1356 | 0 | 4022 |
| `aten::bitwise_and` | 1384 | 2120 | 12842 |
| `aten::where` | 1391 | 2089 | 10717 |
| `Unrecognized` | 2 | 0 | 1855 |
| `aten::_to_copy` | 616 | 0 | 1797 |

## batch 16

render **122.54 ms** median (min 121.11, max 123.26, sd 0.62) = **131 fps**  ·  set_state 0.19 ms  ·  peak VRAM 26 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 14966 | 54.8 | 5999 | 10206 |
| other | 6612 | 24.2 | 90749 | 26406 |
| kernel/raster | 5738 | 21.0 | 27731 | 3572 |
| allocation | 0 | 0.0 | 12232 | 3601 |
| synchronization | 0 | 0.0 | 17 | 4 |
| output conversion | 0 | 0.0 | 4 | 4 |

43793 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8173 | 0 | 45297 |
| `aten::empty` | 1715 | 0 | 6279 |
| `aten::empty_strided` | 1020 | 0 | 4805 |
| `aten::resize_` | 1356 | 0 | 4004 |
| `aten::where` | 1391 | 2287 | 10401 |
| `aten::bitwise_and` | 1384 | 2195 | 12414 |
| `aten::_to_copy` | 616 | 0 | 1805 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(bool, unsigned long, unsigned long)#1}, std::array<char*, 4ul>, false>(int, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(bool, unsigned long, unsigned long)#1}, std::array<char*, 4ul>)` | 1025 | 1560 | 0 |

## batch 64

render **123.18 ms** median (min 121.83, max 127.44, sd 1.45) = **520 fps**  ·  set_state 0.20 ms  ·  peak VRAM 103 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 17126 | 55.5 | 6010 | 10205 |
| other | 7415 | 24.0 | 91619 | 26406 |
| kernel/raster | 6325 | 20.5 | 27809 | 3572 |
| allocation | 0 | 0.0 | 12476 | 3601 |
| synchronization | 0 | 0.0 | 380 | 4 |
| output conversion | 0 | 0.0 | 4 | 4 |

43792 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8173 | 0 | 45329 |
| `aten::empty` | 1715 | 0 | 6485 |
| `aten::empty_strided` | 1020 | 0 | 4836 |
| `aten::resize_` | 1356 | 0 | 4013 |
| `aten::where` | 1391 | 2834 | 10397 |
| `aten::bitwise_and` | 1384 | 2390 | 12391 |
| `aten::_to_copy` | 616 | 0 | 1832 |
| `aten::copy_` | 659 | 1708 | 3460 |

## batch 256

render **127.19 ms** median (min 126.16, max 131.85, sd 1.56) = **2013 fps**  ·  set_state 0.20 ms  ·  peak VRAM 411 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 26629 | 57.4 | 6090 | 10205 |
| other | 10903 | 23.5 | 93467 | 26406 |
| kernel/raster | 8881 | 19.1 | 28321 | 3572 |
| allocation | 0 | 0.0 | 12570 | 3601 |
| synchronization | 0 | 0.0 | 3126 | 4 |
| output conversion | 0 | 0.0 | 4 | 4 |

43792 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8173 | 0 | 45949 |
| `aten::empty` | 1715 | 0 | 6385 |
| `aten::where` | 1391 | 5113 | 10537 |
| `aten::empty_strided` | 1020 | 0 | 4959 |
| `aten::resize_` | 1356 | 0 | 4134 |
| `aten::copy_` | 659 | 3438 | 3484 |
| `aten::bitwise_and` | 1384 | 3225 | 12657 |
| `cudaDeviceSynchronize` | 2 | 0 | 3121 |

## batch 1024

render **141.23 ms** median (min 140.62, max 141.70, sd 0.35) = **7251 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1643 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 57496 | 59.3 | 6042 | 10204 |
| other | 22091 | 22.8 | 92785 | 26406 |
| kernel/raster | 17307 | 17.9 | 28208 | 3572 |
| allocation | 0 | 0.0 | 12627 | 3601 |
| synchronization | 0 | 0.0 | 16310 | 4 |
| output conversion | 0 | 0.0 | 4 | 4 |

43791 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 8173 | 0 | 45637 |
| `cudaDeviceSynchronize` | 2 | 0 | 16305 |
| `aten::where` | 1391 | 12707 | 10469 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 3 | 10288 | 0 |
| `aten::copy_` | 659 | 9063 | 3501 |
| `aten::empty` | 1715 | 0 | 6410 |
| `aten::bitwise_and` | 1384 | 5734 | 12707 |
| `Memcpy DtoD (Device -> Device)` | 3 | 5248 | 0 |

## batch 4096

render **227.49 ms** median (min 226.21, max 233.92, sd 2.43) = **18005 fps**  ·  set_state 0.22 ms  ·  peak VRAM 6565 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 199018 | 59.8 | 6103 | 10213 |
| other | 75718 | 22.8 | 94521 | 26415 |
| kernel/raster | 57955 | 17.4 | 28480 | 3572 |
| allocation | 0 | 0.0 | 29669 | 3604 |
| synchronization | 0 | 0.0 | 73600 | 4 |
| output conversion | 0 | 0.0 | 4 | 4 |

43812 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 0 | 73594 |
| `aten::where` | 1391 | 46250 | 10581 |
| `cudaLaunchKernel` | 8182 | 0 | 46106 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 6 | 42025 | 0 |
| `aten::copy_` | 659 | 32687 | 3476 |
| `Memcpy DtoD (Device -> Device)` | 3 | 21021 | 0 |
| `aten::bitwise_and` | 1384 | 19078 | 12744 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::CompareFunctor<long> >(at::TensorIteratorBase&, at::native::(anonymous namespace)::CompareFunctor<long> const&)::{lambda(int)#1})` | 39 | 18514 | 0 |

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
