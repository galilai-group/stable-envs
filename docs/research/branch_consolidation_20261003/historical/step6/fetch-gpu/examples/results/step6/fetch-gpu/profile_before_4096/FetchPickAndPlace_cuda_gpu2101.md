# Profile note — FetchPickAndPlace on cuda

- workstream: `step6/fetch-gpu`
- baseline commit: ``
- host: gpu2101.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 480px  ·  state_dim: 25
- states: `examples/results/step4/states/FetchPickAndPlace.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **1.74 ms** median (min 1.73, max 1.93, sd 0.06) = **576 fps**  ·  set_state 0.12 ms  ·  peak VRAM 4 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 339 | 72.9 | 1948 | 253 |
| host->device | 89 | 19.0 | 209 | 83 |
| kernel/raster | 38 | 8.1 | 250 | 13 |
| allocation | 0 | 0.0 | 94 | 15 |
| synchronization | 0 | 0.0 | 48 | 11 |

375 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Unrecognized` | 1 | 0 | 1052 |
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 291 | 0 |
| `cudaMemcpyAsync` | 18 | 0 | 232 |
| `cudaLaunchKernel` | 8 | 0 | 153 |
| `aten::slice` | 36 | 0 | 96 |
| `cudaGraphLaunch` | 1 | 0 | 85 |
| `aten::empty_strided` | 10 | 0 | 66 |
| `aten::view` | 29 | 0 | 57 |

## batch 16

render **4.95 ms** median (min 4.92, max 4.98, sd 0.02) = **3233 fps**  ·  set_state 0.14 ms  ·  peak VRAM 41 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 3314 | 85.2 | 1700 | 252 |
| host->device | 336 | 8.6 | 219 | 81 |
| kernel/raster | 238 | 6.1 | 165 | 15 |
| allocation | 0 | 0.0 | 70 | 15 |
| synchronization | 0 | 0.0 | 50 | 11 |

374 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 3223 | 0 |
| `Unrecognized` | 1 | 0 | 1058 |
| `cudaLaunchKernel` | 16 | 0 | 158 |
| `cudaMemcpyAsync` | 10 | 0 | 116 |
| `aten::slice` | 36 | 0 | 90 |
| `aten::index` | 1 | 80 | 19 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 80 | 0 |
| `aten::where` | 1 | 55 | 15 |

## batch 64

render **15.22 ms** median (min 15.21, max 15.27, sd 0.02) = **4204 fps**  ·  set_state 0.15 ms  ·  peak VRAM 157 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 12783 | 87.5 | 1055 | 252 |
| host->device | 1046 | 7.2 | 200 | 81 |
| kernel/raster | 784 | 5.4 | 158 | 15 |
| allocation | 0 | 0.0 | 68 | 15 |
| synchronization | 0 | 0.0 | 670 | 11 |

374 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 12591 | 0 |
| `cudaDeviceSynchronize` | 3 | 0 | 643 |
| `Unrecognized` | 1 | 0 | 423 |
| `aten::index` | 1 | 274 | 19 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 274 | 0 |
| `aten::add` | 1 | 209 | 12 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 1 | 209 | 0 |
| `aten::where` | 1 | 189 | 13 |

## batch 256

render **49.75 ms** median (min 49.63, max 50.21, sd 0.19) = **5146 fps**  ·  set_state 0.16 ms  ·  peak VRAM 622 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 44628 | 86.5 | 1237 | 252 |
| host->device | 3921 | 7.6 | 207 | 81 |
| kernel/raster | 3033 | 5.9 | 174 | 15 |
| allocation | 0 | 0.0 | 71 | 15 |
| synchronization | 0 | 0.0 | 3222 | 11 |

374 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 44043 | 0 |
| `cudaDeviceSynchronize` | 3 | 0 | 3197 |
| `aten::index` | 1 | 1090 | 21 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 1090 | 0 |
| `aten::add` | 1 | 830 | 13 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 1 | 830 | 0 |
| `aten::where` | 1 | 749 | 14 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 1 | 749 | 0 |

## batch 1024

render **195.53 ms** median (min 195.34, max 195.68, sd 0.12) = **5237 fps**  ·  set_state 0.22 ms  ·  peak VRAM 1355 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 178654 | 86.5 | 2610 | 650 |
| host->device | 15677 | 7.6 | 465 | 201 |
| kernel/raster | 12098 | 5.9 | 507 | 39 |
| allocation | 0 | 0.0 | 132 | 27 |
| synchronization | 0 | 0.0 | 10981 | 17 |

934 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 4 | 176315 | 0 |
| `cudaDeviceSynchronize` | 6 | 0 | 10937 |
| `aten::index` | 4 | 4362 | 83 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 4 | 4362 | 0 |
| `aten::add` | 4 | 3320 | 56 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 4 | 3320 | 0 |
| `aten::where` | 4 | 2998 | 59 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 2998 | 0 |

## batch 4096

render **779.72 ms** median (min 779.41, max 855.82, sd 22.74) = **5253 fps**  ·  set_state 0.39 ms  ·  peak VRAM 3380 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 713339 | 86.5 | 6823 | 2238 |
| host->device | 62682 | 7.6 | 1464 | 682 |
| kernel/raster | 48361 | 5.9 | 1594 | 135 |
| allocation | 0 | 0.0 | 348 | 76 |
| synchronization | 0 | 0.0 | 43771 | 41 |

3172 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 16 | 703982 | 0 |
| `cudaDeviceSynchronize` | 18 | 0 | 43659 |
| `aten::index` | 16 | 17453 | 283 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 16 | 17453 | 0 |
| `aten::add` | 16 | 13289 | 189 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 16 | 13289 | 0 |
| `aten::where` | 16 | 11982 | 203 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 16 | 11982 | 0 |

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
