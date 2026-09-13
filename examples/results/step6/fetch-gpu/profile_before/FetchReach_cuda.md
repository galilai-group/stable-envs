# Profile note — FetchReach on cuda

- workstream: `step6/fetch-gpu`
- baseline commit: `d148f1a582d9`
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 8
- frame: 480px  ·  state_dim: 18
- states: `examples/results/step4/states/FetchReach.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **1.64 ms** median (min 1.63, max 1.68, sd 0.02) = **610 fps**  ·  set_state 0.12 ms  ·  peak VRAM 4 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 333 | 72.7 | 1569 | 244 |
| host->device | 87 | 19.1 | 200 | 81 |
| kernel/raster | 38 | 8.2 | 233 | 13 |
| allocation | 0 | 0.0 | 96 | 15 |
| synchronization | 0 | 0.0 | 49 | 11 |

364 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Unrecognized` | 1 | 0 | 793 |
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 286 | 0 |
| `cudaMemcpyAsync` | 17 | 0 | 208 |
| `cudaLaunchKernel` | 8 | 0 | 109 |
| `aten::slice` | 33 | 0 | 87 |
| `aten::empty_strided` | 10 | 0 | 66 |
| `cudaGraphLaunch` | 1 | 0 | 56 |
| `aten::view` | 29 | 0 | 54 |

## batch 16

render **4.88 ms** median (min 4.87, max 4.90, sd 0.01) = **3279 fps**  ·  set_state 0.14 ms  ·  peak VRAM 40 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 3307 | 85.3 | 972 | 243 |
| host->device | 332 | 8.6 | 393 | 79 |
| kernel/raster | 238 | 6.1 | 151 | 15 |
| allocation | 0 | 0.0 | 64 | 15 |
| synchronization | 0 | 0.0 | 59 | 11 |

363 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 3219 | 0 |
| `Unrecognized` | 1 | 0 | 392 |
| `aten::_to_copy` | 10 | 0 | 237 |
| `cudaLaunchKernel` | 15 | 0 | 139 |
| `cudaMemcpyAsync` | 10 | 0 | 108 |
| `aten::slice` | 33 | 0 | 81 |
| `aten::index` | 1 | 80 | 18 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 80 | 0 |

## batch 64

render **13.44 ms** median (min 13.33, max 15.06, sd 0.81) = **4762 fps**  ·  set_state 0.14 ms  ·  peak VRAM 156 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 11201 | 86.1 | 2120 | 243 |
| host->device | 1032 | 7.9 | 198 | 79 |
| kernel/raster | 779 | 6.0 | 156 | 15 |
| allocation | 0 | 0.0 | 66 | 15 |
| synchronization | 0 | 0.0 | 664 | 11 |

363 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 11018 | 0 |
| `Unrecognized` | 1 | 0 | 1471 |
| `cudaDeviceSynchronize` | 3 | 0 | 639 |
| `aten::index` | 1 | 274 | 19 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 274 | 0 |
| `aten::add` | 1 | 209 | 13 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 1 | 209 | 0 |
| `aten::where` | 1 | 188 | 13 |

## batch 256

render **49.53 ms** median (min 49.17, max 49.91, sd 0.21) = **5168 fps**  ·  set_state 0.15 ms  ·  peak VRAM 621 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 44318 | 86.4 | 1355 | 243 |
| host->device | 3923 | 7.6 | 187 | 79 |
| kernel/raster | 3037 | 5.9 | 162 | 15 |
| allocation | 0 | 0.0 | 68 | 15 |
| synchronization | 0 | 0.0 | 3242 | 11 |

363 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 43735 | 0 |
| `cudaDeviceSynchronize` | 3 | 0 | 3216 |
| `aten::index` | 1 | 1092 | 19 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 1092 | 0 |
| `aten::add` | 1 | 830 | 13 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 1 | 830 | 0 |
| `aten::where` | 1 | 751 | 14 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 1 | 751 | 0 |

## batch 1024

render **193.84 ms** median (min 193.78, max 193.98, sd 0.06) = **5283 fps**  ·  set_state 0.19 ms  ·  peak VRAM 1355 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 177230 | 86.4 | 3111 | 614 |
| host->device | 15679 | 7.6 | 408 | 193 |
| kernel/raster | 12123 | 5.9 | 437 | 39 |
| allocation | 0 | 0.0 | 134 | 27 |
| synchronization | 0 | 0.0 | 11554 | 17 |

890 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 4 | 174899 | 0 |
| `cudaDeviceSynchronize` | 6 | 0 | 11510 |
| `aten::index` | 4 | 4374 | 72 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 4 | 4374 | 0 |
| `aten::add` | 4 | 3323 | 47 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 4 | 3323 | 0 |
| `aten::where` | 4 | 3006 | 49 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 3006 | 0 |

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
