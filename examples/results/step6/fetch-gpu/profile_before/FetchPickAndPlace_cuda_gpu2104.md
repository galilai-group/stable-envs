# Profile note — FetchPickAndPlace on cuda

- workstream: `step6/fetch-gpu`
- baseline commit: `d148f1a582d9`
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 8
- frame: 480px  ·  state_dim: 25
- states: `examples/results/step4/states/FetchPickAndPlace.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **2.99 ms** median (min 2.45, max 3.20, sd 0.34) = **334 fps**  ·  set_state 0.25 ms  ·  peak VRAM 4 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 333 | 72.4 | 6356 | 253 |
| host->device | 90 | 19.4 | 352 | 83 |
| kernel/raster | 38 | 8.2 | 426 | 13 |
| allocation | 0 | 0.0 | 146 | 15 |
| synchronization | 0 | 0.0 | 67 | 11 |

375 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `aten::abs` | 2 | 0 | 1971 |
| `aten::expand_as` | 2 | 0 | 1813 |
| `Unrecognized` | 1 | 0 | 1264 |
| `cudaMemcpyAsync` | 18 | 0 | 381 |
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 285 | 0 |
| `cudaLaunchKernel` | 8 | 0 | 190 |
| `aten::slice` | 36 | 0 | 177 |
| `aten::empty_strided` | 10 | 0 | 101 |

## batch 16

render **4.93 ms** median (min 4.90, max 4.97, sd 0.02) = **3248 fps**  ·  set_state 0.22 ms  ·  peak VRAM 40 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 3340 | 85.3 | 1960 | 252 |
| host->device | 337 | 8.6 | 206 | 81 |
| kernel/raster | 240 | 6.1 | 173 | 15 |
| allocation | 0 | 0.0 | 66 | 15 |
| synchronization | 0 | 0.0 | 52 | 11 |

374 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 3249 | 0 |
| `Unrecognized` | 1 | 0 | 1340 |
| `cudaLaunchKernel` | 16 | 0 | 150 |
| `cudaMemcpyAsync` | 10 | 0 | 117 |
| `aten::slice` | 36 | 0 | 91 |
| `aten::index` | 1 | 80 | 18 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 80 | 0 |
| `aten::where` | 1 | 55 | 12 |

## batch 64

render **13.57 ms** median (min 13.49, max 14.89, sd 0.61) = **4717 fps**  ·  set_state 0.14 ms  ·  peak VRAM 156 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 11355 | 86.2 | 1894 | 252 |
| host->device | 1038 | 7.9 | 206 | 81 |
| kernel/raster | 784 | 6.0 | 171 | 15 |
| allocation | 0 | 0.0 | 69 | 15 |
| synchronization | 0 | 0.0 | 680 | 11 |

374 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 11170 | 0 |
| `Unrecognized` | 1 | 0 | 1265 |
| `cudaDeviceSynchronize` | 3 | 0 | 655 |
| `aten::index` | 1 | 275 | 18 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 275 | 0 |
| `aten::add` | 1 | 210 | 12 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 1 | 210 | 0 |
| `aten::where` | 1 | 189 | 13 |

## batch 256

render **49.91 ms** median (min 49.60, max 49.98, sd 0.15) = **5130 fps**  ·  set_state 0.17 ms  ·  peak VRAM 621 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 44786 | 86.5 | 1873 | 252 |
| host->device | 3929 | 7.6 | 200 | 81 |
| kernel/raster | 3038 | 5.9 | 178 | 15 |
| allocation | 0 | 0.0 | 69 | 15 |
| synchronization | 0 | 0.0 | 3213 | 11 |

374 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 44201 | 0 |
| `cudaDeviceSynchronize` | 3 | 0 | 3187 |
| `aten::index` | 1 | 1091 | 25 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 1091 | 0 |
| `cudaMemcpyAsync` | 10 | 0 | 963 |
| `aten::add` | 1 | 831 | 13 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 1 | 831 | 0 |
| `aten::where` | 1 | 750 | 15 |

## batch 1024

render **195.60 ms** median (min 195.46, max 195.82, sd 0.10) = **5235 fps**  ·  set_state 0.22 ms  ·  peak VRAM 1355 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 178903 | 86.5 | 3416 | 650 |
| host->device | 15699 | 7.6 | 430 | 201 |
| kernel/raster | 12124 | 5.9 | 443 | 39 |
| allocation | 0 | 0.0 | 124 | 27 |
| synchronization | 0 | 0.0 | 11332 | 17 |

934 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 4 | 176564 | 0 |
| `cudaDeviceSynchronize` | 6 | 0 | 11289 |
| `aten::index` | 4 | 4373 | 74 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 4 | 4373 | 0 |
| `aten::add` | 4 | 3323 | 47 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 4 | 3323 | 0 |
| `aten::where` | 4 | 3008 | 50 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 3008 | 0 |

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
