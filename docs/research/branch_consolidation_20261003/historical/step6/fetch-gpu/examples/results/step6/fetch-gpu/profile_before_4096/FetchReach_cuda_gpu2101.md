# Profile note — FetchReach on cuda

- workstream: `step6/fetch-gpu`
- baseline commit: ``
- host: gpu2101.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- frame: 480px  ·  state_dim: 18
- states: `examples/results/step4/states/FetchReach.npz`
- torch 2.14.0+cu126  ·  triton 3.8.0

## batch 1

render **1.65 ms** median (min 1.63, max 1.86, sd 0.07) = **607 fps**  ·  set_state 0.12 ms  ·  peak VRAM 4 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 334 | 72.9 | 2511 | 244 |
| host->device | 86 | 18.8 | 311 | 81 |
| kernel/raster | 38 | 8.2 | 327 | 13 |
| allocation | 0 | 0.0 | 137 | 15 |
| synchronization | 0 | 0.0 | 60 | 11 |

364 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `Unrecognized` | 1 | 0 | 1289 |
| `cudaMemcpyAsync` | 17 | 0 | 383 |
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 287 | 0 |
| `cudaLaunchKernel` | 8 | 0 | 163 |
| `aten::slice` | 33 | 0 | 141 |
| `aten::empty_strided` | 10 | 0 | 91 |
| `aten::view` | 29 | 0 | 82 |
| `aten::as_strided` | 51 | 0 | 73 |

## batch 16

render **4.95 ms** median (min 4.87, max 4.98, sd 0.03) = **3233 fps**  ·  set_state 0.14 ms  ·  peak VRAM 41 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 3080 | 85.0 | 1489 | 243 |
| host->device | 318 | 8.8 | 196 | 79 |
| kernel/raster | 226 | 6.2 | 164 | 15 |
| allocation | 0 | 0.0 | 69 | 15 |
| synchronization | 0 | 0.0 | 43 | 11 |

363 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 2995 | 0 |
| `Unrecognized` | 1 | 0 | 860 |
| `cudaLaunchKernel` | 15 | 0 | 148 |
| `cudaMemcpyAsync` | 10 | 0 | 118 |
| `aten::slice` | 33 | 0 | 84 |
| `aten::index` | 1 | 74 | 19 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 74 | 0 |
| `aten::add` | 1 | 54 | 13 |

## batch 64

render **14.06 ms** median (min 14.05, max 14.08, sd 0.01) = **4551 fps**  ·  set_state 0.14 ms  ·  peak VRAM 157 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 11216 | 86.1 | 930 | 243 |
| host->device | 1033 | 7.9 | 195 | 79 |
| kernel/raster | 782 | 6.0 | 156 | 15 |
| allocation | 0 | 0.0 | 68 | 15 |
| synchronization | 0 | 0.0 | 669 | 11 |

363 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 11032 | 0 |
| `cudaDeviceSynchronize` | 3 | 0 | 643 |
| `Unrecognized` | 1 | 0 | 327 |
| `aten::index` | 1 | 274 | 18 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 274 | 0 |
| `aten::add` | 1 | 209 | 12 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 1 | 209 | 0 |
| `aten::where` | 1 | 189 | 13 |

## batch 256

render **49.18 ms** median (min 49.15, max 49.49, sd 0.11) = **5206 fps**  ·  set_state 0.15 ms  ·  peak VRAM 622 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 44183 | 86.4 | 1190 | 243 |
| host->device | 3915 | 7.7 | 207 | 79 |
| kernel/raster | 3033 | 5.9 | 177 | 15 |
| allocation | 0 | 0.0 | 70 | 15 |
| synchronization | 0 | 0.0 | 3212 | 11 |

363 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 1 | 43600 | 0 |
| `cudaDeviceSynchronize` | 3 | 0 | 3187 |
| `aten::index` | 1 | 1090 | 20 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 1 | 1090 | 0 |
| `aten::add` | 1 | 830 | 23 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 1 | 830 | 0 |
| `aten::where` | 1 | 748 | 14 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 1 | 748 | 0 |

## batch 1024

render **193.56 ms** median (min 193.28, max 193.71, sd 0.13) = **5290 fps**  ·  set_state 0.20 ms  ·  peak VRAM 1355 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 176882 | 86.4 | 3057 | 614 |
| host->device | 15650 | 7.6 | 419 | 193 |
| kernel/raster | 12099 | 5.9 | 467 | 39 |
| allocation | 0 | 0.0 | 123 | 27 |
| synchronization | 0 | 0.0 | 11435 | 17 |

890 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 4 | 174549 | 0 |
| `cudaDeviceSynchronize` | 6 | 0 | 11393 |
| `aten::index` | 4 | 4360 | 74 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 4 | 4360 | 0 |
| `aten::add` | 4 | 3324 | 47 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 4 | 3324 | 0 |
| `aten::where` | 4 | 2994 | 50 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 4 | 2994 | 0 |

## batch 4096

render **771.43 ms** median (min 771.26, max 771.77, sd 0.15) = **5310 fps**  ·  set_state 0.28 ms  ·  peak VRAM 3380 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| other | 711221 | 86.5 | 6006 | 2094 |
| host->device | 62673 | 7.6 | 1278 | 649 |
| kernel/raster | 48463 | 5.9 | 1541 | 135 |
| allocation | 0 | 0.0 | 340 | 76 |
| synchronization | 0 | 0.0 | 44827 | 41 |

2995 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `render__locals___render_megakernel_cbc9e69b_cuda_kernel_forward` | 16 | 701887 | 0 |
| `cudaDeviceSynchronize` | 18 | 0 | 44721 |
| `aten::index` | 16 | 17506 | 281 |
| `void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<1> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})` | 16 | 17506 | 0 |
| `aten::add` | 16 | 13288 | 194 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctorOnSelf_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctorOnSelf_add<int> const&)::{lambda(int)#1})` | 16 | 13288 | 0 |
| `aten::where` | 16 | 12032 | 190 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::where_kernel_impl(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(bool, unsigned char, unsigned char)#1} const&)::{lambda(int)#1})` | 16 | 12032 | 0 |

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
