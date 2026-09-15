# Profile note — DMCManipulatorBringBall on cuda

- workstream: `step7/dmc-manipulator`
- baseline commit: ``
- host: gpu2104.oscar.ccv.brown.edu  ·  gpu: NVIDIA GeForce RTX 3090  ·  cpu: x86_64  ·  threads: 4
- tree measured: `/oscar/scratch/sboughan/se7/dmc-manipulator/stable_envs/__init__.py`  ·  load average 38.0
- frame: 224px  ·  state_dim: 15
- states: `env.sample_states (no frozen corpus for this env)`
- torch 2.14.0+cu126  ·  triton 3.8.0
- measured tree: `$SCR/se7/dmc-manipulator/stable_envs/__init__.py`

## batch 1

render **5.49 ms** median (min 5.47, max 5.51, sd 0.01) = **182 fps**  ·  set_state 0.12 ms  ·  peak VRAM 6 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 536 | 48.2 | 290 | 351 |
| kernel/raster | 361 | 32.4 | 2340 | 264 |
| other | 214 | 19.2 | 4612 | 1280 |
| output conversion | 2 | 0.2 | 14 | 2 |
| allocation | 0 | 0.0 | 344 | 71 |
| synchronization | 0 | 0.0 | 55 | 17 |

1985 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 243 | 0 | 1671 |
| `Unrecognized` | 1 | 0 | 858 |
| `aten::select` | 113 | 0 | 247 |
| `aten::empty` | 34 | 0 | 169 |
| `cudaMemcpyAsync` | 12 | 0 | 147 |
| `aten::empty_strided` | 24 | 0 | 137 |
| `aten::as_strided` | 229 | 0 | 136 |
| `aten::view` | 69 | 0 | 115 |

## batch 16

render **7.88 ms** median (min 7.86, max 8.13, sd 0.09) = **2030 fps**  ·  set_state 0.13 ms  ·  peak VRAM 53 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 1023 | 45.6 | 285 | 432 |
| kernel/raster | 695 | 30.9 | 2350 | 254 |
| other | 503 | 22.4 | 4779 | 1426 |
| output conversion | 25 | 1.1 | 15 | 3 |
| allocation | 0 | 0.0 | 370 | 74 |
| synchronization | 0 | 0.0 | 166 | 17 |

2206 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 327 | 0 | 2014 |
| `Unrecognized` | 1 | 0 | 995 |
| `_tri_cast` | 4 | 297 | 0 |
| `aten::select` | 113 | 0 | 253 |
| `aten::mul` | 97 | 228 | 844 |
| `aten::empty` | 35 | 0 | 168 |
| `cudaStreamSynchronize` | 11 | 0 | 156 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 56 | 154 | 0 |

## batch 64

render **9.67 ms** median (min 9.64, max 9.69, sd 0.01) = **6620 fps**  ·  set_state 0.14 ms  ·  peak VRAM 237 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 3531 | 45.0 | 269 | 432 |
| kernel/raster | 2676 | 34.1 | 2335 | 254 |
| other | 1555 | 19.8 | 4071 | 1426 |
| output conversion | 93 | 1.2 | 12 | 3 |
| allocation | 0 | 0.0 | 330 | 74 |
| synchronization | 0 | 0.0 | 1149 | 17 |

2206 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaLaunchKernel` | 327 | 0 | 1993 |
| `aten::mul` | 97 | 1084 | 838 |
| `_tri_cast` | 4 | 1081 | 0 |
| `cudaStreamSynchronize` | 11 | 0 | 786 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 56 | 781 | 0 |
| `aten::add` | 50 | 604 | 397 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 40 | 549 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 358 |

## batch 256

render **23.65 ms** median (min 23.54, max 23.88, sd 0.14) = **10823 fps**  ·  set_state 0.14 ms  ·  peak VRAM 936 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 11962 | 45.0 | 365 | 432 |
| kernel/raster | 9192 | 34.6 | 2439 | 254 |
| other | 5034 | 19.0 | 5113 | 1426 |
| output conversion | 365 | 1.4 | 13 | 3 |
| allocation | 0 | 0.0 | 347 | 74 |
| synchronization | 0 | 0.0 | 10516 | 17 |

2206 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaStreamSynchronize` | 11 | 0 | 8439 |
| `aten::mul` | 97 | 3817 | 856 |
| `_tri_cast` | 4 | 3652 | 0 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 56 | 2716 | 0 |
| `aten::add` | 50 | 2250 | 412 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 40 | 2091 | 0 |
| `cudaDeviceSynchronize` | 2 | 0 | 2070 |
| `cudaLaunchKernel` | 327 | 0 | 2044 |

## batch 1024

render **93.33 ms** median (min 92.98, max 95.96, sd 0.83) = **10972 fps**  ·  set_state 0.18 ms  ·  peak VRAM 1316 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 47437 | 45.0 | 1213 | 1674 |
| kernel/raster | 36357 | 34.5 | 9287 | 1007 |
| other | 20198 | 19.2 | 24014 | 5437 |
| output conversion | 1464 | 1.4 | 45 | 12 |
| allocation | 0 | 0.0 | 1232 | 278 |
| synchronization | 0 | 0.0 | 37378 | 50 |

8458 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaStreamSynchronize` | 35 | 0 | 37356 |
| `aten::mul` | 385 | 15048 | 3369 |
| `_tri_cast` | 16 | 14687 | 0 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 224 | 10704 | 0 |
| `cudaMemcpyAsync` | 35 | 0 | 8970 |
| `aten::add` | 200 | 8890 | 1586 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 160 | 8262 | 0 |
| `cudaLaunchKernel` | 1305 | 0 | 7884 |

## batch 4096

render **355.43 ms** median (min 354.45, max 356.24, sd 0.64) = **11524 fps**  ·  set_state 0.38 ms  ·  peak VRAM 1756 MiB

| phase (op-level attribution) | device us | device % | cpu us | calls |
|---|---|---|---|---|
| host->device | 189502 | 45.0 | 3363 | 5401 |
| kernel/raster | 145635 | 34.6 | 30185 | 3266 |
| other | 80047 | 19.0 | 80829 | 17474 |
| output conversion | 5851 | 1.4 | 135 | 39 |
| allocation | 0 | 0.0 | 3830 | 891 |
| synchronization | 0 | 0.0 | 149973 | 149 |

27220 operator calls per render. Top by device time:

| op | calls | device us | cpu us |
|---|---|---|---|
| `cudaStreamSynchronize` | 107 | 0 | 149395 |
| `aten::mul` | 1249 | 60496 | 10972 |
| `_tri_cast` | 52 | 58343 | 0 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1})` | 728 | 43024 | 0 |
| `aten::add` | 650 | 35839 | 5230 |
| `cudaMemcpyAsync` | 107 | 0 | 35134 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, false>(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)` | 520 | 33356 | 0 |
| `cudaLaunchKernel` | 4239 | 0 | 25419 |

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
