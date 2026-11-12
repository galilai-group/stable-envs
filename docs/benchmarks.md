# Benchmarks

The benchmarks below time how long stable-envs takes to turn a [state](terms.md#state) into a uint8 [frame](terms.md#frame) and compare it with the upstream renderer, with the commands that reproduce each number. Run the harnesses from a clone of the repository. The installed wheel leaves out the `benchmarks` package.

```bash
python -m benchmarks.render --envs TwoRoom --device cpu --batches 64   # prints one JSON record
```

## PushT

On CPU, PushT draws with a Numba rasterizer whose frames equal the pygame frames of the pinned stable-worldmodel implementation. The useful comparison is the cost per frame against `backend="source"`, which draws with that pygame code in up to 8 render worker processes on Linux for batches of 64 or more.

| Backend | Threads | Batch 256, ms per frame | Batch 1,024, ms per frame |
|---|---:|---:|---:|
| `"auto"` (Numba) | 1 | 0.095 | 0.093 |
| `"auto"` (Numba) | 64 | 0.014 | 0.0082 |
| `"source"` (pygame) | 64 | 0.184 | 0.129 |

With 64 threads at batch 1,024, Numba takes 0.0082 ms per frame, about 16× less than the pygame path.

??? note "Hardware and command"

    One node with a Xeon Gold 6548Y+ and 64 CPUs allocated, median of 3 runs, with N = 1 or 64:

    ```bash
    NUMBA_NUM_THREADS=N OMP_NUM_THREADS=N MKL_NUM_THREADS=N python -m benchmarks.render \
      --envs PushT --device cpu --size 224 --batches 256,1024 --warmup 2 --repeats 7 --threads N
    ```

    Numba takes its thread count from `NUMBA_NUM_THREADS`, not from `--threads`.

`benchmarks.render` leaves the first render of a batch out of its timings. That render builds one Pymunk scene per batch row, and the first Numba render in a process also loads the compiled kernels.

## DM Control ray casting

`DMCBallInCupCatch` at batch 64, rendered from saved states by the [ray caster](contributing/renderers.md#ray-casting-and-cpu3d) and by the upstream renderer (NVIDIA OpenGL on the GPU row), with the same states, output destination and device class:

| Device and output | stable-envs FPS | Upstream FPS | Ratio | MSE (0-255), mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| One Xeon Platinum 8268 thread, CPU output | 15.6 | 14.9 | 1.05× | 3.1 ± 1.2 |
| One RTX 3090, CPU output | 2,190 | 240 | 9.1× | 3.6 ± 1.5 |

Sixteen states were repeated to fill each batch, and each timing used three warmed samples. Timing excludes the rollout, state loading, restoring and render-state preparation.

## DM Control physics

`benchmarks/dmc_mjwarp.py` measures the latency and memory of native and MuJoCo Warp DM Control physics on CUDA. It times construction, reset, restore, raw physics, public stepping and rendering separately, and its headline number is `public_step_no_pixels`, a full public step without frames or history. By default it runs all 17 classes on both backends at batches 1, 16, 64, 256 and 1024, with reset seeds 91000, 91010 and 91020, each case in a fresh process. `--batches` and `--seeds` change the matrix, `--extended-batches` adds 2048, 4096 and 8192, and `--measurement-profile focused` skips the repeated reset, restore and rendering phases. `--dry-run` prints the matrix without touching CUDA:

```bash
python benchmarks/dmc_mjwarp.py --dry-run --pretty \
  --envs DMCPendulumSwingup,DMCCheetahRun --batches 1024 --extended-batches
```

Remove `--dry-run` and add `--output results/dmc-mjwarp.json` to measure it. Measuring needs `psutil` and `nvidia-ml-py`. Each step replays the same seeded random actions on both backends after 20 warmup actions and a checkpoint restore, for 100 timed calls, and the results keep the hashes of the actions and the start checkpoint so that a pair with different inputs is marked as an error. The RSS peaks are process high-water marks, and the torch memory peaks are measured per phase after the warmup. `python benchmarks/dmc_mjwarp.py --help` lists every option.

## OGBench Cube and Scene on CUDA

The [MJWarp renderer](contributing/renderers.md#mjwarp) at `supersample=2` and 64 × 64, against the stable-envs NVIDIA OpenGL path with the same compiled appearance, on one H200 with MuJoCo and MuJoCo Warp 3.17. The default for these classes is `supersample=1`, which uses OpenGL. MJWarp returns CUDA tensors. The OpenGL path returns NumPy images, and its time includes the GPU readback.

| Class | Batch | MJWarp FPS | OpenGL FPS | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Cube | 1 | 237 | 261 | 0.907× |
| Cube | 64 | 8,147 | 264 | 30.8× |
| Cube | 256 | 13,506 | 264 | 51.1× |
| Scene | 1 | 184 | 230 | 0.803× |
| Scene | 64 | 6,160 | 230 | 26.7× |
| Scene | 256 | 9,971 | 230 | 43.3× |

The run used 32 saved states, each reached independently and repeated to fill the larger batches, with three warm-up and six timed samples. Timing includes the public `render` call and synchronization, and excludes the rollout, loading, projection, the first transfer and batch indexing. FPS values are rounded, and the ratios come from the unrounded timings.

These ratios are specific to this H200. They are amplified because the OpenGL baseline ran about 6–7× slower here than in earlier RTX 3090 runs, and the GPU, driver, host and MuJoCo version all changed, so the comparison doesn't isolate a hardware effect.

## nvdiffrast on CUDA

The [nvdiffrast renderer](contributing/renderers.md#nvdiffrast) drew a textured antmaze-medium scene at 12,992 FPS at batch 256 and 15,994 FPS at batch 4,096, at 224 px on one H200. The poses were already on the GPU, and the timing leaves out simulation and state preparation.

## Method

Each run takes states from the environment's own reset distribution or from saved episodes and renders `(H, W, 3)` uint8 frames at the class's native supersampling. The two harnesses measure different things:

- `python -m benchmarks.render` times `env.render` on a reset state over a range of batch sizes. It measures the throughput and batch scaling of stable-envs alone, without the upstream renderer.
- `python -m benchmarks.pipeline` times the path from a host render state to a frame: the copy to the device and the render call. With `--jpeg-dir`, it adds a single-process Pillow baseline that decodes real JPEG files.

Neither harness imports stable-worldmodel. [benchmarks/README.md](https://github.com/galilai-group/stable-envs/blob/main/benchmarks/README.md) has their options and output schema. Keep these in mind when you read the numbers:

- The ray-cast and CPU3D families (OGBench mazes, DM Control, navigation) depend on the host CPU, and their numbers change with machine load.
- The upstream renderers run in one process and have no batched entry point.
- Use the class's native supersampling. Forcing `1` on a class that declares `2` measures a cheaper picture than `snv.make(name)` returns.

## Reproduce

```bash
python -m benchmarks.render --help
python -m benchmarks.pipeline --help
python examples/bench_pusht_source.py --help
python examples/bench_gpu_vs_cpu.py --out results.json
```

`bench_gpu_vs_cpu.py` times `env.render(state)` for every environment in `snv.list_envs()` on CPU and, when it's available, on CUDA, from states reset with the same seed on each device.
