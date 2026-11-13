# Benchmarks

The benchmarks below time how long stable-envs takes to turn a [state](terms.md#state) into a uint8 [frame](terms.md#frame) and compare it with the upstream renderer. This page gives the commands and what they measure. It gives no results, because results change with the hardware and the versions. Run the harnesses from a clone of the repository. The installed wheel leaves out the `benchmarks` package.

```bash
python -m benchmarks.render --envs TwoRoom --device cpu --batches 64   # prints one JSON record
```

## PushT

On CPU, PushT draws with a Numba rasterizer whose frames equal the pygame frames of the pinned stable-worldmodel implementation. The useful comparison is the cost per frame against `backend="source"`, which draws with that pygame code in up to 8 render worker processes on Linux for batches of 64 or more. Set the thread count with `NUMBA_NUM_THREADS`, because Numba does not read `--threads`:

```bash
NUMBA_NUM_THREADS=N OMP_NUM_THREADS=N MKL_NUM_THREADS=N python -m benchmarks.render \
  --envs PushT --device cpu --size 224 --batches 256,1024 --warmup 2 --repeats 7 --threads N
```

`benchmarks.render` leaves the first render of a batch out of its timings. That render builds one Pymunk scene per batch row, and the first Numba render in a process also loads the compiled kernels.

## DM Control ray casting

The [ray caster](contributing/renderers.md#ray-casting-and-cpu3d) draws DM Control frames from saved states on CPU and CUDA. To compare it with the upstream renderer, give both the same states, output destination and device class, and leave the rollout, state loading, restoring and render-state preparation out of the timing. The [comparison with the upstream libraries](#comparison-with-the-upstream-libraries) does this and also writes the frame MSE.

## DM Control physics

`benchmarks/dmc_mjwarp.py` measures the latency and memory of native and MuJoCo Warp DM Control physics on CUDA. It times construction, reset, restore, public stepping and rendering separately, with the public API only, and its headline number is `public_step_no_pixels`, a full public step without frames or history. By default it runs all 17 classes on both backends at batches 1, 16, 64, 256 and 1024, with reset seeds 91000, 91010 and 91020, each case in a fresh process. `--batches` and `--seeds` change the matrix, `--extended-batches` adds 2048, 4096 and 8192, and `--measurement-profile focused` skips the repeated reset, restore and rendering phases. `--dry-run` prints the matrix without touching CUDA:

```bash
python benchmarks/dmc_mjwarp.py --dry-run --pretty \
  --envs DMCPendulumSwingup,DMCCheetahRun --batches 1024 --extended-batches
```

Remove `--dry-run` and add `--output results/dmc-mjwarp.json` to measure it. Measuring needs `psutil` and `nvidia-ml-py`. Each step replays the same seeded random actions on both backends after 20 warmup actions and a checkpoint restore, for 100 timed calls, and the results keep the hashes of the actions and the start checkpoint so that a pair with different inputs is marked as an error. The RSS peaks are process high-water marks, and the torch memory peaks are measured per phase after the warmup. `python benchmarks/dmc_mjwarp.py --help` lists every option.

## OGBench Cube and Scene on CUDA

The default for these classes is `supersample=1`, which draws with the NVIDIA OpenGL path. `supersample=2` draws with the [MJWarp renderer](contributing/renderers.md#mjwarp). MJWarp returns CUDA tensors. The OpenGL path returns NumPy images, so a timing of it includes the GPU readback. The OpenGL time changes with the GPU, the driver, the host and the MuJoCo version, so compare the two paths on the same machine.

## Method

Each run takes states from the environment's own reset distribution or from saved episodes and renders `(H, W, 3)` uint8 frames at the class's native supersampling. The two harnesses measure different things:

- `python -m benchmarks.render` times `env.render` on a reset state over a range of batch sizes. It measures the throughput and batch scaling of stable-envs alone, without the upstream renderer.
- `python -m benchmarks.pipeline` times the path from a host state to a frame: the copy to the device and the `env.render` call. With `--jpeg-dir`, it adds a single-process Pillow baseline that decodes real JPEG files.

Neither harness imports stable-worldmodel. [benchmarks/README.md](https://github.com/galilai-group/stable-envs/blob/main/benchmarks/README.md) has their options and output schema. Keep these in mind when you read the results:

- The ray-cast and CPU3D families (OGBench mazes, DM Control, navigation) depend on the host CPU, and their numbers change with machine load.
- The upstream renderers run in one process and have no batched entry point.
- Use the class's native supersampling. Forcing `1` on a class that declares `2` measures a cheaper picture than `snv.make(name)` returns.

## Comparison with the upstream libraries

`benchmarks/paper/run.sh` measures the speed of stable-envs against the upstream library of each family and the frame MSE between them. It pins each measurement to the same CPU cores and uses only the public `stable_envs` API. It has three parts:

- `offline`: stable-envs draws stored inputs with `env.render` at each batch size. The inputs are rows from `sample_states`, or full states from random-action rollouts for a class without rows. The upstream library draws the same inputs in 1 process and in one process for each core. The frame MSE compares the two on the 0-255 scale.
- `step`: physics steps at each batch size, against the upstream library stepping the same number of environments on CPU.
- `frames`: steps with a frame after each step.

`benchmarks/offline_sweep.py`, `benchmarks/online_ladder.py` and `benchmarks/draw.py` do the measurements. `benchmarks/paper/summarize.py` writes Markdown tables from one results directory, and `benchmarks/paper/sanity.py` checks the spread and the status of each cell. [benchmarks/README.md](https://github.com/galilai-group/stable-envs/blob/main/benchmarks/README.md#paper-harness) gives the options, the environment variables and the output files.

## Reproduce

```bash
python -m benchmarks.render --help
python -m benchmarks.pipeline --help
python examples/bench_pusht_source.py --help
python examples/bench_gpu_vs_cpu.py --out results.json
```

`bench_gpu_vs_cpu.py` times `env.render(state)` for every environment in `snv.list_envs()` on CPU and, when it's available, on CUDA, from states reset with the same seed on each device.
