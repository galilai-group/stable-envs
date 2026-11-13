# Benchmarks

This page covers how long it takes to turn a State into a uint8 frame. The
[test suite](testing.md) covers correctness.

## Method

Each run takes States from the env's own reset distribution or saved episodes and renders `(H, W, 3)`
uint8 frames at the class's native supersampling. The maintained harnesses measure different things:

- `python -m benchmarks.render` times `env.render` on a reset State over a range of batch sizes. It
  measures throughput and batch scaling. It does not run a reference renderer.
- `python -m benchmarks.pipeline` times the path from a host State to a frame: the copy to the device
  and the `env.render` call. With `--jpeg-dir`, it adds a single-process Pillow baseline that
  decodes real JPEG files.

Neither harness imports stable-worldmodel. See
[benchmarks/README.md](https://github.com/galilai-group/stable-envs/blob/main/benchmarks/README.md)
for their options and output schema.

## PushT

On CPU, PushT draws with a Numba rasterizer (see [PushT](environments/pusht.md)). Its frames equal
the pygame frames of the pinned stable-worldmodel source, so the useful comparison is cost per frame
against `backend="source"`, which draws with that pygame code in 8 render worker processes for batches
of 64 or more. `NUMBA_NUM_THREADS=N OMP_NUM_THREADS=N MKL_NUM_THREADS=N python -m benchmarks.render
--envs PushT --device cpu --size 224 --batches 256,1024 --warmup 2 --repeats 7 --threads N` with N = 1
or 64 (Numba takes its thread count from `NUMBA_NUM_THREADS`, not `--threads`) on one node (Xeon Gold
6548Y+, 64 CPUs allocated), median of 3 runs:

| backend | threads | batch 256 ms/frame | batch 1,024 ms/frame |
|---|---:|---:|---:|
| `"auto"` (Numba) | 1 | 0.095 | 0.093 |
| `"auto"` (Numba) | 64 | 0.014 | 0.0082 |
| `"source"` (pygame) | 64 | 0.184 | 0.129 |

`benchmarks.render` does not time the first render of a batch. It builds one pymunk scene per lane,
and the first Numba render in a process also loads the compiled kernels.

## Notes

- Ray-cast and CPU3D families (OGBench mazes, DM Control, navigation) depend on the host CPU. Their
  numbers change with machine load.
- Some classes draw with a different rasterizer from the source. Their frames differ from the source
  frames. `env.describe()["visual_fidelity"]` gives the measured agreement.
- The official renderers run in one process. They have no batched entry point.
- Use the class's native supersampling. Forcing `1` on a class that declares `2` measures a cheaper
  picture than `snv.make(name)` returns.

## Reproduce

```bash
python -m benchmarks.render --help
python -m benchmarks.pipeline --help
python examples/bench_pusht_source.py --help
python examples/bench_gpu_vs_cpu.py --out results.json
```

`bench_gpu_vs_cpu.py` times `env.render(state)` for every env in `snv.list_envs()` on CPU and, when
available, on CUDA, from the same States.
