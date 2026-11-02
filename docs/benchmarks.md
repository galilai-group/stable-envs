# Benchmarks

This page covers how long it takes to turn a State into a uint8 frame. The
[test suite](testing.md) covers correctness.

## Method

Each run takes States from the env's own reset distribution or saved episodes and renders `(H, W, 3)`
uint8 frames at the class's native supersampling. The maintained harnesses measure different things:

- `python -m benchmarks.render` times `env.render` on a reset State over a range of batch sizes. It
  measures throughput and batch scaling. It does not run a reference renderer.
- `python -m benchmarks.pipeline` times the path from a host render State to a frame: the copy to
  the device and the render call. With `--jpeg-dir`, it adds a single-process Pillow baseline that
  decodes real JPEG files.

Neither harness imports stable-worldmodel. See
[benchmarks/README.md](https://github.com/galilai-group/stable-envs/blob/main/benchmarks/README.md)
for their options and output schema.

## PushT

PushT draws with the source's pygame code on CPU. `device="cuda"` only places the output. Its frames
equal the pinned stable-worldmodel source, so the useful comparison is cost per frame against that
source on the same core. `examples/bench_pusht_source.py` renders States saved from four episodes
both ways, on one thread, at 224 px:

| CPU | batch | first `env.render` ms/frame | warm `env.render` ms/frame | source ms/frame | warm vs source |
|---|---:|---:|---:|---:|---:|
| Apple M5 Pro | 64 | 1.08 | 0.55 | 0.66 | 1.22× |
| Apple M5 Pro | 1,024 | 0.88 | 0.59 | 0.67 | 1.13× |
| Xeon Platinum 8268 | 64 | 4.98 | 3.69 | 3.04 | 0.83× |
| Xeon Platinum 8268 | 1,024 | 5.19 | 4.02 | 3.28 | 0.82× |

The first render of a batch builds one pymunk scene per lane. Later renders reuse it. The pygame
drawing is the same on both sides. The difference is Python work per frame.

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
