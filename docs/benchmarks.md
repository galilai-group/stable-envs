# Benchmarks

This page covers speed: how long it takes to turn a State into a uint8 frame. Correctness is covered
by the [test suite](testing.md).

## Method

Each run takes States from the env's own reset distribution or saved episodes and renders `(H, W, 3)`
uint8 frames. The GPU side calls `env.render(state)` at the class's native supersampling. The CPU side
uses the env's official reference renderer, where one exists. Both sides do the same job on the same
machine.

The maintained harnesses are `python -m benchmarks.render` and `python -m benchmarks.pipeline`. See
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
- Visual-tier 3-D families are for live training data, not a substitute for stored official pixels.
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
