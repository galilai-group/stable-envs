# Examples

The getting-started examples use only the public `stable_envs` API:

| script | demonstrates |
|---|---|
| [`render.py`](render.py) | construct an env, `reset`, and `render(state)` |
| [`batched_env.py`](batched_env.py) | `num_envs`, specs, short batches, lifecycle |
| [`action_sequences.py`](action_sequences.py) | single actions, known action chunks, independent State/frame slices and endpoint reuse |

```bash
python examples/render.py --help
python examples/batched_env.py --help
python examples/action_sequences.py
```

## Training from stored State

Current applications store complete State snapshots in their own dataset/loader
and call `env.render(batch)` after transfer. Rendering does not require calling
`set_state` first. See the [State contract](../docs/concepts/state.md).
stable-envs does not own dataset storage or policy rollouts.

## Other current examples

- [`domain_randomization.py`](domain_randomization.py) — batched Variation / configuration
- [`gallery.py`](gallery.py) — side-by-side media on identical states
- [`bench_gpu_vs_cpu.py`](bench_gpu_vs_cpu.py) — renderer throughput
- [`bench_pusht_source.py`](bench_pusht_source.py) — PushT `env.render` against the pinned source, same core
- [`rendernet/`](rendernet/README.md) — residual corrector training
- [`robocasa_visual/`](robocasa_visual/README.md) — RoboCasa visual pipeline
