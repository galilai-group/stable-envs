# Examples

| script | shows |
|---|---|
| [`render.py`](render.py) | construct an env, `reset`, `render(state)` |
| [`batched_env.py`](batched_env.py) | `num_envs`, specs, short batches, lifecycle |
| [`action_sequences.py`](action_sequences.py) | single actions, action chunks, State and frame selection |
| [`domain_randomization.py`](domain_randomization.py) | batched Variation |
| [`gallery.py`](gallery.py) | side-by-side images on identical States |
| [`bench_gpu_vs_cpu.py`](bench_gpu_vs_cpu.py) | renderer throughput |
| [`bench_pusht_source.py`](bench_pusht_source.py) | PushT `env.render` against the pinned source, same core |

```bash
python examples/render.py --help
python examples/batched_env.py --help
python examples/action_sequences.py
```

## Training from stored State

Store full State snapshots in your own dataset. Call `env.render(batch)` on them after transfer. You
do not need to call `set_state` first. See the [State contract](../docs/concepts/state.md).

## LeWM world model

[`lewm/`](lewm/README.md) trains and evaluates LeWM (stable-worldmodel) on stable-envs PushT. The training
script draws the frames on the training GPU from the state rows of the dataset.
