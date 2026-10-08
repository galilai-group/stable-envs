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

[`train_from_states.py`](train_from_states.py) and
[`ddp_train_from_states.py`](ddp_train_from_states.py) are legacy experiments using
compact rows and the former chaining API. They require migration before use with
the current full-State contract; they are not validated getting-started examples.
Current applications store complete State snapshots in their own dataset/loader
and call `env.render(batch)` after transfer. Rendering does not require calling
`set_state` first. See the [State contract](../docs/concepts/state.md).
stable-envs does not own dataset storage or policy rollouts.

## Other current examples

- [`domain_randomization.py`](domain_randomization.py) — batched Variation / configuration
- [`dump_frames.py`](dump_frames.py) — write frames from stored states
- [`gallery.py`](gallery.py) — side-by-side media on identical states
- [`bench_gpu_vs_cpu.py`](bench_gpu_vs_cpu.py) / [`bench_vs_decode.py`](bench_vs_decode.py) — renderer throughput
- [`bench_storage.py`](bench_storage.py) / [`bench_lance.py`](bench_lance.py) — storage pipeline baselines
- [`dino_agreement.py`](dino_agreement.py) — DINO agreement measurement
- [`rendernet/`](rendernet/README.md) — residual corrector training
- [`robocasa_visual/`](robocasa_visual/README.md) — RoboCasa visual pipeline
