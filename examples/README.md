# Examples

The getting-started examples use only the public `stable_envs` API:

| script | demonstrates |
|---|---|
| [`render.py`](render.py) | construct an env, `reset`, and `render(state)` |
| [`batched_env.py`](batched_env.py) | `num_envs`, specs, short batches, lifecycle |
| [`action_sequences.py`](action_sequences.py) | single actions, batch-first sequences and capture |

```bash
python examples/render.py --help
python examples/batched_env.py --help
python examples/action_sequences.py
```

## Training from compact state

[`train_from_states.py`](train_from_states.py) and
[`ddp_train_from_states.py`](ddp_train_from_states.py) show the intended training path: store compact
states in your own dataset/loader and call `env.render(env.unpack_render_state(batch))` after transfer. stable-envs
does not own dataset storage or policy rollouts; those responsibilities belong to your data stack and
sibling stable-* projects.

## Other current examples

- [`domain_randomization.py`](domain_randomization.py) — batched Variation / configuration
- [`dump_frames.py`](dump_frames.py) — write frames from stored states
- [`gallery.py`](gallery.py) — side-by-side media on identical states
- [`bench_gpu_vs_cpu.py`](bench_gpu_vs_cpu.py) / [`bench_vs_decode.py`](bench_vs_decode.py) — renderer throughput
- [`bench_storage.py`](bench_storage.py) / [`bench_lance.py`](bench_lance.py) — storage pipeline baselines
- [`dino_agreement.py`](dino_agreement.py) — DINO agreement measurement
- [`rendernet/`](rendernet/README.md) — residual corrector training
- [`robocasa_visual/`](robocasa_visual/README.md) — RoboCasa visual pipeline

Frozen measurement harnesses, paper tables, and historical result dumps live under
[`docs/research/artifacts/`](../docs/research/artifacts/README.md).
