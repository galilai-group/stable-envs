# Examples

The getting-started examples use only the public `stable_envs` API:

| script | demonstrates |
|---|---|
| [`render.py`](render.py) | construct an env, `reset`, and `render(state)` |
| [`batched_env.py`](batched_env.py) | `num_envs`, specs, short batches, lifecycle |
| [`functional_dynamics.py`](functional_dynamics.py) | stateful `step` vs pure `transition` / `unroll` |

```bash
python examples/render.py --help
python examples/batched_env.py --help
python examples/functional_dynamics.py --help
```

## Training from compact state

[`train_from_states.py`](train_from_states.py) and
[`ddp_train_from_states.py`](ddp_train_from_states.py) show the intended training path: store compact
states in your own dataset/loader and call `env.render(env.unpack_render_state(batch))` after transfer. stable-envs
does not own dataset storage or policy rollouts; those responsibilities belong to your data stack and
sibling stable-* projects.

## Measurement and research scripts

The remaining scripts are benchmark/research harnesses used to produce the evidence in
[`results/`](results/) and the research archive. They are not part of the getting-started API. In
particular, fidelity and throughput claims should be read from the generated artifacts rather than
reproduced from prose in this file.

Useful entry points include:

- [`bench_gpu_vs_cpu.py`](bench_gpu_vs_cpu.py) — renderer throughput vs official CPU references;
- [`bench_vs_decode.py`](bench_vs_decode.py) — regeneration vs JPEG decode;
- [`bench_step3.py`](bench_step3.py) / [`bench_step3_pipeline.py`](bench_step3_pipeline.py) — pipeline benches;
- [`fidelity_step4.py`](fidelity_step4.py) — frozen-corpus fidelity/conformance measurement;
- [`bench_storage.py`](bench_storage.py) / [`bench_lance.py`](bench_lance.py) — storage pipeline baselines;
- [`gallery.py`](gallery.py) — side-by-side media on identical states;
- [`domain_randomization.py`](domain_randomization.py) — batched Variation / configuration;
- [`dump_frames.py`](dump_frames.py) — write frames from stored states;
- [`dino_agreement.py`](dino_agreement.py) — DINO agreement measurement.

Historical milestone-specific scripts and large calibration artifacts are preserved in release/tag
archives rather than defining the public API. Step-5 SWM paired-eval harnesses live under
[`docs/research/artifacts/`](../docs/research/artifacts/README.md), not in this directory.
