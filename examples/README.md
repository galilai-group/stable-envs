# Examples

Each script runs on CPU with small defaults, and the scripts with options list them with `--help`. The scripts aren't part of the PyPI package. Clone the repository and run them from its root:

```bash
git clone https://github.com/galilai-group/stable-envs
cd stable-envs
```

`gallery.py` and `bench_gpu_vs_cpu.py` without `--envs` run every registered class and skip the classes whose extra is missing. `render.py --out` needs Pillow (`pip install pillow`).

| Script | Shows |
|---|---|
| [`render.py`](render.py) | construct an environment, `reset`, and `render(state)` (PushT by default, so it needs the `pusht` extra) |
| [`batched_env.py`](batched_env.py) | `num_envs`, specs, a short last batch, and closing with a `with` block |
| [`action_sequences.py`](action_sequences.py) | single actions, action chunks, and picking which states and frames a step returns |
| [`domain_randomization.py`](domain_randomization.py) | rendering one state under many sampled variations in one call |
| [`gallery.py`](gallery.py) | PNG sheets of reset frames, a short rollout and sampled variation for each environment |
| [`bench_gpu_vs_cpu.py`](bench_gpu_vs_cpu.py) | `env.render(state)` throughput on CPU and CUDA, from states reset with the same seed |
| [`bench_pusht_source.py`](bench_pusht_source.py) | PushT `env.render(state)` against the pinned stable-worldmodel PushT with one torch thread (set `NUMBA_NUM_THREADS=1` for one CPU thread; needs stable-worldmodel on `PYTHONPATH`) |
| [`lewm/`](lewm/README.md) | training and evaluating stable-worldmodel's LeWM world model on PushT, with frames that `env.render(state)` draws on the training device from the state rows of the dataset |

```bash
python examples/render.py --help
python examples/batched_env.py --help
python examples/action_sequences.py
```

## Training from stored state

Store full state snapshots in your own dataset, then call `env.render(batch)` on them after you load or transfer them. You don't need to call `set_state` first, because `render` draws whatever state you pass in without touching the running environments. See [Store states, render later](../docs/tutorials/store-and-render.md).
