# Train a world model (LeWM)

This tutorial trains LeWM, the latent world model of [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), on PushT [frames](../terms.md#frame) that stable-envs draws during training. The dataset keeps only the state rows and the actions. After a batch moves to the training device, `env.render(state)` draws its frames on that device. The scripts are in [`examples/lewm`](https://github.com/galilai-group/stable-envs/blob/main/examples/lewm/README.md):

| Script | What it does |
|---|---|
| `make_dataset.py` | Copies a LeWM HDF5 dataset without its pixels (`--states-only`), or with new pixels that stable-envs draws from the state column. |
| `train.py` | Trains LeWM. `--render online` (the default) draws the frames from the state rows on the training device. `--render offline` reads the `pixels` column. |
| `eval.py` | Plans with CEM and the trained model on stable-envs environments, through stable-worldmodel's `World`. |
| `state_render.py` | Writes dataset state rows into a full stable-envs state and calls `env.render(state)`. |
| `config/pusht.yaml` | The environment, the state-row layout, the model, the training values and the planning values. |

The model, the loss, the optimizer, the data split and the planning values are those of stable-worldmodel `scripts/train/lewm.py` and `scripts/plan/eval_wm.py` at commit `21446f1`. The scripts use only the public stable-envs API: `snv.make`, `reset`, `get_state`, `step` and `render`.

## Prerequisites

Use Linux with an NVIDIA GPU for training. A CPU is sufficient for the small check at the end. Install stable-envs with the `pusht` extra, then stable-worldmodel at the pinned commit:

```bash
pip install -e ".[pusht]"        # from the stable-envs repository root
pip install "stable-worldmodel[train,format] @ git+https://github.com/galilai-group/stable-worldmodel@21446f1ede6d5284e981bd7b47f432b994e6d812" \
    "stable-pretraining==0.1.8" scikit-learn
```

On CUDA, PushT draws with Triton, which comes with the Linux PyTorch wheels.

## Draw frames from state rows

A LeWM PushT dataset holds one state row per step: agent x and y, block x and y, block angle, and agent velocity. `state_render.py` copies a full state from `env.reset(seed=0)` and writes the columns of each row into these leaves (`env.leaves` in the config):

| PushT state leaf | Row columns |
|---|---|
| `current.body.agent.position` | 0, 1 |
| `current.body.block.position` | 2, 3 |
| `current.body.block.angle` | 4 |

The other leaves (goal, colors, shapes) keep their reset values. This block does the same steps on CPU:

```python
import torch
import stable_envs as snv

rows = torch.tensor([[200.0, 300.0, 250.0, 250.0, 0.5], [320.0, 180.0, 260.0, 240.0, 1.2]])
leaves = {("current", "body", "agent", "position"): (0, 2),
          ("current", "body", "block", "position"): (2, 4),
          ("current", "body", "block", "angle"): (4, 5)}
with snv.PushT(num_envs=1, device="cpu") as env:
    env.reset(seed=0)
    state = env.get_state().expand(len(rows)).clone()  # one batch row per state row
    for key, (start, stop) in leaves.items():
        leaf = state.get(key)
        leaf.copy_(rows[:, start:stop].reshape(leaf.shape).to(leaf.dtype))
    for key in (("current", "render_segments"), ("current", "render_segment_count")):
        state.get(key).zero_()                         # the rows hold no contact points
    frames = env.render(state, size=224)
print(frames.shape, frames.dtype)  # torch.Size([2, 224, 224, 3]) torch.uint8
```

`render` draws the state that you give it. It does not change the running environment, so you do not call `set_state` first. The dataset rows hold no contact points, thus the frames show no contact markers. The source frames show a marker where the agent touches the block.

## Make the dataset

Get the LeWM PushT dataset (46 GB after decompression):

```bash
hf download quentinll/lewm-pusht pusht_expert_train.h5.zst --repo-type dataset --local-dir data
zstd -d data/pusht_expert_train.h5.zst -o data/pusht_expert_train.h5
```

Make the state dataset (about 160 MB). It keeps every column except `pixels`:

```bash
python examples/lewm/make_dataset.py data/pusht_expert_train.h5 data/pusht_state.h5 --states-only
```

## Train

Train with frames drawn on the GPU (half-size model, bf16, `torch.compile`):

```bash
python examples/lewm/train.py --data data/pusht_state.h5 --half --compile
```

The weights go to `runs/checkpoints/lewm-pusht/weights_epoch_N.pt`. `--out` and `--name` change this folder. Without `--half`, the model has the full size of stable-worldmodel's `lewm.yaml`. `--precision 32` trains in float32. Arguments after the options change the config, for example `trainer.max_epochs=10 loader.num_workers=4`. `--wandb PROJECT` logs to Weights & Biases.

To train from stored stable-envs frames, draw them once and use `--render offline`:

```bash
python examples/lewm/make_dataset.py data/pusht_expert_train.h5 data/pusht_snv.h5 --device cuda
python examples/lewm/train.py --data data/pusht_snv.h5 --render offline --half --compile
```

`--render offline` also reads the original LeWM file, with the frames of the source renderer.

## Evaluate

Plan with the trained model on 50 start and goal pairs from the dataset:

```bash
python examples/lewm/eval.py --data data/pusht_state.h5 --policy runs/checkpoints/lewm-pusht/weights_epoch_100.pt
```

`--policy random` evaluates random actions. `--num-eval` sets the number of pairs, `--env-device` the device of the stable-envs environments and `--device` the device of the model and the planner. The script prints the success rate as JSON. `--out` also appends it as one JSON line to a file.

## Small check

This runs a few hundred steps on a small part of the data:

```bash
python examples/lewm/make_dataset.py data/pusht_expert_train.h5 data/pusht_small.h5 --states-only --episodes 200
python examples/lewm/train.py --data data/pusht_small.h5 --half --max-steps 300 --precision 32 --accelerator cpu \
    loader.batch_size=16 loader.num_workers=2
python examples/lewm/eval.py --data data/pusht_small.h5 --policy runs/checkpoints/lewm-pusht/weights_epoch_1.pt \
    --device cpu --num-eval 8 eval.eval_budget=20
```

The [README](https://github.com/galilai-group/stable-envs/blob/main/examples/lewm/README.md#epoch-time) gives the measured epoch times with stored frames, with the source renderer and with stable-envs on the GPU.

## Checklist

- Copy a full state from `reset` and write only the leaves that the dataset columns hold. The other leaves keep their reset values.
- Set to zero the leaves that the dataset does not hold and that change the frame, such as the PushT contact points.
- Draw after the batch moves to the training device, so that the frames do not go across the bus.
- For another environment, make a new config with new `env.id`, `env.leaves`, `env.zero`, `eval` values and `data.frameskip`. `eval.py` calls `env.reset(seed, options)` with `eval.reset_options`, so the environment must accept them.

Next: [Store states, render later](store-and-render.md) stores states and renders them under any variation, which gives a model more looks of the same episodes.
