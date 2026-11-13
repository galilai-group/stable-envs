# LeWM on stable-envs

These scripts train and evaluate LeWM, the JEPA world model of
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), on stable-envs PushT.

The training script can draw the frames during training. The DataLoader reads only the states and the
actions from the dataset. After a batch moves to the training device, `Env.render(state)` draws the frames
on that device. Thus the dataset does not store pixels, and the CPU does not decode or resize images.

| file | what it does |
|---|---|
| [`make_dataset.py`](make_dataset.py) | Copies a LeWM HDF5 dataset without its pixels (`--states-only`), or with new pixels that stable-envs draws from the state column. |
| [`train.py`](train.py) | Trains LeWM. `--render online` (default) draws the frames from the state rows on the training device. `--render offline` reads the `pixels` column. |
| [`eval.py`](eval.py) | Plans with CEM and the trained model on stable-envs environments, through stable-worldmodel's `World`. |
| [`state_render.py`](state_render.py) | Writes dataset state rows into a full stable-envs State and calls `env.render(state)`. |
| [`config/pusht.yaml`](config/pusht.yaml) | The environment, the state-row layout, the model, training and planning values. |

The model, the loss, the optimizer, the data split and the planning values are those of stable-worldmodel
`scripts/train/lewm.py` and `scripts/plan/eval_wm.py` at commit `21446f1`. The scripts use the public
stable-envs API only: `snv.make`, `reset`, `get_state`, `step` and `render`.

## Install

Use Linux with an NVIDIA GPU for training. The CPU works for the small check below.

```bash
pip install -e ".[pusht]"        # from the stable-envs repository root
pip install "stable-worldmodel[train,format] @ git+https://github.com/galilai-group/stable-worldmodel@21446f1ede6d5284e981bd7b47f432b994e6d812" \
    "stable-pretraining==0.1.8" scikit-learn
```

On CUDA, PushT draws with Triton, which comes with the Linux PyTorch wheels.

The numbers below come from torch 2.14.1 (CUDA 12.6), lightning 2.6.6, stable-pretraining 0.1.8 and
pymunk 7.2.0 on Python 3.12.

## PushT

Get the LeWM PushT dataset (46 GB after decompression):

```bash
hf download quentinll/lewm-pusht pusht_expert_train.h5.zst --repo-type dataset --local-dir data
zstd -d data/pusht_expert_train.h5.zst -o data/pusht_expert_train.h5
```

1. Make the state dataset (about 160 MB). It keeps every column except `pixels`:

   ```bash
   python examples/lewm/make_dataset.py data/pusht_expert_train.h5 data/pusht_state.h5 --states-only
   ```

2. Train with frames drawn on the GPU (half-size model, bf16, `torch.compile`):

   ```bash
   python examples/lewm/train.py --data data/pusht_state.h5 --half --compile
   ```

   The weights go to `runs/checkpoints/lewm-pusht/weights_epoch_N.pt`. Without `--half`, the model has the
   full size of `lewm.yaml`. `--precision 32` trains in float32. Arguments after the options change the
   config, for example `trainer.max_epochs=10 loader.num_workers=4`. `--wandb PROJECT` logs to Weights &
   Biases.

3. Evaluate the model on 50 start and goal pairs from the dataset:

   ```bash
   python examples/lewm/eval.py --data data/pusht_state.h5 --policy runs/checkpoints/lewm-pusht/weights_epoch_100.pt
   ```

   `--policy random` evaluates random actions. The script prints the success rate as JSON.

To train from stored stable-envs frames instead, draw them once and use `--render offline`:

```bash
python examples/lewm/make_dataset.py data/pusht_expert_train.h5 data/pusht_snv.h5 --device cuda
python examples/lewm/train.py --data data/pusht_snv.h5 --render offline --half --compile
```

`--render offline` also reads the original LeWM file, with the frames of the source renderer.

### Small check

This runs a few hundred steps on a small part of the data:

```bash
python examples/lewm/make_dataset.py data/pusht_expert_train.h5 data/pusht_small.h5 --states-only --episodes 200
python examples/lewm/train.py --data data/pusht_small.h5 --half --max-steps 300 --precision 32 --accelerator cpu \
    loader.batch_size=16 loader.num_workers=2
python examples/lewm/train.py --data data/pusht_small.h5 --half --compile --max-steps 300
python examples/lewm/eval.py --data data/pusht_small.h5 --policy runs/checkpoints/lewm-pusht/weights_epoch_1.pt \
    --num-eval 8 eval.eval_budget=20
```

## Frames

`state_render.py` copies a State from `env.reset(seed=0)` and writes the state row of each sample into
these leaves (`env.leaves` in the config):

| PushT State leaf | state columns |
|---|---|
| `current.body.agent.position` | 0, 1 |
| `current.body.block.position` | 2, 3 |
| `current.body.block.angle` | 4 |

The other leaves (goal, colours, shapes) keep their reset values. The dataset rows hold no contact points, so
the frames show no contact markers. The source frames show a marker where the agent touches the block.

On a PushT State, `render` draws the same bytes as the source pygame renderer, on CPU and on CUDA (see
the [PushT page](../../docs/environments/pusht.md)). A comparison of model results with source frames and
with stable-envs frames is in the stable-envs paper.

## Epoch time

One epoch of the half-size model with bf16 and `torch.compile` on PushT (13,933 steps of 128 samples) on
one NVIDIA L40S. The training process and its W DataLoader workers had W + 1 CPU cores (AMD EPYC 9374F).
The time excludes the process start and the compile time of the first 200 steps.

- stored frames: the frames from `pusht_expert_train.h5`, decoded and resized in the DataLoader workers.
- source renderer: the stable-worldmodel PushT renderer (pygame) draws the frames in the DataLoader workers.
- stable-envs on GPU: `render` draws the frames on the GPU after the batch moves there.

| W | CPU cores | stored frames | source renderer | stable-envs on GPU |
|---|---|---|---|---|
| 1 | 2 | 189.3 min (157 samples/s) | 223.7 min (133 samples/s) | 20.7 min (1439 samples/s) |
| 2 | 3 | 103.9 min (286 samples/s) | 109.5 min (271 samples/s) | 20.9 min (1421 samples/s) |
| 4 | 5 | 58.3 min (509 samples/s) | 72.5 min (410 samples/s) | 20.8 min (1426 samples/s) |
| 8 | 9 | 32.0 min (928 samples/s) | not measured | not measured |

The training step alone (random batches on the GPU, no DataLoader) runs at 1644 samples/s on this GPU. The
stable-envs column uses 86% to 88% of that rate. The GPU render takes about 7.5% of each step. The workers
read only the states and the actions, so the stable-envs column does not change with W. The runs that
did not finish the epoch in the time limit give a projected time: 13,933 times the steady step time.

These times come from the experiment code of the paper, which drew the frames from compact render rows.
`train.py` draws them from full States with the public `render`. The model, data and training settings
are the same. Measure on your own hardware before you compare.

## Other environments

Make a new config with these changes:

- `env.id`: the stable-envs class name, and `env.kwargs` for its constructor.
- `env.state_column`, `env.leaves` and `env.zero`: the dataset column with the state rows, and the State
  leaves that its columns go into.
- `eval.observation_leaf`, `eval.proprio_columns` and `eval.reset_options`: the State leaf that holds the
  task observation, and the reset options that the dataset columns give. `eval.py` calls
  `env.reset(seed, options)` with these options, so the environment must accept them.
- `data.frameskip` and the `eval` values of the matching stable-worldmodel configs.
