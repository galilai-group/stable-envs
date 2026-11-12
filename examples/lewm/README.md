# LeWM on PushT

LeWM (`stable_worldmodel.wm.lewm.LeWM`) is a latent world model: a ViT encoder turns each frame into an embedding, and a transformer predicts the next embedding from the last few embeddings and actions. It trains with a mean squared error on the predicted embedding plus SIGReg, a regularizer that pulls the embeddings toward an isotropic Gaussian so they don't collapse to a constant.

`train_lewm.py` collects PushT episodes with uniform random actions, cuts them into windows of `--history` + 1 frames, and trains a small LeWM on them with stable-worldmodel 0.1.1. It prints the loss as it trains, and `--out` saves the weights. This directory is its own uv project, so run it from `examples/lewm`:

```bash
uv run python train_lewm.py
uv run python train_lewm.py --num-envs 4 --steps 8 --size 64 --iters 2
uv run python train_lewm.py --help
```

The default run collects 16 episodes of 32 steps at 64 by 64 pixels and trains for 200 iterations on CPU. On an Apple M1 Pro it collects in under a second and trains in about 35 seconds, and the loss goes from 1.17 to 0.20. The first run in a new environment takes longer because Numba compiles the PushT kernels, and on macOS the synced `.venv` is about 1.9 GB, mostly PyTorch. A macOS warning that SDL is implemented twice (OpenCV and pygame both ship it) is harmless.

`--device cuda` trains the model on a GPU. PushT still collects on CPU, because the script builds it with `device="cpu"`.

## Flags and the upstream recipe

stable-worldmodel trains LeWM with `scripts/train/lewm.py` and `scripts/train/config/lewm.yaml` at tag `0.1.1`. This script keeps the loss and the window layout but shrinks the data and the model so the defaults run on a laptop CPU.

| Flag | Default | Upstream `lewm.yaml` at 0.1.1 |
|---|---|---|
| `--num-envs`, `--steps` | 16 episodes of 32 random actions | the recorded `pusht_expert_train` dataset |
| `--size` | 64 | `img_size: 224` |
| `--iters` | 200 | `trainer.max_epochs: 100` over the dataset |
| `--batch-size` | 32 | `loader.batch_size: 128` |
| `--device` | `cpu` | `trainer.accelerator: gpu`, `precision: bf16` |
| `--history` | 3 | `wm.history_size: 3`, `wm.num_preds: 1` |
| `--frameskip` | 1 | `frameskip: 5` in `data/pusht.yaml` |
| `--seed` | 0 | `seed: 3072` |
| `--out` | not saved | a checkpoint per epoch |

| Fixed in the script | `train_lewm.py` | Upstream `lewm.yaml` at 0.1.1 |
|---|---|---|
| encoder | Hugging Face ViT with 2 layers, width 192, 3 heads, patch 8 | ViT-tiny, patch 14 |
| embedding size | 192 | 192 |
| predictor | depth 2, 4 heads, MLP width 512 | depth 6, 16 heads, MLP width 2048, dropout 0.1 |
| projectors | MLP width 512 with BatchNorm | MLP width 2048 with BatchNorm |
| optimizer | AdamW, lr 3e-4, weight decay 1e-3, constant lr | AdamW, lr 5e-5, weight decay 1e-3, warmup then cosine |
| SIGReg | weight 0.09, 17 knots, 1024 projections | weight 0.09, 17 knots, 1024 projections |
| gradient clipping | 1.0 | 1.0 |
| inputs | pixels scaled to [0, 1], raw actions | pixels normalized with ImageNet statistics, actions normalized per column |

With `--frameskip k`, consecutive frames of a window are `k` env steps apart, and each action row joins the `k` actions between them into one vector of `2 * k` values, as stable-worldmodel datasets do.

## Why a separate project

This directory has its own `pyproject.toml` and `uv.lock`, and it takes stable-envs from the repository as an editable path dependency. A stable-envs extra would publish a `stable-envs -> stable-worldmodel` edge in the package metadata, and a dependency group would pull stable-worldmodel and its dependencies into the root `uv.lock`. The stable-envs wheel and sdist leave `examples/` out.

The pin is `stable-worldmodel==0.1.1` because it's the first LeWM release whose `SIGReg` runs on CPU (0.1.0 creates its projections with `device='cuda'`). stable-worldmodel 0.1.1 doesn't depend on stable-envs, so the pin keeps the example installable if a later release does. The PyPI package named `swm` is an unrelated project, and `swm` is only the name of stable-worldmodel's command-line tool.
