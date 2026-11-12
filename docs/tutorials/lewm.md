# Train a world model (LeWM)

This tutorial trains LeWM, stable-worldmodel's latent world model, on PushT [frames](../terms.md#frame) from stable-envs. You collect episodes, cut them into short windows, encode the frames, predict the next embedding and compute the loss, following [`examples/lewm/train_lewm.py`](https://github.com/galilai-group/stable-envs/blob/main/examples/lewm/train_lewm.py).

## Prerequisites

You need a CPU and the `examples/lewm` uv project, which installs stable-envs with the `pusht` extra, `stable-worldmodel==0.1.1` and `transformers`. Its [README](https://github.com/galilai-group/stable-envs/blob/main/examples/lewm/README.md) explains why it's a separate project. Run the code with `uv run python` from `examples/lewm`. Each code block continues from the one before it, so the page runs top to bottom as one script.

## Collect episodes

`step(actions, states=None, frames=slice(None))` runs a sequence of actions and returns every frame without the states. `step` doesn't return the frame before the first action, so render it after `reset`:

```python
# runs in the examples/lewm environment
import torch
import stable_envs as snv

torch.manual_seed(0)
actions = torch.rand(16, 32, 2, generator=torch.Generator().manual_seed(0)) * 2 - 1  # (B, T, 2) in [-1, 1]
with snv.PushT(num_envs=16, size=64, device="cpu") as env:
    env.reset(seed=0)
    first = env.render()                                      # the frame before the first action
    out = env.step(actions, states=None, frames=slice(None))  # every frame, no states
frames = torch.cat([first[:, None], out["frames"]], dim=1)
print(frames.shape, frames.dtype)     # torch.Size([16, 33, 64, 64, 3]) torch.uint8
print(bool(out["frame_valid"].all()))  # True, no episode reached the goal early
```

Action `t` moves an episode from frame `t` to frame `t + 1`. If a [batch row](../terms.md#batch-row) reaches the goal inside the sequence, its later frames repeat the last one and `out["frame_valid"]` marks them false, so the script drops that episode.

## Cut windows

LeWM reads a window of `history + 1` frames and the `history` actions between them. With a frameskip of `k`, consecutive frames of a window are `k` steps apart, and each action row joins the `k` actions between them, as stable-worldmodel datasets do. Pixels go to `(B, T, C, H, W)` floats in [0, 1]:

```python
history, frameskip, batch_size = 3, 1, 32
span = history * frameskip
generator = torch.Generator().manual_seed(0)
rows = torch.randint(0, 16, (batch_size, 1), generator=generator)
starts = torch.randint(0, 32 - span + 1, (batch_size, 1), generator=generator)
pixels = frames[rows, starts + torch.arange(history + 1) * frameskip].permute(0, 1, 4, 2, 3).float() / 255
window_actions = actions[rows, starts + torch.arange(span)].reshape(batch_size, history, frameskip * 2)
print(pixels.shape)          # torch.Size([32, 4, 3, 64, 64])
print(window_actions.shape)  # torch.Size([32, 3, 2])
```

## Build LeWM

`LeWM` takes a Hugging Face ViT encoder, whose CLS token is the frame embedding, a `Predictor` transformer, an `Embedder` for the actions and two `MLP` projectors. The sizes here are the script's CPU defaults, which are smaller than stable-worldmodel's recipe:

```python
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor
from stable_worldmodel.wm.loss import SIGReg
from transformers import ViTConfig, ViTModel

dim = 192
encoder = ViTModel(
    ViTConfig(image_size=64, patch_size=8, hidden_size=dim, num_hidden_layers=2, num_attention_heads=3, intermediate_size=4 * dim),
    add_pooling_layer=False,
)
predictor = Predictor(num_frames=history, depth=2, heads=4, mlp_dim=512, input_dim=dim, hidden_dim=dim, output_dim=dim)
model = LeWM(
    encoder=encoder,
    predictor=predictor,
    action_encoder=Embedder(input_dim=frameskip * 2, emb_dim=dim),
    projector=MLP(dim, 512, dim, norm_fn=torch.nn.BatchNorm1d),
    pred_proj=MLP(dim, 512, dim, norm_fn=torch.nn.BatchNorm1d),
)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")  # 2.73M
```

The [README](https://github.com/galilai-group/stable-envs/blob/main/examples/lewm/README.md#flags-and-the-upstream-recipe) compares every size with `lewm.yaml` at stable-worldmodel 0.1.1.

## Encode, predict and compute the loss

`encode` embeds every frame and every action row, and `predict` maps the first `history` embeddings and their actions to the embeddings one frame later. The loss is the mean squared error against the embeddings of frames 1 to `history`, plus SIGReg, which pulls the embeddings toward an isotropic Gaussian so they don't collapse to a constant:

```python
sigreg = SIGReg(knots=17, num_proj=1024)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

info = model.encode({"pixels": pixels, "action": window_actions})
emb = info["emb"]                                        # (32, 4, 192), one per frame
pred = model.predict(emb[:, :history], info["act_emb"])  # (32, 3, 192), one frame later
pred_loss = (pred - emb[:, 1:]).pow(2).mean()
sigreg_loss = sigreg(emb.transpose(0, 1))                # SIGReg reads (T, B, D)
loss = pred_loss + 0.09 * sigreg_loss

opt.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
print(emb.shape, pred.shape)  # torch.Size([32, 4, 192]) torch.Size([32, 3, 192])
```

The script repeats this step with a fresh batch of windows each iteration.

## Run the script

From `examples/lewm`:

```bash
uv run python train_lewm.py
```

The default run collects 16 episodes of 32 steps and trains for 200 iterations. On an Apple M1 Pro CPU it takes about 35 seconds, and the loss goes from 1.17 to 0.20. `--frameskip 5` uses the frameskip of stable-worldmodel's recipe (it needs `--steps` of at least 15), `--device cuda` trains the model on a GPU, and `--out lewm.pt` saves the weights.

## Checklist

- Render the frame before the first action after `reset`, because `step` returns only the frames after each action.
- Keep `out["frame_valid"]` with the frames. An episode that reaches the goal repeats its last frame.
- Give `LeWM.encode` pixels as `(B, T, C, H, W)` floats and actions as `(B, T, A)`, where `A` is `frameskip` times the action size.
- Pass `SIGReg` the embeddings as `(T, B, D)`, so transpose the `(B, T, D)` output of `encode`.
- Set `Predictor(num_frames=...)` to the history length.

Next: [Store states, render later](store-and-render.md) stores states and renders them under any variation, which gives a model more looks of the same episodes.
