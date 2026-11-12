"""Train a LeWM world model on random PushT rollouts.

Collects PushT episodes with uniform random actions, then trains stable-worldmodel's LeWM to predict the
next frame embedding from the previous ``--history`` frame embeddings and actions.

    uv run python train_lewm.py
    uv run python train_lewm.py --num-envs 64 --steps 64 --frameskip 5 --iters 1000 --out lewm.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor
from stable_worldmodel.wm.loss import SIGReg
from transformers import ViTConfig, ViTModel

import stable_envs as snv

EMBED_DIM = 192
LR = 3e-4
WEIGHT_DECAY = 1e-3
SIGREG_WEIGHT = 0.09


def collect(num_envs: int, steps: int, size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames ``[B, steps + 1, H, W, 3]`` and actions ``[B, steps, 2]`` of the episodes that never end early."""
    actions = torch.rand(num_envs, steps, 2, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    with snv.PushT(num_envs=num_envs, size=size, device="cpu") as env:
        env.reset(seed=seed)
        first = env.render()
        out = env.step(actions, states=None, frames=slice(None))
    # An episode that reaches the goal repeats its last frame for the remaining actions.
    keep = out["frame_valid"].all(dim=1)
    return torch.cat([first[:, None], out["frames"]], dim=1)[keep], actions[keep]


def sample_windows(frames, actions, batch_size: int, history: int, frameskip: int, generator):
    """Random windows: pixels ``[N, history + 1, 3, H, W]`` in [0, 1] and actions ``[N, history, frameskip * 2]``."""
    num_envs, steps, action_dim = actions.shape
    span = history * frameskip
    rows = torch.randint(0, num_envs, (batch_size, 1), generator=generator)
    starts = torch.randint(0, steps - span + 1, (batch_size, 1), generator=generator)
    pixels = frames[rows, starts + torch.arange(history + 1) * frameskip].permute(0, 1, 4, 2, 3).float() / 255
    # Action row k joins the frameskip actions between frames k and k + 1, as stable-worldmodel datasets do.
    window_actions = actions[rows, starts + torch.arange(span)].reshape(batch_size, history, frameskip * action_dim)
    return pixels, window_actions


def build_model(size: int, history: int, action_dim: int) -> LeWM:
    encoder = ViTModel(
        ViTConfig(
            image_size=size,
            patch_size=8,
            hidden_size=EMBED_DIM,
            num_hidden_layers=2,
            num_attention_heads=3,
            intermediate_size=4 * EMBED_DIM,
        ),
        add_pooling_layer=False,
    )
    predictor = Predictor(
        num_frames=history,
        depth=2,
        heads=4,
        mlp_dim=512,
        input_dim=EMBED_DIM,
        hidden_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
    )
    return LeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=Embedder(input_dim=action_dim, emb_dim=EMBED_DIM),
        projector=MLP(EMBED_DIM, 512, EMBED_DIM, norm_fn=torch.nn.BatchNorm1d),
        pred_proj=MLP(EMBED_DIM, 512, EMBED_DIM, norm_fn=torch.nn.BatchNorm1d),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train LeWM on random PushT rollouts.")
    p.add_argument("--num-envs", type=int, default=16, help="PushT episodes to collect")
    p.add_argument("--steps", type=int, default=32, help="random actions per episode")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu", help="where the model trains; PushT collects on CPU")
    p.add_argument("--history", type=int, default=3, help="context frames per window")
    p.add_argument("--frameskip", type=int, default=1, help="env steps between two frames of a window")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="optional path for the trained weights")
    args = p.parse_args(argv)

    if args.history * args.frameskip > args.steps:
        print("--steps must be at least --history * --frameskip", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    frames, actions = collect(args.num_envs, args.steps, args.size, args.seed)
    print(f"collected frames {tuple(frames.shape)}  actions {tuple(actions.shape)}  {time.perf_counter() - t0:.1f}s")
    if len(frames) == 0:
        print("every episode ended early; collect fewer --steps", file=sys.stderr)
        return 1

    model = build_model(args.size, args.history, args.frameskip * actions.shape[-1]).to(args.device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"LeWM  {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters  device={args.device}")

    generator = torch.Generator().manual_seed(args.seed)
    log_every = max(1, args.iters // 10)
    t0 = time.perf_counter()
    for it in range(args.iters):
        pixels, window_actions = sample_windows(
            frames, actions, args.batch_size, args.history, args.frameskip, generator
        )
        info = model.encode({"pixels": pixels.to(args.device), "action": window_actions.to(args.device)})
        emb = info["emb"]
        pred = model.predict(emb[:, : args.history], info["act_emb"])
        pred_loss = (pred - emb[:, 1:]).pow(2).mean()
        sigreg_loss = sigreg(emb.transpose(0, 1))
        loss = pred_loss + SIGREG_WEIGHT * sigreg_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % log_every == 0 or it == args.iters - 1:
            print(
                f"iter {it:4d}  loss {loss.item():.4f}  pred {pred_loss.item():.4f}  sigreg {sigreg_loss.item():.4f}"
            )
    elapsed = time.perf_counter() - t0
    print(f"trained {args.iters} iterations in {elapsed:.1f}s ({args.iters / elapsed:.1f} it/s)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        config = {"size": args.size, "history": args.history, "frameskip": args.frameskip}
        torch.save({"model": model.state_dict(), "config": config}, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
