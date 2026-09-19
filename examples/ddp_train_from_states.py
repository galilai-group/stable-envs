"""The same loop under DistributedDataParallel, and the three things that actually change.

The short answer to "does this work with DDP out of the box" is yes, and the reason is that there is
nothing here for DDP to interact with: the dataset is a ``TensorDataset`` of states, so
``DistributedSampler`` shards it exactly as it shards an image folder's file list. Nothing in the
rendering path is collective, no rank talks to another to produce a frame, and FSDP is orthogonal — it
shards the model, and this is a question about the input pipeline.

What genuinely differs from a stored-image pipeline, measured by ``tests/gpu/test_distributed.py``:

  **1. The env must be built on the rank's own device.** ``device="cuda"`` resolves to the *current*
  device, so the ordinary DDP preamble ``torch.cuda.set_device(local_rank)`` is what makes this correct,
  and it has to happen BEFORE the env is constructed. Building it first puts every rank's renderer on
  cuda:0 -- which still runs, and still trains, and quietly serialises the whole job through one GPU.
  Passing ``device=f"cuda:{local_rank}"`` explicitly is the version that cannot be got wrong. This is
  now the ONLY rank-aware object in the file: the loader carries states on the host and knows nothing
  about devices, which is exactly why the recommended idiom is the loader you already have.

  **2. ``num_workers`` is a free choice here, and 0 is the right default.** There is no disk in this
  path and no decode, so the usual DDP advice to raise it is advice about a bottleneck that does not
  exist; states are 20 B a row. (Were the *dataset* doing the rendering, as ``StateDataset(frames=True)``
  does, 0 would be mandatory rather than merely sensible — a worker would build a second CUDA context
  and ship frames back over IPC that never needed to leave the device.)

  **3. The renderer shares the GPU with the model.** This is the one real cost and it does not exist for
  a stored dataset, where decoding happens on the host. Peak memory is the batch of frames plus the
  model's own, and the render occupies SMs the backward pass would otherwise have. Both are reported
  below per rank rather than asserted to be negligible.

Sampling is the one place to be careful, and it is the same care a stored dataset needs: give
``DistributedSampler`` a seed and call ``set_epoch``, or every rank draws the same permutation.
``drop_last`` is left False on both the sampler and the loader — ``DistributedSampler`` pads its shards
to equal length, so the ranks still agree on the batch count, and ``set_state`` renders a short final
batch rather than refusing it.

    torchrun --nproc_per_node=2 examples/ddp_train_from_states.py
    torchrun --nproc_per_node=2 examples/ddp_train_from_states.py --env PushT --steps 200
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import stable_envs as se


class TinyNet(nn.Module):
    """uint8 (B,H,W,3) -> state_dim. Normalisation on device, from the uint8 the renderer produced.

    Pools to a $4\\times4$ grid rather than to $1\\times1$. Global average pooling destroys exactly the
    quantity this task needs -- where things are -- and the first version of this file used it: the loss
    sat at 1.00, which is what predicting the per-dimension mean costs on a standardised target. It
    looked like a pipeline problem and was an architecture problem, which is the failure mode a
    plumbing example is most likely to be blamed for.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64 * 16, out_dim))

    def forward(self, frames):
        x = frames.permute(0, 3, 1, 2).float().div_(255.0)
        return self.body(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PushT")
    ap.add_argument("--n-states", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=128, help="per rank")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--size", type=int, default=0, help="0 = the env's native size")
    ap.add_argument("--out", default="examples/results")
    a = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # set_device BEFORE anything builds a tensor: `device="cuda"` means "the current device", and the
    # dataset resolves it at construction time.
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = f"cuda:{local_rank}"

    # Every rank holds the same states on the HOST; the sampler is what makes the work disjoint. This
    # is the same arrangement as an image folder whose file list every rank knows in full.
    rng = np.random.default_rng(0)
    proto = se.make(a.env, num_envs=1, device=dev, **({"size": a.size} if a.size else {}))
    states = proto.sample_states(a.n_states, rng=rng)
    del proto

    # The whole input pipeline: a tensor of states, torch's own dataset, torch's own sampler, torch's
    # own collate. Nothing from stable_envs appears until the render inside the step, which is the
    # point -- and it is also why the render cost below can be timed at all, since it is not hidden
    # inside a __getitems__ the loader calls.
    ds = TensorDataset(torch.as_tensor(np.asarray(states, np.float32)))
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=0,
                                 drop_last=False)
    loader = DataLoader(ds, batch_size=a.batch, sampler=sampler, drop_last=False, num_workers=0)

    env = se.make(a.env, num_envs=a.batch, device=dev, **({"size": a.size} if a.size else {}))
    net = DDP(TinyNet(env.state_dim).to(dev), device_ids=[local_rank])
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)

    # Standardise the regression target once, on the host, from the full state set. Per-batch
    # standardisation would make the target depend on the shard and give the two ranks different
    # objectives -- which trains, and converges to something that is not the same model.
    st_all = torch.as_tensor(np.asarray(states, np.float32))
    mu, sd = st_all.mean(0).to(dev), st_all.std(0).clamp_min(1e-6).to(dev)

    torch.cuda.reset_peak_memory_stats(dev)
    t_render = t_step = 0.0
    seen, step, losses = 0, 0, []
    for epoch in range(1000):
        sampler.set_epoch(epoch)                       # without this every epoch is the same order
        for (batch_states,) in loader:
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            frames = env.set_state(batch_states).render()
            torch.cuda.synchronize(dev)
            t_render += time.perf_counter() - t0

            t0 = time.perf_counter()
            # get_state(): on this rank's device, and already trimmed if this was a short final batch.
            target = (env.get_state() - mu) / sd
            loss = torch.nn.functional.mse_loss(net(frames), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            torch.cuda.synchronize(dev)
            t_step += time.perf_counter() - t0

            losses.append(float(loss))
            seen += frames.shape[0]
            step += 1
            if step >= a.steps:
                break
        if step >= a.steps:
            break

    peak = torch.cuda.max_memory_allocated(dev) / 1e9
    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    local = torch.tensor([seen, t_render, t_step, peak, first, last], device=dev,
                         dtype=torch.float64)
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)

    if rank == 0:
        rows = [g.tolist() for g in gathered]
        total_frames = sum(r[0] for r in rows)
        wall = max(r[1] + r[2] for r in rows)
        summary = {
            "env": a.env, "world_size": world, "batch_per_rank": a.batch, "steps": step,
            "frames_total": total_frames, "frames_per_s": total_frames / wall,
            # The render's share of a training step is the number that says whether generating pixels
            # still matters. Reported rather than argued: on a small net it is large because the net is
            # small, and that is the honest reading rather than a flattering one.
            "render_frac_of_step": sum(r[1] for r in rows) / sum(r[1] + r[2] for r in rows),
            "render_fps_per_rank": [r[0] / r[1] for r in rows],
            "loss_first5": [r[4] for r in rows], "loss_last5": [r[5] for r in rows],
            "per_rank": [{"rank": i, "frames": r[0], "render_s": r[1], "step_s": r[2],
                          "peak_gb": r[3]} for i, r in enumerate(rows)],
        }
        print(json.dumps(summary, indent=2))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "ddp_train_from_states.json").write_text(json.dumps(summary, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
