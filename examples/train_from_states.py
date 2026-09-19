"""A real training loop with no image dataset: states in the loader, ``set_state().render()`` in the step.

This is the usage the repo is for, end to end and actually run — and note what it does NOT import.
The dataset is ``torch.utils.data.TensorDataset`` and the loader is a plain ``DataLoader``: slicing a
``(N, state_dim)`` float32 array is something torch already does, and stable-envs supplies the renderer,
not the data plumbing.

    ds = TensorDataset(torch.as_tensor(states))               # your states, your dataset
    loader = DataLoader(ds, batch_size=256, shuffle=True)     # your loader, default collate

    for (states,) in loader:                      # (B, state_dim) float32, 20 B per sample
        frames = env.set_state(states).render()   # (B, 96, 96, 3) uint8, one kernel launch
        loss = mse(net(frames), target(env.get_state()))

``drop_last`` is left at its default of False on purpose. The last batch of an epoch is short, and
``set_state`` takes a short batch and renders exactly those rows — measured at 3.371 ms for 4,095 PushT
states against 3.341 ms for 4,096, i.e. free. Telling every user to drop data to fit a fixed buffer was
the alternative.

``env.get_state()`` rather than the loop variable is not decoration either: it is the states as the env
holds them, on the device, already the right number of rows after a short batch, so the target cannot
drift out of step with the frames.

The task is deliberately the inverse of the renderer: recover the compact state from the frame. It is
the trivial task the brief asks for, but it is not arbitrary — if a frame really is a function of a
5-number state, that function should be invertible from pixels, and a network that gets there confirms
the frames carry the state rather than a constant. Convergence is reported as held-out R^2 per state
dimension plus the position error in pixels, because an MSE on normalised targets is not interpretable
and "the loss went down" is not a result.

The angle is regressed as ``(cos, sin)``. Regressing the angle directly puts a discontinuity at +-pi
inside the target, and the network then averages across the wrap and reports a plausible-looking loss
while being maximally wrong on a measure-zero set that the dataset visits constantly.

What the timing here is for. Rendering is now cheap enough that it is worth knowing whether it still
matters at all, so every step is split with explicit ``cuda.synchronize`` into render and
forward/backward, and the same loop is run against frames already resident in device memory — the
fastest data path that could possibly exist, since there is no I/O of any kind in it. The gap between
the two IS the cost of generating the pixels, measured rather than inferred from the split.

    python examples/train_from_states.py
    python examples/train_from_states.py --batch 512 --steps 800
    python examples/train_from_states.py --batch 4096 --batch-sampler
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

import stable_envs as se

SCREEN = 512.0


class StateNet(nn.Module):
    """Small conv net, 96x96x3 uint8 -> 6 numbers. Sized to be trained in seconds, not to be good.

    Normalisation happens inside the module, on the GPU, from the uint8 the renderer produced. Doing it
    in the loader would move float32 across the same boundary that the renderer just avoided, i.e. 4x
    the bytes for a conversion that costs one fused kernel here.
    """

    def __init__(self, out_dim: int = 6, width: int = 32):
        super().__init__()
        c = [3, width, width * 2, width * 4, width * 4]
        self.body = nn.Sequential(
            *[
                layer
                for i in range(4)
                for layer in (
                    nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1),
                    nn.GroupNorm(8, c[i + 1]),
                    nn.SiLU(),
                )
            ]
        )
        self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(256), nn.SiLU(), nn.Linear(256, out_dim))

    def forward(self, frames_uint8: torch.Tensor) -> torch.Tensor:
        x = frames_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        return self.head(self.body(x))


def targets(states: torch.Tensor) -> torch.Tensor:
    """``(B,5)`` PushT state -> ``(B,6)`` regression target, all components O(1).

    Positions are mapped to [-1,1] from the 512-unit board rather than standardised from batch
    statistics: the board is a fixed constant of the env, so a fixed map keeps the reported pixel errors
    comparable across runs and across batch sizes.
    """
    xy = states[:, :4] / SCREEN * 2.0 - 1.0
    ang = states[:, 4]
    return torch.cat([xy, torch.cos(ang)[:, None], torch.sin(ang)[:, None]], dim=1)


def decode(pred: torch.Tensor) -> torch.Tensor:
    """Inverse of ``targets``: back to ``(B,5)`` in board units and radians."""
    xy = (pred[:, :4] + 1.0) * 0.5 * SCREEN
    ang = torch.atan2(pred[:, 5], pred[:, 4])
    return torch.cat([xy, ang[:, None]], dim=1)


@torch.no_grad()
def evaluate(net, env, states: torch.Tensor, batch: int) -> dict:
    """Held-out R^2 per target dimension, plus errors in the units the env is defined in."""
    net.eval()
    preds, gts = [], []
    for i in range(0, states.shape[0] - batch + 1, batch):
        s = states[i : i + batch].cuda()
        f = env.set_state(s).render()
        preds.append(net(f).float())
        gts.append(targets(s))
    net.train()
    p, g = torch.cat(preds), torch.cat(gts)
    ss_res = ((p - g) ** 2).sum(0)
    ss_tot = ((g - g.mean(0)) ** 2).sum(0)
    r2 = (1 - ss_res / ss_tot).cpu().numpy()

    ps, gs = decode(p), decode(g)
    agent_px = (ps[:, :2] - gs[:, :2]).norm(dim=1).mean().item()
    block_px = (ps[:, 2:4] - gs[:, 2:4]).norm(dim=1).mean().item()
    d = (ps[:, 4] - gs[:, 4] + math.pi) % (2 * math.pi) - math.pi
    return {
        "r2": {
            n: float(v) for n, v in zip(["agent_x", "agent_y", "block_x", "block_y", "cos_angle", "sin_angle"], r2)
        },
        "mse": float(((p - g) ** 2).mean()),
        "agent_err_px": agent_px,
        "block_err_px": block_px,
        "angle_err_deg": float(d.abs().mean() * 180 / math.pi),
        "n": int(p.shape[0]),
    }


def timed_steps(env, net, opt, states_pool, batch: int, steps: int, source=None, warmup: int = 5) -> dict:
    """``steps`` optimisation steps, with render and forward/backward timed separately.

    ``source`` supplies pre-rendered frames instead of rendering, which is the control: the same
    optimisation with the pixels already in device memory. Synchronising twice per step costs a little
    of both numbers, and it is the only way to attribute the time rather than guess at it.

    The first steps are run untimed. cuDNN picks its algorithms on the first call of each shape and the
    rasterizer bakes its static prefix once, so timing from step 0 charges both to whichever batch size
    happens to come first in a sweep.
    """
    render_s = train_s = 0.0
    n = states_pool.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(warmup):
        idx = torch.randint(0, n, (batch,), generator=g)
        s = states_pool[idx].cuda()
        frames = source[idx.cuda()] if source is not None else env.set_state(s).render()
        loss = ((net(frames) - targets(s)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t_all = time.perf_counter()
    for i in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        s = states_pool[idx].cuda(non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        frames = source[idx.cuda()] if source is not None else env.set_state(s).render()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = ((net(frames) - targets(s)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        render_s += t1 - t0
        train_s += t2 - t1
    wall = time.perf_counter() - t_all
    return {
        "batch": batch,
        "steps": steps,
        "wall_s": wall,
        "render_ms": render_s / steps * 1e3,
        "train_ms": train_s / steps * 1e3,
        "step_ms": wall / steps * 1e3,
        "render_share": render_s / (render_s + train_s),
        "frames_per_s": batch * steps / wall,
        "final_loss": float(loss.detach()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PushT")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--train", type=int, default=32768)
    ap.add_argument("--val", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sweep", default="64,128,256,512,1024")
    ap.add_argument(
        "--batch-sampler",
        action="store_true",
        help="index the states a batch at a time instead of a row at a time; worth it "
        "above ~B=1024, no custom collate either way",
    )
    ap.add_argument("--out", type=Path, default=Path("examples/results/train_from_states.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    proto = se.make(args.env, num_envs=1, size=args.size, device="cuda")
    all_states = proto.sample_states(args.train + args.val, rng)
    train_states, val_states = all_states[: args.train], all_states[args.train :]

    # `StateDataset` is used for ONE line here — the storage report, which needs the env to measure the
    # frame shape rather than assume it is square. It is not in the training path below; the loader is
    # a plain torch DataLoader over a plain TensorDataset.
    comp = se.StateDataset(proto, train_states, size=args.size, frames=False).compression
    print(
        f"{args.env}: {comp['n']:,} training states, {comp['stored_bytes']:,} B "
        f"({comp['state_bytes'] / comp['n']:.0f} B/sample) against "
        f"{comp['frame_bytes']:,} B of {comp['frame_shape']} frames -> {comp['ratio']:.0f}x smaller"
    )
    print(
        f"the whole training set fits in {comp['stored_bytes'] / 1024:.0f} kB; the frames it stands "
        f"in for are {comp['frame_bytes'] / 1e6:.0f} MB"
    )

    env = se.make(args.env, num_envs=args.batch, size=args.size, device="cuda")
    net = StateNet().cuda()
    net(env.set_state(torch.as_tensor(train_states[: args.batch], device="cuda")).render())  # LazyLinear
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"net: {n_params / 1e6:.2f} M parameters\n")

    # A plain DataLoader over a plain TensorDataset. No collate_fn, no library dataset class, and
    # `drop_last=False` — the short final batch is rendered, not dropped and not padded.
    #
    # At this batch size the per-item indexing torch does here is invisible. It is not free at large
    # batches: measured on one H200 at B=4,096, fetching a PushT batch this way costs 14.9 ms against
    # 0.79 ms for a BatchSampler, because `[dataset[i] for i in idx]` is ~3.4 us of python per row. The
    # fix needs no custom collate — see the --batch-sampler flag and the note in stable_envs/data.py.
    train_ds = TensorDataset(torch.as_tensor(np.asarray(train_states, np.float32)))
    if args.batch_sampler:
        loader = DataLoader(
            train_ds,
            batch_size=None,
            sampler=BatchSampler(RandomSampler(range(len(train_ds))), args.batch, drop_last=False),
        )
    else:
        loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    print(f"DataLoader: {len(loader)} batches/epoch over TensorDataset, torch's default collate, drop_last=False")

    val = torch.as_tensor(val_states)
    before = evaluate(net, env, val, args.batch)
    print(
        f"before training: val MSE {before['mse']:.4f}, block error {before['block_err_px']:.1f} px, "
        f"R^2 {min(before['r2'].values()):+.3f}..{max(before['r2'].values()):+.3f}"
    )

    render_s = train_s = 0.0
    seen = 0
    t_start = time.perf_counter()
    step = 0
    log = []
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            # TensorDataset yields a 1-tuple per sample, so the collated batch is a 1-list; the
            # BatchSampler path hands the batch straight through. Both end up as (B, state_dim).
            states = batch[0] if isinstance(batch, (list, tuple)) else batch
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            frames = env.set_state(states).render()  # <- the whole point
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            # get_state(), not `states`: on device, and already the right number of rows if this was
            # the short final batch.
            loss = ((net(frames) - targets(env.get_state())) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            torch.cuda.synchronize()
            render_s += t1 - t0
            train_s += time.perf_counter() - t1
            seen += int(frames.shape[0])
            step += 1
            if step % 200 == 0 or step == 1:
                print(f"  step {step:4d}  loss {loss.item():.5f}")
            if step % 10 == 0 or step == 1:
                log.append({"step": step, "loss": float(loss.detach())})
    wall = time.perf_counter() - t_start

    after = evaluate(net, env, val, args.batch)
    print(f"\nafter {args.steps} steps ({seen:,} frames, {wall:.1f} s):")
    print(f"  val MSE {before['mse']:.4f} -> {after['mse']:.5f}")
    for k, v in after["r2"].items():
        print(f"  R^2 {k:<10s} {v:+.4f}")
    print(
        f"  agent position error {after['agent_err_px']:.2f} px, "
        f"block position error {after['block_err_px']:.2f} px, "
        f"block angle error {after['angle_err_deg']:.2f} deg  (512-unit board, rendered at "
        f"{args.size} px, so one output pixel is {SCREEN / args.size:.2f} board units)"
    )

    share = render_s / (render_s + train_s)
    print(
        f"\nthroughput: {seen / wall:,.0f} frames/s end to end "
        f"({wall / args.steps * 1e3:.2f} ms/step at batch {args.batch})"
    )
    print(f"  render          {render_s / args.steps * 1e3:.3f} ms/step  ({share * 100:.1f}% of step)")
    print(f"  fwd+bwd+step    {train_s / args.steps * 1e3:.3f} ms/step  ({(1 - share) * 100:.1f}%)")

    # Control: the same optimisation with frames already in VRAM. No renderer, no I/O, nothing to beat.
    pool = torch.as_tensor(train_states[:8192])
    cached = se.render_states(env, pool.numpy(), size=args.size, chunk=4096)
    ctl_net = StateNet().cuda()
    ctl_net(cached[:2])
    ctl_opt = torch.optim.Adam(ctl_net.parameters(), lr=args.lr)
    ctl = timed_steps(env, ctl_net, ctl_opt, pool, args.batch, 100, source=cached)
    live_net = StateNet().cuda()
    live_net(env.set_state(pool[: args.batch].cuda()).render())
    live_opt = torch.optim.Adam(live_net.parameters(), lr=args.lr)
    live = timed_steps(env, live_net, live_opt, pool, args.batch, 100)
    print(f"\ncontrol, batch {args.batch}, 100 steps:")
    print(f"  frames already in VRAM  {ctl['step_ms']:.2f} ms/step   {ctl['frames_per_s']:,.0f} frames/s")
    print(f"  rendered every step     {live['step_ms']:.2f} ms/step   {live['frames_per_s']:,.0f} frames/s")
    print(
        f"  generating the pixels costs {live['step_ms'] - ctl['step_ms']:.2f} ms/step "
        f"({(live['step_ms'] / ctl['step_ms'] - 1) * 100:.0f}% over the fastest data path there is), "
        f"and replaces {cached.numel() / 1e6:.0f} MB of resident pixels with "
        f"{pool.numel() * 4 / 1e3:.0f} kB of state"
    )
    del cached
    torch.cuda.empty_cache()

    # How the render share moves with batch size: the scene build is ~2 ms of Python regardless of B.
    sweep = []
    for b in [int(x) for x in args.sweep.split(",") if x]:
        e = se.make(args.env, num_envs=b, size=args.size, device="cuda")
        n = StateNet().cuda()
        n(e.set_state(torch.as_tensor(train_states[:b], device="cuda")).render())
        o = torch.optim.Adam(n.parameters(), lr=args.lr)
        sweep.append(timed_steps(e, n, o, torch.as_tensor(train_states), b, 50))
        del e, n, o
        torch.cuda.empty_cache()
    print("\n| batch | render ms | fwd+bwd ms | step ms | render share | frames/s |")
    print("|---|---|---|---|---|---|")
    for r in sweep:
        print(
            f"| {r['batch']} | {r['render_ms']:.2f} | {r['train_ms']:.2f} | {r['step_ms']:.2f} | "
            f"{r['render_share'] * 100:.1f}% | {r['frames_per_s']:,.0f} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "env": args.env,
                "size": args.size,
                "batch": args.batch,
                "steps": args.steps,
                "n_train": args.train,
                "n_val": args.val,
                "params": n_params,
                "compression": comp,
                "loss_log": log,
                "before": before,
                "after": after,
                "throughput": {
                    "frames_per_s": seen / wall,
                    "wall_s": wall,
                    "render_ms": render_s / args.steps * 1e3,
                    "train_ms": train_s / args.steps * 1e3,
                    "render_share": share,
                },
                "control_cached_vram": ctl,
                "control_live_render": live,
                "batch_sweep": sweep,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
