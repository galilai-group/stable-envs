"""Fit a corrector for one Meta-World task -- or for SEVERAL AT ONCE, which is the family's question.

The recipe is ``scratch/altrender/rendernet/train.py``'s, unchanged where it does not have to change:
architectures and the zero-initialised head from ``rendernet/nets.py`` (so the model IS the bilinear
baseline at step 0 and every number is a departure from a measured point), L1 on the un-rounded
output because mean |diff| is the metric the tier declares, bf16 autocast with gradient clipping,
OneCycle, and the held-out number recomputed on the ROUNDED uint8 output after every epoch.

Two things are new and both are about what Meta-World declares.

**INTERIOR-EXACTNESS IS IN THE TRAINING LOOP.** Meta-World is a PARITY family: its declaration is
``interior_exact >= 0.978`` and ``all-pixel exact >= 0.80``, not a mean. Downsampling barely touches
interior-exactness (a flat region survives a resize), so the bilinear baseline this corrector starts
from is already at 0.997 and a corrector that lowers the mean while adding half a level of residual
everywhere would destroy it. It is therefore measured every epoch, against a mask computed once from
the OFFICIAL frames, and it is reported next to the mean rather than after the fact.

**--envs TAKES A LIST.** Meta-World is 50 tasks, so "one corrector or fifty" is the question that
decides whether this is a technique or a chore. A multi-task run concatenates the training splits of
several tasks and reports the held-out number PER TASK, so a model that is good on average and bad on
one task cannot hide.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C
import data as D                      # rendernet's loader; ROOT is repointed below
import nets                           # rendernet's architectures and ZOO

D.ROOT = C.DATA
C.extend_zoo()
CKPT = os.path.join(C.LOGS, "ckpt")


def interior_masks(hr: np.ndarray, block: int = 256) -> torch.Tensor:
    """``(N, H-1, W-1)`` bool on device: the repo's own interior mask of the OFFICIAL frames."""
    from stable_envs.testing.conformance import interior_mask
    out = [interior_mask(hr[i:i + block]) for i in range(0, len(hr), block)]
    return torch.as_tensor(np.concatenate(out), device="cuda")


def quick_metrics(pred_u8: torch.Tensor, hr_u8: torch.Tensor, imask: torch.Tensor | None = None):
    """mean / within-8 / exact / interior-exact / tail, on device, over the max-of-channels diff."""
    d = (pred_u8.to(torch.int16) - hr_u8.to(torch.int16)).abs().amax(-1).float()
    hist = torch.histc(d, bins=256, min=0, max=255)
    cdf = torch.cumsum(hist, 0) / hist.sum()
    out = {"mean_abs": float(d.mean()), "within8_frac": float((d <= 8).float().mean()),
           "exact_frac": float((d == 0).float().mean()), "max_abs": float(d.max()),
           "p999": float(torch.searchsorted(cdf, torch.tensor(0.999, device=cdf.device)))}
    if imask is not None:
        di = d[:, :-1, :-1][imask]
        out["interior_exact_frac"] = float((di == 0).float().mean())
        out["interior_mean_abs"] = float(di.mean())
    return out


def _forward_u8(model, u8, shrink: float = 0.0):
    """``forward`` with an optional soft-shrink on the residual, in units of 1/255.

    On this family shrinkage is not a nicety. The cheap frame's interior is already 0.997-0.9999
    bit-exact, because a downsample-then-upsample of a flat region reproduces it, so "the corrector
    has nothing to say here" has to produce a bit-identical pixel or the parity metric falls.
    """
    x = u8.permute(0, 3, 1, 2).to(model.dtype).mul_(1.0 / 255.0)
    x = x.contiguous(memory_format=torch.channels_last)
    base = F.interpolate(x, size=(model.out_size, model.out_size), mode="bilinear",
                         align_corners=False)
    xin = x
    if model.pre_size and x.shape[-1] != model.in_size:
        xin = F.interpolate(x, size=(model.in_size, model.in_size), mode="bilinear",
                            align_corners=False)
    res = model.residual(xin)
    if shrink > 0:
        res = F.softshrink(res, shrink / 255.0)
    y = (base + res).mul_(255.0).round_().clamp_(0, 255)          # stays in the model's dtype
    return y.to(torch.uint8).permute(0, 2, 3, 1)


@torch.no_grad()
def predict(model, lr_u8, batch: int = 128, shrink: float = 0.0):
    return torch.cat([_forward_u8(model, lr_u8[i:i + batch], shrink)
                      for i in range(0, len(lr_u8), batch)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("envs", help="one env name, or several separated by commas")
    ap.add_argument("--net", default="unet_112t128_w48_d2_u2")
    ap.add_argument("--tag", default="r112")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-limit", type=int, default=512)
    ap.add_argument("--shrink", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    envs = [e for e in a.envs.split(",") if e]

    tr_lr, tr_hr, metas = [], [], {}
    ho: dict = {}
    for e in envs:
        lr, hr, _, _, meta, _, _ = D.load_split(e, "train", a.tag, a.limit or None)
        tr_lr.append(torch.as_tensor(lr, device="cuda"))
        tr_hr.append(torch.as_tensor(hr, device="cuda"))
        metas[e] = meta
        hlr, hhr, _, _, hmeta, _, _ = D.load_split(e, "heldout", a.tag, a.eval_limit or None)
        ho[e] = (torch.as_tensor(hlr, device="cuda"), torch.as_tensor(hhr, device="cuda"),
                 interior_masks(hhr), hmeta)
        print(f"{e}: train {lr.shape} heldout {hlr.shape}", flush=True)
    tr_lr_g = torch.cat(tr_lr)
    tr_hr_g = torch.cat(tr_hr)
    del tr_lr, tr_hr

    model = nets.build(a.net).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr_lr_g) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1),
                                                pct_start=0.15)

    def evaluate(shrink=a.shrink):
        model.eval()
        per = {}
        for e, (hlr, hhr, im, _) in ho.items():
            per[e] = quick_metrics(predict(model, hlr, shrink=shrink), hhr, im)
        agg = {k: float(np.mean([v[k] for v in per.values()])) for k in next(iter(per.values()))}
        return per, agg

    base_per, base_agg = evaluate()
    print("epoch  -1 (bilinear baseline) " + json.dumps({k: round(v, 5) for k, v in base_agg.items()}),
          flush=True)

    hist = []
    best = (1e9, None, -1)
    g = torch.Generator(device="cuda").manual_seed(a.seed)
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr_lr_g), device="cuda", generator=g)
        tot, nb, gmax = 0.0, 0, 0.0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            idx = perm[i:i + a.batch]
            x = tr_lr_g[idx].permute(0, 3, 1, 2).float().mul_(1 / 255.0)
            x = x.contiguous(memory_format=torch.channels_last)
            y = tr_hr_g[idx].permute(0, 3, 1, 2).float().mul_(1 / 255.0)
            y = y.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.forward_float(x)
                loss = F.l1_loss(p.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gmax = max(gmax, float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)))
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        per, agg = evaluate()
        agg.update(train_l1_255=255.0 * tot / max(nb, 1), grad_norm_max=gmax, epoch=ep,
                   lr=float(sched.get_last_lr()[0]), seconds=time.perf_counter() - t0)
        hist.append({"agg": agg, "per_env": per})
        print(f"epoch {ep:3d} " + json.dumps({k: round(v, 5) for k, v in agg.items()}), flush=True)
        if agg["mean_abs"] < best[0]:
            best = (agg["mean_abs"], {k: v.detach().clone() for k, v in model.state_dict().items()},
                    ep)

    model.load_state_dict(best[1])
    final_per, final_agg = evaluate()
    shr_per, shr_agg = evaluate(shrink=0.5) if a.shrink == 0 else (None, None)
    os.makedirs(CKPT, exist_ok=True)
    name = a.out or f"{'+'.join(envs) if len(envs) < 4 else f'multi{len(envs)}'}_{a.net}_{a.tag}"
    torch.save({"net": a.net, "envs": envs, "tag": a.tag, "state_dict": best[1],
                "args": vars(a), "best_epoch": best[2], "heldout": final_per,
                "bilinear_baseline": base_per}, os.path.join(CKPT, name + ".pt"))
    with open(os.path.join(CKPT, name + ".json"), "w") as f:
        json.dump({"args": vars(a), "envs": envs, "params": nets.n_params(model),
                   "gflops_per_frame": nets.gflops_per_frame(a.net),
                   "bilinear_baseline": {"per_env": base_per, "agg": base_agg},
                   "final": {"per_env": final_per, "agg": final_agg},
                   "final_shrink0.5": None if shr_agg is None else
                   {"per_env": shr_per, "agg": shr_agg},
                   "best_epoch": best[2], "history": hist, "train_meta": metas}, f, indent=1)
    print("wrote", os.path.join(CKPT, name + ".pt"), flush=True)


if __name__ == "__main__":
    main()
