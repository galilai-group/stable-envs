"""Fit an ss=1 -> official corrector, and watch the ONE metric that can silently break the cell.

Everything is GPU-resident: 8,192 pairs at 256 px is 3.2 GB of uint8, which is 2 % of an H200, so
there is no loader and no augmentation pipeline to get wrong. Three things are different from
``rendernet/train.py``, and each of them is because these are PARITY cells rather than visual-tier
ones.

**The zero-initialised head starts the model at the ss=1 frame BYTE FOR BYTE.** Input and output are
the same resolution here, so the skip is an identity rather than a bilinear upsample: epoch -1 is not
an approximation of the baseline, it IS the baseline, and every later number is a departure from a row
of the results table.

**Interior-exactness is reported every epoch, against a mask computed ONCE.** The parity assertion is
that every pixel whose 3x3 neighbourhood in the OFFICIAL frame is flat must be bit-identical. A
corrector that lowers mean |diff| while adding half a level of residual to a flat surface destroys
every one of those pixels and the mean will not say so. The mask is
``testing.conformance.interior_mask`` -- the repo's own function, on the held-out official frames,
cached -- so the in-loop number is the same quantity the test asserts and not a re-derivation of it.

**``--shrink`` is the interior lever and it is swept, not defaulted.** Soft-shrinking the residual by
half a level makes "the corrector has nothing to say here" quantise back to the input's own byte. On
a visual-tier cell that was worth a third of a point and was turned off; on a parity cell it is the
difference between a claim that survives and one that does not.

THREE FAILURES THAT COST A DAY EACH AND ARE INVISIBLE FROM THE LOSS CURVE
------------------------------------------------------------------------

All three produce a run that finishes, reports the baseline to five decimal places, and looks like a
negative result about the problem. They are facts about the fit.

1. **L1 never leaves the origin when the target is nearly the input.** 93-98 % of pixels here are
   already bit-identical, so the L1 subgradient is exactly zero on almost every pixel and carries no
   magnitude anywhere else. The residual settles at 0.04-0.09/255 -- a fifth of a quantisation level,
   so not one uint8 changes -- and ``train_loss`` sits flat at the do-nothing value for 400 epochs.
   Charbonnier at eps 1e-3 does the same. **L2 learns; check ``res_mean_255``, not the loss.** That
   diagnostic exists because ``mean_abs`` cannot tell "learned nothing" from "learned something worth
   a fifth of a level", and those are opposite answers.

2. **``softshrink`` in the TRAINING forward zeroes every gradient.** It is exactly zero below its
   threshold and so is its derivative, and the head is zero-initialised, so the model starts inside
   the flat region and never leaves: ``grad_norm_max`` reads 0.0 at epoch 0 and at epoch 400.
   Shrinkage is a quantisation-time decision; it is applied in ``metrics``/``predict`` and nowhere
   else. The signature to look for is a gradient norm of exactly 0.0, which no clip and no learning
   rate will explain.

3. **The trunk dies above lr ~ 8e-4 and does not come back.** With OneCycle peaking at 2e-3 the fit
   is real for 40 epochs -- training loss down 5.6x, held-out p99.9 86 -> 31, past supersample 2's
   41 -- and then at the schedule's peak the loss jumps back to the do-nothing value *to four
   significant figures* and stays there for the remaining 160 epochs. It reads as convergence to "the
   optimum is zero". It is not; peak lr 3e-4 with clip 0.05 is stable and reaches the same place.
   ``rendernet``'s ReLU-death has the same shape for a different reason, and both are quiet because a
   zero-initialised head gives a dead trunk a perfectly good baseline to fall back to.

AND THE RESULT THEY WERE HIDING, kept because it was measured and then not needed
--------------------------------------------------------------------------------

With L2 plus ``--res-l1`` and an evaluation-time shrink, a 415 k-parameter corrector on
``DMCCartpoleBalance`` at 84 px reaches interior-exact 1.0 -- zero disagreeing interior pixels over 41
independent draws -- while beating the ss=2 render it replaces on mean (0.127 against 0.289),
all-pixel exactness (0.9840 against 0.9826) and p99.9 (18 against 44), at 1.24x its throughput.
``DMCFingerSpin`` at shrink 8 does the same (0.505 against 0.547, p99.9 33 against 41) at 1.45x.

It is not shipped, and the reasons are the honest ones. **It never fixes ``max_abs``** -- 102-120
against supersample 2's 52 -- because an ss=1 frame is a ONE-BIT QUANTISATION of the coverage field,
so sub-pixel position survives only in the staircase pattern along a locally straight edge and is
simply absent at a corner, a thin feature or a silhouette crossing, which is where the extreme
pixels are. **It loses outright on ``MujocoPusher`` at 256 px**, worse than shipped on every metric at
the shrink that restores interior-exactness. And the shrink threshold has to be picked from a DRAW
COUNT: FingerSpin at shrink 4 reads 0.99999694 on 2,048 held-out states -- 5 pixels in 1.63 M -- and
fails 4 of 41 independent draws. Shrink 8 is zero across all 41.

The family took ``default_supersample = 1`` instead, which keeps every assertion with margin and needs
no checkpoint. See ``probe_parity_draws.py`` for the survey that decided it.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import common as K  # noqa: E402

import numpy as np
import torch
import torch.nn.functional as F

import nets_ss  # noqa: E402


def load(cell: str, size: int, split: str):
    d = os.path.join(K.ROOT, cell, f"s{size}", split)
    out = {k: np.load(os.path.join(d, f"{k}.npy")) for k in ("ss1", "ss2", "official")}
    out["meta"] = json.load(open(os.path.join(d, "meta.json")))
    return out


def interior_mask_cached(cell: str, size: int, split: str, official: np.ndarray) -> np.ndarray:
    """``(N, H-1, W-1)`` bool from the repo's own ``interior_mask``, computed once and stored.

    scipy's per-frame minimum filter over 2,048 frames is a minute; recomputing it every epoch would
    dominate a six-minute fit. It is a pure function of the official frames, so it is cached beside
    them and the cache is keyed by the split directory it was built from.
    """
    p = os.path.join(K.ROOT, cell, f"s{size}", split, "interior_mask.npy")
    if os.path.exists(p):
        m = np.load(p)
        if m.shape[0] == len(official):
            return m
    from stable_envs.testing import conformance as C
    t0 = time.perf_counter()
    m = np.concatenate([C.interior_mask(official[i:i + 256]) for i in range(0, len(official), 256)])
    np.save(p, m)
    print(f"interior mask {m.shape} ({m.mean():.4f} of pixels) in {time.perf_counter()-t0:.1f}s",
          flush=True)
    return m


@torch.no_grad()
def metrics(pred: torch.Tensor, target: torch.Tensor, imask: torch.Tensor | None,
            block: int = 1024) -> dict:
    """Every quantity the parity test reads, on device, over the max-of-channels difference.

    ``torch.quantile`` refuses inputs above 2**24 elements, so the tail comes off a 256-bin histogram
    of an integer-valued quantity, which is exact rather than sampled.
    """
    n = len(pred)
    hist = torch.zeros(256, device=pred.device, dtype=torch.float64)
    tot = 0.0
    n_int = n_bad = 0
    int_sum = 0.0
    for lo in range(0, n, block):
        d = (pred[lo:lo + block].to(torch.int16)
             - target[lo:lo + block].to(torch.int16)).abs().amax(-1)
        hist += torch.histc(d.float(), bins=256, min=0, max=255).double()
        tot += float(d.sum())
        if imask is not None:
            di = d[:, :-1, :-1][imask[lo:lo + block]]
            n_int += int(di.numel())
            n_bad += int((di != 0).sum())
            int_sum += float(di.sum())
    npx = float(hist.sum())
    cdf = torch.cumsum(hist, 0) / npx
    lev = torch.arange(256, device=pred.device, dtype=torch.float64)
    out = {"mean_abs": tot / npx,
           "exact_frac": float(hist[0] / npx),
           "within8_frac": float(hist[:9].sum() / npx),
           "max_abs": int(lev[hist > 0].max()),
           "p999": float(torch.searchsorted(cdf, torch.tensor(0.999, device=cdf.device,
                                                              dtype=torch.float64))),
           "p9999": float(torch.searchsorted(cdf, torch.tensor(0.9999, device=cdf.device,
                                                              dtype=torch.float64)))}
    if imask is not None:
        out.update(interior_px=n_int, interior_bad_px=n_bad,
                   interior_exact_frac=None if n_int == 0 else 1.0 - n_bad / n_int,
                   interior_mean_abs=None if n_int == 0 else int_sum / n_int)
    return out


@torch.no_grad()
def residual_stats(model, x_u8: torch.Tensor) -> dict:
    """How big is the learned residual, in units of 1/255, BEFORE quantisation kills it.

    The uint8 metrics cannot distinguish "the network learned nothing" from "the network learned
    something worth a fifth of a quantisation level", and those are opposite answers to the question
    this study is asking. The residual in float is the only place the difference is visible.
    """
    x = x_u8.permute(0, 3, 1, 2).to(model.dtype).mul_(1.0 / 255.0)
    r = model.residual_float(x.contiguous(memory_format=torch.channels_last)).float().abs() * 255.0
    return {"res_mean_255": float(r.mean()), "res_max_255": float(r.max()),
            "res_frac_over_half": float((r > 0.5).float().mean())}


@torch.no_grad()
def predict(model, x_u8: torch.Tensor, shrink: float = 0.0, batch: int = 256) -> torch.Tensor:
    outs = []
    for i in range(0, len(x_u8), batch):
        outs.append(model(x_u8[i:i + batch], shrink))
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--net", default="espcn_w64_d6_u4")
    ap.add_argument("--target", default="official", choices=("official", "ss2"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.0)
    ap.add_argument("--loss", default="l1", choices=("l1", "l2", "charb"))
    ap.add_argument("--res-l1", type=float, default=0.0,
                    help="L1 penalty on the residual; pushes it to EXACTLY zero where the "
                         "data does not pay for a correction, which is what preserves the "
                         "bit-exact pixels a parity cell is asserted on")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True

    tr = load(a.cell, a.size, "train")
    ho = load(a.cell, a.size, "heldout")
    if a.limit:
        for k in ("ss1", "ss2", "official"):
            tr[k] = tr[k][:a.limit]
    imask = interior_mask_cached(a.cell, a.size, "heldout", ho["official"])

    x_tr = torch.as_tensor(tr["ss1"], device="cuda")
    y_tr = torch.as_tensor(tr[a.target], device="cuda")
    x_ho = torch.as_tensor(ho["ss1"], device="cuda")
    y_ho = torch.as_tensor(ho["official"], device="cuda")
    ss2_ho = torch.as_tensor(ho["ss2"], device="cuda")
    im = torch.as_tensor(imask, device="cuda")
    print(f"{a.cell} {a.size}px train {tuple(x_tr.shape)} target={a.target} heldout "
          f"{tuple(x_ho.shape)} interior_frac={float(im.float().mean()):.4f}", flush=True)

    model = nets_ss.build(a.net).cuda().to(memory_format=torch.channels_last)
    ref = {"ss1": metrics(x_ho, y_ho, im), "ss2": metrics(ss2_ho, y_ho, im)}
    print("ss1  (do nothing) " + json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                             for k, v in ref["ss1"].items()}), flush=True)
    print("ss2  (shipped)    " + json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                             for k, v in ref["ss2"].items()}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    steps = a.epochs * (len(x_tr) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1),
                                                pct_start=0.15)
    g = torch.Generator(device="cuda").manual_seed(a.seed)
    hist, best = [], (1e9, None, -1)
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(x_tr), device="cuda", generator=g)
        tot, nb, gmax = 0.0, 0, 0.0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            idx = perm[i:i + a.batch]
            x = x_tr[idx].permute(0, 3, 1, 2).float().mul_(1 / 255.0)
            x = x.contiguous(memory_format=torch.channels_last)
            y = y_tr[idx].permute(0, 3, 1, 2).float().mul_(1 / 255.0)
            y = y.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # NO shrink in the training forward, and that is a bug fix rather than a preference.
                # ``softshrink`` is exactly zero on |res| < t and so is its gradient, and the head is
                # zero-initialised -- so a shrink applied here starts the model in the flat region and
                # it never leaves: measured, ``grad_norm_max`` is 0.0 at every epoch and the fit is a
                # no-op that reports the baseline. Shrinkage is a quantisation-time decision and it
                # belongs at evaluation, where ``metrics`` applies it.
                p = model.forward_float(x, 0.0).float()
                if a.loss == "l1":
                    loss = F.l1_loss(p, y)
                elif a.loss == "l2":
                    loss = F.mse_loss(p, y)
                else:                        # Charbonnier: L2 near zero, L1 in the tail
                    loss = torch.sqrt((p - y) ** 2 + (1e-3) ** 2).mean()
                if a.res_l1:
                    # A sparsity penalty on the RESIDUAL, aimed at the measured failure mode rather
                    # than at regularisation in general. Under plain L2 the corrector fixes the edge
                    # tail (p99.9 86 -> 38, past ss=2's 41) and pays for it by spreading ~0.4-1.0/255
                    # over the whole frame, which quantises 93 % of already-bit-exact pixels away.
                    # L1 on the residual is the term whose optimum is EXACTLY zero wherever the data
                    # does not pay for a correction, which is the property a parity cell needs.
                    loss = loss + a.res_l1 * (p - x).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gmax = max(gmax, float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)))
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        model.eval()
        m = metrics(predict(model, x_ho, a.shrink), y_ho, im)
        m.update(epoch=ep, train_loss=tot / max(nb, 1), grad_norm_max=gmax,
                 lr=float(sched.get_last_lr()[0]), seconds=time.perf_counter() - t0,
                 **residual_stats(model, x_ho[:64]))
        hist.append(m)
        print(f"epoch {ep:3d} " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                              for k, v in m.items()}), flush=True)
        if m["mean_abs"] < best[0]:
            best = (m["mean_abs"], {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)

    os.makedirs(K.CKPT, exist_ok=True)
    name = a.tag or f"{a.cell}_s{a.size}_{a.net}_{a.target}_sh{a.shrink:g}_{a.loss}"
    torch.save({"net": a.net, "cell": a.cell, "size": a.size, "args": vars(a),
                "state_dict": best[1], "best_epoch": best[2], "baseline": ref,
                "heldout": hist[-1], "best_mean_abs": best[0]},
               os.path.join(K.CKPT, name + ".pt"))
    with open(os.path.join(K.CKPT, name + ".json"), "w") as f:
        json.dump({"args": vars(a), "baseline": ref, "history": hist,
                   "params": nets_ss.n_params(model),
                   "gflops_per_frame": nets_ss.gflops_per_frame(a.net, a.size),
                   "train_meta": tr["meta"], "heldout_meta": ho["meta"]}, f, indent=1)
    print("wrote", os.path.join(K.CKPT, name + ".pt"), flush=True)


if __name__ == "__main__":
    main()
