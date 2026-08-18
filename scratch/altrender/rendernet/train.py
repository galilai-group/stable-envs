"""Fit a corrector on the training split and report the held-out number after every epoch.

Everything is GPU-resident: 16,384 pairs at 112 px in and 256 px out is 3.6 GB of uint8, which is
2.5 % of an H200, so there is no loader, no worker process and no augmentation pipeline to get wrong.
The whole run is a few minutes, which is what makes it possible to sweep architectures rather than
argue about them.

Three choices are load-bearing and none of them is a default:

**The head is zero-initialised, so training starts EXACTLY at the bilinear baseline.** Every number
this produces is therefore a departure from a known, already-measured point, and a run that fails
degrades to that point rather than to noise.

**The loss is L1 on the un-rounded output.** Mean |diff| against the official frame is the metric the
tier declares, so the training objective is that metric and not a proxy for it. Rounding stays out of
the graph and comes back in ``forward``, which is the tensor a consumer receives.

**The held-out set is a different set of RECORDED STATES, not a different draw from the same ones.**
See ``gen_pairs.py``. The training loop never sees it, and the number printed each epoch is measured
on the rounded uint8 output through the repo's own ``pixel_agreement``, not on the float tensor the
loss was computed from.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data as D
import nets

CKPT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/ckpt"


def quick_metrics(pred_u8: torch.Tensor, hr_u8: torch.Tensor, dyn: torch.Tensor | None = None):
    """mean |diff| / within-8 / exact, on device, over the max-of-channels difference.

    Same reduction as ``testing.conformance.pixel_agreement`` (max over channels, then mean over
    pixels) so the in-loop number and the reported one are the same quantity.
    """
    d = (pred_u8.to(torch.int16) - hr_u8.to(torch.int16)).abs().amax(-1).float()
    # ``torch.quantile`` refuses inputs above 2**24 elements and a held-out set is 1.3e8, so the tail
    # is read off a 256-bin histogram of an integer-valued quantity -- exact, not sampled.
    hist = torch.histc(d, bins=256, min=0, max=255)
    cdf = torch.cumsum(hist, 0) / hist.sum()
    out = {"mean_abs": float(d.mean()), "within8_frac": float((d <= 8).float().mean()),
           "exact_frac": float((d == 0).float().mean()), "max_abs": float(d.max()),
           "p999": float(torch.searchsorted(cdf, torch.tensor(0.999, device=cdf.device)))}
    if dyn is not None:
        out["mean_abs_dynamic"] = float(d[:, dyn].mean())
        out["mean_abs_static"] = float(d[:, ~dyn].mean())
    return out


@torch.no_grad()
def predict(model, lr_u8: torch.Tensor, batch: int = 256, shrink: float = 0.0,
            depth=None, ids=None):
    outs = []
    for i in range(0, len(lr_u8), batch):
        sl = slice(i, i + batch)
        outs.append(_forward_u8(model, lr_u8[sl], shrink,
                                None if depth is None else depth[sl],
                                None if ids is None else ids[sl]))
    return torch.cat(outs)


def _forward_u8(model, u8, shrink: float = 0.0, depth=None, ids=None):
    """``forward`` with an optional soft-shrink on the residual, in units of 1/255.

    Shrinkage is not a regularisation trick, it is the interior-exactness lever. MJWarp reproduces a
    flat interior EXACTLY -- 0.41-0.68 of RoboCasa's interior pixels are bit-identical to MuJoCo's --
    and a corrector that adds a residual of 0.2/255 everywhere destroys every one of them at the
    rounding step while improving nothing. Shrinking by half a level makes "the corrector has nothing
    to say here" produce a bit-identical pixel by construction.
    """
    x = u8.permute(0, 3, 1, 2).to(model.dtype).mul_(1.0 / 255.0)
    x = x.contiguous(memory_format=torch.channels_last)
    base = F.interpolate(x, size=(model.out_size, model.out_size), mode="bilinear",
                         align_corners=False)
    xin = x
    if model.pre_size and x.shape[-1] != model.in_size:
        xin = F.interpolate(x, size=(model.in_size, model.in_size), mode="bilinear",
                            align_corners=False)
    if model.aux_dim():
        xin = model._augment(xin, None if depth is None else
                             depth[:, None].to(model.dtype).mul(1.0 / 255.0), ids)
    res = model.residual(xin)
    if shrink > 0:
        res = F.softshrink(res, shrink / 255.0)
    # STAY IN THE MODEL'S DTYPE. An earlier version wrote ``(base + res).float()`` here, which
    # materialises an fp32 (1024, 3, 256, 256) tensor -- 805 MB -- for a quantisation step that fp16
    # performs exactly (every integer up to 2048 is representable). It cost 3.3 us per frame, 18 % of
    # the whole corrector budget, and it was invisible until the end-to-end timing disagreed with the
    # standalone one by exactly that much.
    y = (base + res).mul_(255.0).round_().clamp_(0, 255)
    return y.to(torch.uint8).permute(0, 2, 3, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--net", default="espcn_112t128_w96_d6_u2")
    ap.add_argument("--tag", default="r112_tex2")
    ap.add_argument("--eval-tag", default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shrink", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dyn-frac", type=float, default=0.25)
    ap.add_argument("--aux", default="none",
                    choices=("none", "depth", "descriptor", "learned"),
                    help="extra per-pixel inputs: depth only, depth + the model-derived per-geom "
                         "descriptor (transferable), or depth + a learned per-geom embedding "
                         "(maximum capacity, no transfer by construction)")
    ap.add_argument("--emb-dim", type=int, default=8)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True

    tr_lr, tr_hr, _, _, meta, tr_d, tr_s = D.load_split(a.env, "train", a.tag, a.limit or None)
    ho_lr, ho_hr, _, _, hmeta, ho_d, ho_s = D.load_split(a.env, "heldout", a.eval_tag or a.tag)
    # The aux spec is decided HERE and stored in the checkpoint, so a corrector can never be rebuilt
    # with different inputs from the ones it was fitted on.
    auxspec = None
    if a.aux != "none":
        if tr_d is None:
            raise SystemExit(f"--aux {a.aux} but {a.tag} has no depth/segid planes; run gen_aux.py")
        tab = D.geom_table(a.env)
        auxspec = {"depth": True}
        if a.aux == "descriptor":
            auxspec.update(geom="descriptor", table=tab)
        elif a.aux == "learned":
            auxspec.update(geom="learned", n_rows=int(tab.shape[0]), dim=a.emb_dim)
        elif a.aux != "depth":
            raise SystemExit(f"unknown --aux {a.aux}")
    dyn = torch.as_tensor(D.dynamic_mask(ho_hr, a.dyn_frac), device="cuda")
    print(f"train {tr_lr.shape} -> {tr_hr.shape} | heldout {ho_lr.shape} | "
          f"dynamic pixels {float(dyn.float().mean()):.3f}", flush=True)

    tr_lr_g = torch.as_tensor(tr_lr, device="cuda")
    tr_hr_g = torch.as_tensor(tr_hr, device="cuda")
    ho_lr_g = torch.as_tensor(ho_lr, device="cuda")
    ho_hr_g = torch.as_tensor(ho_hr, device="cuda")
    g_ = lambda x: None if x is None or auxspec is None else torch.as_tensor(x, device="cuda")
    tr_d_g, tr_s_g = g_(tr_d), g_(tr_s)
    ho_d_g, ho_s_g = g_(ho_d), g_(ho_s)

    model = nets.build(a.net, aux=auxspec).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr_lr_g) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1),
                                                pct_start=0.15)

    # The bilinear baseline, measured on the same held-out frames the model will be measured on. A
    # corrector that does not beat this has learned nothing, and the zero-initialised head means the
    # model IS this at step 0.
    base = quick_metrics(predict(model, ho_lr_g, depth=ho_d_g, ids=ho_s_g), ho_hr_g, dyn)
    print("epoch  -1 (bilinear baseline) " + json.dumps({k: round(v, 4) for k, v in base.items()}),
          flush=True)

    hist = []
    best = (1e9, None)
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
            # bf16 autocast with NO loss scaler: bf16 carries fp32's exponent range, so there is
            # nothing to scale, and a scaler that silently skips steps would hide the divergence this
            # loop is now guarded against. The guard is the clip: one spike is what killed the ReLU
            # sweep, and it arrives as a single step with a gradient norm two orders of magnitude
            # above the rest.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.forward_float(
                    x,
                    None if tr_d_g is None else tr_d_g[idx][:, None].float().mul_(1 / 255.0),
                    None if tr_s_g is None else tr_s_g[idx])
                loss = F.l1_loss(p.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            gmax = max(gmax, float(gn))
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        model.eval()
        m = quick_metrics(predict(model, ho_lr_g, shrink=a.shrink, depth=ho_d_g, ids=ho_s_g),
                          ho_hr_g, dyn)
        m["train_l1_255"] = 255.0 * tot / max(nb, 1)
        m["grad_norm_max"] = gmax
        m["lr"] = float(sched.get_last_lr()[0])
        m["epoch"] = ep
        m["seconds"] = time.perf_counter() - t0
        hist.append(m)
        print(f"epoch {ep:3d} " + json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
        if m["mean_abs"] < best[0]:
            best = (m["mean_abs"], {k: v.detach().clone() for k, v in model.state_dict().items()})

    os.makedirs(CKPT, exist_ok=True)
    name = a.out or f"{a.env}_{a.net}_{a.tag}"
    torch.save({"net": a.net, "env": a.env, "tag": a.tag, "aux": a.aux,
                "aux_spec": None if auxspec is None else
                {k: v for k, v in auxspec.items() if k != "table"},
                "emb_dim": a.emb_dim, "state_dict": best[1],
                "args": vars(a), "heldout": hist[-1], "best_mean_abs": best[0],
                "bilinear_baseline": base, "train_meta": meta, "heldout_meta": hmeta},
               os.path.join(CKPT, name + ".pt"))
    with open(os.path.join(CKPT, name + ".json"), "w") as f:
        json.dump({"args": vars(a), "bilinear_baseline": base, "history": hist,
                   "train_meta": meta, "heldout_meta": hmeta,
                   "params": nets.n_params(model), "aux": a.aux,
                   "gflops_per_frame": nets.gflops_per_frame(a.net, aux=auxspec)}, f, indent=1)
    print("wrote", os.path.join(CKPT, name + ".pt"), flush=True)


if __name__ == "__main__":
    main()
