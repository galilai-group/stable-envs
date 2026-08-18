"""Two experiments ``train.py`` cannot express, in one fit loop, because they share a data path.

**(a) ONE CORRECTOR ACROSS SEVERAL CELLS.** ``--envs A,B,C`` concatenates several cells' training
splits and reports a held-out number PER CELL after every epoch. The three room cameras of one
RoboCasa kitchen see one scene from three poses, and ``README.md`` §4 only ever measured a corrector
trained on ONE of them and pointed at another -- which fails. Training on all three at once is a
different question and the cheapest way to ask what the network is actually learning: if one
checkpoint covers a kitchen, the thing being fitted is the kitchen; if it does not, it is the view.

``--per-env-limit`` is what makes the cost claim measurable rather than asserted. Trained on the FULL
union a joint model has three times the data and three times the fitting of one specialist, so beating
them costs the same and saves only a checkpoint. Trained on ``16384 // n_envs`` rows per cell it has
exactly ONE specialist's budget spread over three cells, and beating them there is the 3x.

**(b) RANDOMISING THE AUX TABLE ROWS THE WAY THE STYLE IS RANDOMISED.** ``README.md`` §4b diagnosed
the descriptor's out-of-distribution collapse precisely: ``background frac 0.000`` in the generation
log means the model was never once shown the background row of its own table, and by extension never
a row it had not fitted, so a table swapped to another kitchen is entirely unseen input and it emits
86/255 of residual on a grey frame. Three augmentations attack exactly that, and none of them was
tried:

* ``--row-dropout p`` -- each GEOM (not each pixel) is independently remapped to the background row
  with probability ``p`` for the duration of a step. This is the one that makes ``background frac``
  nonzero, and it is per geom rather than per pixel because a per-pixel drop is salt-and-pepper noise
  a convolution filters out, while a per-geom drop removes a whole surface's identity the way an
  unknown material would.
* ``--row-jitter s`` -- Gaussian noise of ``s`` times each column's own standard deviation across
  rows, so a column that is 0/1 (``texgen_gap``) and a column that is a log size are perturbed on
  their own scales.
* ``--row-resample p`` -- a fraction of rows is replaced by a row drawn from the COLUMN-WISE marginals
  (each column permuted independently). Those are rows that exist nowhere in any kitchen: the joint
  structure is destroyed and the marginals are kept, which is as close as one can get to "a table row
  this model has never fitted" without smuggling in the test scene's own table.

All three are applied to a COPY of the frozen table for the step and the pristine table is restored
before anything is saved, so the checkpoint carries the table it will be evaluated with.
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
from train import CKPT, predict, quick_metrics


def load_many(envs, split, tag, limit_per_env=0):
    """Concatenated ``(lr, hr, depth, segid, offsets, tables)`` over several cells.

    Geom ids are OFFSET per cell and the cells' descriptor tables are stacked, so one frozen table
    serves several scenes exactly: cell ``k``'s geom ``g`` is row ``offset_k + g``. Two kitchens'
    ids collide otherwise, and a collision is not an error -- it is a wrong material, silently.
    """
    lrs, hrs, deps, segs, offs, tabs, metas = [], [], [], [], [], [], []
    off = 0
    for e in envs:
        lr, hr, _, _, meta, dep, seg = D.load_split(e, split, tag, limit_per_env or None)
        t = D.geom_table(e)
        if seg is not None:
            if t is None:
                raise SystemExit(f"{e} has segid planes but no geom_features.npy")
            seg = seg + off
        lrs.append(lr)
        hrs.append(hr)
        deps.append(dep)
        segs.append(seg)
        offs.append(off)
        if t is not None:
            tabs.append(t)
            off += int(t.shape[0])
        metas.append(meta)
    cat = lambda xs: None if any(x is None for x in xs) else np.concatenate(xs)
    return (np.concatenate(lrs), np.concatenate(hrs), cat(deps), cat(segs), offs,
            (np.concatenate(tabs) if tabs else None), metas)


class RowAug:
    """Row-level augmentation of a FROZEN descriptor table, rebuilt per step.

    Holds the pristine table and hands back ``(table', remap)``: the caller installs ``table'`` on
    the model's ``GeomTable`` buffer for one step and gathers ``remap[ids]`` to apply the dropout.
    """

    def __init__(self, table: torch.Tensor, dropout: float, jitter: float, resample: float,
                 seed: int = 0):
        self.t = table
        self.dropout, self.jitter, self.resample = float(dropout), float(jitter), float(resample)
        self.g = torch.Generator(device=table.device).manual_seed(seed)
        self.col_std = table[1:].std(0, keepdim=True).clamp_min(1e-6)  # row 0 is background
        self.n = int(table.shape[0])

    def active(self) -> bool:
        return self.dropout > 0 or self.jitter > 0 or self.resample > 0

    def step(self):
        t = self.t
        if self.resample > 0:
            t = t.clone()
            body = self.t[1:]
            pick = torch.rand(self.n - 1, device=t.device, generator=self.g) < self.resample
            if int(pick.sum()) > 0:
                k = int(pick.sum())
                # Each column permuted INDEPENDENTLY: the marginals survive, the joint structure does
                # not, so these are rows no compiled kitchen would ever produce.
                cols = [body[torch.randint(0, len(body), (k,), device=t.device, generator=self.g), j]
                        for j in range(body.shape[1])]
                t[1:][pick] = torch.stack(cols, 1)
        if self.jitter > 0:
            t = t if t is not self.t else t.clone()
            noise = torch.randn(t.shape, device=t.device, generator=self.g) * self.col_std
            t = t + self.jitter * noise
            t[0] = self.t[0]                                   # background stays the background
        remap = None
        if self.dropout > 0:
            remap = torch.arange(self.n, device=t.device, dtype=torch.int32)
            drop = torch.rand(self.n, device=t.device, generator=self.g) < self.dropout
            drop[0] = False
            remap = torch.where(drop, torch.zeros_like(remap), remap)
        return t, remap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True, help="comma-separated cells to fit JOINTLY")
    ap.add_argument("--net", default="unet_112t128_w48_d2_u2")
    ap.add_argument("--tag", default="r112_tex2_sty")
    ap.add_argument("--eval-tag", default="", help="tag of the per-cell held-out splits")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--per-env-limit", type=int, default=0,
                    help="rows kept per cell; 0 = all. 16384//n_envs is ONE specialist's budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shrink", type=float, default=0.0)
    ap.add_argument("--dyn-frac", type=float, default=0.25)
    ap.add_argument("--aux", default="none", choices=("none", "depth", "descriptor", "learned"))
    ap.add_argument("--emb-dim", type=int, default=8)
    ap.add_argument("--row-dropout", type=float, default=0.0)
    ap.add_argument("--row-jitter", type=float, default=0.0)
    ap.add_argument("--row-resample", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    envs = [e for e in a.envs.split(",") if e]
    etag = a.eval_tag or a.tag

    tr_lr, tr_hr, tr_d, tr_s, offs, table, metas = load_many(envs, "train", a.tag, a.per_env_limit)
    print(f"train {tr_lr.shape} over {envs} (offsets {offs})", flush=True)

    auxspec = None
    if a.aux != "none":
        if tr_d is None:
            raise SystemExit(f"--aux {a.aux} but {a.tag} has no depth/segid planes; run gen_aux.py")
        auxspec = {"depth": True}
        if a.aux == "descriptor":
            auxspec.update(geom="descriptor", table=table)
        elif a.aux == "learned":
            auxspec.update(geom="learned", n_rows=int(table.shape[0]), dim=a.emb_dim)

    tr_lr_g = torch.as_tensor(tr_lr, device="cuda")
    tr_hr_g = torch.as_tensor(tr_hr, device="cuda")
    del tr_lr, tr_hr
    tr_d_g = None if tr_d is None or auxspec is None else torch.as_tensor(tr_d, device="cuda")
    tr_s_g = None if tr_s is None or auxspec is None else torch.as_tensor(tr_s, device="cuda")

    # Per-cell held-out sets, each with its OWN dynamic mask, kept separate: a mean over three cells
    # would hide a corrector that is excellent on two and harmful on the third, which is exactly the
    # failure mode a joint model is at risk of.
    ho = []
    for k, e in enumerate(envs):
        lr, hr, _, _, _, dep, seg = D.load_split(e, "heldout", etag, 512)
        ho.append({"env": e,
                   "lr": torch.as_tensor(lr, device="cuda"),
                   "hr": torch.as_tensor(hr, device="cuda"),
                   "dyn": torch.as_tensor(D.dynamic_mask(hr, a.dyn_frac), device="cuda"),
                   "d": None if dep is None or auxspec is None else
                        torch.as_tensor(dep, device="cuda"),
                   "s": None if seg is None or auxspec is None else
                        torch.as_tensor(seg.astype(np.int32) + offs[k], device="cuda")})

    model = nets.build(a.net, aux=auxspec).cuda().to(memory_format=torch.channels_last)
    aug = None
    if model.geom is not None and not model.geom.learned:
        aug = RowAug(model.geom.table.clone(), a.row_dropout, a.row_jitter, a.row_resample, a.seed)
        if not aug.active():
            aug = None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (len(tr_lr_g) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1),
                                                pct_start=0.15)

    def eval_all():
        model.eval()
        out = []
        for h in ho:
            m = quick_metrics(predict(model, h["lr"], shrink=a.shrink, depth=h["d"], ids=h["s"]),
                              h["hr"], h["dyn"])
            m["env"] = h["env"]
            out.append(m)
        return out

    base = eval_all()
    print("epoch  -1 (bilinear baseline) "
          + json.dumps([{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                        for m in base]), flush=True)

    hist, best = [], (1e9, None)
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
            ids = None if tr_s_g is None else tr_s_g[idx]
            if aug is not None:
                tab, remap = aug.step()
                model.geom.table = tab
                if remap is not None and ids is not None:
                    ids = remap[ids.long()]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model.forward_float(
                    x,
                    None if tr_d_g is None else tr_d_g[idx][:, None].float().mul_(1 / 255.0),
                    ids)
                loss = F.l1_loss(p.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gmax = max(gmax, float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)))
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        if aug is not None:
            model.geom.table = aug.t                      # evaluate on the PRISTINE table, always
        rows = eval_all()
        mean_over_cells = float(np.mean([r["mean_abs"] for r in rows]))
        rec = {"epoch": ep, "per_env": rows, "mean_over_envs": mean_over_cells,
               "train_l1_255": 255.0 * tot / max(nb, 1), "grad_norm_max": gmax,
               "lr": float(sched.get_last_lr()[0]), "seconds": time.perf_counter() - t0}
        hist.append(rec)
        print(f"epoch {ep:3d} mean_over_envs {mean_over_cells:.4f} "
              + json.dumps({r['env']: round(r['mean_abs'], 3) for r in rows})
              + f" train {rec['train_l1_255']:.3f}", flush=True)
        if mean_over_cells < best[0]:
            best = (mean_over_cells, {k: v.detach().clone() for k, v in model.state_dict().items()})

    os.makedirs(CKPT, exist_ok=True)
    if aug is not None:                                   # the checkpoint carries the REAL table
        best[1]["geom_table.table"] = aug.t.clone()
    torch.save({"net": a.net, "env": envs[0], "envs": envs, "geom_offsets": offs, "tag": a.tag,
                "aux": a.aux,
                "aux_spec": None if auxspec is None else
                {k: v for k, v in auxspec.items() if k != "table"},
                "emb_dim": a.emb_dim, "state_dict": best[1],
                "args": vars(a), "heldout": hist[-1], "best_mean_over_envs": best[0],
                "bilinear_baseline": base, "train_meta": metas},
               os.path.join(CKPT, a.out + ".pt"))
    with open(os.path.join(CKPT, a.out + ".json"), "w") as f:
        json.dump({"args": vars(a), "envs": envs, "geom_offsets": offs,
                   "bilinear_baseline": base, "history": hist, "train_meta": metas,
                   "params": nets.n_params(model), "aux": a.aux,
                   "gflops_per_frame": nets.gflops_per_frame(a.net, aux=auxspec)}, f, indent=1)
    print("wrote", os.path.join(CKPT, a.out + ".pt"), flush=True)


if __name__ == "__main__":
    main()
