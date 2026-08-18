"""Two-rate rendering for HumanoidBench: measure BOTH halves of the frame, not one and a subtraction.

The premise on record is that 78% of the kernel's time is inside the 25% of the frame the robot's box
covers, so a foveated scheme -- robot at full rate, far field cheap -- has 22% to play with and a
ceiling of 1/0.78 = 1.28x. Every number in that sentence deserves to be measured directly rather than
inferred from a centred crop, and ``rc.ray`` makes that possible: the megakernel indexes a FLAT array
of camera-frame directions, so a context can be handed the robot's rays, or the complement of them,
or every other pixel of them, and each is a direct timing of that set.

What is measured here:

  ``full``          the frame as shipped
  ``box``           only the rays inside the movers' union box, as a rectangle (a real window)
  ``outer``         only the rays OUTSIDE it -- an arbitrary, non-rectangular ray set
  ``outer_half``    the outer field at every second pixel in each axis, nearest-upsampled
  ``two_rate``      box at full rate composited with ``outer_half``: throughput and all four metrics

``box`` + ``outer`` against ``full`` is the check that the split is a partition of the same work; the
gap between them is the launch overhead a two-pass scheme has to pay twice.

The box itself is measured, per state and unioned over a chunk, with and without sorting the batch by
where the box lands -- because the ray table is shared by every world in a launch, so what a window
costs is set by the UNION over the chunk and not by the per-state box.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_common as RP  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/reproj"


def mover_mask(c, mv_t, n: int, res: int):
    """Per-world boolean mask of the pixels the movers occupy, from the segmentation-free route: a
    render with everything else hidden writes depth > 0 exactly where a mover was hit."""
    return c.depth[:n] > 0


def boxes_from_mask(mask: torch.Tensor):
    """``(N,H,W)`` -> ``(N,4)`` [c0,c1,r0,r1], the tight box of the true pixels (whole frame if none)."""
    n, h, w = mask.shape
    anyr = mask.any(2)
    anyc = mask.any(1)
    idx_r = torch.arange(h, device=mask.device).unsqueeze(0)
    idx_c = torch.arange(w, device=mask.device).unsqueeze(0)
    big = torch.full_like(idx_r, h)
    r0 = torch.where(anyr, idx_r, big).amin(1)
    r1 = torch.where(anyr, idx_r, torch.zeros_like(idx_r)).amax(1) + 1
    c0 = torch.where(anyc, idx_c, big).amin(1)
    c1 = torch.where(anyc, idx_c, torch.zeros_like(idx_c)).amax(1) + 1
    return torch.stack([c0, c1, r0, r1], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HumanoidBenchH1Walk")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import stable_envs as se

    tag = a.tag or a.env
    mod = RP.module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    cls = type(env)
    m = RP.model_of(mod, cls)
    cid = RP.cam_id_of(mod, cls, m)
    S = a.size

    st = env.sample_states(max(a.batch, a.nworld))
    qpos = RP.full_qpos(mod, cls, env, m, st)
    _, _, gp, gm = RP.kinematics_cpu(m, qpos[:a.nworld], cid)
    mv = RP.moving_geoms(m, gp, gm, mod.VISIBLE_GROUPS)
    vis = set(int(g) for g in mod.VISIBLE_GROUPS)
    static = np.array([g for g in range(m.ngeom)
                       if int(m.geom_group[g]) in vis and g not in set(mv.tolist())], np.int64)
    qall = torch.as_tensor(qpos, dtype=torch.float32, device="cuda")
    q = qall[:a.nworld]

    # Where the movers are, measured rather than bounded: a movers-only pass writes depth ONLY where
    # one was hit, so the box is the tight box of that mask and carries no bounding-sphere slack.
    cdyn = RP.make_ctx(mod, cls, m, S, a.nworld, hide=static, depth=True, skybox=False)
    boxes = []
    masks0 = None
    for lo in range(0, len(qpos), a.nworld):
        hi = min(lo + a.nworld, len(qpos))
        cdyn.wp.to_torch(cdyn.wd.qpos)[:hi - lo] = qall[lo:hi]
        RP.launch(cdyn)
        mk = cdyn.depth[:hi - lo] > 0
        if masks0 is None:
            masks0 = mk.clone()
        boxes.append(boxes_from_mask(mk).cpu().numpy())
    box_all = np.concatenate(boxes)
    per = ((box_all[:, 1] - box_all[:, 0]) * (box_all[:, 3] - box_all[:, 2]) / float(S * S))

    def chunk_unions(order):
        out = []
        for lo in range(0, len(order), a.nworld):
            b = box_all[order[lo:lo + a.nworld]]
            u = (b[:, 0].min(), b[:, 1].max(), b[:, 2].min(), b[:, 3].max())
            out.append((max(u[1] - u[0], u[3] - u[2]),
                        (u[1] - u[0]) * (u[3] - u[2]) / float(S * S)))
        return out

    plain = chunk_unions(np.arange(len(box_all)))
    key = np.lexsort((box_all[:, 2] + box_all[:, 3], box_all[:, 0] + box_all[:, 1]))
    srt = chunk_unions(key)
    uni = (int(box_all[:a.nworld, 0].min()), int(box_all[:a.nworld, 1].max()),
           int(box_all[:a.nworld, 2].min()), int(box_all[:a.nworld, 3].max()))
    facts = {"env": a.env, "n_movers": int(len(mv)), "batch": len(box_all),
             "mover_pixel_frac": float(masks0.float().mean()),
             "box_frac_mean": float(per.mean()), "box_frac_max": float(per.max()),
             "union_box_first_chunk": uni,
             "chunk_union_frac_unsorted": [float(np.mean([x[1] for x in plain])),
                                           float(np.max([x[1] for x in plain]))],
             "chunk_union_frac_sorted": [float(np.mean([x[1] for x in srt])),
                                         float(np.max([x[1] for x in srt]))],
             "chunk_union_side_sorted_p50": float(np.median([x[0] for x in srt]))}
    print("facts " + json.dumps(facts), flush=True)
    del cdyn
    mod._CTX_CACHE.clear()
    torch.cuda.empty_cache()

    # the full frame, and the ray table every windowed pass gathers from
    cfull = RP.make_ctx(mod, cls, m, S, a.nworld, depth=False)
    cfull.wp.to_torch(cfull.wd.qpos)[:] = q
    RP.launch(cfull)
    full_rgb = cfull.bgra[:, ..., [2, 1, 0]].clone()
    ray = cfull.ray0.clone()
    t_full = RP.timed(lambda: cfull.wp.capture_launch(cfull.graph), a.reps)
    rows = {"full": {"ms": t_full * 1e3, "fps": a.nworld / t_full, "rays": S * S}}
    print(f"full {t_full*1e3:8.3f} ms  {a.nworld/t_full:9,.0f} f/s", flush=True)

    # The union box as a rectangle, rounded up to a multiple of 8. A power-of-two ladder is the wrong
    # quantiser here: this family's union box is 177-181 px on a 256 px frame, and the next power of
    # two is the whole frame. The side is the one a SORTED batch would pay -- the median over its
    # chunks -- because sorting is free and the unsorted union is not what a pipeline would use.
    need = int(np.median([x[0] for x in srt]))
    side = min(S, max(8, int((need + 7) // 8 * 8)))
    x0 = min(max((uni[0] + uni[1]) // 2 - side // 2, 0), S - side)
    y0 = min(max((uni[2] + uni[3]) // 2 - side // 2, 0), S - side)
    sel_box = ((torch.arange(side, device="cuda") + y0).view(-1, 1) * S
               + (torch.arange(side, device="cuda") + x0).view(1, -1)).reshape(-1)

    def timed_rays(res_side: int, sel: torch.Tensor, label: str):
        c = RP.make_ctx(mod, cls, m, res_side, a.nworld, depth=False, share=cfull)
        c.wp.to_torch(c.rc.ray).copy_(ray.index_select(0, sel))
        c.wp.capture_launch(c.graph_draw)
        c.wp.synchronize()
        t = RP.timed(lambda: c.wp.capture_launch(c.graph_draw), a.reps)
        rgb = c.bgra[..., [2, 1, 0]].clone()
        rows[label] = {"ms": t * 1e3, "fps": a.nworld / t, "rays": int(res_side) ** 2,
                       "ray_frac": res_side ** 2 / float(S * S),
                       "speedup_vs_full": t_full / t}
        print(f"{label:12s} {t*1e3:8.3f} ms  {a.nworld/t:9,.0f} f/s  rays "
              f"{res_side**2/float(S*S):.3f}  x{t_full/t:5.2f}", flush=True)
        return c, rgb

    cbox, box_rgb = timed_rays(side, sel_box, f"box{side}")

    # the complement: an arbitrary, non-rectangular ray set, padded to a square count with repeats of
    # its own first ray (a repeat costs what the ray costs, so the timing is if anything pessimistic)
    inside = torch.zeros(S * S, dtype=torch.bool, device="cuda")
    inside[sel_box] = True
    out_idx = torch.nonzero(~inside).squeeze(1)
    if len(out_idx) == 0:
        raise SystemExit("the movers' box IS the frame: there is no far field to make cheap")
    oside = int(np.ceil(np.sqrt(len(out_idx))))
    pad = torch.cat([out_idx, out_idx[:1].expand(oside * oside - len(out_idx))])
    couter, _ = timed_rays(oside, pad, f"outer{oside}")

    # the outer field at every second pixel in each axis
    half = torch.zeros(S * S, dtype=torch.bool, device="cuda")
    hidx = ((torch.arange(0, S, 2, device="cuda").view(-1, 1) * S)
            + torch.arange(0, S, 2, device="cuda").view(1, -1)).reshape(-1)
    half[hidx] = True
    oh_idx = torch.nonzero(half & ~inside).squeeze(1)
    ohside = int(np.ceil(np.sqrt(len(oh_idx))))
    pad2 = torch.cat([oh_idx, oh_idx[:1].expand(ohside * ohside - len(oh_idx))])
    cohalf, _ = timed_rays(ohside, pad2, f"outerhalf{ohside}")

    # The frame at every second pixel in each axis -- the whole frame, robot included -- and the union
    # of the movers' own MASKS at full rate. Together they are the finest two-rate split the shared
    # ray table allows: the ray table is per LAUNCH, so the fovea can be an arbitrary pixel SET but
    # it has to be the same set for all 64 worlds, and the union of 64 robot silhouettes is what that
    # costs. This is the configuration that keeps the robot exact and coarsens everything else.
    hside = S // 2
    chalf, _ = timed_rays(hside, hidx, f"half{hside}")
    munion = torch.zeros(S * S, dtype=torch.bool, device="cuda")
    munion[torch.nonzero(masks0.reshape(a.nworld, -1).any(0)).squeeze(1)] = True
    m_idx = torch.nonzero(munion).squeeze(1)
    mside = int(np.ceil(np.sqrt(len(m_idx))))
    padm = torch.cat([m_idx, m_idx[:1].expand(mside * mside - len(m_idx))])
    cmask, _ = timed_rays(mside, padm, f"maskunion{mside}")
    facts["mask_union_frac"] = float(len(m_idx) / (S * S))
    rows["two_rate_mask_plus_half"] = {
        "ms": rows[f"maskunion{mside}"]["ms"] + rows[f"half{hside}"]["ms"],
        "speedup_vs_full": t_full * 1e3 / (rows[f"maskunion{mside}"]["ms"]
                                           + rows[f"half{hside}"]["ms"])}
    rows["box_plus_outer"] = {"ms": rows[f"box{side}"]["ms"] + rows[f"outer{oside}"]["ms"],
                              "speedup_vs_full": t_full * 1e3 / (rows[f"box{side}"]["ms"]
                                                                 + rows[f"outer{oside}"]["ms"])}
    rows["two_rate_box_plus_outerhalf"] = {
        "ms": rows[f"box{side}"]["ms"] + rows[f"outerhalf{ohside}"]["ms"],
        "speedup_vs_full": t_full * 1e3 / (rows[f"box{side}"]["ms"]
                                           + rows[f"outerhalf{ohside}"]["ms"])}
    rows["ceiling_if_outer_were_free"] = {"ms": rows[f"box{side}"]["ms"],
                                          "speedup_vs_full": t_full * 1e3
                                          / rows[f"box{side}"]["ms"]}

    # the two-rate frame, scored: box at full rate, outer nearest-upsampled from its half grid
    out = torch.zeros((a.nworld, S, S, 3), dtype=torch.uint8, device="cuda")
    oh_rgb = cohalf.bgra[..., [2, 1, 0]].reshape(a.nworld, -1, 3)
    flat = out.reshape(a.nworld, S * S, 3)
    # every pixel takes the sample at (row & ~1, col & ~1), which is what a nearest upsample is
    rr = (torch.arange(S, device="cuda") // 2)
    src_of = (rr.view(-1, 1) * (S // 2) + rr.view(1, -1)).reshape(-1)
    lut = torch.full((S * S,), -1, dtype=torch.long, device="cuda")
    lut[oh_idx] = torch.arange(len(oh_idx), device="cuda")
    coarse_pos = ((torch.arange(0, S, 2, device="cuda").view(-1, 1) * S)
                  + torch.arange(0, S, 2, device="cuda").view(1, -1)).reshape(-1)
    coarse_src = lut[coarse_pos].clamp(min=0)
    flat[:] = oh_rgb.index_select(1, coarse_src.index_select(0, src_of))
    flat[:, sel_box] = cbox.bgra[..., [2, 1, 0]].reshape(a.nworld, -1, 3)
    two = out

    # the mask variant's frame: everything nearest-upsampled from the half grid, movers exact
    m2 = torch.zeros((a.nworld, S, S, 3), dtype=torch.uint8, device="cuda").reshape(a.nworld, -1, 3)
    hrgb = chalf.bgra[..., [2, 1, 0]].reshape(a.nworld, -1, 3)
    m2[:] = hrgb.index_select(1, src_of)
    m2[:, m_idx] = cmask.bgra[..., [2, 1, 0]].reshape(a.nworld, -1, 3)[:, :len(m_idx)]
    m2 = m2.reshape(a.nworld, S, S, 3)

    o = two.cpu().numpy()
    f = full_rgb.cpu().numpy()
    om = m2.cpu().numpy()
    res = {"facts": facts, "rows": rows, "two_rate_vs_full": RP.metrics(o, f),
           "mask_variant_vs_full": RP.metrics(om, f)}
    ref = cls.reference_env(S)
    official = cls.reference_render(ref, st[:a.nworld])
    res["two_rate_vs_official"] = RP.metrics(o, official) | RP.interior_metrics(o, official)
    res["mask_variant_vs_official"] = RP.metrics(om, official) | RP.interior_metrics(om, official)
    res["full_vs_official"] = RP.metrics(f, official) | RP.interior_metrics(f, official)
    res["declared"] = dict(cls.visual_fidelity)
    print("JSON " + json.dumps(res), flush=True)
    with open(os.path.join(OUT, f"fovea_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    from PIL import Image
    amp = lambda d: np.stack([np.clip(d.max(-1) * 8, 0, 255).astype(np.uint8)] * 3, -1)
    Image.fromarray(np.concatenate(
        [official[0], f[0], o[0], amp(np.abs(o[0].astype(np.int32) - f[0].astype(np.int32)))], 1)
    ).save(os.path.join(OUT, f"fovea_{tag}.png"))


if __name__ == "__main__":
    main()
