"""Localise the sub-batch bug the closure sweep found, rather than guessing at it.

`tests/gpu/test_reproducibility.py::test_a_shared_style_broadcasts_to_any_batch` renders rows 0..b-1
of an 8-row stack for b in (1, 3, 8) and requires each to equal those rows of the full render. It
fails on both cells. `check_batch_size_invariance` passes on the same cells -- but it uses sizes
(1, 7, 64), which through `nw = 1 << (n-1).bit_length()` map to nworld 1, 8 and 64, while b=3 maps to
**nworld 4**, a width nothing else exercises. So the first hypothesis is that the frame depends on
nworld, and the second is that it depends on how many worlds were WRITTEN.

This separates them, and separates both from the composite:

* render the same 3 states at nworld 4 (natural) and at nworld 8 (padded to 8 rows, first 3 read);
* compare the raw MJWarp RGB, the segmentation, and the composited output independently, so a
  difference in the mask is not reported as a difference in shading;
* re-render each configuration twice, to separate "depends on nworld" from "not deterministic";
* and try the empty batch, which `test_render_is_pure_and_set_state_is_idempotent` crashes on with
  `zero-size array to reduction`.
"""
from __future__ import annotations

import argparse, json, os, socket, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=["OGBCube", "OGBScene"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    import torch
    import stable_envs as se
    from stable_envs.envs import ogb_manip as M

    rec = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "cells": {}}

    for cname in a.cells:
        env = se.make(cname, batch=8, device="cuda")
        st = env.sample_states(8, np.random.default_rng(0))
        cell = {}

        full = env.render(st).cpu().numpy()
        cell["nw_for_n"] = {str(n): int(1 if n <= 1 else min(M.MJW_CHUNK, 1 << (n - 1).bit_length()))
                            for n in (1, 2, 3, 5, 7, 8)}

        # 1. the failing comparison itself, per b
        rows = {}
        for b in (1, 2, 3, 4, 5, 7, 8):
            got = env.render(st[:b]).cpu().numpy()
            d = np.abs(got.astype(np.int16) - full[:b].astype(np.int16))
            rows[str(b)] = {"nworld": cell["nw_for_n"].get(str(b)),
                            "identical": bool(np.array_equal(got, full[:b])),
                            "max_abs": int(d.max()) if d.size else None,
                            "frac_px_differing": float((d.max(-1) > 0).mean()) if d.size else None}
            print(f"{cname} b={b} nworld={rows[str(b)]['nworld']} identical={rows[str(b)]['identical']} "
                  f"max {rows[str(b)]['max_abs']} frac {rows[str(b)]['frac_px_differing']}", flush=True)
        cell["by_batch"] = rows

        # 2. determinism at a FIXED batch: is the difference nworld, or non-determinism?
        cell["repeat_same_b"] = {}
        for b in (3, 8):
            r1 = env.render(st[:b]).cpu().numpy()
            r2 = env.render(st[:b]).cpu().numpy()
            cell["repeat_same_b"][str(b)] = bool(np.array_equal(r1, r2))
        print(f"{cname} repeat-at-same-b identical: {cell['repeat_same_b']}", flush=True)

        # 3. padding 3 states to 8 rows: same nworld as the full render, only 3 rows meaningful
        pad = np.concatenate([st[:3], np.repeat(st[3:4], 5, axis=0)])
        got_pad = env.render(pad).cpu().numpy()[:3]
        cell["padded_to_8_matches_full"] = bool(np.array_equal(got_pad, full[:3]))
        d = np.abs(got_pad.astype(np.int16) - full[:3].astype(np.int16))
        cell["padded_max_abs"] = int(d.max())
        print(f"{cname} 3-rows-padded-to-8 matches full[:3]: {cell['padded_to_8_matches_full']} "
              f"(max {cell['padded_max_abs']})", flush=True)

        # 4. is it the mask or the shading? compare the raw MJWarp output at both widths
        b3 = M._bundle(cname[3:])
        res = int(env.size)
        raw = {}
        for nw in (4, 8):
            c = M._context(cname[3:], res, nw)
            wp = c.wp
            q = torch.as_tensor(np.ascontiguousarray(st[:3, :b3.nq]), device="cuda",
                                dtype=torch.float32)
            wp.to_torch(c.wd.qpos)[:3] = q
            if b3.nmocap:
                mp = torch.as_tensor(np.ascontiguousarray(
                    st[:3, b3.nq:b3.nq + 3 * b3.nmocap]).reshape(3, b3.nmocap, 3),
                    device="cuda", dtype=torch.float32)
                wp.to_torch(c.wd.mocap_pos)[:3] = mp
                idq = torch.zeros((3, b3.nmocap, 4), dtype=torch.float32, device="cuda")
                idq[..., 0] = 1.0
                wp.to_torch(c.wd.mocap_quat)[:3] = idq
            torch.cuda.synchronize(); wp.capture_launch(c.graph); wp.synchronize()
            raw[nw] = (c.bgra[:3][..., [2, 1, 0]].clone().cpu().numpy(),
                       c.seg[:3].clone().cpu().numpy())
        drgb = np.abs(raw[4][0].astype(np.int16) - raw[8][0].astype(np.int16))
        dseg = raw[4][1] != raw[8][1]
        cell["raw_rgb_identical_across_nworld"] = bool(drgb.max() == 0)
        cell["raw_rgb_max_abs"] = int(drgb.max())
        cell["raw_rgb_frac_px_differing"] = float((drgb.max(-1) > 0).mean())
        cell["seg_identical_across_nworld"] = bool(not dseg.any())
        cell["seg_frac_differing"] = float(dseg.mean())
        print(f"{cname} RAW across nworld 4 vs 8: rgb identical {cell['raw_rgb_identical_across_nworld']} "
              f"(max {cell['raw_rgb_max_abs']}, {cell['raw_rgb_frac_px_differing']*100:.3f}% px) | "
              f"seg identical {cell['seg_identical_across_nworld']} "
              f"({cell['seg_frac_differing']*100:.3f}% px)", flush=True)

        # 5. the empty batch that crashes the purity test
        try:
            e = env.render(st[:0]).cpu().numpy()
            cell["empty_batch"] = {"ok": True, "shape": list(e.shape)}
        except Exception as ex:
            cell["empty_batch"] = {"ok": False, "error": f"{type(ex).__name__}: {ex}"[:200]}
        print(f"{cname} empty batch: {cell['empty_batch']}", flush=True)

        rec["cells"][cname] = cell
        (out / "subbatch_diag.json").write_text(json.dumps(rec, indent=1, sort_keys=True))

    (out / "subbatch_diag.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
