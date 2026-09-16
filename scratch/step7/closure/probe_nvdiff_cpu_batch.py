"""Characterise the CPU batch-size dependence that `tests/unit/test_nvdiff_triangles.py::
test_from_triangles_clip_row_is_independent_of_batch_size` reports on the certification host.

The test is W6's own: it asserts that ``Scene.clip``'s row 0 is bit-identical whether it is drawn
alone or inside a batch of 64, on the CPU path. It PASSES on an Apple M5 and FAILS on Oscar's
x86-64 batch nodes. This probe does not fix anything and touches no library file; it separates the
two products the row goes through so the report can say WHICH one moves:

  A. ``ND.mvp_rows(proj, view, M)``   -- the batch-invariant model-view-projection W6 made float64
                                         and rounded once. If this is bit-identical at every batch
                                         width, W6's fix is intact and the CPU failure is downstream.
  B. ``Scene.clip(...)``              -- the whole thing, including the float32 ``mvp @ vertices``
                                         product, whose BLAS kernel a CPU library may select by
                                         batch width exactly as cuBLAS does on the card.

and records, per batch width, the max absolute delta and how many float32 ulps that is at the
coordinate magnitude involved. Thread count is recorded because a CPU BLAS may also choose its
kernel by the number of threads.

    python scratch/step7/closure/probe_nvdiff_cpu_batch.py --out DIR/nvdiff_cpu_batch.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--widths", default="1,2,7,8,31,32,63",
                    help="batch widths to sample. The closure sweep's original set skips 4-6, and "
                         "W6's CLOSURE-1 follow-up reports the normals moving at widths 4 and 8 "
                         "where this probe recorded width 8 identical -- two committed artifacts "
                         "disagreeing on one width, which a contiguous sweep settles")
    a = ap.parse_args()
    import torch
    from stable_envs.render import nvdiff as ND
    sys.path.insert(0, str(ROOT / "tests" / "unit"))
    from test_nvdiff_triangles import _box_soup                             # the test's own geometry

    from scipy.spatial.transform import Rotation as Rt

    rec = {"host": platform.node(), "machine": platform.machine(),
           "python": platform.python_version(), "torch": torch.__version__,
           "torch_threads": torch.get_num_threads(),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "git_sha": os.environ.get("STABLE_ENVS_GIT_COMMIT"),
           "widths": {}}

    v, n, f = _box_soup((0.5, 0.25, 1.0))
    S = ND.Scene.from_triangles(v, n, f, np.zeros(len(v), np.int64), rgba=[[1, 1, 1, 1]], device="cpu")
    rng = np.random.default_rng(5)
    B = 64
    q = rng.normal(size=(B, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    xmat = torch.as_tensor(Rt.from_quat(q).as_matrix(), dtype=torch.float32)[:, None]
    xpos = torch.as_tensor(rng.normal(size=(B, 1, 3)) * 300.0, dtype=torch.float32)
    eye = rng.normal(size=(B, 3)) * 300.0
    cam = ND.Camera(eye, np.broadcast_to(np.eye(3), (B, 3, 3)).copy(), 90.0, 0.1, 255.0)
    view, proj = cam.matrices()
    view_t = torch.as_tensor(view, dtype=torch.float32)
    proj_t = torch.as_tensor(proj, dtype=torch.float32)

    pos, nrm, wp = S.clip(xpos, xmat, view_t, proj_t)
    # mvp_rows takes proj (4,4), view (B,4,4) and M (B,G,4,4). A per-row identity M isolates the
    # proj @ view @ M product itself from anything the scene's own model matrices contribute.
    eyeM = torch.eye(4).expand(B, 1, 4, 4).contiguous()
    rec["shapes"] = {"view": list(view_t.shape), "proj": list(proj_t.shape)}
    full_mvp = ND.mvp_rows(proj_t, view_t, eyeM)

    for b in [int(x) for x in a.widths.split(",")]:
        row = {}
        p_b, n_b, w_b = S.clip(xpos[:b], xmat[:b], view_t[:b], proj_t)
        for name, got, want in (("clip_pos", p_b, pos[:b]), ("clip_nrm", n_b, nrm[:b]),
                                ("clip_worldpos", w_b, wp[:b])):
            d = (got.double() - want.double()).abs()
            mx = float(d.max())
            scale = float(want.abs().max())
            row[name] = {"identical": bool(torch.equal(got, want)), "max_abs_delta": mx,
                         "n_differing": int((got != want).sum()),
                         "n_elements": int(got.numel()),
                         "max_abs_value": scale,
                         "float32_ulp_at_that_scale": float(np.spacing(np.float32(scale)))
                         if scale else None}
        if full_mvp is not None:
            mvp_b = ND.mvp_rows(proj_t, view_t[:b], eyeM[:b])
            row["mvp_rows"] = {"identical": bool(torch.equal(mvp_b, full_mvp[:b])),
                               "max_abs_delta": float((mvp_b.double() - full_mvp[:b].double()).abs().max())}
        rec["widths"][str(b)] = row
        print(b, json.dumps(row, default=str)[:300], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
