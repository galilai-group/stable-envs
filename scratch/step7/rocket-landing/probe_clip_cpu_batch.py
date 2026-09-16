"""CLOSURE-1: localise the x86-64 CPU batch-size dependence in ``nvdiff.Scene.clip``, and test a fix.

The closure sweep measured (``examples/results/step7/closure/nvdiff_cpu_batch/``, job 6413606,
node1939) that on x86-64 CPU ``Scene.clip``'s POSITIONS differ by one float32 ulp at batch widths 1
and 2, while ``mvp_rows`` -- W6's float64 elementwise product -- is bit-identical at every width on
both architectures, as are ``clip``'s normals and world positions. So the residual is downstream of
the mvp, in the stage that turns per-geom transforms into per-vertex clip coordinates:

    pos = torch.sparse.mm(self.S4, mvp.permute(1, 3, 0, 2).reshape(4 * G, 4 * B))

``S4`` is ``(V, 4G)`` with exactly FOUR nonzeros per row -- vertex ``v``'s homogeneous local position
in geom ``g(v)``'s four columns -- so each output element is a sum of four products, and the order
that sum is taken in is the sparse kernel's choice. The kernel is chosen by the dense operand's column
count, which is ``4 * B``. That is the same class of defect W6 fixed in the mvp, one product later.

This script does three things and writes one JSON:

  1. **Localise.** Per width, compare against the full-batch prefix: ``mvp_rows`` (expected identical),
     the ``spmm`` stage fed a mvp that is bit-identical by construction (expected to differ -- this is
     the claim), and the three ``clip`` outputs.
  2. **Test the candidate fix** -- ``_clip_pos_explicit`` below, the four products summed in a written
     order with a float64 accumulator and one cast at the end. Same value for a row whatever the batch,
     on any backend, because there is no reduction kernel to choose.
  3. **Size it.** V, G and the peak temporary the explicit form needs per cell, because the sparse
     formulation exists to avoid a ``(B, V, 4, 4)`` materialisation and a fix that reintroduces one
     would trade a CPU-only correctness bug for a GPU memory regression.

Run anywhere; x86-64 is where it matters. ``--cells`` also builds the six locomotion Scenes if mujoco
is importable, to report their V and G.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np
import torch


def _box_soup(half=(0.5, 0.25, 1.0)):
    """A triangle soup with enough vertices to exercise the gather; geometry is irrelevant here."""
    hx, hy, hz = half
    v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                 np.float64)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v, n, f


def _clip_pos_explicit(S, mvp):
    """``pos`` as FOUR elementwise products summed in a written order, accumulated in float64.

    ``S4`` row ``v`` is ``(v4[v, 0..3])`` at columns ``4*g(v) + 0..3``, so

        pos[v, c] = sum_k v4[v, k] * D[4*g(v) + k, c]        D = mvp laid out (4G, 4B)

    and this writes that sum out instead of asking a sparse kernel for it. Four index_selects and
    three adds, in a fixed order, with no reduction whose shape depends on ``B`` -- so element
    ``(v, c)`` is bitwise the same whether one world or four thousand share the call.
    """
    G, B = S.G, mvp.shape[0]
    D = mvp.to(torch.float64).permute(1, 3, 0, 2).reshape(4 * G, 4 * B)
    base = S.vgeom * 4
    v4 = torch.cat([S.vert, torch.ones(S.V, 1, device=S.vert.device, dtype=S.vert.dtype)], -1
                   ).to(torch.float64)
    acc = v4[:, 0:1] * D.index_select(0, base + 0)
    acc = acc + v4[:, 1:2] * D.index_select(0, base + 1)
    acc = acc + v4[:, 2:3] * D.index_select(0, base + 2)
    acc = acc + v4[:, 3:4] * D.index_select(0, base + 3)
    return acc.to(mvp.dtype)


def _clip_pos_explicit32(S, mvp):
    """The same four products and three adds, accumulated in the INPUT dtype rather than float64.

    Worth measuring separately because it is the variant that could fix the CPU width dependence at
    ZERO cost elsewhere: a CSR spmm walks each row's nonzeros in ascending column order and
    accumulates, which for ``S4``'s four-per-row is exactly ``((h0*d0 + h1*d1) + h2*d2) + h3*d3``. If
    the shipped kernel already computes that, writing it out changes nothing anywhere and only removes
    the kernel's freedom to choose a different order for a narrow batch. The float64 variant is the
    fallback: it is the correctly rounded value, which is a BETTER number but not the same one, and a
    different number on cuda means the six locomotion cells move again.
    """
    G, B = S.G, mvp.shape[0]
    D = mvp.permute(1, 3, 0, 2).reshape(4 * G, 4 * B)
    base = S.vgeom * 4
    v4 = torch.cat([S.vert, torch.ones(S.V, 1, device=S.vert.device, dtype=S.vert.dtype)], -1)
    acc = v4[:, 0:1] * D.index_select(0, base + 0)
    acc = acc + v4[:, 1:2] * D.index_select(0, base + 1)
    acc = acc + v4[:, 2:3] * D.index_select(0, base + 2)
    acc = acc + v4[:, 3:4] * D.index_select(0, base + 3)
    return acc


def _contract4(rows, base, coef, dtype):
    """``sum_k coef[:, k] * rows(base + k)`` -- four gathers, three adds, written order, in ``dtype``.

    In-place so that only the accumulator and one gathered block are live at a time; the spmm it
    replaces allocates one output of the same shape, so the peak goes from 1x to 2x that, not to the
    ``(B, V, 4, 4)`` the sparse formulation exists to avoid. ``mul_``/``add_`` are plain elementwise
    kernels -- deliberately not ``addcmul_``, which is free to contract into an FMA and would put the
    rounding back under the backend's control.
    """
    acc = rows(base + 0).to(dtype).mul_(coef[:, 0:1])
    for k in (1, 2, 3):
        acc.add_(rows(base + k).to(dtype).mul_(coef[:, k:k + 1]))
    return acc


def _explicit_inplace32(S, mvp):
    """The shipped-candidate form: ``_contract4`` in the input dtype, matching ``_clip_pos_explicit32``."""
    G, B = S.G, mvp.shape[0]
    D = mvp.permute(1, 3, 0, 2).reshape(4 * G, 4 * B)
    v4 = torch.cat([S.vert, torch.ones(S.V, 1, device=S.vert.device, dtype=S.vert.dtype)], -1)
    return _contract4(lambda idx: D.index_select(0, idx), S.vgeom * 4, v4, mvp.dtype)


def _explicit3(S, coef, xmat):
    """The same, three-wide, for the normal / world-offset stages (``S3`` and ``S3p``)."""
    G, B = S.G, xmat.shape[0]
    D = xmat.permute(1, 3, 0, 2).reshape(3 * G, 3 * B)
    base = S.vgeom * 3
    acc = D.index_select(0, base + 0).mul_(coef[:, 0:1])
    for k in (1, 2):
        acc.add_(D.index_select(0, base + k).mul_(coef[:, k:k + 1]))
    return acc


def _cmp(a, b, label):
    d = (a.to(torch.float64) - b.to(torch.float64)).abs()
    mx = float(a.abs().max()) if a.numel() else 0.0
    return {"identical": bool(torch.equal(a, b)), "max_abs_delta": float(d.max()) if d.numel() else 0.0,
            "n_differing": int((d > 0).sum()), "n_elements": int(a.numel()), "max_abs_value": mx,
            "float32_ulp_at_that_scale": float(np.spacing(np.float32(mx))) if mx else 0.0,
            "what": label}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 7, 8, 31, 32, 63])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cells", action="store_true", help="also report V/G for the six locomotion cells")
    a = ap.parse_args()
    import stable_envs.render.nvdiff as ND

    R = {"host": platform.node(), "machine": platform.machine(), "python": sys.version.split()[0],
         "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
         "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
         "slurm_job": os.environ.get("SLURM_JOB_ID"),
         "git_sha": (open(".git_sha").read().strip() if os.path.exists(".git_sha") else None),
         "widths": {}}

    v, n, f = _box_soup()
    S = ND.Scene.from_triangles(v, n, f, np.zeros(len(v), np.int64), rgba=[[1, 1, 1, 1]], device="cpu")
    rng = np.random.default_rng(5)
    B = a.batch
    q = rng.normal(size=(B, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    try:
        from scipy.spatial.transform import Rotation as Rt
        mats = Rt.from_quat(q).as_matrix()
    except Exception:                                                               # pragma: no cover
        mats = np.broadcast_to(np.eye(3), (B, 3, 3)).copy()
    xmat = torch.as_tensor(mats, dtype=torch.float32)[:, None]
    xpos = torch.as_tensor(rng.normal(size=(B, 1, 3)) * 300.0, dtype=torch.float32)
    cam = ND.Camera(rng.normal(size=(B, 3)) * 300.0,
                    np.broadcast_to(np.eye(3), (B, 3, 3)).copy(), 90.0, 0.1, 255.0)
    view, proj = cam.matrices()
    view_t = torch.as_tensor(view, dtype=torch.float32)
    proj_t = torch.as_tensor(proj, dtype=torch.float32)
    R["shapes"] = {"V": int(S.V), "G": int(S.G), "batch": B}

    def build_mvp(b):
        M = torch.zeros(b, S.G, 4, 4, dtype=torch.float32)
        M[:, :, :3, :3] = xmat[:b]
        M[:, :, :3, 3] = xpos[:b]
        M[:, :, 3, 3] = 1.0
        return ND.mvp_rows(proj_t, view_t[:b], M)

    mvp_full = build_mvp(B)
    pos_full, nrm_full, wp_full = S.clip(xpos, xmat, view_t, proj_t)
    spmm_full = torch.sparse.mm(S.S4, mvp_full.permute(1, 3, 0, 2).reshape(4 * S.G, 4 * B))
    expl_full = _clip_pos_explicit(S, mvp_full)
    expl32_full = _clip_pos_explicit32(S, mvp_full)
    inpl32_full = _explicit_inplace32(S, mvp_full)
    xm = xmat.permute(0, 1, 3, 2) if False else xmat
    nrm_sp_full = torch.sparse.mm(S.S3, xmat.permute(1, 3, 0, 2).reshape(3 * S.G, 3 * B))
    wp_sp_full = torch.sparse.mm(S.S3p, xmat.permute(1, 3, 0, 2).reshape(3 * S.G, 3 * B))
    nrm_ex_full = _explicit3(S, S.norm, xmat)
    wp_ex_full = _explicit3(S, S.vert, xmat)

    for b in a.widths:
        if b > B:
            continue
        w = {}
        mvp_b = build_mvp(b)
        w["mvp_rows"] = _cmp(mvp_b, mvp_full[:b], "W6's float64 elementwise proj@view@M")
        # the spmm stage ALONE, fed a mvp that is bit-identical by construction: isolates the kernel
        sp_b = torch.sparse.mm(S.S4, mvp_full[:b].permute(1, 3, 0, 2).reshape(4 * S.G, 4 * b))
        w["spmm_stage_on_identical_input"] = _cmp(
            sp_b.reshape(S.V, b, 4), spmm_full.reshape(S.V, B, 4)[:, :b],
            "torch.sparse.mm(S4, mvp) at width 4*b vs the full-batch prefix")
        w["explicit64_stage_on_identical_input"] = _cmp(
            _clip_pos_explicit(S, mvp_full[:b]).reshape(S.V, b, 4),
            expl_full.reshape(S.V, B, 4)[:, :b], "float64 accumulator, same input")
        w["explicit32_stage_on_identical_input"] = _cmp(
            _clip_pos_explicit32(S, mvp_full[:b]).reshape(S.V, b, 4),
            expl32_full.reshape(S.V, B, 4)[:, :b], "float32 accumulator, same input")
        p, nr, wp = S.clip(xpos[:b], xmat[:b], view_t[:b], proj_t)
        w["clip_pos"] = _cmp(p, pos_full[:b], "Scene.clip positions (what the failing test asserts)")
        w["clip_nrm"] = _cmp(nr, nrm_full[:b], "Scene.clip normals")
        w["clip_worldpos"] = _cmp(wp, wp_full[:b], "Scene.clip world positions")
        R["widths"][str(b)] = w
        print(b, json.dumps({k: v_["identical"] for k, v_ in w.items()}), flush=True)

    # does the candidate fix agree with the shipped one at the FULL batch (the value, not the order)?
    R["explicit64_vs_spmm_full_batch"] = _cmp(expl_full, spmm_full,
                                              "float64 accumulator vs torch.sparse.mm, full batch")
    R["explicit32_vs_spmm_full_batch"] = _cmp(expl32_full, spmm_full,
                                              "float32 accumulator vs torch.sparse.mm, full batch")
    R["inplace32_vs_spmm_full_batch"] = _cmp(inpl32_full, spmm_full,
                                             "the SHIPPED-candidate in-place form vs torch.sparse.mm")
    R["inplace32_vs_outofplace32"] = _cmp(inpl32_full, expl32_full, "in-place vs out-of-place, same order")
    R["nrm_explicit_vs_spmm"] = _cmp(nrm_ex_full, nrm_sp_full, "S3 stage: explicit vs torch.sparse.mm")
    R["wp_explicit_vs_spmm"] = _cmp(wp_ex_full, wp_sp_full, "S3p stage: explicit vs torch.sparse.mm")
    R["sizing"] = {"note": "peak extra temporary of the explicit form, per call, in float64",
                   "bytes_per_temporary": int(S.V * 4 * B * 8),
                   "spmm_output_bytes_float32": int(S.V * 4 * B * 4)}

    if a.cells:
        cells = {}
        try:
            import stable_envs as se
            import mujoco                                                          # noqa: F401
            for name in ("MujocoHalfCheetah", "MujocoHopper", "MujocoWalker2d", "MujocoAnt",
                         "MujocoHumanoid", "MujocoSwimmer"):
                try:
                    cls = se.REGISTRY.get(name)
                    if cls is None:
                        cells[name] = "not registered"
                        continue
                    m = cls._model() if hasattr(cls, "_model") else None
                    cells[name] = "no CPU model accessor" if m is None else {
                        "V": int(ND.Scene(m, device="cpu").V), "G": int(ND.Scene(m, device="cpu").G)}
                except Exception as e:                                             # pragma: no cover
                    cells[name] = f"{type(e).__name__}: {e}"
        except Exception as e:                                                     # pragma: no cover
            cells = {"blocked": f"{type(e).__name__}: {e}"}
        R["locomotion_cells"] = cells

    for k in ("explicit64_vs_spmm_full_batch", "explicit32_vs_spmm_full_batch",
              "inplace32_vs_spmm_full_batch", "inplace32_vs_outofplace32",
              "nrm_explicit_vs_spmm", "wp_explicit_vs_spmm"):
        print(f"  {k:34s} identical={R[k]['identical']} n_differing={R[k]['n_differing']}"
              f"/{R[k]['n_elements']} max {R[k]['max_abs_delta']:.3e}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(R, fh, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
