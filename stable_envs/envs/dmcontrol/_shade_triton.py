"""One Triton kernel for ``_shade(...) * tex * 255`` of DM Control frames, equal to the Torch ops bit for bit.

Each Torch op rounds to float32, so the kernel does the same ops in the same order, unfused, with ``div_rn`` and
``sqrt_rn``. A light configuration it does not cover returns None and the caller uses the Torch ops.
"""

from __future__ import annotations

import numpy as np
import torch

from ...render.raycast3d import HAVE_TRITON
from ._model import LIGHT_DIRECTIONAL, SPECULAR_SCALE

#: False draws with the Torch ops; tests compare both.
ENABLED = True
#: Most lights one launch shades.
MAX_LIGHTS = 4
#: Columns of a light row: direction or position (3), spot axis (3), attenuation (3), ambient, diffuse,
#: specular, spot cosine and spot exponent.
COLUMNS = 14
#: Spot exponents whose ``pow`` ATen replaces with other ops, each with a kernel code; other exponents use ``powf``.
_SPOT_POW = {2.0: 1, 3.0: 2, -2.0: 3}
#: Spot exponents ATen draws with ops this kernel does not repeat (``rsqrt``, ``sqrt``, ``reciprocal``, fills).
_UNSUPPORTED = {0.0, 1.0, 0.5, -0.5, -1.0}

if HAVE_TRITON:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _spot_pow(x, e, MODE: tl.constexpr):
        if MODE == 1:
            r = x * x
        elif MODE == 2:
            r = x * x * x
        elif MODE == 3:
            r = tl.math.div_rn(1.0, x * x)
        else:
            r = libdevice.pow(x, e)
        return r

    @triton.jit
    def _light(
        table,
        i: tl.constexpr,
        px,
        py,
        pz,
        DIRECTIONAL: tl.constexpr,
        SPOT: tl.constexpr,
        POW: tl.constexpr,
        COLS: tl.constexpr,
    ):
        """Direction toward light ``i`` and its attenuation, as ``_light_geometry``."""
        row = table + i * COLS
        if DIRECTIONAL:
            lx = tl.load(row + 0) + px * 0.0
            ly = tl.load(row + 1) + py * 0.0
            lz = tl.load(row + 2) + pz * 0.0
            att = px * 0.0 + 1.0
        else:
            vx = tl.load(row + 0) - px
            vy = tl.load(row + 1) - py
            vz = tl.load(row + 2) - pz
            dist = tl.sqrt_rn(tl.maximum(vx * vx + vy * vy + vz * vz, 1e-24))
            lx = tl.math.div_rn(vx, dist)
            ly = tl.math.div_rn(vy, dist)
            lz = tl.math.div_rn(vz, dist)
            k0 = tl.load(row + 6)
            k1 = tl.load(row + 7)
            k2 = tl.load(row + 8)
            att = tl.math.div_rn(1.0, tl.maximum(k0 + k1 * dist + k2 * dist * dist, 1e-12)) * 1.0
            if SPOT:
                se = tl.load(row + 3) * -lx + tl.load(row + 4) * -ly + tl.load(row + 5) * -lz
                inside = se >= tl.load(row + 12)
                att = tl.where(inside, att * _spot_pow(tl.maximum(se, 0.0), tl.load(row + 13), POW), 0.0)
        return lx, ly, lz, att

    @triton.jit
    def _shade_kernel(
        p_ptr,
        psb,
        psm,
        n_ptr,
        nsb,
        nsm,
        v_ptr,
        vsb,
        vsm,
        idx_ptr,
        isb,
        ism,
        mat_ptr,
        matsb,
        spec_ptr,
        shin_ptr,
        tex_ptr,
        unlit_ptr,
        usb,
        usm,
        table,
        out_ptr,
        total,
        M,
        NL: tl.constexpr,
        DIRS: tl.constexpr,
        SPOTS: tl.constexpr,
        SHADOWS: tl.constexpr,
        POWS: tl.constexpr,
        HAS_UNLIT: tl.constexpr,
        SCALE: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        b = offs // M
        m = offs % M
        po = b * psb + m * psm
        px = tl.load(p_ptr + po + 0, mask=mask, other=0.0)
        py = tl.load(p_ptr + po + 1, mask=mask, other=0.0)
        pz = tl.load(p_ptr + po + 2, mask=mask, other=0.0)
        no = b * nsb + m * nsm
        nx = tl.load(n_ptr + no + 0, mask=mask, other=0.0)
        ny = tl.load(n_ptr + no + 1, mask=mask, other=0.0)
        nz = tl.load(n_ptr + no + 2, mask=mask, other=0.0)
        vo = b * vsb + m * vsm
        vx = tl.load(v_ptr + vo + 0, mask=mask, other=0.0)
        vy = tl.load(v_ptr + vo + 1, mask=mask, other=0.0)
        vz = tl.load(v_ptr + vo + 2, mask=mask, other=0.0)
        k = tl.load(idx_ptr + b * isb + m * ism, mask=mask, other=0)
        mo = b * matsb + k * 3
        m0 = tl.load(mat_ptr + mo + 0, mask=mask, other=0.0)
        m1 = tl.load(mat_ptr + mo + 1, mask=mask, other=0.0)
        m2 = tl.load(mat_ptr + mo + 2, mask=mask, other=0.0)
        spec = tl.load(spec_ptr + k, mask=mask, other=0.0)
        exponent = 128.0 * tl.load(shin_ptr + k, mask=mask, other=0.0)
        if HAS_UNLIT:
            lit_scale = (tl.load(unlit_ptr + b * usb + m * usm, mask=mask, other=0) == 0).to(tl.float32)
        o0 = px * 0.0
        o1 = px * 0.0
        o2 = px * 0.0
        highlight = px * 0.0
        for i in tl.static_range(NL):
            lx, ly, lz, att = _light(
                table, i, px, py, pz, (DIRS >> i) & 1, (SPOTS >> i) & 1, (POWS >> (2 * i)) & 3, COLS
            )
            lit = att
            if HAS_UNLIT:
                if (SHADOWS >> i) & 1:
                    lit = att * lit_scale
            ndl = tl.maximum(nx * lx + ny * ly + nz * lz, 0.0)
            hx = lx + vx
            hy = ly + vy
            hz = lz + vz
            hn = tl.maximum(tl.sqrt_rn(hx * hx + hy * hy + hz * hz), 1e-12)
            hx = tl.math.div_rn(hx, hn)
            hy = tl.math.div_rn(hy, hn)
            hz = tl.math.div_rn(hz, hn)
            ndh = tl.maximum(nx * hx + ny * hy + nz * hz, 0.0)
            row = table + i * COLS
            ambient = att * tl.load(row + 9)
            o0 = o0 + m0 * ambient
            o1 = o1 + m1 * ambient
            o2 = o2 + m2 * ambient
            diffuse = lit * ndl * tl.load(row + 10)
            o0 = o0 + m0 * diffuse
            o1 = o1 + m1 * diffuse
            o2 = o2 + m2 * diffuse
            highlight = highlight + lit * tl.load(row + 11) * libdevice.pow(ndh, exponent)
        white = highlight * spec * SCALE
        oo = offs * 3
        for c in tl.static_range(3):
            if c == 0:
                value = o0
            elif c == 1:
                value = o1
            else:
                value = o2
            value = tl.minimum(tl.maximum(value + white, 0.0), 1.0)
            value = value * tl.load(tex_ptr + k * 3 + c, mask=mask, other=0.0) * 255.0
            tl.store(out_ptr + oo + c, value, mask=mask)


def _table(L: dict):
    """The light rows and flags of ``L``, built once with the Torch ops ``_light_geometry`` uses; None if unsupported."""
    hit = L.get("_triton_table")
    if hit is not None:
        return hit or None
    from ._shading import _norm

    n = len(L["kind"])
    pow_codes = [0] * MAX_LIGHTS
    rows = []
    dirs = spots = shadows = 0
    ok = 0 < n <= MAX_LIGHTS
    for i in range(n if ok else 0):
        row = torch.zeros(COLUMNS, device=L["pos"].device, dtype=torch.float32)
        if L["kind"][i] == LIGHT_DIRECTIONAL:
            dirs |= 1 << i
            row[0:3] = -_norm(L["dir"][i].view(1, 3))[0]
        else:
            row[0:3] = L["pos"][i]
            row[6:9] = L["attenuation"][i]
            if L["cutoff"][i] < 180.0:
                exponent = float(L["exponent"][i])
                if exponent in _UNSUPPORTED:
                    ok = False
                spots |= 1 << i
                row[3:6] = _norm(L["dir"][i].view(1, 3))[0]
                row[12] = float(np.float32(np.cos(np.deg2rad(L["cutoff"][i]))))
                row[13] = exponent
                pow_codes[i] = _SPOT_POW.get(exponent, 0)
        row[9], row[10], row[11] = L["ambient"][i][0], L["diffuse"][i][0], L["specular"][i][0]
        if L["castshadow"][i]:
            shadows |= 1 << i
        rows.append(row)
    table = (torch.stack(rows).contiguous(), n, dirs, spots, shadows, tuple(pow_codes)) if ok else ()
    L["_triton_table"] = table
    return table or None


def _vector_strides(t: torch.Tensor, B: int, M: int):
    """``(batch, ray)`` strides of a float32 ``(B or 1, M or 1, 3)`` tensor with a packed last axis, else None."""
    if t.dim() != 3 or t.shape[-1] != 3 or t.stride(-1) != 1 or t.dtype != torch.float32:
        return None
    if t.shape[0] not in (1, B) or t.shape[1] not in (1, M):
        return None
    return (t.stride(0) if t.shape[0] > 1 else 0), (t.stride(1) if t.shape[1] > 1 else 0)


def shade_textured(p, n, v, L, mat_rgb, idx, spec, shin, tex, unlit=None):
    """``_shade(p, n, v, L, mat_rgb[b, idx], spec[idx], shin[idx], unlit) * tex[idx] * 255`` in one launch.

    Returns None when Triton, the device, the layouts or the lights aren't covered.
    """
    if not ENABLED or not HAVE_TRITON or not p.is_cuda:
        return None
    table = _table(L)
    if table is None or p.dim() != 3 or idx.dim() != 2 or idx.dtype != torch.int64:
        return None
    B, M = (max(t.shape[0] for t in (p, n, v, idx)), max(t.shape[1] for t in (p, n, v, idx)))
    strides = [_vector_strides(t, B, M) for t in (p, n, v)]
    if any(s is None for s in strides) or idx.shape[0] not in (1, B) or idx.shape[1] not in (1, M):
        return None
    if mat_rgb.dim() != 3 or not mat_rgb.is_contiguous() or mat_rgb.dtype != torch.float32:
        return None
    if not all(t.is_contiguous() and t.dtype == torch.float32 for t in (spec, shin, tex)):
        return None
    rows, nl, dirs, spots, shadows, pows = table
    idx = idx.expand(B, M)
    if mat_rgb.shape[0] not in (1, B):
        return None
    if unlit is not None:
        unlit = unlit.expand(B, M)
        if unlit.dtype != torch.bool:
            return None
        unlit_strides = unlit.stride()
        unlit = unlit.view(torch.uint8)
    out = torch.empty((B, M, 3), device=p.device, dtype=torch.float32)
    total = B * M
    if total == 0:
        return out
    block = 256
    with torch.cuda.device(p.device):
        _shade_kernel[(triton.cdiv(total, block),)](
            p,
            *strides[0],
            n,
            *strides[1],
            v,
            *strides[2],
            idx,
            idx.stride(0),
            idx.stride(1),
            mat_rgb,
            mat_rgb.stride(0) if mat_rgb.shape[0] > 1 else 0,
            spec,
            shin,
            tex,
            unlit if unlit is not None else out,
            *(unlit_strides if unlit is not None else (0, 0)),
            rows,
            out,
            total,
            M,
            NL=nl,
            DIRS=dirs,
            SPOTS=spots,
            SHADOWS=shadows,
            POWS=sum(code << (2 * i) for i, code in enumerate(pows)),
            HAS_UNLIT=unlit is not None,
            SCALE=SPECULAR_SCALE,
            COLS=COLUMNS,
            BLOCK=block,
            enable_fp_fusion=False,
        )
    return out
