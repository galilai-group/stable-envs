"""Undo ONLY the device-guard edits, so the parity failure can be attributed or cleared.

``git diff`` on these two files is not a usable A/B: the index holds a much older version, so the diff
carries a session's worth of unrelated work alongside the three edits under test. This reverses exactly
those three by string replacement and reports whether each one matched, so a silent no-op cannot be
mistaken for a clean revert.

    python ab_device_guard.py off    # remove the guards
    python ab_device_guard.py on     # put them back
"""
from __future__ import annotations

import sys
from pathlib import Path

R = Path("/mnt/vast/home/randall.balestriero/code/stable-envs/stable_envs/render")

RASTER_KEY_ON = ('key = (Rh, Rw, scene.world, scene.flip_y, scene.vertices, K, '
                 'str(scene.kind.device), blob)')
RASTER_KEY_OFF = 'key = (Rh, Rw, scene.world, scene.flip_y, scene.vertices, K, blob)'

RASTER_GUARD_ON = """    with torch.cuda.device(scene.kind.device):
        base = _static_base(scene, H * ss, W * ss, block)
        dummy = buf                   # unused pointer when HAS_BASE is False; Triton needs an argument
        grid = (B, triton.cdiv(H * W, block))
        _scene_kernel[grid](
            scene.kind, scene.geom, scene.fill, scene.line, scene.line_w, scene.background,
            base if base is not None else dummy, buf,
            N, int(scene.static_prefix) if base is not None else 0,
            H=H, W=W, SS=ss, V=V,
            wx0=x0, wy0=y0, sx=(x1 - x0) / (W * ss), sy=(y1 - y0) / (H * ss),
            FLIP=bool(scene.flip_y), y_hi=(y0 + y1), HAS_BASE=base is not None,
            BLOCK=block,
        )"""
RASTER_GUARD_OFF = """    base = _static_base(scene, H * ss, W * ss, block)
    dummy = buf                       # unused pointer when HAS_BASE is False; Triton needs an argument
    grid = (B, triton.cdiv(H * W, block))
    _scene_kernel[grid](
        scene.kind, scene.geom, scene.fill, scene.line, scene.line_w, scene.background,
        base if base is not None else dummy, buf,
        N, int(scene.static_prefix) if base is not None else 0,
        H=H, W=W, SS=ss, V=V,
        wx0=x0, wy0=y0, sx=(x1 - x0) / (W * ss), sy=(y1 - y0) / (H * ss),
        FLIP=bool(scene.flip_y), y_hi=(y0 + y1), HAS_BASE=base is not None,
        BLOCK=block,
    )"""

TILE_GUARD_ON = """    with torch.cuda.device(idx.device):
        _blit_kernel[(B * H * T, triton.cdiv(wpr, BLOCK))](
            atlas_w, idx.contiguous(), out.view(-1).view(_WORD_DTYPE[w]),
            H=H, W=W, T=T, WPTR=T * 3 // w, WPR=wpr, WPT=T * T * 3 // w, BLOCK=BLOCK, num_warps=4)"""
TILE_GUARD_OFF = """    _blit_kernel[(B * H * T, triton.cdiv(wpr, BLOCK))](
        atlas_w, idx.contiguous(), out.view(-1).view(_WORD_DTYPE[w]),
        H=H, W=W, T=T, WPTR=T * 3 // w, WPR=wpr, WPT=T * T * 3 // w, BLOCK=BLOCK, num_warps=4)"""

TILE_KEY_ON = """        key = (int(tile_size), str(_resolve(device)))
        if key not in cls._mem:
            cls._mem[key] = cls(tile_size, _resolve(device))"""
TILE_KEY_OFF = """        key = (int(tile_size), str(device))
        if key not in cls._mem:
            cls._mem[key] = cls(tile_size, device)"""

EDITS = [
    ("raster2d.py", "raster cache key", RASTER_KEY_ON, RASTER_KEY_OFF),
    ("raster2d.py", "raster launch guard", RASTER_GUARD_ON, RASTER_GUARD_OFF),
    ("tilemap.py", "tilemap launch guard", TILE_GUARD_ON, TILE_GUARD_OFF),
    ("tilemap.py", "tilemap atlas key", TILE_KEY_ON, TILE_KEY_OFF),
]


def main() -> None:
    want = sys.argv[1] if len(sys.argv) > 1 else "off"
    ok = True
    for fname, label, on, off in EDITS:
        p = R / fname
        s = p.read_text()
        src, dst = (on, off) if want == "off" else (off, on)
        if src not in s:
            print(f"  {label}: source form NOT FOUND (already {want}?)")
            ok = False
            continue
        p.write_text(s.replace(src, dst, 1))
        print(f"  {label}: -> {want}")
    print("all four edits toggled" if ok else "SOME EDITS DID NOT APPLY -- do not trust the A/B")


if __name__ == "__main__":
    main()
