"""Procedural textures of the world, generated in memory and handed to MuJoCo as model assets.

``assets()`` returns ``{file name: PNG bytes}`` for ``MjModel.from_xml_string(xml, assets)``; the model XML
refers to the textures by these names. Every texture is drawn from a fixed seed, so the world looks the
same in every process; they are built once per process.
"""

from __future__ import annotations

import functools
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .util import font

TERRAIN_SIZE = (2000, 1200)  # pixels for the 1000 x 600 m ground -> 0.5 m / px


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _smooth_noise(rng, shape, cells):
    small = rng.random((cells[1], cells[0]))
    img = Image.fromarray((small * 255).astype(np.uint8)).resize(shape, Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def terrain(seed: int = 7) -> Image.Image:
    """Non-repeating ground covering the whole ground once, so a pixel observation identifies the place.

    Low-frequency colour variation, scattered landmark patches and a faint 10 m grid.
    """
    rng = np.random.default_rng(seed)
    W, H = TERRAIN_SIZE
    base = np.array([0.80, 0.71, 0.55])
    n1 = _smooth_noise(rng, TERRAIN_SIZE, (12, 8))
    n2 = _smooth_noise(rng, TERRAIN_SIZE, (40, 24))
    tint = np.stack([n1 * 0.10, n2 * 0.08, (1 - n1) * 0.08], -1)
    img = np.clip(base + tint - 0.08 + 0.08 * n2[..., None], 0, 1)
    im = Image.fromarray((img * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    palette = [(120, 95, 70), (95, 110, 80), (160, 140, 110), (140, 90, 70), (90, 90, 100), (190, 175, 140)]
    for _ in range(900):  # landmark patches, 2-16 m across
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(2, 16) * 2
        col = palette[rng.integers(len(palette))]
        if rng.random() < 0.6:
            dr.ellipse([cx - r, cy - r * rng.uniform(0.5, 1), cx + r, cy + r], fill=col)
        else:
            dr.rectangle([cx - r, cy - r / 2, cx + r, cy + r / 2], fill=col)
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    dr = ImageDraw.Draw(im)
    for x in range(0, W, 20):  # 10 m grid
        dr.line([x, 0, x, H], fill=(150, 130, 100), width=1)
    for y in range(0, H, 20):
        dr.line([0, y, W, y], fill=(150, 130, 100), width=1)
    return im


def facade() -> Image.Image:
    """Glass-and-concrete facade with a few lit windows."""
    rng = np.random.default_rng(3)
    W, H = 256, 512
    im = Image.new("RGB", (W, H), (70, 76, 86))
    dr = ImageDraw.Draw(im)
    for y in range(8, H, 24):
        for x in range(6, W, 20):
            lit = rng.random() < 0.18
            col = (
                (235, 215, 150) if lit else tuple(int(c) for c in rng.integers(95, 140, 3) * np.array([0.8, 0.9, 1.0]))
            )
            dr.rectangle([x, y, x + 13, y + 15], fill=col)
    return im


def glass_facade() -> Image.Image:
    rng = np.random.default_rng(11)
    W, H = 256, 512
    base = np.zeros((H, W, 3), np.float32)
    grad = np.linspace(0.0, 1.0, H)[:, None]
    base[..., 0] = 0.30 + 0.15 * grad
    base[..., 1] = 0.45 + 0.15 * grad
    base[..., 2] = 0.62 + 0.12 * grad
    im = Image.fromarray((base * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for y in range(0, H, 16):
        dr.line([0, y, W, y], fill=(40, 50, 62), width=2)
        for x in range(0, W, 32):
            if rng.random() < 0.08:
                dr.rectangle([x + 2, y + 2, x + 29, y + 13], fill=(250, 225, 160))
    for x in range(0, W, 32):
        dr.line([x, 0, x, H], fill=(55, 65, 78), width=2)
    return im


def stone_facade() -> Image.Image:
    rng = np.random.default_rng(12)
    W, H = 256, 512
    im = Image.new("RGB", (W, H), (196, 182, 160))
    dr = ImageDraw.Draw(im)
    for y in range(10, H, 28):
        dr.rectangle([0, y + 20, W, y + 23], fill=(170, 156, 135))
        for x in range(8, W, 26):
            lit = rng.random() < 0.15
            dr.rectangle([x, y, x + 14, y + 17], fill=(245, 220, 150) if lit else (60, 70, 82))
            dr.rectangle([x, y, x + 14, y + 2], fill=(150, 138, 120))
    return im


def banner() -> Image.Image:
    im = Image.new("RGB", (512, 64), (20, 20, 22))
    dr = ImageDraw.Draw(im)
    for i in range(0, 512, 16):
        for j in range(0, 64, 16):
            if (i // 16 + j // 16) % 2 == 0:
                dr.rectangle([i, j, i + 15, j + 15], fill=(240, 240, 240))
    dr.rectangle([150, 8, 362, 56], fill=(200, 20, 30))
    dr.text((256, 32), "LAUNCH", font=font(36), fill=(255, 255, 255), anchor="mm")
    return im.transpose(Image.FLIP_LEFT_RIGHT)  # the box face maps it mirrored


def crowd() -> Image.Image:
    rng = np.random.default_rng(5)
    im = Image.new("RGB", (256, 128), (70, 72, 80))
    dr = ImageDraw.Draw(im)
    pal = [(220, 40, 40), (40, 90, 210), (240, 200, 40), (240, 240, 240), (40, 160, 70), (230, 120, 30), (30, 30, 30)]
    for _ in range(1400):
        x, y = rng.uniform(0, 256), rng.uniform(0, 128)
        c = pal[rng.integers(len(pal))]
        dr.ellipse([x - 2, y - 3, x + 2, y + 3], fill=c)
    return im


def sign() -> Image.Image:
    im = Image.new("RGB", (512, 128), (18, 110, 60))
    dr = ImageDraw.Draw(im)
    dr.rectangle([4, 4, 507, 123], outline=(240, 240, 240), width=4)
    dr.text((256, 44), "SPACEPORT  ->", font=font(44), fill=(255, 255, 255), anchor="mm")
    dr.text((256, 96), "EXIT 42   2 km", font=font(30), fill=(255, 255, 255), anchor="mm")
    return im


def grass() -> Image.Image:
    rng = np.random.default_rng(9)
    small = rng.random((16, 32))
    n = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((512, 256), Image.BICUBIC)) / 255.0
    img = np.stack([0.30 + 0.12 * n, 0.52 + 0.12 * n, 0.22 + 0.06 * n], -1)
    return Image.fromarray((img * 255).astype(np.uint8))


_TEXTURES = {
    "terrain.png": terrain,
    "facade.png": facade,
    "facade_glass.png": glass_facade,
    "facade_stone.png": stone_facade,
    "banner.png": banner,
    "crowd.png": crowd,
    "sign.png": sign,
    "grass.png": grass,
}


@functools.cache
def assets() -> dict[str, bytes]:
    """``{file name: PNG bytes}`` of every texture the model XML names."""
    return {name: _png(make()) for name, make in _TEXTURES.items()}
