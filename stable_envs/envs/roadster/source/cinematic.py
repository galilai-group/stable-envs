"""The cinematic look (visual only): classic MuJoCo dark space, plus deterministic post-processing.

`patch_space(xml)` rewrites a model built by `model.build_xml`: a near-black gradient sky, a dark checker
grid floor that reaches the horizon and fades into black (MuJoCo's fog; render with mjRND_FOG on), no
decorative scenery (trees, hills, verges, crowd; the task objects stay), the sun moved into the chassis as
a tracking directional light so its shadow map always covers the car, 8x MSAA and dim headlight fill.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageFilter

SUN_DIR = np.array([-0.32, 0.26, -0.91])  # direction the sunlight travels (world frame)
FOG = (0.42, 1.15)  # fog start / end as fractions of the model extent (100 m): fully black beyond ~115 m
SCENERY_MATS = ("leaf", "leaf2", "bark", "hill", "grass", "crowd")


def patch_space(xml: str) -> str:
    xml = re.sub(
        r'<texture name="sky"[^>]*/>',
        '<texture name="sky" type="skybox" builtin="gradient" rgb1="0.09 0.11 0.15" rgb2="0 0 0" '
        'width="512" height="512"/>',
        xml,
    )
    xml = re.sub(
        r'<texture name="ground"[^>]*/>',
        '<texture name="ground" type="2d" builtin="checker" rgb1="0.13 0.15 0.18" rgb2="0.1 0.115 0.14" '
        'mark="edge" markrgb="0.22 0.25 0.3" width="300" height="300"/>',
        xml,
    )
    xml = re.sub(
        r'<material name="ground"[^>]*/>',
        '<material name="ground" texture="ground" texuniform="true" texrepeat="0.25 0.25" '
        'reflectance="0.15" specular="0.3" shininess="0.3"/>',
        xml,
    )
    xml = re.sub(r"<quality[^>]*/>", '<quality shadowsize="8192" offsamples="8"/>', xml)
    xml = re.sub(
        r"<map [^>]*/>",
        f'<map znear="0.002" zfar="25" haze="0.55" shadowclip="0.3" shadowscale="0.6" '
        f'fogstart="{FOG[0]}" fogend="{FOG[1]}"/>',
        xml,
    )
    # the ground reaches the horizon (the default 1000 x 600 m plane ends inside the view)
    xml = xml.replace(
        '<geom name="ground" type="plane" size="500 300 1" pos="100 0 0"',
        '<geom name="ground" type="plane" size="3000 3000 1" pos="100 0 0"',
    )
    xml = re.sub(r'<rgba haze="[^"]*"/>', '<rgba haze="0 0 0 1" fog="0 0 0 1"/>', xml)
    xml = re.sub(
        r"<headlight[^>]*/>",
        '<headlight ambient="0.33 0.34 0.37" diffuse="0.12 0.12 0.12" specular="0.05 0.05 0.05"/>',
        xml,
    )
    # move the sun into the chassis as a tracking light: same direction, shadow map centered on the car
    xml = re.sub(r'\s*<light name="sun"[^>]*/>', "", xml)
    d = SUN_DIR / np.linalg.norm(SUN_DIR)
    p = -d * 60
    sun = (
        f'<light name="sun" mode="track" directional="true" pos="{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}" '
        f'dir="{d[0]:.3f} {d[1]:.3f} {d[2]:.3f}" diffuse="0.95 0.9 0.82" specular="0.5 0.5 0.45" castshadow="true"/>'
    )
    xml = xml.replace('<freejoint name="root"/>', '<freejoint name="root"/>\n' + sun, 1)
    mats = "|".join(SCENERY_MATS)
    return re.sub(rf'\s*<geom[^>]*material="({mats})"[^>]*/>', "", xml)


def postfx(frame: np.ndarray, bloom: float = 0.55) -> np.ndarray:
    """Bloom on bright pixels (jets, lamps, lights), a gentle filmic grade and a vignette; deterministic."""
    img = frame.astype(np.float32) / 255.0
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    thr = 0.8
    bright = img * np.clip((lum - thr) / (1 - thr), 0, 1)[..., None]
    b8 = Image.fromarray((np.clip(bright, 0, 1) * 255).astype(np.uint8))
    h = frame.shape[0]
    glow = sum(
        np.asarray(b8.filter(ImageFilter.GaussianBlur(h * r)), np.float32) / 255.0 * w
        for r, w in ((0.006, 0.6), (0.02, 0.5), (0.05, 0.35))
    )
    img = img + bloom * glow
    # filmic tone curve + split toning (cool shadows, warm highlights)
    img = img / (1 + 0.18 * img)
    img = np.clip(img * 1.14, 0, 1)
    img = img * img * (3 - 2 * img) * 0.35 + img * 0.65  # gentle S-curve
    lum = (img @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    img = img + (1 - lum) * np.array([-0.012, 0.0, 0.025]) + lum * np.array([0.02, 0.006, -0.018])
    # vignette
    H, W = frame.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    img = img * (1 - 0.32 * np.clip(r2 - 0.25, 0, None))[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
