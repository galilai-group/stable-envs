"""contrib.pusht_torch: pygame/OpenCV-exact primitives and frames equal to the public PushT render."""

from __future__ import annotations

import itertools
import os
import platform
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest
import torch


def _installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


pytestmark = pytest.mark.skipif(
    _installed("pygame") != "2.6.1" or _installed("pymunk") != "7.2.0",
    reason="matches pygame 2.6.1 drawing; the public PushT requires pymunk 7.2.0",
)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
LINUX_X86 = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")


def _pg_mask(draw, base=None):
    import pygame

    surface = pygame.Surface((512, 512))
    if base is None:
        surface.fill((0, 0, 0))
    else:
        pygame.surfarray.blit_array(surface, base.transpose(1, 0, 2))
    draw(surface)
    return pygame.surfarray.array3d(surface).transpose(1, 0, 2)


def _ours(parts_of):
    from contrib.pusht_torch.pgdraw import SpanRaster

    raster = SpanRaster(512, 512, "cpu")
    colors = np.array([[[0, 0, 0], [255, 0, 0]]], np.uint8)
    return raster.paint(colors, parts_of(raster))[0].numpy()


def test_primitives_match_pygame():
    import pygame

    from contrib.pusht_torch.pgdraw import line_width

    rng = np.random.default_rng(0)
    meta = np.array([[0, 1]])
    for t in range(150):
        x, y, r = (int(v) for v in (*rng.integers(-20, 532, 2), rng.integers(1, 80)))
        ours = _ours(lambda raster: raster.circles(meta, np.array([[x, y, r]])))
        assert np.array_equal(ours, _pg_mask(lambda s: pygame.draw.circle(s, (255, 0, 0), (x, y), r))), (x, y, r)

        n = int(rng.integers(3, 9))
        pts = rng.integers(0, 512, (n, 2)) if t % 3 == 0 else rng.integers(40, 470, 2) + rng.integers(-60, 60, (n, 2))
        if t % 7 == 0:
            pts[1, 1] = pts[0, 1]
        ours = _ours(lambda raster: raster.polys(meta, pts[None, :, 0], pts[None, :, 1]))
        ref = _pg_mask(lambda s: pygame.draw.polygon(s, (255, 0, 0), [tuple(map(int, p)) for p in pts]))
        assert np.array_equal(ours, ref), pts.tolist()

        a = rng.integers(0, 512, 2)
        b = np.clip(a + rng.integers(-200, 200, 2), 0, 511)
        if t % 11 == 0:
            b[0] = a[0]
        w = int(rng.integers(1, 9))
        p = np.array([[*a, *b, w]])
        ours = _ours(lambda raster: raster.lines(meta, p))
        ref = _pg_mask(lambda s: pygame.draw.lines(s, (255, 0, 0), False, [tuple(map(int, a)), tuple(map(int, b))], w))
        assert np.array_equal(ours, ref), p.tolist()

        c = a - 40 if t % 2 else a + 60
        ours = _ours(lambda raster: raster.spans((0, 1), line_width(*map(int, c), *map(int, b), w, 512, 512)))
        ref = _pg_mask(lambda s: pygame.draw.lines(s, (255, 0, 0), False, [tuple(map(int, c)), tuple(map(int, b))], w))
        assert np.array_equal(ours, ref), (c.tolist(), b.tolist(), w)


def test_antialiased_lines_match_pygame():
    import pygame

    from contrib.pusht_torch.pgdraw import SpanRaster

    rng = np.random.default_rng(1)
    raster = SpanRaster(512, 512, "cpu")
    for t in range(150):
        base = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        a = rng.integers(0, 512, 2)
        b = a.copy() if t % 17 == 0 else np.clip(a + rng.integers(-40, 40, 2), 0, 511)
        rgb = tuple(int(v) for v in rng.integers(0, 256, 3))
        ref = _pg_mask(lambda s: pygame.draw.aalines(s, rgb, False, [tuple(map(int, a)), tuple(map(int, b))]), base)
        ours = raster.blend_aalines(
            torch.from_numpy(base.copy())[None], [(0, 0, tuple(map(float, a)), tuple(map(float, b)), rgb)]
        )
        assert np.array_equal(ours[0].numpy(), ref), (a.tolist(), b.tolist())


@pytest.mark.parametrize("size", [64, 96, 128] + ([224, 300] if LINUX_X86 else []))
def test_resize_matches_opencv(size):
    cv2 = pytest.importorskip("cv2")
    from contrib.pusht_torch.cvresize import LinearResize

    imgs = np.random.default_rng(size).integers(0, 256, (3, 512, 512, 3), dtype=np.uint8)
    ours = LinearResize(512, size, "cpu")(torch.from_numpy(imgs)).numpy()
    assert np.array_equal(ours, np.stack([cv2.resize(i, (size, size)) for i in imgs]))


def _frames_match(device, size, seed):
    import stable_envs as snv
    from contrib.pusht_torch import TorchRenderer

    rng = np.random.default_rng(seed)
    renderer = TorchRenderer(device, chunk=29)
    try:
        for name, target, marker in itertools.product(("PushT", "PushTDiscrete"), (True, False), (False, True)):
            env = snv.make(name, num_envs=8, size=size, with_target=target, render_action=marker)
            keep = [n for n in env.variation_names if n.split(".")[1] not in ("shape", "scale")]
            states = []
            for episode in range(6):
                choices = (env.default_variation(), env.sample_variation(), env.sample_variation(paths=keep))
                env.set_variation(choices[episode % 3])
                env.reset(seed=episode)
                for _ in range(int(rng.integers(0, 12))):
                    if env.discrete:
                        action = torch.as_tensor(rng.integers(0, 16, (env.B, 2)), dtype=torch.int64)
                    else:
                        action = torch.as_tensor(rng.uniform(-1, 1, (env.B, 2)), dtype=torch.float32)
                    env.step(action, frames=None)
                states.append(env.get_state())
            for state in states:
                expected = env.render(state).cpu()
                got = renderer.render(env, state).cpu()
                assert torch.equal(got, expected), (name, target, marker)
            env.close()
    finally:
        renderer.close()


def test_frames_equal_the_public_render():
    _frames_match("cpu", 96, 0)
