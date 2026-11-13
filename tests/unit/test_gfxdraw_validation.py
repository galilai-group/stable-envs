from types import SimpleNamespace

import pytest
import torch

from stable_envs.render import gfxdraw as G
from stable_envs.render.gfxdraw import classic as C
from stable_envs.render.gfxdraw import fused as F


@pytest.mark.parametrize("count", [0, 1, 2, 5, 8])
def test_classic_polygon_rejects_unsupported_vertex_counts(count):
    ops = C.Ops(2, 32, 32, "cpu")
    with pytest.raises(ValueError, match="3 or 4 vertices"):
        ops.polygon(torch.zeros(2, count, 2), torch.zeros(2, 3))
    assert not ops.verts and not ops.meta


def test_style_canvas_honors_explicit_blend_color():
    gfx = G.Gfx(3, 3, "cpu")
    cv = G.StyleCanvas(gfx, 2, torch.zeros(2, 3)).use(torch.zeros(2, 3))
    color = torch.tensor([[20, 90], [50, 120], [80, 150]], dtype=torch.int64)
    b = torch.arange(2)
    xy = torch.ones(2, dtype=torch.long)
    cv.blend_at(b, xy, xy, color, torch.full((2,), G.SOLID), torch.ones(2, dtype=torch.bool))
    assert torch.equal(cv.rgb()[:, 1, 1], color.t().to(torch.uint8))


@pytest.mark.parametrize("installed", [False, True])
def test_cuda_only_entry_points_refuse_cpu(monkeypatch, installed):
    monkeypatch.setattr(F, "HAVE_TRITON", installed)
    monkeypatch.setattr(C, "HAVE_TRITON", installed)
    with pytest.raises(ValueError, match="CUDA"):
        C.render_classic(C.Ops(1, 4, 4, "cpu"))
    with pytest.raises(ValueError, match="CUDA"):
        F.render_surface(torch.zeros(1, 1, 4, 2), None, None, None, 4, 4, 1, 1)
    with pytest.raises(ValueError, match="CUDA"):
        F.shrink(torch.zeros(1, 4, 3), SimpleNamespace())
