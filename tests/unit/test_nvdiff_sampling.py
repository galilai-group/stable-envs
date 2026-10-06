import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from stable_envs.render import nvdiff


def test_anisotropic_taps_follow_screen_axis(monkeypatch):
    calls = []
    dr = ModuleType("nvdiffrast.torch")

    def texture(tex, uv, uv_da, **kwargs):
        calls.append(uv.clone())
        return torch.zeros_like(uv)

    dr.texture = texture
    package = ModuleType("nvdiffrast")
    package.torch = dr
    monkeypatch.setitem(sys.modules, "nvdiffrast", package)
    monkeypatch.setitem(sys.modules, "nvdiffrast.torch", dr)
    # Grazing footprint: screen X moves (u, v) by (0.4, 0.3); Y by (0.05, 0.02).
    uv = torch.tensor([[[[0.5, 0.5]]]])
    derivatives = torch.tensor([[[[0.4, 0.05, 0.3, 0.02]]]])
    nvdiff.BatchRenderer._sample(SimpleNamespace(aniso=2), torch.zeros(4, 4, 3), uv, derivatives)
    torch.testing.assert_close(calls[0], torch.tensor([[[[0.4, 0.425]]]]))
    torch.testing.assert_close(calls[1], torch.tensor([[[[0.6, 0.575]]]]))


def test_filtered_empty_scene_has_clear_error():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="1"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match="no visible geometry"):
        nvdiff.Scene(model, groups=(), device="cpu")
