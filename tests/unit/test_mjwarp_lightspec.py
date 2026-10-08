from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("mujoco_warp")
from stable_envs.render.mjwarp import lightspec as lights


@pytest.mark.parametrize("version,shared", [("3.12.0", True), ("3.11.0", False)])
def test_unsupported_render_falls_back_before_touching_context(monkeypatch, version, shared):
    monkeypatch.setattr(lights.mjw, "__version__", version)
    stock = Mock(return_value="stock frame")
    monkeypatch.setattr(lights, "_STOCK_RENDER", stock)
    model = SimpleNamespace(light_castshadow=SimpleNamespace(shape=(1 if shared else 2, 3)))
    data, context = object(), object()
    assert lights.render(model, data, context) == "stock frame"
    stock.assert_called_once_with(model, data, context)


def test_context_restores_after_error_and_refuses_nesting(monkeypatch):
    monkeypatch.delenv("STABLE_ENVS_LIGHTSPEC", raising=False)
    monkeypatch.setattr(lights.mjw, "__version__", lights.UPSTREAM_VERSION)
    original = lights._upstream.render, lights.mjw.render
    with pytest.raises(ValueError, match="construction failed"):
        with lights.specialised_lights() as active:
            assert active
            assert lights._upstream.render is lights.mjw.render is lights.render
            with pytest.raises(RuntimeError, match="overlap"):
                with lights.specialised_lights():
                    pass
            raise ValueError("construction failed")
    assert (lights._upstream.render, lights.mjw.render) == original
    with lights.specialised_lights() as active:
        assert active


@pytest.mark.parametrize(
    "version,override,enabled", [("3.12.0", "1", True), ("3.11.0", "0", True), ("3.11.0", None, False)]
)
def test_disabled_context_leaves_upstream_untouched(monkeypatch, version, override, enabled):
    monkeypatch.setattr(lights.mjw, "__version__", version)
    if override is None:
        monkeypatch.delenv("STABLE_ENVS_LIGHTSPEC", raising=False)
    else:
        monkeypatch.setenv("STABLE_ENVS_LIGHTSPEC", override)
    original = lights._upstream.render, lights.mjw.render
    with lights.specialised_lights(enabled=enabled) as active:
        assert not active
        assert (lights._upstream.render, lights.mjw.render) == original
