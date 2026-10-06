import pytest
import torch

pytestmark = pytest.mark.gpu


def test_lightspec_context_and_graph_match_stock(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("MJWarp rendering requires CUDA")
    mujoco = pytest.importorskip("mujoco")
    mjw = pytest.importorskip("mujoco_warp")
    wp = pytest.importorskip("warp")
    from stable_envs.render import mjwarp_lightspec as lights
    from stable_envs.render.mjwarp import BatchRenderer

    monkeypatch.delenv("STABLE_ENVS_LIGHTSPEC", raising=False)
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <light pos="0 0 3" dir="0 0 -1" castshadow="true"/>
          <light pos="2 1 2" dir="-1 -1 -1" castshadow="false"/>
          <camera pos="0 -3 2" xyaxes="1 0 0 0 0.55 0.83"/>
          <geom type="plane" size="3 3 .1" rgba=".5 .5 .5 1"/>
          <body pos="0 0 .4"><joint type="slide" axis="1 0 0"/>
            <geom type="sphere" size=".4" rgba=".8 .2 .1 1"/>
          </body>
        </worldbody></mujoco>""")
    qpos = torch.tensor([[-0.3], [0.0], [0.3]], device="cuda")
    with wp.ScopedDevice("cuda:0"):
        with BatchRenderer(model, 3, res=32, prepare=False) as source:
            expected = source.render(qpos).clone()
            wp.synchronize()
        with lights.specialised_lights() as active:
            assert active == (mjw.__version__ == lights.UPSTREAM_VERSION)
            renderer = BatchRenderer(model, 3, res=32, prepare=False)
        with renderer:
            actual = renderer.render(qpos)
            wp.synchronize()
            assert torch.equal(actual, expected)
