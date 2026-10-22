from types import SimpleNamespace

import pytest
import torch

from stable_envs.render.nvdiff import BatchRenderer

pytestmark = pytest.mark.gpu


def test_anisotropic_texture_matches_known_footprint():
    if not torch.cuda.is_available():
        pytest.skip("nvdiffrast requires CUDA")
    dr = pytest.importorskip("nvdiffrast.torch")
    texture = torch.rand(16, 16, 3, generator=torch.Generator().manual_seed(17)).cuda()
    uv = torch.tensor([[[[0.5, 0.5]]]], device="cuda")
    derivatives = torch.tensor([[[[0.4, 0.05, 0.3, 0.02]]]], device="cuda")
    options = dict(filter_mode="linear-mipmap-linear", boundary_mode="wrap")
    first = dr.texture(texture[None], torch.tensor([[[[0.4, 0.425]]]], device="cuda"), derivatives * 0.5, **options)
    second = dr.texture(texture[None], torch.tensor([[[[0.6, 0.575]]]], device="cuda"), derivatives * 0.5, **options)
    actual = BatchRenderer._sample(SimpleNamespace(aniso=2), texture, uv, derivatives)
    torch.testing.assert_close(actual, (first + second) / 2, rtol=0, atol=1e-6)
    actual = BatchRenderer._sample(SimpleNamespace(aniso=1), texture, uv, derivatives)
    torch.testing.assert_close(actual, dr.texture(texture[None], uv, derivatives, **options), rtol=0, atol=0)


def test_rectangular_rgb_depth_and_semantic_ids_use_same_coverage():
    import numpy as np

    from stable_envs.render.nvdiff import Camera, Scene, _plane

    if not torch.cuda.is_available():
        pytest.skip("nvdiffrast requires CUDA")
    pytest.importorskip("nvdiffrast.torch")
    v, n, f = _plane([1, 1, 0], 1)
    with Scene.from_triangles(v, n, f, np.zeros(len(v), np.int64), [[1, 0, 0, 1]], device="cuda") as scene:
        with BatchRenderer(scene=scene, res=(48, 80), shading="tiny", textures=False, skybox=False) as renderer:
            pos = torch.zeros(2, 1, 3, device="cuda")
            mat = torch.eye(3, device="cuda").expand(2, 1, 3, 3)
            cam = Camera(np.array([0.0, 0.0, 2.0]), np.eye(3), 60, 0.1, 10)
            rgb = renderer.render(pos, mat, cam)
            depth = renderer.render(pos, mat, cam, mode="depth")
            ids = renderer.render(pos, mat, cam, mode="ids")
            assert rgb.shape == (2, 48, 80, 3)
            assert depth.shape == ids.shape == (2, 48, 80)
            assert torch.equal(rgb[..., 0] > 0, ids == 1)
            torch.testing.assert_close(depth[ids == 1], torch.full_like(depth[ids == 1], 2), rtol=0, atol=1e-6)
            assert not depth[ids == 0].any()


@pytest.mark.parametrize("shading", ["tiny", "mujoco"])
def test_multi_output_shares_coverage_and_validates_all_buffers_before_writing(monkeypatch, shading):
    import numpy as np

    from stable_envs.render.nvdiff import Camera, Lights, Scene, _plane

    if not torch.cuda.is_available():
        pytest.skip("nvdiffrast requires CUDA")
    pytest.importorskip("nvdiffrast.torch")
    v, n, f = _plane([1, 1, 0], 1)
    texture = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)
    with Scene.from_triangles(
        v,
        n,
        f,
        np.zeros(len(v), np.int64),
        [[1, 1, 1, 1]],
        uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
        textures={0: texture},
        tex_of_geom=[0],
    ) as scene:
        with BatchRenderer(
            scene=scene, res=(48, 80), shading=shading, textures=True, skybox=False, aniso=2
        ) as renderer:
            if shading == "mujoco":
                renderer.set_lights(
                    Lights(
                        pos=np.array([[0.0, 0.0, 2.0]]),
                        dir=np.array([[0.0, 0.0, -1.0]]),
                        directional=np.array([False]),
                        ambient=np.full((1, 3), 0.2),
                        diffuse=np.full((1, 3), 0.7),
                        specular=np.full((1, 3), 0.1),
                        attenuation=np.array([[1.0, 0.0, 0.0]]),
                        cutoff=np.array([180.0]),
                        exponent=np.array([0.0]),
                        headlight=np.array([False]),
                    )
                )
            pos = torch.zeros(2, 1, 3, device="cuda")
            mat = torch.eye(3, device="cuda").expand(2, 1, 3, 3)
            cam = Camera(np.array([0.0, 0.0, 2.0]), np.eye(3), 60, 0.1, 10)
            calls = []
            rasterize = renderer._rasterize

            def counted(*args):
                calls.append(1)
                return rasterize(*args)

            monkeypatch.setattr(renderer, "_rasterize", counted)
            outputs = renderer.render_many(pos, mat, cam)
            assert len(calls) == 1
            for mode, image in outputs.items():
                assert torch.equal(image, renderer.render(pos, mat, cam, mode=mode))
            buffers = {m: torch.full_like(v, 19) for m, v in outputs.items()}
            broken = dict(buffers, depth=buffers["depth"].double())
            count = len(calls)
            with pytest.raises(ValueError, match="out"):
                renderer.render_many(pos, mat, cam, out=broken)
            assert len(calls) == count and all((v == 19).all() for v in buffers.values())
            assert renderer.render_many(pos, mat, cam, out=buffers) is buffers
            assert all(torch.equal(buffers[m], outputs[m]) for m in outputs)
            renderer.ss = 2
            with pytest.raises(ValueError, match="supersample"):
                renderer.render_many(pos, mat, cam)
