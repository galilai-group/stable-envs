"""The Step-7 generic pieces of ``stable_envs.render.nvdiff``, on the CPU, against MEASURED numbers.

``Scene.from_triangles``, ``gather_material``, ``shade_tiny``, ``offset_clip`` and ``prepare_texture_array``
are pure torch/numpy and need neither mujoco nor nvdiffrast. The shading numbers here are not chosen:
they are the pixel values Bullet's TinyRenderer produced on synthetic scenes
(``examples/results/step7/rocket-landing/phase0/tiny_law.json``, ``tiny_law2.json``,
``phase0_light_sweep.json``), so a change to the law is caught by the renderer it emulates rather than
by a number somebody typed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs.render import nvdiff as ND  # noqa: E402


def _box_soup(half=(1.0, 1.0, 1.0)):
    v, n, f = ND._box(half)
    return v, n, f


def test_from_triangles_builds_the_same_members_as_the_model_path():
    v, n, f = _box_soup()
    q, qn, qf = ND._plane([2.0, 2.0, 0.1], 1.0)
    vert = np.concatenate([v, q])
    norm = np.concatenate([n, qn])
    face = np.concatenate([f, qf + len(v)])
    vg = np.concatenate([np.zeros(len(v), np.int64), np.ones(len(q), np.int64)])
    S = ND.Scene.from_triangles(vert, norm, face, vg, rgba=[[1, 0, 0, 1], [0, 1, 0, 1]], device="cpu")
    assert S.V == len(vert) and S.G == 2 and S.geoms == [0, 1]
    assert tuple(S.face.shape) == (len(face), 3) and S.fgeom.tolist() == vg[face[:, 0]].tolist()
    assert S.mat_rgb.tolist() == [[1, 0, 0], [0, 1, 0]] and S.alpha.tolist() == [1, 1]
    for name in (
        "vert",
        "norm",
        "uv",
        "vgeom",
        "face",
        "fgeom",
        "mat_rgb",
        "alpha",
        "spec",
        "shininess",
        "emission",
        "texid",
        "S4",
        "S3",
        "S3p",
        "textures",
        "mjgeom",
    ):
        assert hasattr(S, name), name


def test_from_triangles_clip_matches_a_dense_transform():
    """The sparse per-geom transform is the render's first operation; check it against the obvious one."""
    v, n, f = _box_soup((0.5, 0.25, 1.0))
    S = ND.Scene.from_triangles(v, n, f, np.zeros(len(v), np.int64), rgba=[[1, 1, 1, 1]], device="cpu")
    rng = np.random.default_rng(0)
    B = 3
    q = rng.normal(size=(B, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    from scipy.spatial.transform import Rotation as Rt

    xmat = torch.as_tensor(Rt.from_quat(q).as_matrix(), dtype=torch.float32)[:, None]  # (B,1,3,3)
    xpos = torch.as_tensor(rng.normal(size=(B, 1, 3)), dtype=torch.float32)
    cam = ND.Camera(np.array([0.0, -5.0, 1.0]), np.eye(3), 60.0, 0.1, 50.0)
    view, proj = cam.matrices()
    view_t = torch.as_tensor(view, dtype=torch.float32)
    proj_t = torch.as_tensor(proj, dtype=torch.float32)
    pos, nrm, wp = S.clip(xpos, xmat, view_t, proj_t)
    # dense: world = R v + t, clip = P V world
    world = torch.einsum("bij,vj->bvi", xmat[:, 0], S.vert) + xpos
    world4 = torch.cat([world, torch.ones(B, S.V, 1)], -1)
    clip = torch.einsum("ij,bvj->bvi", proj_t @ view_t[0], world4)
    assert torch.allclose(pos, clip, atol=1e-4)
    assert torch.allclose(wp, world, atol=1e-5)
    assert torch.allclose(nrm, torch.einsum("bij,vj->bvi", xmat[:, 0], S.norm), atol=1e-5)


def test_from_triangles_refuses_a_triangle_that_straddles_two_geoms():
    v, n, f = _box_soup()
    vg = np.zeros(len(v), np.int64)
    vg[f[0, 1]] = 1
    with pytest.raises(ValueError, match="one geom"):
        ND.Scene.from_triangles(v, n, f, vg, rgba=[[1, 1, 1, 1], [1, 1, 1, 1]], device="cpu")


def test_from_triangles_carries_a_texture_through_the_shared_preparation():
    v, n, f = ND._plane([1.0, 1.0, 0.0], 1.0)
    tex = np.zeros((64, 64, 4), np.uint8)
    tex[..., 0] = 255
    S = ND.Scene.from_triangles(
        v,
        n,
        f,
        np.zeros(len(v), np.int64),
        rgba=[[1, 1, 1, 1]],
        uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
        textures={7: tex},
        tex_of_geom=[7],
        device="cpu",
    )
    assert S.texid.tolist() == [7] and 7 in S.textures
    spec = S.textures[7]
    assert spec["shape"] == (64, 64, 3) and spec["mipmap_ok"] and spec["resampled"] is None
    assert float(spec["tex"][..., 0].min()) == 1.0 and float(spec["tex"][..., 1:].max()) == 0.0
    assert tuple(S.uv.shape) == (4, 3) and S.uv[2].tolist() == [1.0, 1.0, 0.0]


def test_gather_material_shared_and_per_row():
    gid = torch.tensor([[[0, 1], [1, 0]], [[1, 1], [0, 0]]])  # (B=2,2,2)
    shared = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = ND.gather_material(shared, gid)
    assert out[0, 0, 0].tolist() == [1, 0, 0] and out[0, 0, 1].tolist() == [0, 1, 0]
    per_row = torch.stack([shared, shared[[1, 0]]])  # row 1 swaps
    out = ND.gather_material(per_row, gid)
    assert out[0, 0, 0].tolist() == [1, 0, 0]
    assert out[1, 0, 0].tolist() == [1, 0, 0]  # row 1, geom 1 -> swapped table -> red
    assert out[1, 1, 0].tolist() == [0, 1, 0]  # row 1, geom 0 -> green


def _px(col):
    """TinyRenderer's quantisation: int(255 * value), clamped."""
    return int(ND.QUANT(col, "trunc").reshape(-1)[0])


@pytest.mark.parametrize(
    "law,normal,expected",
    [
        # phase0_light_sweep.json / tiny_law.json: a horizontal white surface under the DEFAULT law reads 239
        (ND.TinyShading(), (0, 0, 1), 239),
        # tiny_law2.json tilt 0: diffuse only -> int(255 * 0.8639) = 220; specular only -> int(255 * 0.8639^2) = 190
        (ND.TinyShading(ambient=0.0, diffuse=1.0, specular=0.0), (0, 0, 1), 220),
        (ND.TinyShading(ambient=0.0, diffuse=0.0, specular=1.0), (0, 0, 1), 190),
        # tilt 10 deg about y: r_z 0.664 -> 112 (specular only), n.l 0.7757 -> 197 (diffuse only)
        (ND.TinyShading(ambient=0.0, diffuse=0.0, specular=1.0), "tilt10y", 112),
        (ND.TinyShading(ambient=0.0, diffuse=1.0, specular=0.0), "tilt10y", 197),
        # a face turned AWAY from the light is not ambient-only (153): TinyRenderer builds the reflection
        # from the UNCLAMPED n.l, so r_z is +0.8639 on the downward face and the specular term survives.
        # Measured 162 and 158 (tiny_law4_backface.json), predicted 153 by the clamped law, 162 / 158 here.
        (ND.TinyShading(), (0, 0, -1), 162),
        (ND.TinyShading(), "away150x", 158),
    ],
)
def test_shade_tiny_reproduces_the_measured_tinyrenderer_pixels(law, normal, expected):
    if normal in ("tilt10y", "away150x"):
        from scipy.spatial.transform import Rotation as Rt

        rv = [0.0, np.radians(10.0), 0.0] if normal == "tilt10y" else [np.radians(150.0), 0.0, 0.0]
        normal = Rt.from_rotvec(rv).apply([0.0, 0.0, 1.0])
    n = torch.as_tensor(np.asarray(normal, np.float32)).view(1, 1, 1, 3)
    base = torch.ones(1, 1, 1, 3)
    assert _px(ND.shade_tiny(n, base, law)) == expected


def test_shade_tiny_is_world_fixed_and_multiplies_the_specular_into_the_material():
    """Two facts measured in tiny_law2.json: the light does not follow the camera (no camera enters the
    law at all), and a black material stays black however specular the law is (TinyRenderer multiplies
    ``specular`` into ``color``, unlike MuJoCo's additive white highlight)."""
    law = ND.TinyShading(ambient=0.0, diffuse=0.0, specular=1.0)
    n = torch.as_tensor(np.asarray(law.unit_dir(), np.float32)).view(1, 1, 1, 3)
    assert _px(ND.shade_tiny(n, torch.ones(1, 1, 1, 3), law)) == int(255 * law.unit_dir()[2] ** 2)
    assert _px(ND.shade_tiny(n, torch.zeros(1, 1, 1, 3), law)) == 0


def test_offset_clip_moves_by_exactly_the_requested_pixels():
    R = 480
    pos = torch.tensor([[[0.0, 0.0, 0.5, 2.0]]])  # ndc (0, 0) after the divide
    out = ND.offset_clip(pos, R, (0.5, 0.5))
    ndc = out[..., :2] / out[..., 3:4]
    assert torch.allclose(ndc, torch.tensor([[[0.5 * 2 / R, 0.5 * 2 / R]]]))
    assert ND.offset_clip(pos, R, (0.0, 0.0)) is pos  # the default is a no-op, not a copy


def test_quant_trunc_is_what_tinyrenderer_does():
    x = torch.tensor([0.9999, 0.5, 254.6 / 255.0])
    assert ND.QUANT(x, "trunc").tolist() == [254, 127, 254]
    assert ND.QUANT(x, "round").tolist() == [255, 128, 255]


def test_prepare_texture_array_accepts_uint8_rgba_and_flags_non_pow2():
    a = np.full((100, 100, 4), 128, np.uint8)
    spec = ND.prepare_texture_array(a, "cpu", "pot")
    assert spec["shape"] == (128, 128, 3) and spec["resampled"] == "100x100 -> 128x128 (nearest)"
    assert abs(float(spec["tex"].mean()) - 128 / 255) < 1e-6
    spec = ND.prepare_texture_array(a, "cpu", "none")
    assert spec["shape"] == (100, 100, 3) and not spec["mipmap_ok"] and spec["resampled"] is None


def test_mvp_rows_is_bitwise_independent_of_batch_composition():
    """Row 0's model-view-projection must not change with how many rows share the call (defect A of Step 7
    W6: cuBLAS's batched 4x4 GEMM changed kernel between 13*24 and 13*31 products and flipped pixels)."""
    rng = np.random.default_rng(3)
    B, G = 4096, 13
    proj = torch.as_tensor(rng.normal(size=(4, 4)), dtype=torch.float32)
    view = torch.as_tensor(rng.normal(size=(B, 4, 4)) * 100.0, dtype=torch.float32)
    M = torch.as_tensor(rng.normal(size=(B, G, 4, 4)) * 400.0, dtype=torch.float32)
    full = ND.mvp_rows(proj, view, M)
    assert tuple(full.shape) == (B, G, 4, 4) and full.dtype == torch.float32
    for b in (1, 7, 24, 31, 64, 1024):
        part = ND.mvp_rows(proj, view[:b], M[:b])
        assert torch.equal(part, full[:b]), f"prefix batch {b} differs from the full batch"
    # a row in the middle, alone vs in place
    i = 1234
    assert torch.equal(ND.mvp_rows(proj, view[i : i + 1], M[i : i + 1])[0], full[i])
    # and it IS the product, correctly rounded from float64
    ref = (proj.double() @ view.double())[:, None] @ M.double()
    assert torch.equal(full, ref.to(torch.float32))
    # the unbatched-view form
    one = ND.mvp_rows(proj, view[0], M)
    assert torch.equal(one[0], full[0])


def test_from_triangles_clip_row_is_independent_of_batch_size():
    """``Scene.clip`` row ``i`` must not depend on how many rows share the call -- as CERTIFIED.

    This test originally asserted bitwise invariance at every width on whatever device it ran on, and
    that over-claimed: the Step-7 closure sweep found it failing on x86-64 CPU. Corrected here to
    assert what the repo certifies, per the CLOSURE-1 ruling (no second re-freeze in Step 7).

    **cuda, when available: bitwise at every width.** This is the certified device -- every nvdiff cell
    renders on it -- and it is enforced per env by the mandatory ``check_batch_size_invariance``
    (measured 0 at 1, 3, 7, 8, 64, 256 for the six locomotion cells and at 1, 7, 64, 4096 for
    ``PFRocketLanding``).

    **cpu: within ONE float32 ulp at every width**, which is the portable claim. Bitwise-ness on CPU is
    a property of the machine, not of this repo's arithmetic, so it is measured and recorded rather
    than asserted -- asserting it is exactly the mistake this test is correcting. Measured on x86-64
    (login010, torch 2.14.0+cu126, 1/2/4 threads;
    ``examples/results/step7/rocket-landing/closure1/probe_clip_cpu_batch_x86_widthsweep.json``):

    ==========  ===================================  ==================================
    quantity    bitwise on x86-64 CPU                otherwise
    ==========  ===================================  ==================================
    positions   widths >= 7                          widths 1-6, one ulp (3.05e-05)
    normals     every width except 4 and 8           widths 4, 8, one ulp (5.96e-08)
    world pos   every width                          --
    ==========  ===================================  ==================================

    Everything is bitwise at every width on arm64 (Apple M5) and on cuda. Two earlier width samples
    missed part of this and are superseded by the sweep above: the closure sweep's (1, 2, 7, 8, 31, 32,
    63) and this workstream's first probe's (1, 2, 3, 7, 8, 31, 32, 63) both skip 4-6, so both reported
    the positions as recovering at width 3 and the normals as never moving.

    The cause is measured and is NOT this repo's arithmetic: ``mvp_rows`` -- the float64 elementwise
    ``proj @ view @ M`` -- is bit-identical at every width on every machine measured. The residual is
    ``torch.sparse.mm``, whose four-term row sums (``S4``/``S3`` carry four and three nonzeros per row)
    are reordered by a kernel chosen from the dense operand's column count.

    DOCUMENTED_LIMITATION (Step 7), also on ``rocket_landing.DOCUMENTED_LIMITATIONS`` and so in
    ``describe()``: this reaches no shipped frame, because nothing in the registry draws through nvdiff
    on ``device="cpu"``. A fix for the POSITIONS is written and measured --
    ``scratch/step7/rocket-landing/closure1_explicit_pos_rows.patch`` -- and is not applied in Step 7
    because on cuda it moves four of the six re-frozen locomotion cells by 2.7e-07 of mean_abs and
    1e-09 of SSIM, and a re-freeze needs a certification failure with a causal reason rather than an
    implementation that merely differs. It is the Step-8 candidate
    (``closure1/digest_vs_refreeze.json``). Note that it addresses the positions only; the normals
    share the same sparse stage and would need the same treatment.
    """
    v, n, f = _box_soup((0.5, 0.25, 1.0))
    rng = np.random.default_rng(5)
    B = 64
    from scipy.spatial.transform import Rotation as Rt

    q = rng.normal(size=(B, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    mats = torch.as_tensor(Rt.from_quat(q).as_matrix(), dtype=torch.float32)[:, None]
    poss = torch.as_tensor(rng.normal(size=(B, 1, 3)) * 300.0, dtype=torch.float32)
    cam = ND.Camera(rng.normal(size=(B, 3)) * 300.0, np.broadcast_to(np.eye(3), (B, 3, 3)).copy(), 90.0, 0.1, 255.0)
    view, proj = cam.matrices()

    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        S = ND.Scene.from_triangles(v, n, f, np.zeros(len(v), np.int64), rgba=[[1, 1, 1, 1]], device=dev)
        xmat, xpos = mats.to(dev), poss.to(dev)
        view_t = torch.as_tensor(view, dtype=torch.float32, device=dev)
        proj_t = torch.as_tensor(proj, dtype=torch.float32, device=dev)
        full = S.clip(xpos, xmat, view_t, proj_t)
        for b in (1, 2, 3, 4, 5, 6, 7, 8, 16, 31, 63):
            got = S.clip(xpos[:b], xmat[:b], view_t[:b], proj_t)
            for name, g, fl in zip(("positions", "normals", "world positions"), got, full):
                ref = fl[:b]
                if dev == "cuda":
                    assert torch.equal(g, ref), (
                        f"cuda: clip {name} depend on the batch width at {b} -- this is a "
                        f"certification failure; check_batch_size_invariance enforces it per env"
                    )
                else:
                    # the portable CPU claim: one float32 ulp at the quantity's own scale. An
                    # assertion, not an exemption -- it fails the moment the residual grows past what
                    # a reordered four-term sum can account for.
                    d = float((g.double() - ref.double()).abs().max())
                    ulp = float(np.spacing(np.float32(float(ref.abs().max()))))
                    assert d <= ulp, (
                        f"cpu: clip {name} moved by {d:.3e} at width {b}, more than the one float32 "
                        f"ulp ({ulp:.3e}) the sparse kernel's summation order can account for -- that "
                        f"is no longer the documented limitation, it is a defect"
                    )
