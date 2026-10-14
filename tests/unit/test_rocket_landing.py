"""CPU checks of ``envs.rocket.legacy``: state layout, colour law, tessellation, camera and link lag.

Nothing here renders (nvdiffrast is CUDA-only); these cover everything between a state and a clip-space triangle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
from stable_envs.envs.rocket import legacy as RL  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "tests" / "fixtures" / "states" / "rocket_landing"


def _corpus():
    if not (REF / "states.npz").exists():
        pytest.skip("the rocket reference corpus is not checked out")
    return np.load(REF / "states.npz")


def test_state_is_the_env_observation_prefix():
    assert RL.PyFlytRocketLanding.state_dim == 13 and len(RL.PyFlytRocketLanding.state_names) == 13
    assert RL.PyFlytRocketLanding.state_names[3:7] == ("quat_x", "quat_y", "quat_z", "quat_w")
    S = _corpus()
    assert np.array_equal(S["state13"], S["obs30"][:, :13].astype(np.float32))


def test_lagged_pose_reproduces_the_cached_link_pose_of_the_corpus():
    """Links and camera are drawn one substep behind the base; ``lagged_pose`` must land on PyBullet's link 0 pose."""
    S = _corpus()
    pos, q, pos_prev, q_prev, v, w = RL.lagged_pose(S["obs30"][:, :13])
    cached = S["cached_link0"]
    assert np.abs(pos_prev - cached[:, :3]).max() < 1e-9
    qerr = np.minimum(np.abs(q_prev - cached[:, 3:]).max(1), np.abs(q_prev + cached[:, 3:]).max(1))
    assert qerr.max() < 1e-9
    assert np.abs(v - S["world_vel"][:, :3]).max() < 1e-6 and np.abs(w - S["world_vel"][:, 3:]).max() < 1e-6
    # and from the float32 state the env actually holds, to well under a pixel (1 cm at 3 m)
    pos32, _, prev32, _, _, _ = RL.lagged_pose(S["state13"])
    assert np.abs(prev32 - cached[:, :3]).max() < 1e-3


def test_style_layout_is_the_spaces_and_the_corpus_converter_addresses_it_by_path():
    lay = RL.style_layout()
    assert list(lay) == sorted(lay) and RL.style_dim() == 16  # gymnasium sorts Dict keys
    assert set(lay) == set(RL.CORPUS_STYLE_PATHS)
    corpus = np.array([[255, 204, 0, 51, 51, 51, 0, 0, 0, 51, 51, 51, 200, 200, 200, 1]], np.float64)
    env = RL.corpus_style_to_env(corpus)[0]
    o, _ = lay["pad.color"]
    assert env[o : o + 3].tolist() == [200, 200, 200]
    o, _ = lay["flame.ignited"]
    assert env[o] == 1 and env[lay["rocket.leg_color"][0]] == 0


def test_material_table_is_the_reference_colour_law():
    row = RL.corpus_style_to_env(np.array([255, 204, 0, 51, 51, 51, 0, 0, 0, 51, 51, 51, 200, 200, 200, 0]))[0]
    m = RL.material_table(row)[0]
    assert m.shape == (13, 3)
    assert np.allclose(m[1], [1.0, 0.8, 0.0]) and np.allclose(m[3], [0.2, 0.2, 0.2]) and np.allclose(m[8], 0)
    assert m[12, 0] == 0.784 and m[12, 0] != 200 / 255  # the URDF rewrite's 3 decimals
    assert np.allclose(m[0], 1.0) and np.allclose(m[2], 1.0)  # ground material white, fuel tank white
    assert np.allclose(m[11], RL.FLAME_RGBA[:3])
    row[RL.style_layout()["flame.ignited"][0]] = 1
    assert np.allclose(RL.material_table(row)[0, 11], RL.IGNITION_RGBA[:3])


def test_material_table_agrees_with_the_corpus_rgba_validation():
    meta = json.load(open(REF / "meta.json")) if (REF / "meta.json").exists() else pytest.skip("no corpus")
    assert meta["validation"]["rgba_err"]["max"] < 1e-6
    assert meta["style_layout"].startswith("body_rgb, fin_rgb, leg_rgb, booster_rgb, pad_rgb")


def test_tiny_cylinder_is_bullets_support_hull():
    vert, norm, face = RL.tiny_cylinder(4.77, 0.185)
    assert len(vert) == 42 and len(face) == 80  # every direction lands on its own point
    r = np.hypot(vert[:, 0], vert[:, 1])
    assert r.min() >= 0.185 - 1e-9 and r.max() <= 0.185 + RL.HULL_MARGIN + 1e-9
    assert np.abs(vert[:, 2]).min() >= 4.77 / 2 - 1e-9 and np.abs(vert[:, 2]).max() <= 4.77 / 2 + RL.HULL_MARGIN + 1e-9
    assert np.allclose(np.linalg.norm(norm, axis=1), 1.0)
    assert np.allclose(norm, vert / np.linalg.norm(vert, axis=1, keepdims=True))
    for tri in face:  # outward winding
        a, b, c = vert[tri]
        assert np.dot(np.cross(b - a, c - a), (a + b + c) / 3) > 0


def test_scene_arrays_are_thirteen_geoms_with_a_textured_ground():
    vert, norm, face, vg, uv = RL.build_scene_arrays()
    assert len(vert) == 382 and len(face) == 486 and vg.max() == 12 and vg.min() == 0
    assert (uv[vg == 0] != 0).any() and not (uv[vg != 0] != 0).any()
    assert np.allclose(uv[vg == 0], [[100, 0], [100, 100], [0, 100], [0, 0]])  # plane100.obj's vt
    assert list(RL.GEOM_NAMES[:2]) == ["ground", "body"] and RL.GEOM_NAMES[-1] == "pad"
    assert np.all(vg[face[:, 0]] == vg[face[:, 1]]) and np.all(vg[face[:, 0]] == vg[face[:, 2]])


def test_checker_texture_is_flipped_for_nvdiffrast():
    pytest.importorskip("pybullet_data")
    pytest.importorskip("PIL")
    t = RL.checker_texture()
    assert t.shape == (512, 512, 3) and t.dtype == np.uint8
    assert len(np.unique(t.reshape(-1, 3), axis=0)) == 2  # white and checker blue, nothing else


def test_geom_poses_put_the_body_at_t_and_the_links_one_substep_behind():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    state = np.concatenate([[0, 0, 0], q, [0, 0, -24.0], [1.0, 2.0, 100.0]])  # 24 m/s straight down
    xp, xm = RL.geom_poses(state[None])
    assert np.allclose(xp[0, 1], [1, 2, 100])  # the body: pose t
    assert np.allclose(xp[0, 3], [1, 2, 100 + 0.1 - 2.0])  # booster: t - dt, z - 2
    assert np.allclose(xp[0, 11], [1, 2, 100 + 0.1 - 3.4])  # flame
    assert np.allclose(xp[0, 12], RL.PAD_POS) and np.allclose(xp[0, 0], 0)
    for g in (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12):  # every unrotated geom
        assert np.allclose(xm[0, g], np.eye(3)), g
    # leg_2's URDF yaw of 4.188 rad
    leg2 = RL.quat_to_mat(RL.quat_from_euler(np.array([0, 0, 4.188])))
    assert np.allclose(xm[0, 9], leg2)


def test_camera_is_pyflyts_rocket_camera():
    pos = np.array([[3.0, -1.0, 50.0]])
    q = np.array([[0.0, 0.0, 0.0, 1.0]])
    eye, mat = RL.camera_pose(pos, q)
    assert np.allclose(eye[0], pos[0] + RL.CAM_OFFSET)
    m = mat[0]
    assert np.allclose(m.T @ m, np.eye(3), atol=1e-12)  # orthonormal
    fwd = -m[:, 2]
    assert np.allclose(fwd, (pos[0] - eye[0]) / np.linalg.norm(pos[0] - eye[0]))  # tracking the link
    assert np.linalg.det(m) > 0


def test_euler_round_trip_is_pybullets_convention():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(64, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    back = RL.quat_from_euler(RL.euler_from_quat(q))
    assert np.minimum(np.abs(back - q).max(1), np.abs(back + q).max(1)).max() < 1e-12
    assert np.allclose(
        RL.quat_to_mat(RL.quat_from_euler(np.array([0, 0, np.pi / 2]))), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    )


def test_sample_states_have_the_obs_layout_and_the_corpus_ranges():
    env = object.__new__(RL.PyFlytRocketLanding)
    env.rng = np.random.default_rng(1)
    st = env.sample_states(512)
    assert st.shape == (512, 13) and st.dtype == np.float32 and np.isfinite(st).all()
    assert np.allclose(np.linalg.norm(st[:, 3:7], axis=1), 1.0, atol=1e-5)
    h = st[:, 12]
    assert h.min() >= 1.8 and h.max() <= 480.5 and (h > 300).mean() > 0.35 and (h < 50).mean() > 0.1


def test_default_style_row_is_the_variation_space_default():
    env = object.__new__(RL.PyFlytRocketLanding)
    space = RL.PyFlytRocketLanding.make_variation_space(env)
    assert sorted(space.names()) == sorted(RL.CORPUS_STYLE_PATHS)
    assert np.array_equal(space.pack().astype(np.float64), RL._default_style_row())
    lay = RL.style_layout()
    o, _ = lay["rocket.body_color"]
    assert RL._default_style_row()[o : o + 3].tolist() == [255, 204, 0]
    assert RL._default_style_row()[lay["pad.color"][0] : lay["pad.color"][0] + 3].tolist() == [255, 255, 255]


def test_lagged_pose_renormalises_a_float32_quaternion():
    """A float32 state carries |q| = 1 +- 3e-8; the lag law must still see a unit quaternion."""
    S = _corpus()
    q32 = S["state13"][:, 3:7]
    assert np.abs(np.linalg.norm(q32.astype(np.float64), axis=1) - 1).max() > 1e-9  # it really is off
    pos, q, pos_prev, q_prev, _, _ = RL.lagged_pose(S["state13"])
    assert np.abs(np.linalg.norm(q, axis=1) - 1).max() < 1e-15
    assert np.abs(np.linalg.norm(q_prev, axis=1) - 1).max() < 1e-12


def test_visual_declaration_is_complete_with_margin():
    """Every required key is a number, the thresholds leave margin over the measurement, and the registry accepts it.

    The values themselves are asserted against a card in tests/gpu/test_conformance.py.
    """
    from stable_envs.env import REQUIRED_VISUAL_FIDELITY
    from stable_envs.registry import check_fidelity_declaration

    vf = RL.PyFlytRocketLanding.visual_fidelity
    assert all(k in vf for k in REQUIRED_VISUAL_FIDELITY)
    assert all(vf[k] is not None for k in REQUIRED_VISUAL_FIDELITY)
    assert not vf["measured_at"].startswith("UNMEASURED")
    for token in ("480 px", "supersample 1", "worst of 4", "PyFlyt 0.29.0", "pybullet 3.2.7", "RTX 3090"):
        assert token in vf["measured_at"], f"measured_at does not say {token!r}"
    assert vf["reference_versions"] == {"PyFlyt": "0.29.0", "pybullet": "3.2.7"}
    assert set(vf["measured_with"]) == {"nvdiffrast", "torch"}
    assert vf["mean_abs"] < vf["mean_abs_max"]
    assert vf["within8_frac"] > vf["within8_min"]
    assert vf["ssim"] > vf["ssim_min"]
    assert vf["mean_abs_max"] >= 1.2 * vf["mean_abs"], "a ceiling sitting on the measurement is fragile"
    assert vf["within8_min"] <= 0.97 * vf["within8_frac"]
    assert vf["ssim_min"] <= 0.97 * vf["ssim"]
    lo, hi = vf["draw_spread"]["mean_abs"]
    assert lo <= hi and abs(vf["mean_abs"] - hi) < 0.01
    assert vf["reference_unreachable_share"] == RL.REFERENCE_UNREACHABLE_SHARE == 1.0 / 960.0
    assert vf["reference_unreachable_rows_in_draws"] == 0
    check_fidelity_declaration(RL.PyFlytRocketLanding)


def test_missing_backend_names_every_optional_dependency():
    msg = RL.missing_backend()
    if RL.NVDIFF_IMPORT_ERROR is not None:
        assert msg is not None and msg.startswith("nvdiffrast:")
    import importlib.util

    for mod in ("scipy", "pybullet_data", "PIL"):
        if importlib.util.find_spec(mod) is None:
            assert msg is not None and mod in msg


def test_registered_iff_the_public_class_imports():
    import stable_envs as se

    errors = se.import_errors()
    assert ("PyFlytRocketLanding" in se.registry.REGISTRY) == ("rocket_semantics" not in errors)
    assert "rocket_landing" not in errors  # the render backend is absent cleanly, not broken
    if RL.missing_backend() is None:
        with pytest.raises(NotImplementedError, match="CPU"):
            RL.PyFlytRocketLanding(num_envs=1, device="cpu")


def test_reference_render_returns_owned_flags_and_preserves_default_frames(monkeypatch):
    from stable_envs.testing.conformance import reference_accepts_style

    cls = RL.PyFlytRocketLanding
    assert reference_accepts_style(cls)
    monkeypatch.setattr(cls, "last_reference_unreachable", np.zeros(0, bool))
    monkeypatch.setattr(
        RL,
        "_reference_frame_flagged",
        lambda env, state, style: (np.full((2, 2, 3), int(state[0]), np.uint8), bool(state[1])),
    )
    states = np.zeros((2, 13))
    states[:, :2] = [[11, 0], [22, 1]]
    default = cls.reference_render(None, states)
    frames, flags = cls.reference_render(None, states, return_flags=True)
    np.testing.assert_array_equal(default, frames)
    np.testing.assert_array_equal(frames[:, 0, 0, 0], [11, 22])
    np.testing.assert_array_equal(flags, [False, True])
    np.testing.assert_array_equal(cls.last_reference_unreachable, flags)
    cls.reference_render(None, np.zeros((1, 13)))
    np.testing.assert_array_equal(flags, [False, True])
    np.testing.assert_array_equal(cls.last_reference_unreachable, [False])
