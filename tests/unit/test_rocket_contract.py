"""PyFlytRocketLanding onto the Env contract, checked on the class and the module.

The cell cannot be constructed without nvdiffrast, so these check the constructor's signature, the CPU
refusal's wording, the state layout and the declarations. Renderer pixels are covered by
``test_rocket_landing.py``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import stable_envs as snv
from stable_envs.envs.rocket import legacy as RL


def test_the_rocket_constructor_takes_num_envs():
    """Checked on the signature: the cell cannot be built without a CUDA host."""
    p = inspect.signature(RL.PyFlytRocketLanding.__init__).parameters
    assert list(p)[1] == "num_envs", "num_envs must be the first positional parameter"
    assert p["num_envs"].default == 1
    assert "batch" not in p
    assert p["device"].default is None


def test_the_rocket_state_is_one_rigid_body_grouped_by_what_it_is():
    """Checked on the class: the cell cannot be built without a CUDA host."""
    lay = RL.PyFlytRocketLanding.state_layout
    assert [tuple(k) for k, _, _ in lay] == [
        ("rocket", "angular_velocity"),
        ("rocket", "orientation"),
        ("rocket", "linear_velocity"),
        ("rocket", "position"),
    ]
    assert [w for _, w, _ in lay] == [3, 4, 3, 3]
    assert [o for _, _, o in lay] == [0, 3, 7, 10]
    assert sum(w for _, w, _ in lay) == RL.PyFlytRocketLanding.state_dim == 13
    # The offsets follow the frozen observation layout.
    n = RL.PyFlytRocketLanding.state_names
    assert n[0:3] == ("ang_vel_x", "ang_vel_y", "ang_vel_z")
    assert n[3:7] == ("quat_x", "quat_y", "quat_z", "quat_w")
    assert n[7:10] == ("lin_vel_x", "lin_vel_y", "lin_vel_z")
    assert n[10:13] == ("pos_x", "pos_y", "pos_z")


def test_the_rocket_refuses_a_cpu_device_by_design():
    """A ``device="cpu"`` request is refused, never served by CUDA.

    With nvdiffrast installed this raises ``NotImplementedError`` naming the missing CPU rasterizer;
    without it the ``ImportError`` for the unreachable renderer fires first. Both are accepted.
    """
    with pytest.raises((NotImplementedError, ImportError)) as e:
        RL.PyFlytRocketLanding(num_envs=1, device="cpu")
    msg = str(e.value)
    if isinstance(e.value, NotImplementedError):
        assert "has no CPU backend" in msg and "never served" in msg
    else:
        assert "cannot render in this interpreter" in msg


def test_the_rocket_is_registered_exactly_when_it_can_render():
    """With the PyFlyt extra the semantic class replaces the renderer-only one."""
    registered = snv.registry.REGISTRY.get("PyFlytRocketLanding")
    if "rocket_semantics" not in snv.import_errors():
        assert registered is not None and registered.__module__ == "stable_envs.envs.rocket.semantics"
    else:
        assert (registered is RL.PyFlytRocketLanding) == (RL.missing_backend() is None)
        assert (registered is None) == (RL.missing_backend() is not None)


def test_the_rocket_colour_exactness_declaration_survives_verbatim():
    d = RL.PyFlytRocketLanding.color_exactness_applicability
    assert d["leaves"] == (
        "pad.color",
        "rocket.body_color",
        "rocket.booster_color",
        "rocket.fin_color",
        "rocket.leg_color",
    )
    assert len(d["leaves"]) == 5
    assert d["max_shading_factor"] == 0.993567
    assert d["reachable_ceiling_for_the_probe_targets"] == {"200": 198, "33": 32, "17": 16}
    assert d["official_renderer_fails_the_same_check"] is True
    assert d["status"].startswith("not_applicable")


def test_the_rocket_mse_and_reference_versions_survive():
    cls = RL.PyFlytRocketLanding
    v = cls.visual_fidelity
    assert v["reference_versions"] == {"PyFlyt": "0.29.0", "pybullet": "3.2.7"}
    assert (v["mse_mean"], v["mse_p95"], v["mse_max"]) == (0.0091, 0.0772, 0.2354)
    assert cls.default_size == 480 and cls.default_supersample == 1


def test_the_rocket_flame_and_ignition_semantics_survive():
    assert RL.FLAME_RGBA == (1.0, 0.8, 0.0, 0.1)
    assert RL.IGNITION_RGBA == (1.0, 0.3, 0.3, 0.8)
    lay = RL.style_layout()
    assert lay["flame.ignited"] == (0, 1)
    assert set(RL.CORPUS_STYLE_PATHS) <= set(lay)
    assert RL.style_dim() == len(RL._default_style_row())


def test_the_batch_invariant_float64_mvp_does_not_depend_on_the_batch():
    """The camera is computed on the host in float64 and must not vary with how it is chunked."""
    rng = np.random.default_rng(0)
    st = np.asarray(rng.normal(size=(9, 13)), np.float64)
    st[:, 3:7] /= np.linalg.norm(st[:, 3:7], axis=1, keepdims=True)
    _, _, pos, q, _, _ = RL.lagged_pose(st)
    eye, cmat = RL.camera_pose(pos, q)
    assert eye.dtype == np.float64 and cmat.dtype == np.float64
    for lo, hi in ((0, 4), (4, 9), (2, 7)):
        _, _, p2, q2, _, _ = RL.lagged_pose(st[lo:hi])
        e2, c2 = RL.camera_pose(p2, q2)
        assert np.array_equal(e2, eye[lo:hi]) and np.array_equal(c2, cmat[lo:hi])
