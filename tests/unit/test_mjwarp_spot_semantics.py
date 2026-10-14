"""``prepare_model`` makes MJWarp discard an out-of-range spot cutoff, as MuJoCo does.

OpenGL's ``GL_SPOT_CUTOFF`` accepts ``[0, 90]`` or the sentinel ``180``; MuJoCo leaves any other cutoff
at 180 (not a spotlight), while ``mujoco_warp`` always applies the cone and ``pow(cos_theta, exponent)``.
``swm/OGBCube-v0`` has one spot at ``cutoff = 100, exponent = 10``.

A spot with an illegal cutoff loses its cone and exponent; legal spots and non-spot lights are untouched;
the light type is never rewritten, because in MuJoCo a point light renders like a switched-off light.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from stable_envs.render.mjwarp import (
    GL_SPOT_CUTOFF_MAX,
    GL_SPOT_CUTOFF_OFF,
    neutralise_illegal_spots,
    prepare_model,
    restore_model,
)

_XML = """<mujoco><worldbody>
  <light name="bad"  type="spot" pos="0 0 1" dir="0 0 -1" cutoff="{bad}" exponent="10"/>
  <light name="good" type="spot" pos="0 0 1" dir="0 0 -1" cutoff="45"  exponent="7"/>
  <light name="dirl" type="directional" dir="0 0 -1"/>
  <geom type="plane" size="1 1 .1"/>
</worldbody></mujoco>"""


def _m(bad="100"):
    return mujoco.MjModel.from_xml_string(_XML.format(bad=bad))


def test_the_constants_are_the_gl_rule():
    assert GL_SPOT_CUTOFF_MAX == 90.0
    assert GL_SPOT_CUTOFF_OFF == 180.0, "the sentinel GL leaves an illegal cutoff at"


@pytest.mark.parametrize("bad", ["100", "120", "180", "-5"])
def test_an_out_of_range_cutoff_loses_BOTH_halves_of_its_attenuation(bad):
    """Both the exponent and the cone are cleared.

    ``mujoco_warp`` returns early when ``cos_theta < cos(cutoff_rad)``, before the exponent, so clearing
    the exponent alone would leave the cone (a 5-degree one at ``cutoff = -5``).
    """
    m = _m(bad)
    hit, restore = neutralise_illegal_spots(m)
    assert hit == [0], f"cutoff {bad} is outside [0, 90] and must be neutralised"
    assert float(m.light_exponent[0]) == 0.0, "pow(cos, 0) == 1, i.e. no angular attenuation"
    assert float(m.light_cutoff[0]) == GL_SPOT_CUTOFF_OFF, "cos(180) = -1, so the cone never fires"
    assert float(m.light_exponent[1]) == 7.0, "the legal spot must be untouched"
    assert float(m.light_cutoff[1]) == 45.0, "the legal spot keeps its cone"
    assert float(restore["light_exponent"][0]) == 10.0, "the original exponent must be recoverable"
    assert float(restore["light_cutoff"][0]) == float(bad), "the original cutoff must be recoverable"


def test_a_negative_cutoff_would_have_left_a_five_degree_cone():
    """A ``-5`` cutoff, a 5-degree cone in ``mujoco_warp``, is replaced by one with ``cos(cutoff) == -1``."""
    m = _m("-5")
    assert float(np.cos(np.deg2rad(-5.0))) > 0.99, "a -5 cutoff is a tight cone, not a wide one"
    neutralise_illegal_spots(m)
    assert float(np.cos(np.deg2rad(float(m.light_cutoff[0])))) == -1.0


@pytest.mark.parametrize("ok", ["0", "45", "89", "90"])
def test_a_legal_cutoff_is_left_exactly_alone(ok):
    """The boundary matters: 90 is legal, 90.0001 is not. MuJoCo honours the former."""
    m = mujoco.MjModel.from_xml_string(_XML.format(bad=ok))
    before = np.array(m.light_exponent, copy=True), np.array(m.light_cutoff, copy=True)
    hit, _ = neutralise_illegal_spots(m)
    assert hit == []
    assert np.array_equal(m.light_exponent, before[0])
    assert np.array_equal(m.light_cutoff, before[1])


def test_just_past_the_boundary_is_neutralised():
    """90 is honoured and 90.0001 is not, so the rule is tested at the discontinuity itself."""
    m = mujoco.MjModel.from_xml_string(_XML.format(bad="90.0001"))
    assert neutralise_illegal_spots(m)[0] == [0]
    assert float(m.light_cutoff[0]) == GL_SPOT_CUTOFF_OFF


def test_non_spot_lights_are_never_touched():
    m = _m()
    before_dir = float(m.light_exponent[2])
    neutralise_illegal_spots(m)
    assert float(m.light_exponent[2]) == before_dir
    assert int(m.light_type[2]) == 1, "the directional light keeps its type"


def test_the_light_type_is_never_rewritten():
    """The light type is never rewritten.

    In MuJoCo a point light in OGBCube renders like the light switched off, so re-typing an illegal spot
    to point would not match the reference.
    """
    m = _m()
    types_before = np.array(m.light_type, copy=True)
    neutralise_illegal_spots(m)
    assert np.array_equal(m.light_type, types_before), "the light type must not be rewritten"


def test_prepare_model_restores_the_cutoff_too():
    m = _m()
    before = np.array(m.light_cutoff, copy=True)
    p = prepare_model(m)
    assert float(m.light_cutoff[0]) == GL_SPOT_CUTOFF_OFF
    restore_model(m, p)
    assert np.array_equal(m.light_cutoff, before)


def test_prepare_model_applies_it_and_reports_it():
    m = _m()
    p = prepare_model(m)
    assert p.neutralised_spots == [0]
    assert float(m.light_exponent[0]) == 0.0
    assert "out-of-range cutoff" in p.summary()


def test_prepare_model_restores_it():
    m = _m()
    before = np.array(m.light_exponent, copy=True)
    p = prepare_model(m)
    restore_model(m, p)
    assert np.array_equal(m.light_exponent, before)


def test_prepare_model_can_opt_out():
    """``spot_semantics=False`` leaves the spot unchanged."""
    m = _m()
    p = prepare_model(m, spot_semantics=False)
    assert p.neutralised_spots == []
    assert float(m.light_exponent[0]) == 10.0
    assert float(m.light_cutoff[0]) == 100.0


def test_a_model_with_no_lights_is_fine():
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
    assert neutralise_illegal_spots(m)[0] == []
    assert prepare_model(m).neutralised_spots == []
