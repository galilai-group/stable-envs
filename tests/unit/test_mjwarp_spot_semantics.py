"""MuJoCo discards an out-of-range spot cutoff; ``prepare_model`` must make MJWarp discard it too.

WHY THIS FILE EXISTS. ``mujoco_warp``'s shader attenuates a spot light with a hard cone plus
``pow(cos_theta, exponent)``. MuJoCo does neither when the cutoff is out of range, because OpenGL's
``GL_SPOT_CUTOFF`` accepts ``[0, 90]`` or the sentinel ``180`` and rejects everything else with
``GL_INVALID_VALUE``, leaving the default 180 = not a spotlight. ``swm/OGBCube-v0`` has one spot at
``cutoff = 100, exponent = 10``, and the disagreement is the single largest fidelity term in that
scene: neutralising it moves interior exactness from 0.0091 to 0.3507 on an RTX 3090.

Two properties are asserted, and the second is the one that keeps this safe for five other families:

* a spot whose cutoff MuJoCo would REJECT loses its angular term;
* a spot whose cutoff MuJoCo HONOURS is left exactly alone, as is every non-spot light.

The counter-finding is asserted too, because it is the mistake a reader is most likely to make: on
the MuJoCo side a POINT light is NOT equivalent to "spot with the cutoff discarded" -- it renders
identically to the light being switched off. That asymmetry is why the correction touches only what
MJWarp reads, and a test that silently allowed re-typing would invite it back.
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
    """Exponent AND cone. The first version of this rule cleared only the exponent.

    ``mujoco_warp`` runs ``if cos_theta < cos(cutoff_rad): return`` before it applies the exponent,
    so clearing the exponent alone leaves the cone. At ``cutoff = 100`` that still discards
    everything past 100 degrees, which MuJoCo does not -- and at a NEGATIVE cutoff it is far worse,
    because ``cos(-5 deg) = 0.996`` is a 5-degree cone that extinguishes almost the whole scene.
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
    """The regression the reviewer caught, asserted as a number rather than as a rule.

    Written against ``np.cos`` directly so it fails if someone reverts to clearing the exponent
    only: with ``cutoff = -5`` left in place, ``mujoco_warp``'s test is ``cos_theta < 0.996``, i.e.
    everything more than 5 degrees off-axis is unlit.
    """
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
    """The counter-finding, asserted so it cannot creep back in.

    Re-typing an illegal-cutoff spot to POINT is the obvious reading of "GL disables the spot", and
    it is measurably wrong on the REFERENCE side: in MuJoCo a POINT light in OGBCube renders
    identically to the light being switched off, 9.5 levels of mean darker than shipped
    (``m1/m1_mujoco_spot_cutoff.json``, ``verdict_point_equals_disabled``).
    """
    m = _m()
    types_before = np.array(m.light_type, copy=True)
    neutralise_illegal_spots(m)
    assert np.array_equal(m.light_type, types_before), "only light_exponent may be written"


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
    """A caller reproducing a number taken without the correction must be able to turn it off."""
    m = _m()
    p = prepare_model(m, spot_semantics=False)
    assert p.neutralised_spots == []
    assert float(m.light_exponent[0]) == 10.0
    assert float(m.light_cutoff[0]) == 100.0


def test_a_model_with_no_lights_is_fine():
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
    assert neutralise_illegal_spots(m)[0] == []
    assert prepare_model(m).neutralised_spots == []
