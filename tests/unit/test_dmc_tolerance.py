"""The DM Control batch tolerances give dm_control's ``rewards.tolerance`` values bit for bit, with its dtype."""

import itertools
import zlib

import numpy as np
import pytest

pytest.importorskip("dm_control")

from dm_control.utils import rewards  # noqa: E402

from stable_envs.envs.dmcontrol.simulation.batch import scalar_tolerance, tolerance  # noqa: E402

pytestmark = pytest.mark.family("dmc")

INF = float("inf")
BOUNDS = [(0.0, 0.0), (0, 0), (-0.25, 0.25), (0.995, 1), (1.0, INF), (-INF, 0.5), (0.6, 2)]
SIGMOIDS = {
    "gaussian": (0.1, 0.5),
    "linear": (0.0, 0.5),
    "quadratic": (0.0, 0.3),
    "long_tail": (0.1,),
    "hyperbolic": (0.1,),
    "reciprocal": (0.1,),
    "cosine": (0.0, 0.1),
    "tanh_squared": (0.1,),
}
SPECIAL = [0.0, -0.0, np.nan, INF, -INF, 1.0, -1.0, 0.25, -0.25, 1.0 + 1e-12]


def _values(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, rng.choice([0.01, 1.0, 10.0]), shape)
    flat = x.reshape(-1)
    flat[: min(len(SPECIAL), flat.size)] = SPECIAL[: flat.size]
    return x


def _seed(*parts):
    return zlib.crc32(repr(parts).encode())


def _same(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected, strict=True)
    numbers = ~np.isnan(expected)
    np.testing.assert_array_equal(np.signbit(actual[numbers]), np.signbit(expected[numbers]))


@pytest.mark.parametrize("sigmoid", sorted(SIGMOIDS))
@pytest.mark.parametrize("bounds", BOUNDS, ids=str)
@pytest.mark.parametrize("margin", [0.0, 0.5, 2, 5.0])
def test_tolerance_of_arrays_matches_dm_control(sigmoid, bounds, margin):
    for value_at_margin, shape in itertools.product(SIGMOIDS[sigmoid], [(1,), (16,), (128, 3)]):
        x = _values(shape, _seed(sigmoid, bounds, margin, shape))
        kwargs = dict(bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=value_at_margin)
        _same(tolerance(x, **kwargs), rewards.tolerance(x, **kwargs))


@pytest.mark.parametrize("sigmoid", ["gaussian", "long_tail"])
@pytest.mark.parametrize("bounds", BOUNDS, ids=str)
@pytest.mark.parametrize("margin", [0.02, 0.5, 2.0])
def test_scalar_tolerance_matches_dm_control_on_each_scalar(sigmoid, bounds, margin):
    x = _values((512,), _seed(sigmoid, bounds, margin))
    actual = scalar_tolerance(x, bounds, margin, sigmoid)
    for convert in (np.float64, float):
        _same(actual, np.array([rewards.tolerance(convert(v), bounds, margin, sigmoid) for v in x]))


def test_scalar_tolerance_takes_per_environment_bounds():
    size = np.array([0.01, 0.02, 0.05])
    x = np.array([0.0, 0.03, 0.2])
    expected = np.array([rewards.tolerance(v, (0, s), s) for v, s in zip(x, size)])
    _same(scalar_tolerance(x, (0, size), size), expected)


def test_scalar_tolerance_refuses_other_sigmoids():
    with pytest.raises(ValueError, match="Unsupported scalar sigmoid"):
        scalar_tolerance(np.zeros(2), margin=1.0, sigmoid="linear")
