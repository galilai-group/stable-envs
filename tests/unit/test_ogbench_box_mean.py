"""The integer supersample box filter equals the rounded float mean it replaces."""

import numpy as np
import pytest

from tests.helpers.ogbench_opengl import opengl  # noqa: F401


@pytest.mark.parametrize("ss", [1, 2, 3, 4])
def test_box_mean_equals_rint_of_the_float_mean(opengl, ss):  # noqa: F811
    module, _ = opengl
    rng = np.random.default_rng(ss)
    for trial in range(60):
        size = int(rng.integers(1, 24))
        shape = (size * ss, size * ss, 3)
        if trial % 3 == 0:
            image = rng.integers(0, 256, shape, dtype=np.uint8)
        elif trial % 3 == 1:
            image = rng.choice(np.array([0, 1, 2, 3, 127, 128, 254, 255], np.uint8), shape)
        else:
            image = (rng.integers(0, 2, shape) * 255).astype(np.uint8)
        want = np.rint(image.reshape(size, ss, size, ss, 3).mean((1, 3))).astype(np.uint8)
        out = np.empty((size, size, 3), np.uint8)
        module.box_mean(image, ss, out)
        np.testing.assert_array_equal(out, want)


def test_box_mean_rounds_every_two_by_two_sum(opengl):  # noqa: F811
    module, _ = opengl
    sums = np.arange(4 * 255 + 1)
    values = np.clip(sums[:, None] - 255 * np.arange(4), 0, 255).astype(np.uint8)
    image = np.repeat(values.reshape(len(sums), 1, 2, 2, 1).transpose(0, 2, 1, 3, 4), 3, axis=4)
    image = image.reshape(len(sums) * 2, 2, 3)
    out = np.empty((len(sums), 1, 3), np.uint8)
    module.box_mean(image, 2, out)
    np.testing.assert_array_equal(out[:, 0, 0], np.rint(sums / 4).astype(np.uint8))
