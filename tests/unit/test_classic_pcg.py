"""The array port of NumPy's SeedSequence seeding of PCG64 and of its doubles."""

import numpy as np
import pytest

from stable_envs.envs.classic_control.simulation import pcg

pytestmark = pytest.mark.family("classic")

SEEDS = [0, 1, 7, 12345, 2**31, 2**32 - 1, 2**32, 2**32 + 5, 2**40 + 3, 2**63, 2**64 - 1]


def test_seeds_and_doubles_equal_numpy():
    state, inc = pcg.seed(np.asarray(SEEDS, dtype=np.uint64))
    draws, after = pcg.doubles(state, inc, 5)
    for i, s in enumerate(SEEDS):
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(s)))
        start = generator.bit_generator.state["state"]
        assert (int(state[0][i]) << 64 | int(state[1][i])) == start["state"], s
        assert (int(inc[0][i]) << 64 | int(inc[1][i])) == start["inc"], s
        assert [float(d[i]) for d in draws] == generator.random(5).tolist(), s
        assert (int(after[0][i]) << 64 | int(after[1][i])) == generator.bit_generator.state["state"]["state"]


def test_limbs_round_trip():
    state, _ = pcg.seed(np.asarray(SEEDS, dtype=np.uint64))
    back = pcg.from_limbs(pcg.to_limbs(state))
    assert np.array_equal(back[0], state[0]) and np.array_equal(back[1], state[1])
