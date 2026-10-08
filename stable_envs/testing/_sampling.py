"""Fixed-seed state and style draws shared by conformance checks and the test runner."""

import numpy as np

STATE_SEED = 20260809
STYLE_SEED = 20260810


def sample_inputs(env, n: int):
    """Draw the fixed-seed inputs shared by the parent test and worker."""
    states = np.asarray(env.sample_states(n, rng=np.random.default_rng(STATE_SEED)), np.float32)
    style = (
        np.asarray(env.sample_style(n, rng=np.random.default_rng(STYLE_SEED)), np.float32)
        if env.style_dim
        else np.zeros((n, 0), np.float32)
    )
    return states, style
