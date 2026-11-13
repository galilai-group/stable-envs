"""Shared check that render(state) reads only the drawn State fields."""

import pytest
import torch


def assert_render_checks_drawn_fields_only(env, state):
    """A non-finite restore-only field renders the same frames and fails set_state; a drawn one fails render."""
    drawn = {key for key, _ in env._drawn_fields()}
    hidden = [k for k, _, off, dt in env._state_plan() if off is None and dt.is_floating_point and k not in drawn]
    assert hidden, "the State has no floating restore-only field"
    frames = env.render(state)
    bad = state.clone()
    bad[hidden[0]][0, 0] = float("nan")
    assert torch.equal(env.render(bad), frames)
    with pytest.raises(ValueError):
        env.set_state(bad)
    for key in drawn:
        bad = state.clone()
        bad[key][-1, 0] = float("nan")
        with pytest.raises(ValueError, match=f"{'.'.join(key)} contains non-finite"):
            env.render(bad)
