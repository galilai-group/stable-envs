from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs import spaces
from stable_envs.testing import conformance as C

SIZE = 48
DISC = ((18, 18), 6)
DISC_RGB = (200, 40, 40)


def _checker(size: int = SIZE, cell: int = 2) -> np.ndarray:
    """Create a texture with no flat 3x3 interior, leaving the interior-agreement metric undefined."""
    y, x = np.mgrid[0:size, 0:size]
    a = (((y // cell) + (x // cell)) % 2).astype(np.uint8)
    return np.stack([80 + 60 * a, 80 + 60 * a, 90 + 60 * a], -1).astype(np.uint8)


def _frame(draw_disc: bool, jitter: int = 0) -> np.ndarray:
    """Draw an optional disc, then offset every channel so neither appearance need match exactly."""
    f = _checker().astype(np.int16)
    if draw_disc:
        y, x = np.mgrid[0:SIZE, 0:SIZE]
        (cy, cx), r = DISC
        m = (y - cy) ** 2 + (x - cx) ** 2 <= r * r
        f[m] = np.asarray(DISC_RGB, np.int16)
    return np.clip(f + int(jitter), 0, 255).astype(np.uint8)


class FakeEnv:
    """Control candidate visibility independently from the synthetic reference.

    A per-row style column selects the candidate disc; official_draws controls
    the source. This isolates object presence from rasterization differences.
    """

    reference_id = "synthetic/disc"
    fidelity_tier = "parity"

    def __init__(self, n: int = 8, declared: int = 1, official_draws: bool = True, jitter: int = 1):
        self.B, self.size, self.n = n, SIZE, n
        self.official_draws = bool(official_draws)
        # Nonzero jitter exercises a finite error ratio; jitter=0 tests the undefined-ratio branch.
        self.jitter = int(jitter)
        self.variation_space = spaces.Dict(
            {
                "visibility": spaces.Dict({"target": spaces.Discrete(2, init_value=int(declared))}),
            }
        )
        self._layout = {p: (o, w) for p, o, w in self.variation_space.flat_layout()}

    @property
    def style_dim(self) -> int:
        return self.variation_space.flat_dim

    def default_style(self) -> np.ndarray:
        return self.variation_space.pack()

    def sample_states(self, n, rng=None):
        return np.zeros((int(n), 1), np.float32)

    def _render(self, states, variation=None, style=None):
        n = len(states)
        off, _ = self._layout["visibility.target"]
        if style is not None:
            col = np.asarray(style, np.float32).reshape(n, -1)[:, off]
        elif variation is not None and "visibility.target" in variation:
            col = np.full(n, float(variation["visibility.target"]), np.float32)
        else:
            col = np.full(n, float(self.variation_space["visibility"]["target"].value), np.float32)
        return torch.as_tensor(np.stack([_frame(c >= 0.5, self.jitter) for c in col]))

    @classmethod
    def reference_env(cls, size):
        return None

    def _official(self, n: int) -> np.ndarray:
        return np.stack([_frame(self.official_draws, 0)] * n)


@pytest.fixture(autouse=True)
def _reference(monkeypatch):
    """Route reference rendering through the fixture without native GL contexts."""
    holder = {}
    monkeypatch.setattr(C, "reference_env", lambda cls, size: holder["env"])
    monkeypatch.setattr(
        FakeEnv, "reference_render", classmethod(lambda cls, ref, states: ref._official(len(states))), raising=False
    )
    return holder


def _run(reference, *, declared=1, official_draws=True, n=8, jitter=1):
    env = FakeEnv(n=n, declared=declared, official_draws=official_draws, jitter=jitter)
    reference["env"] = env
    r = C.check_task_object_visibility(env, states=env.sample_states(n))
    return r["objects"]["visibility.target"]


def test_the_interior_metric_does_not_move_when_the_object_appears():
    official = np.stack([_frame(False, 0)] * 4)
    with_disc = np.stack([_frame(True, 0)] * 4)
    without = np.stack([_frame(False, 0)] * 4)
    a = C.pixel_agreement(with_disc, official)
    b = C.pixel_agreement(without, official)
    assert a["mean_abs"] > b["mean_abs"], "the disc must change the frame at all"
    assert a["interior_exact_frac"] == b["interior_exact_frac"], (
        f"the fixture no longer reproduces the blind spot: interior exactness moved from "
        f"{b['interior_exact_frac']} to {a['interior_exact_frac']} when a whole object appeared"
    )
    assert a["interior_px"] == b["interior_px"]


def test_the_guard_passes_when_both_renderers_draw_the_object(_reference):
    o = _run(_reference, declared=1, official_draws=True)
    assert o["footprint_px"] > 0
    assert o["presence_margin"] > 0.9, o


def test_the_guard_passes_when_neither_renderer_draws_the_object(_reference):
    o = _run(_reference, declared=0, official_draws=False)
    assert o["declared_value"] == 0
    assert o["footprint_px"] > 0, "the knob must still be able to draw the object when asked"
    assert o["presence_margin"] > 0.9, o


def test_the_guard_FIRES_when_we_draw_an_object_the_reference_does_not(_reference):
    o = _run(_reference, declared=1, official_draws=False)
    assert o["presence_margin"] < 0, (
        f"we paint an object the reference does not paint and the guard did not notice: {o}"
    )


def test_a_flipped_arm_that_is_bit_exact_is_reported_rather_than_divided_by(_reference):
    o = _run(_reference, declared=1, official_draws=False, jitter=0)
    assert o["presence_margin"] is None
    assert "why_no_margin" in o, o
    # Interior agreement remains blind to the same missing-object error.
    env = FakeEnv(n=8, declared=1, official_draws=False, jitter=0)
    ours = env._render(env.sample_states(8)).cpu().numpy()
    blind = C.pixel_agreement(ours, env._official(8))
    same = C.pixel_agreement(np.stack([_frame(False, 0)] * 8), env._official(8))
    assert blind["interior_exact_frac"] == same["interior_exact_frac"], (
        "the fixture is no longer demonstrating the blind spot; the guard's justification has "
        "changed and this file has to be re-argued"
    )


def test_the_guard_FIRES_when_the_reference_draws_an_object_we_do_not(_reference):
    o = _run(_reference, declared=0, official_draws=True)
    assert o["presence_margin"] < 0, (
        f"the reference paints an object we do not paint and the guard did not notice: {o}"
    )


def test_the_guard_reports_a_per_row_collapse(_reference):
    env = FakeEnv(n=8, declared=1, official_draws=True)
    _reference["env"] = env
    clean = C.check_task_object_visibility(env, states=env.sample_states(8))
    assert clean["objects"]["visibility.target"]["per_row_max_abs_diff"] == 0

    real = FakeEnv._render

    def collapse(self, states, variation=None, style=None):
        if style is not None:
            style = np.tile(np.asarray(style, np.float32).reshape(len(states), -1)[:1], (len(states), 1))
        return real(self, states, variation=variation, style=style)

    env._render = collapse.__get__(env, FakeEnv)
    r = C.check_task_object_visibility(env, states=env.sample_states(8))
    assert r["objects"]["visibility.target"]["per_row_max_abs_diff"] > 0, (
        "a style collapse to row 0 went unreported: every uniform render still agrees, which is "
        "exactly why check_per_env_variation exists and exactly why it cannot cover a Discrete leaf"
    )
