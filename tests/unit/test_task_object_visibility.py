"""The fidelity blind spot, and the guard that closes it — asserted on a synthetic scene, on CPU.

``interior_exact_frac`` is this repository's strongest per-pixel claim and it has a hole with a name:
``conformance.interior_mask`` keeps only pixels whose 3x3 neighbourhood in the REFERENCE is one flat
colour, so a disc that lands on a textured floor contributes no interior pixels at all. A whole
task-relevant object can then appear or disappear and the metric does not move.

Measured, on real renderers, before this file existed: Step 5's reacher probe compared the same 32
states under two target placements that differ by an entire red disc — mean |d| 0.459 against 0.171 —
and both reported ``interior_exact_frac = 0.9998668941566535`` and ``interior_px = 82641``, identical
to the last digit (``examples/results/step5/reacher/parity_probe.json``). At the shipped 84 px the
same scene reports ``interior_frac = 0.000000``, so the assertion does not even apply.

``check_task_object_visibility`` is the complement. This file proves it FIRES, which is the only thing
a guard has to prove and the thing no passing run can show. The instrument is synthetic on purpose,
for the reason ``tests/certification/test_declarations.py`` gives: a rule is proved by something that breaks
it, and no env in this repository does. A fake env draws a disc over a checkerboard, a fake official
renderer decides independently whether IT draws one, and the four combinations are put through both
metrics side by side — so the test also records, as an assertion rather than as a claim, that the
interior metric is unmoved by the same event.

Nothing here touches ``interior_mask``, ``pixel_agreement`` or any declared floor.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stable_envs import spaces
from stable_envs.testing import conformance as C

SIZE = 48
#: Where the disc sits, and how big. Small, like a real one: DM Control reacher's target is ~240 px of
#: an 84 px frame on the hard task, and the whole difficulty is that a small object on a textured
#: ground is invisible to a metric that only looks at flat neighbourhoods.
DISC = ((18, 18), 6)
DISC_RGB = (200, 40, 40)


def _checker(size: int = SIZE, cell: int = 2) -> np.ndarray:
    """A floor with NO flat 3x3 neighbourhood anywhere — the condition that blinds the interior mask.

    ``cell = 2`` is deliberate: at one pixel per cell the diff between neighbours is non-zero
    everywhere too, but a 2 px checker is what dm_control's ground actually looks like at 84 px after
    the camera has minified it, and the point of the fixture is to reproduce the real scene's
    property, not to find the most extreme texture.
    """
    y, x = np.mgrid[0:size, 0:size]
    a = (((y // cell) + (x // cell)) % 2).astype(np.uint8)
    return np.stack([80 + 60 * a, 80 + 60 * a, 90 + 60 * a], -1).astype(np.uint8)


def _frame(draw_disc: bool, jitter: int = 0) -> np.ndarray:
    """One frame: the checker floor, optionally with the disc on it, the whole thing off by
    ``jitter`` levels. The jitter is applied LAST, to the disc as well as to the floor, so a fixture
    built with it is never bit-exact against one built without -- which is what a real pair of
    rasterizers looks like and what keeps ``presence_margin``'s denominator away from zero."""
    f = _checker().astype(np.int16)
    if draw_disc:
        y, x = np.mgrid[0:SIZE, 0:SIZE]
        (cy, cx), r = DISC
        m = (y - cy) ** 2 + (x - cx) ** 2 <= r * r
        f[m] = np.asarray(DISC_RGB, np.int16)
    return np.clip(f + int(jitter), 0, 255).astype(np.uint8)


class FakeEnv:
    """The smallest object ``check_task_object_visibility`` can be pointed at.

    It declares one ``visibility.target`` leaf, renders per row from the style column that leaf owns,
    and answers ``reference_render`` from ``official_draws`` — which is set INDEPENDENTLY of what the
    env draws, because the disagreement between those two is the whole subject.
    """

    reference_id = "synthetic/disc"
    fidelity_tier = "parity"

    def __init__(self, n: int = 8, declared: int = 1, official_draws: bool = True,
                 jitter: int = 1):
        self.B, self.size, self.n = n, SIZE, n
        self.official_draws = bool(official_draws)
        # A one-level offset on OUR floor, so neither arm is ever bit-exact against the official
        # frame. Without it the flipped arm's mean error is exactly zero and presence_margin's
        # denominator vanishes -- a real case, tested separately, but not the ordinary one, and a
        # fixture that only exercises the degenerate path proves nothing about the usual one.
        self.jitter = int(jitter)
        self.variation_space = spaces.Dict({
            "visibility": spaces.Dict({"target": spaces.Discrete(2, init_value=int(declared))}),
        })
        self._layout = {p: (o, w) for p, o, w in self.variation_space.flat_layout()}

    # -- the surface the check uses -------------------------------------------------

    @property
    def style_dim(self) -> int:
        return self.variation_space.flat_dim

    def default_style(self) -> np.ndarray:
        return self.variation_space.pack()

    def sample_states(self, n, rng=None):
        return np.zeros((int(n), 1), np.float32)

    def render(self, states, variation=None, style=None):
        n = len(states)
        off, _ = self._layout["visibility.target"]
        if style is not None:
            col = np.asarray(style, np.float32).reshape(n, -1)[:, off]
        elif variation is not None and "visibility.target" in variation:
            col = np.full(n, float(variation["visibility.target"]), np.float32)
        else:
            col = np.full(n, float(self.variation_space["visibility"]["target"].value), np.float32)
        return torch.as_tensor(np.stack([_frame(c >= 0.5, self.jitter) for c in col]))

    # -- the official side ----------------------------------------------------------

    @classmethod
    def reference_env(cls, size):
        return None

    def _official(self, n: int) -> np.ndarray:
        return np.stack([_frame(self.official_draws, 0)] * n)


@pytest.fixture(autouse=True)
def _reference(monkeypatch):
    """Route ``conformance``'s reference plumbing at the fixture. No GL, no dm_control, no cache."""
    holder = {}
    monkeypatch.setattr(C, "reference_env", lambda cls, size: holder["env"])
    monkeypatch.setattr(FakeEnv, "reference_render",
                        classmethod(lambda cls, ref, states: ref._official(len(states))),
                        raising=False)
    return holder


def _run(reference, *, declared=1, official_draws=True, n=8, jitter=1):
    env = FakeEnv(n=n, declared=declared, official_draws=official_draws, jitter=jitter)
    reference["env"] = env
    r = C.check_task_object_visibility(env, states=env.sample_states(n))
    return r["objects"]["visibility.target"]


# --------------------------------------------------------------- the blind spot itself


def test_the_interior_metric_does_not_move_when_the_object_appears():
    """The premise. If this ever fails, the guard below is no longer needed — and that is a fact
    worth finding out from a test rather than from a docstring."""
    official = np.stack([_frame(False, 0)] * 4)
    with_disc = np.stack([_frame(True, 0)] * 4)
    without = np.stack([_frame(False, 0)] * 4)
    a = C.pixel_agreement(with_disc, official)
    b = C.pixel_agreement(without, official)
    assert a["mean_abs"] > b["mean_abs"], "the disc must change the frame at all"
    assert a["interior_exact_frac"] == b["interior_exact_frac"], (
        f"the fixture no longer reproduces the blind spot: interior exactness moved from "
        f"{b['interior_exact_frac']} to {a['interior_exact_frac']} when a whole object appeared")
    assert a["interior_px"] == b["interior_px"]


# --------------------------------------------------------------- the guard fires


def test_the_guard_passes_when_both_renderers_draw_the_object(_reference):
    o = _run(_reference, declared=1, official_draws=True)
    assert o["footprint_px"] > 0
    assert o["presence_margin"] > 0.9, o


def test_the_guard_passes_when_neither_renderer_draws_the_object(_reference):
    """The SWM configuration: ``render_target = 0`` on both sides. Agreement, at the other value."""
    o = _run(_reference, declared=0, official_draws=False)
    assert o["declared_value"] == 0
    assert o["footprint_px"] > 0, "the knob must still be able to draw the object when asked"
    assert o["presence_margin"] > 0.9, o


def test_the_guard_FIRES_when_we_draw_an_object_the_reference_does_not(_reference):
    """The Step-5 situation, and the one the interior metric is blind to."""
    o = _run(_reference, declared=1, official_draws=False)
    assert o["presence_margin"] < 0, (
        f"we paint an object the reference does not paint and the guard did not notice: {o}")


def test_a_flipped_arm_that_is_bit_exact_is_reported_rather_than_divided_by(_reference):
    """The degenerate end of the same failure: our OTHER appearance matches the reference exactly,
    so the denominator is zero. There is no ratio to report and the answer is not 'no data' -- it is
    that the official renderer is drawing the configuration this env does NOT declare."""
    o = _run(_reference, declared=1, official_draws=False, jitter=0)
    assert o["presence_margin"] is None
    assert "why_no_margin" in o, o
    # and the interior metric, on the same frames, does NOT notice — which is the whole argument for
    # this check existing rather than for tightening that one.
    env = FakeEnv(n=8, declared=1, official_draws=False, jitter=0)
    ours = env.render(env.sample_states(8)).cpu().numpy()
    blind = C.pixel_agreement(ours, env._official(8))
    same = C.pixel_agreement(np.stack([_frame(False, 0)] * 8), env._official(8))
    assert blind["interior_exact_frac"] == same["interior_exact_frac"], (
        "the fixture is no longer demonstrating the blind spot; the guard's justification has "
        "changed and this file has to be re-argued")


def test_the_guard_FIRES_when_the_reference_draws_an_object_we_do_not(_reference):
    """The mirror image, which is just as wrong and just as invisible to the interior metric."""
    o = _run(_reference, declared=0, official_draws=True)
    assert o["presence_margin"] < 0, (
        f"the reference paints an object we do not paint and the guard did not notice: {o}")


def test_the_guard_reports_a_per_row_collapse(_reference):
    """Visibility lives in the style so it can differ per environment. If a renderer took row 0's
    value for the whole batch — the collapse ``core/style.py`` records three times — every other
    number here would stay exactly where it is."""
    env = FakeEnv(n=8, declared=1, official_draws=True)
    _reference["env"] = env
    clean = C.check_task_object_visibility(env, states=env.sample_states(8))
    assert clean["objects"]["visibility.target"]["per_row_max_abs_diff"] == 0

    real = FakeEnv.render

    def collapse(self, states, variation=None, style=None):      # row 0's appearance for everybody
        if style is not None:
            style = np.tile(np.asarray(style, np.float32).reshape(len(states), -1)[:1],
                            (len(states), 1))
        return real(self, states, variation=variation, style=style)

    env.render = collapse.__get__(env, FakeEnv)
    r = C.check_task_object_visibility(env, states=env.sample_states(8))
    assert r["objects"]["visibility.target"]["per_row_max_abs_diff"] > 0, (
        "a style collapse to row 0 went unreported: every uniform render still agrees, which is "
        "exactly why check_per_env_variation exists and exactly why it cannot cover a Discrete leaf")
