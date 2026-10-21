"""``tests.helpers.rollout.sample`` on envs that ship a batched GPU simulator.

Covers what the closed-form integrator in ``tests/unit/test_rollout_api.py`` cannot: ``render=True``
with a real renderer, and a real ``step`` that writes through the state buffer (PushT replays a CUDA
graph at a fixed buffer address). 4 environments, 4 steps.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

from _shared import family_params, make_or_skip, resolve_family

import stable_envs as se
from tests.helpers.rollout import RandomPolicy, ScriptedPolicy, sample

N, STEPS, SEED = 4, 4, 20260812

#: Envs whose class itself defines ``_step`` (not inherited).
STEPPING = tuple(n for n, c in sorted(se.registry.REGISTRY.items()) if "_step" in c.__dict__)

#: Env modules that ship a simulator; their import failure explains an empty ``STEPPING``.
STEP_MODULES = ("classic_control", "pusht")


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs"))
def test_a_rollout_has_the_documented_shape_and_its_actions_account_for_its_states(name):
    """A rollout has ``steps+1`` states, ``steps`` actions and one style row per env, all on device."""
    resolve_family(name, family="stepping envs", modules=STEP_MODULES, subject="the batched-simulator rollout")
    env = make_or_skip(name, num_envs=N, device="cuda")
    tr = sample(env, steps=STEPS, batch=N, seed=SEED)

    assert tr["states"].shape == (N, STEPS + 1, env.state_dim)
    assert tr["states"].device.type == "cuda", "the trajectory came back on the host"
    assert tr["style"].shape == (N, env.style_dim)
    if tr["actions"] is not None:
        assert tr["actions"].shape[:2] == (N, STEPS)
        assert tr["actions"].device.type == "cuda"
    assert "frames" not in tr, (
        "frames must be off by default; frame buffers are orders of magnitude larger than the states"
    )


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs"))
def test_the_same_seed_gives_the_same_trajectory_and_a_different_one_does_not(name):
    """The same seed reproduces a rollout in-process, and a different seed changes it.

    ``test_reproducibility.py`` checks the cross-process case.
    """
    resolve_family(name, family="stepping envs", modules=STEP_MODULES, subject="seeded rollouts")
    env = make_or_skip(name, num_envs=N, device="cuda")
    a = sample(env, steps=STEPS, batch=N, seed=SEED)["states"].clone()
    b = sample(env, steps=STEPS, batch=N, seed=SEED)["states"]
    assert torch.equal(a, b), f"{name}: the same seed produced two different trajectories"

    c = sample(env, steps=STEPS, batch=N, seed=SEED + 1)["states"]
    if getattr(env, "seed_independent_trajectory", False):
        # Authored nominal video moves over time but consumes no randomness.
        # Assert the declared property instead of requiring artificial diversity.
        assert env.action_low == env.action_high
        assert torch.equal(a, c), f"{name}: a declared seed-independent trajectory changed with the seed"
        return
    if torch.equal(a, c):
        # A degenerate env (single-point reset, actions that do not move it) legitimately agrees.
        assert float(a.std()) == 0.0, (
            f"{name}: two different seeds gave the same non-degenerate trajectory, so the seed reaches "
            f"neither the initial states nor the policy"
        )


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs"))
def test_rendered_frames_are_exactly_the_renders_of_the_states_beside_them(name):
    """``frames[:, t] == render(states[:, t])`` byte for byte, for every t including 0.

    ``frames[:, t]`` is a strided view, which ``render(out=)`` refuses, so the rollout must assign
    each frame rather than render into the slice.
    """
    resolve_family(name, family="stepping envs", modules=STEP_MODULES, subject="rendered rollouts")
    env = make_or_skip(name, num_envs=N, device="cuda")
    tr = sample(env, steps=STEPS, batch=N, seed=SEED, render=True)
    frames, states, style = tr["frames"], tr["states"], tr["style"]

    probe = env._render(states[:, 0], style=style)
    assert frames.shape == (N, STEPS + 1, *probe.shape[1:]), (
        f"{name}: frame buffer {tuple(frames.shape)} does not match {(N, STEPS + 1, *probe.shape[1:])}"
    )
    assert frames.dtype == torch.uint8
    for t in range(STEPS + 1):
        want = env._render(states[:, t], style=style)
        d = (frames[:, t].to(torch.int16) - want.to(torch.int16)).abs()
        assert int(d.max()) == 0, (
            f"{name}: frame {t} differs from render(states[:, {t}]) by max {int(d.max())}/255 on "
            f"{int((d > 0).any(-1).sum())} pixels"
        )


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs"))
def test_a_scripted_policy_drives_the_real_simulator(name):
    """A scripted action sequence is what the rollout records in ``actions``."""
    resolve_family(name, family="stepping envs", modules=STEP_MODULES, subject="scripted rollouts")
    env = make_or_skip(name, num_envs=N, device="cuda")
    adim = int(getattr(env, "action_dim", 1))
    lo, hi = float(getattr(env, "action_low", -1.0)), float(getattr(env, "action_high", 1.0))
    want = [torch.full((N, adim), lo + (hi - lo) * (0.1 + 0.2 * t), device="cuda") for t in range(STEPS)]

    tr = sample(env, steps=STEPS, batch=N, seed=SEED, policy=ScriptedPolicy(lambda s, t: want[t]))
    assert tr["actions"] is not None, f"{name}: a scripted rollout returned no actions"
    for t in range(STEPS):
        assert torch.allclose(tr["actions"][:, t], want[t], atol=1e-6), (
            f"{name}: the action stored at t={t} is not the one the policy returned"
        )


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs"))
def test_the_random_policy_stays_inside_the_env_s_declared_action_bounds(name):
    """``RandomPolicy`` actions lie in ``[action_low, action_high)``.

    Discrete envs declare ``[0, n)`` and floor, so returning exactly ``n`` would index past the end.
    """
    resolve_family(name, family="stepping envs", modules=STEP_MODULES, subject="action bounds")
    env = make_or_skip(name, num_envs=N, device="cuda")
    if not hasattr(env, "action_dim"):
        pytest.skip(f"{name} declares a step but no action_dim, so there are no bounds to check")
    p = RandomPolicy(seed=SEED)
    p.set_env(env)
    p.reset(N)
    a = p.get_action(env._get_render_state())
    assert a.shape == (N, int(env.action_dim))
    lo, hi = float(env.action_low), float(env.action_high)
    assert float(a.min()) >= lo, f"{name}: action below action_low {lo}"
    assert float(a.max()) < hi or lo == hi, f"{name}: action reached the exclusive bound {hi}"


def test_an_env_without_a_simulator_refuses_and_says_what_to_do_instead():
    """Rolling out an env without ``_step`` raises ``NotImplementedError`` pointing to supplied states."""
    without = [n for n, c in sorted(se.registry.REGISTRY.items()) if "_step" not in c.__dict__]
    assert without, "every registered env declares a step, which makes this check vacuous"
    env = make_or_skip(without[0], num_envs=2, device="cuda")
    with pytest.raises(NotImplementedError) as e:
        sample(env, steps=2, batch=2)
    msg = str(e.value)
    assert "states you already have" in msg, f"the refusal must name what to do instead: {msg}"


def test_the_registry_still_contains_envs_that_can_be_rolled_out():
    """``STEPPING`` is non-empty unless a simulator module failed to import.

    Moving ``_step`` out of the classes' own ``__dict__`` (e.g. into a mixin) would empty every sweep here.
    """
    if not STEPPING:
        errs = {m: e for m, e in se.import_errors().items() if m in STEP_MODULES}
        assert errs, (
            "no registered class declares its own step(), and none of the env modules that ship one "
            f"({', '.join(STEP_MODULES)}) failed to import to explain it, so every rollout sweep is empty"
        )
        pytest.skip(f"the simulator-bearing modules are not importable here: {errs}")
    assert len(STEPPING) >= 2, f"only {list(STEPPING)} declares a step; the rollout sweeps are near-empty"
