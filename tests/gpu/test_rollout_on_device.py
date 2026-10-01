"""Private ``testing._rollout.sample`` against the envs that really ship a batched GPU simulator.

``tests/unit/test_rollout_api.py`` pins the contract on an integrator whose trajectory is known in
closed form — shapes, the actions/states correspondence, the seeding, the refusals. What it cannot
reach is what happens when the simulator is a real one and the renderer is a real one:

  * ``render=True`` is exercised NOWHERE else in this suite, and the loop that fills its frame buffer
    is the one place in the repo that deliberately ASSIGNS rather than rendering into ``out=``. The
    reason is measured: ``frames[:, t]`` of a ``(B, T+1, H, W, 3)`` buffer is a strided view, and
    ``render(out=)`` refuses one — after a period in which it accepted one and scattered the frames
    across the caller's buffer, because ``Tensor.view`` returns a strided tensor unchanged when the
    requested shape is the one it already has. If anybody "optimises" that assignment back into an
    ``out=``, this is what fails.
  * a real ``step`` writes THROUGH the state buffer, and PushT captures the whole control step into a
    CUDA graph that reads that buffer at a fixed address. A rollout that rebound it anywhere would
    replay against dead memory and produce plausible, frozen trajectories.

Kept small on purpose: 4 environments, 4 steps. Every property here is falsified at that size as surely
as at 256x100, and the suite is a gate that has to finish.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

from _shared import family_params, make_or_skip, resolve_family

import stable_envs as se
from stable_envs.testing._rollout import RandomPolicy, ScriptedPolicy, sample

N, STEPS, SEED = 4, 4, 20260812

#: Envs that DECLARE a step of their own. ``hasattr`` is too loose — it is inherited — and the claim
#: being made is that this class ported its dynamics, not that some ancestor did.
STEPPING = tuple(n for n, c in sorted(se.registry.REGISTRY.items()) if "_step" in c.__dict__)

#: Env modules that ship a simulator. If none of them imported, an empty sweep is a fact about the
#: interpreter; if they did import and the sweep is still empty, the ports have gone missing.
STEP_MODULES = ("minigrid", "classic_control", "pusht", "lunarlander")


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs", modules=STEP_MODULES))
def test_a_rollout_has_the_documented_shape_and_its_actions_account_for_its_states(name):
    """``steps+1`` states, ``steps`` actions, one style row per environment — and all on device.

    The initial state is recorded BEFORE the first step, so the off-by-one that loses ``t=0`` shows up
    as a shape. The style is returned because it is part of the state: without it the stored trajectory
    does not determine its own frames, which is the whole claim the corpus rests on.
    """
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
        "frames must be OFF by default — 256x100 at 96px is 707 MB of pixels against 512 kB of "
        "states, and materialising them during collection re-creates the problem this repo removes"
    )


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs", modules=STEP_MODULES))
def test_the_same_seed_gives_the_same_trajectory_and_a_different_one_does_not(name):
    """Both halves, because a rollout that ignores the seed satisfies the first half perfectly.

    The cross-PROCESS version of this is in ``test_reproducibility.py`` and is the stronger claim; this
    one is what localises a failure to the seed rather than to the process, and it is the only one that
    can see a policy whose generator carries on from where the previous rollout left it.
    """
    resolve_family(name, family="stepping envs", modules=STEP_MODULES, subject="seeded rollouts")
    env = make_or_skip(name, num_envs=N, device="cuda")
    a = sample(env, steps=STEPS, batch=N, seed=SEED)["states"].clone()
    b = sample(env, steps=STEPS, batch=N, seed=SEED)["states"]
    assert torch.equal(a, b), f"{name}: the same seed produced two different trajectories"

    c = sample(env, steps=STEPS, batch=N, seed=SEED + 1)["states"]
    if torch.equal(a, c):
        # Not automatically a defect: an env whose reset distribution is a single point and whose
        # actions do not move it would legitimately agree. Say which, rather than failing blind.
        assert float(a.std()) == 0.0, (
            f"{name}: two different seeds gave the same non-degenerate trajectory, so the seed reaches "
            f"neither the initial states nor the policy"
        )


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs", modules=STEP_MODULES))
def test_rendered_frames_are_exactly_the_renders_of_the_states_beside_them(name):
    """``frames[:, t] == render(states[:, t])``, byte for byte, for every t including 0.

    This is the only exercise ``render=True`` gets anywhere. The buffer is filled one timestep at a
    time and ``frames[:, t]`` is a STRIDED destination, which ``render(out=)`` refuses by design — so
    the loop assigns, at the cost of one copy of the frame bytes per timestep against a step dominated
    by the simulator. An implementation that rendered into the slice would either raise here or, before
    the refusal existed, scatter the frames across the buffer and still return the right shape.
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
            f"{int((d > 0).any(-1).sum())} pixels — the rollout's frame buffer is not the render of "
            f"the trajectory it is stored beside"
        )


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs", modules=STEP_MODULES))
def test_a_scripted_policy_drives_the_real_simulator(name):
    """A fixed action sequence must actually reach ``step``, and must be what comes back in ``actions``.

    ``RandomPolicy`` cannot show this: its actions are whatever the rollout says they are, so a loop
    that recorded the proposal and stepped with something else would agree with itself. A scripted
    policy is chosen by the TEST, so the two can disagree.
    """
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


@pytest.mark.parametrize("name", family_params(STEPPING, family="stepping envs", modules=STEP_MODULES))
def test_the_random_policy_stays_inside_the_env_s_declared_action_bounds(name):
    """Declared bounds are a contract with the simulator, and the discrete ones are HALF-OPEN.

    MiniGrid's seven actions, CartPole's two, MountainCar's and Acrobot's three all declare ``[0, n)``
    and floor, so a policy that could return exactly ``n`` would index one past the end of a transition
    table — which for a table-driven port is an out-of-bounds read rather than an error.
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
    """Most of the registry ships a renderer and no ``step``, and that is a design statement.

    Rendering and simulation are separable; an env earns a ``step()`` only where its dynamics can be
    ported AND checked against the reference. So the refusal has to point at the alternative — bring
    your own states — rather than reading like a missing feature, and it must arrive before any work
    is done rather than as an ``AttributeError`` from inside a loop.
    """
    without = [n for n, c in sorted(se.registry.REGISTRY.items()) if "_step" not in c.__dict__]
    assert without, (
        "every registered env now declares a step, which would make this check vacuous; if that is "
        "really true, delete it"
    )
    env = make_or_skip(without[0], num_envs=2, device="cuda")
    with pytest.raises(NotImplementedError) as e:
        sample(env, steps=2, batch=2)
    msg = str(e.value)
    assert "states you already have" in msg, f"the refusal must name what to do instead: {msg}"


def test_the_registry_still_contains_envs_that_can_be_rolled_out():
    """The zero-collection rule for this file, stated once rather than per-sweep.

    Every test above is parameterised over ``STEPPING``. If a refactor stopped classes from declaring
    ``step`` in their own ``__dict__`` — moving it to a mixin would do it — all five sweeps would
    collect one skip apiece and this file would keep passing while asserting nothing about any
    simulator in the repository.
    """
    if not STEPPING:
        errs = {m: e for m, e in se.import_errors().items() if m in STEP_MODULES}
        assert errs, (
            "no registered class declares its own step(), and none of the env modules that ship one "
            f"({', '.join(STEP_MODULES)}) failed to import to explain it — so every rollout sweep in "
            "this file is empty and green"
        )
        pytest.skip(f"the simulator-bearing modules are not importable here: {errs}")
    assert len(STEPPING) >= 2, f"only {list(STEPPING)} declares a step; the rollout sweeps are near-empty"
