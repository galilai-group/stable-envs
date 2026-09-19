"""Policies and batched rollouts: ``env.sample(steps=100, batch=256)`` -> 256 trajectories.

A CONVENIENCE FOR WHEN THERE ARE NO STATES YET, not the recommended way to build a training set. The
recommended way is that you already have states — an offline dataset, a rollout of the official CPU
env, a corpus someone else published — and you feed them to the renderer from your own ``DataLoader``
with ``env.set_state(batch).render()``. That path works on every env in the registry. This one works
only on the minority that ship ``step``, and it is here for the case where a state corpus does not
exist yet and generating one on device beats writing frames to disk.

The policy interface mirrors ``stable_worldmodel.policy.BasePolicy`` — ``get_action(obs, **kwargs)``
plus ``set_env(env)`` — so a policy written for one works here. The one deliberate addition is that
``get_action`` receives a BATCH: ``obs`` carries B environments and the return is ``(B, action_dim)``.
A per-environment policy call would put a Python loop inside the hot path and give back the launch
overhead the rest of the repo spends its effort removing; a policy that genuinely cannot batch can
still loop internally, and that cost is then visibly its own rather than the framework's.

Rollout requires ``env.step`` — a batched GPU simulator — which many envs in this repo do NOT have, by
design. Rendering and simulation are separable, so an env is free to ship a renderer alone; ``sample``
says so precisely rather than failing somewhere deep, and points at the alternative (bring your own
states).

What an env must declare to be rolled out is ``step(actions)`` plus ``action_dim``, ``action_low`` and
``action_high``, so ``RandomPolicy`` works with no arguments. Discrete-action envs (MiniGrid's seven,
CartPole's two, MountainCar's and Acrobot's three) declare HALF-OPEN bounds ``[0, n)`` and floor, so a
uniform float is a uniform action and an integer tensor works unchanged.

A simulator's fidelity claim is NOT a renderer's, and the three regimes among the envs that ship one
must not borrow each other's language:

  * MiniGrid's transition is a finite decision table and the port is EXACT — 0 mismatches over all
    seven actions from each of 256 sampled states, on all six registered grids.
  * classic control is a published closed form; the ports agree with the float64 reference to ~1e-13
    over 100 steps, and the residue is a 1-ulp difference between CUDA's and glibc's ``sin``/``cos``.
  * PushT's contact solver tracks pymunk to a median 0.68 px over 100 steps and is explicitly not
    bit-exact, because contact is chaotic and no reimplementation of it can be.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .dynamics import has_dynamics


class BasePolicy:
    """``get_action(obs) -> (B, action_dim)``. Mirrors ``stable_worldmodel.policy.BasePolicy``."""

    def __init__(self, **kwargs: Any) -> None:
        self.env = None

    def set_env(self, env) -> None:
        self.env = env

    def reset(self, batch: int) -> None:
        """Called once at the start of a rollout. Stateful policies clear their buffers here."""

    def get_action(self, obs: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Uniform over the env's action bounds. The default, and the honest baseline for any comparison."""

    def __init__(self, low=None, high=None, seed: int = 0, **kw):
        super().__init__(**kw)
        self.low, self.high, self.seed = low, high, int(seed)
        self._gen = None

    def set_env(self, env) -> None:
        self.env = env
        if self.low is None:
            self.low = getattr(env, "action_low", -1.0)
        if self.high is None:
            self.high = getattr(env, "action_high", 1.0)
        self._gen = torch.Generator(device=env.device).manual_seed(self.seed)

    def reset(self, batch: int) -> None:
        if self.env is not None:
            self._gen = torch.Generator(device=self.env.device).manual_seed(self.seed)

    def get_action(self, obs, **kw) -> torch.Tensor:
        env = self.env
        lo = torch.as_tensor(self.low, device=env.device, dtype=torch.float32)
        hi = torch.as_tensor(self.high, device=env.device, dtype=torch.float32)
        u = torch.rand(env.B, env.action_dim, generator=self._gen, device=env.device)
        return lo + u * (hi - lo)


class ScriptedPolicy(BasePolicy):
    """Wraps a plain callable ``fn(state, t) -> (B, action_dim)``. For fixed or hand-written controllers."""

    def __init__(self, fn, **kw):
        super().__init__(**kw)
        self.fn = fn

    def get_action(self, obs, t: int = 0, **kw) -> torch.Tensor:
        return self.fn(obs, t)


@torch.no_grad()
def sample(
    env,
    steps: int = 100,
    batch: int | None = None,
    policy: BasePolicy | None = None,
    style=None,
    render: bool = False,
    seed: int | None = None,
    init_states=None,
) -> dict:
    """``batch`` trajectories of ``steps`` steps, entirely on device.

    Returns a dict of stacked tensors with time on axis 1 — ``states (B, steps+1, state_dim)``,
    ``actions (B, steps, action_dim)``, ``style (B, style_dim)``, and ``frames (B, steps+1, H, W, 3)``
    only when ``render=True``.

    Frames are OFF by default and that is the point of the repo: a rollout of 256x100 at 96px is 707 MB
    of pixels against 512 kB of states, and the states can be re-rendered at any resolution, in any
    style, at millions of frames per second whenever they are actually needed. Materialising frames
    during collection re-creates the problem the repo exists to remove.

    ``style`` is carried in the return value because it is part of the state: without it the stored
    trajectory does not determine its own frames.
    """
    if not has_dynamics(env):
        raise NotImplementedError(
            f"{type(env).__name__} ships a renderer but no batched GPU simulator, so it cannot generate "
            f"trajectories. That is a deliberate split: rendering and simulation are separable, and an "
            f"env earns a step() only where its dynamics can be ported AND checked. Either supply "
            f"states you already have "
            f"(stable_envs.render_states / StateDataset), or roll out the official CPU env and render "
            f"the states it produces."
        )

    B = int(batch or env.B)
    if B != env.B:
        raise ValueError(f"batch {B} != env batch {env.B}; construct the env with num_envs={B}")
    if seed is not None:
        env.rng = np.random.default_rng(seed)

    policy = policy or RandomPolicy(seed=seed or 0)
    policy.set_env(env)
    policy.reset(B)

    env.set_state(init_states if init_states is not None else env.sample_states(B), style=style)
    st = [env.get_state().clone()]
    acts = []
    for t in range(int(steps)):
        a = policy.get_action(env.get_state(), t=t)
        a = torch.as_tensor(a, device=env.device, dtype=torch.float32).reshape(B, -1)
        env.step(a)
        acts.append(a)
        st.append(env.get_state().clone())

    out = {
        "states": torch.stack(st, dim=1),
        "actions": torch.stack(acts, dim=1) if acts else None,
        "style": env.get_style().clone(),
    }
    if render:
        # Assign rather than render into ``out=``. ``frames[:, t]`` is a strided view of this buffer
        # and ``render(out=)`` refuses a non-contiguous destination — deliberately, and now
        # explicitly: it used to accept one and scatter the frames across the caller's buffer, because
        # ``Tensor.view`` returns a strided tensor unchanged when the shape it is given is the shape it
        # already has. (This comment used to claim the strided view "raises". It did not. That is the
        # defect ``core/env.require_contiguous_out`` closes, and the claim is now true.)
        #
        # The alternative layout — ``(steps+1, B, H, W, 3)``, whose ``frames[t]`` IS contiguous — would
        # let the renderer write in place, but it hands back a permuted view under the documented
        # ``(B, steps+1, ...)`` shape, and a caller who then calls ``.view()`` on it gets an error
        # instead of a dataset. The copy is the frame bytes once more per timestep (7 MB at B=256,
        # 96px) against a step dominated by the simulator, so it is not the thing to optimise here.
        # The shape is measured, not assumed square: highway-env draws 150x600, Craftax 112x144.
        first = env.render(out["states"][:, 0], style=out["style"])
        frames = torch.empty((B, int(steps) + 1, *first.shape[1:]), dtype=torch.uint8, device=env.device)
        frames[:, 0] = first
        for t in range(1, int(steps) + 1):
            frames[:, t] = env.render(out["states"][:, t], style=out["style"])
        out["frames"] = frames
    return out
