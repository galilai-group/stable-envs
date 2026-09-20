"""Replace the Reacher renderer under ``stable_worldmodel``'s own evaluation, and nothing else.

Same three surfaces as the PushT arm (``examples/lewm_pusht/swap_renderer.py``), for the same
reason -- swapping only some of them would have the world model comparing our frames against
theirs:

  1. ``DMControlWrapper.render`` -- the live observation at every step, reached through
     ``AddPixelsWrapper._get_pixels -> env.render()``.
  2. ``World._extract_init_goal``'s ``pixels`` -- the STORED dataset frame at the start step, which
     is the planner's first observation before the env has been stepped once.
  3. ``World._extract_init_goal``'s ``goal`` -- the stored frame at ``start + goal_offset``, the
     target the world model's cost is measured against and re-injected on every step.

(2) and (3) are re-rendered from the dataset's own ``qpos`` column. That is only possible because a
reacher frame is a function of ``qpos`` alone once the target is not drawn -- which the parity probe
MEASURES rather than assumes, and which is why ``target_mode`` is an argument here instead of a
constant. The arm, the floor and the lighting are all that remain, and every one of them is fixed
for the episode.

Nothing about the policy, the solver, the seeds, the episode selection, the transform pipeline or
the success criterion is touched. Success for ``qpos_match`` is decided by
``ReacherQPosMatchTask.get_termination`` from ``physics.data.qpos`` and never reads a pixel, so the
renderer can only move the metric through the planner's decisions -- which is the channel under
test.
"""

from __future__ import annotations

import numpy as np

#: SUPERSEDED, and kept only so the Step-5 arm stays reproducible. Where a geom the reference does not
#: draw USED to be put instead: Step 5 ran with ``stable_envs``' DM Control appearance space carrying
#: colour, light and fovy but no visibility, so a geom outside the camera frustum was the one way this
#: renderer could be told not to paint it. Chosen far outside a scene whose arm reaches 0.22 m.
#: ``target_mode="hidden"`` says the same thing directly and is the default; see ``VISIBILITY``.
OFFSCREEN = (10.0, 10.0, 0.01)

#: The style leaf that replaced it. ``DMCReacher*`` declares ``visibility.target`` — a binary knob that
#: zeroes the target geom's rgba alpha, which is exactly what ``stable_worldmodel``'s
#: ``render_target = 0`` does to the material and exactly what MuJoCo's own renderer honours. The state
#: then keeps the target position the episode actually drew instead of carrying 10 m as a lie, and the
#: geom is not built, transformed or cast against at all.
VISIBILITY = "visibility.target"

MODES = ("stable_envs", "blank", "official_regen")
#: ``hidden`` and ``offscreen`` differ in MECHANISM, not in intent: both say "the reference does not
#: draw its target, so neither should we". ``offscreen`` is the Step-5 mechanism and is retained so
#: ``tests/fixtures/regression/join/step5/reacher/`` can be reproduced exactly; ``hidden`` is the representation and
#: is the default.
TARGET_MODES = ("hidden", "offscreen", "as_placed")


class Renderer:
    """stable-envs ``DMCReacherHard``, answering the two shapes the swm eval asks for."""

    def __init__(
        self,
        size: int = 224,
        mode: str = "stable_envs",
        device: str = "cuda",
        target_mode: str = "hidden",
        dataset_target_mode: str | None = None,
        env_name: str = "DMCReacherHard",
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        # PER SURFACE, because the two surfaces do not contain the same picture. The live env hides
        # its target (measured: moving the geom changes 0/255). The stored corpus DRAWS it -- every
        # sampled frame carries ~285 saturated-red pixels -- because the data was recorded before
        # the env's default changed. Reproducing one policy on both surfaces means regenerating a
        # goal image without the landmark the model was trained to steer by, which is a change to
        # the objective and not to the renderer.
        dataset_target_mode = target_mode if dataset_target_mode is None else dataset_target_mode
        for name, v in (("target_mode", target_mode), ("dataset_target_mode", dataset_target_mode)):
            if v not in TARGET_MODES:
                raise ValueError(f"{name} must be one of {TARGET_MODES}, got {v!r}")
        self.size, self.mode, self.target_mode = int(size), mode, target_mode
        self.dataset_target_mode = dataset_target_mode
        self.n_live = self.n_dataset = 0
        self.env = None
        self.slots = None
        self.official = None
        if mode == "official_regen":
            # THE CONTROL that separates "our renderer" from "regenerated at all". The corpus was
            # recorded with a floor texture the current environment no longer draws, so a stored
            # frame is ~18/255 away from ANY re-render of its own state -- MuJoCo's included. This
            # mode regenerates the stored frames with the OFFICIAL renderer, so an arm that drops
            # here drops for reasons that have nothing to do with stable-envs.
            from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

            self.official = ReacherDMControlWrapper(task="qpos_match", seed=0)
            self.official.reset(seed=0)
        elif mode != "blank":
            from state_bridge import slot_names

            import stable_envs as se

            self.env = se.make(env_name, num_envs=1, size=self.size, device=device)
            self.slots = slot_names(self.env)
            self.nq = self.env.nq
            if VISIBILITY not in self.env.style_names:
                raise RuntimeError(
                    f"{env_name} declares no {VISIBILITY!r} leaf ({self.env.style_names}), so "
                    f"'the reference does not draw its target' has no representation here. Either "
                    f"this checkout predates that knob or the env is not a DM Control reacher; "
                    f"target_mode='offscreen' is the Step-5 fallback and says so in the arm."
                )

    # ---- the one place a hidden target is expressed -----------------------

    def _variation(self, mode: str) -> dict | None:
        """The style override a target mode asks for. ``None`` when the target is drawn.

        The knob is a STYLE leaf, so it is per row and it is part of what a dataset would store
        beside the state; ``render(..., variation={...})`` broadcasts one value over the batch, which
        is what these two surfaces want (a whole surface is drawn one way or the other).
        """
        return {VISIBILITY: 0} if mode == "hidden" else None

    # ---- the two shapes ---------------------------------------------------

    def blank(self, n: int) -> np.ndarray:
        """The destructive control: shape, dtype and pipeline intact, information gone."""
        return np.full((n, self.size, self.size, 3), 255, np.uint8)

    def from_live_env(self, swm_env) -> np.ndarray:
        """``(size, size, 3)`` for the state the official env is in right now."""
        self.n_live += 1
        if self.mode == "blank":
            return self.blank(1)[0]
        from state_bridge import read_state

        # Under "hidden" the state is read STRAIGHT off the official env -- the target keeps the
        # position the episode drew, and only the appearance says it is not painted. Under the Step-5
        # "offscreen" mechanism the position itself was overwritten, which is the thing this replaces.
        override = {"target": OFFSCREEN} if self.target_mode == "offscreen" else None
        st = read_state(self.env, swm_env, slots=self.slots, target_override=override)
        return self.env.render(st.reshape(1, -1), variation=self._variation(self.target_mode))[0].cpu().numpy()

    def from_state(self, qpos, target_pos=None) -> np.ndarray:
        """``(N, nq)`` stored joint positions (and the episode's target) -> ``(N, size, size, 3)``.

        ``ReacherDMControlWrapper.info`` publishes ``target_pos`` -- ``geom_xpos['target', :2]``,
        which for this model is the same two numbers ``initialize_episode`` wrote into
        ``geom_pos`` -- so a corpus recorded through it carries the whole state our renderer needs,
        and a stored frame really is regenerable from the numbers stored beside it. Where the
        reference draws no target the column is still written with whatever the corpus recorded and
        the APPEARANCE declines to paint it ("hidden"); the Step-5 mechanism overwrote the column
        instead ("offscreen"), and both are kept so that arm stays reproducible.
        """
        q = np.asarray(qpos, np.float64).reshape(len(qpos), -1)
        self.n_dataset += 1
        if self.mode == "blank":
            return self.blank(len(q))
        if self.mode == "official_regen":
            return self._official_frames(q, target_pos)
        t = None
        if self.dataset_target_mode != "offscreen":
            if target_pos is None:
                if self.dataset_target_mode == "as_placed":
                    raise KeyError(
                        "the corpus has no target_pos column, so a stored frame cannot be "
                        "regenerated with the target where the episode put it; the arms would not "
                        "be comparable"
                    )
            else:
                t = np.asarray(target_pos, np.float64).reshape(len(q), -1)
        st = np.zeros((len(q), self.env.state_dim), np.float32)
        st[:, : self.nq] = q[:, : self.nq]
        for j, (_arr, name, comp) in enumerate(self.slots):
            if self.dataset_target_mode == "offscreen":
                st[:, self.nq + j] = OFFSCREEN[comp] if name == "target" else 0.0
            elif t is not None:
                st[:, self.nq + j] = t[:, comp] if comp < t.shape[1] else 0.0
        return self.env.render(st, variation=self._variation(self.dataset_target_mode)).cpu().numpy()

    def _official_frames(self, q, target_pos) -> np.ndarray:
        """Stored frames re-rendered by MuJoCo from the same state our renderer would use."""
        import mujoco

        env = self.official
        p = env.env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target")
        t = None if target_pos is None else np.asarray(target_pos, np.float64).reshape(len(q), -1)
        out = []
        for i in range(len(q)):
            if t is not None and tid >= 0:
                m.geom_pos[tid, :2] = t[i]
            with p.reset_context():
                p.data.qpos[:] = q[i, : p.model.nq]
                p.data.qvel[:] = 0
            out.append(env.render(self.size, self.size).copy())
        return np.stack(out)


def install(
    mode: str = "stable_envs",
    size: int = 224,
    device: str = "cuda",
    target_mode: str = "offscreen",
    dataset_target_mode: str | None = None,
    swap_live: bool = True,
    swap_dataset: bool = True,
) -> Renderer:
    """Monkeypatch the three pixel sources. Returns the renderer, for call counting."""
    import stable_worldmodel.world.world as swm_world
    from stable_worldmodel.envs.dmcontrol.dmcontrol import DMControlWrapper

    r = Renderer(size=size, mode=mode, device=device, target_mode=target_mode, dataset_target_mode=dataset_target_mode)

    if swap_live:

        def render(self, width=224, height=224, camera_id=None):  # noqa: ARG001 -- caller's signature
            return r.from_live_env(self)

        DMControlWrapper.render = render

    if swap_dataset:
        _orig = swm_world._extract_init_goal

        def _extract_init_goal(dataset, episodes_idx, start_steps, goal_offset):
            init_state, goal_state, videos = _orig(dataset, episodes_idx, start_steps, goal_offset)
            if "pixels" in init_state:
                if "qpos" not in init_state:
                    raise KeyError(
                        "the dataset has no qpos column; the stored frames cannot be "
                        "regenerated from state and the arms would not be comparable"
                    )
                init_state["pixels"] = r.from_state(init_state["qpos"], init_state.get("target_pos"))
            if "goal" in goal_state:
                goal_state["goal"] = r.from_state(goal_state["goal_qpos"], goal_state.get("goal_target_pos"))
            return init_state, goal_state, videos

        swm_world._extract_init_goal = _extract_init_goal

    return r
