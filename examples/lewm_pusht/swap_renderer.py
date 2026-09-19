"""Swap the PushT renderer under the official LeWM evaluation, and nothing else.

The claim under test is that a PushT frame is a pure function of five floats. The strongest available
falsification is a pretrained agent: take the official LeWorldModel PushT checkpoint, which was trained
end-to-end on frames produced by the official pygame/pymunk renderer, run its own planning loop
unchanged, and feed it frames produced here instead. If the success rate is unchanged, the frames are
the official frames as far as anything that consumes them is concerned.

Three surfaces produce a pixel in that eval, and all three must be swapped or the model is being fed a
mixture and the comparison means nothing:

  1. ``PushT._render_frame`` — the live observation at every step. Called through
     ``AddPixelsWrapper._get_pixels -> env.render()``.
  2. ``World._extract_init_goal``'s ``pixels`` — the *stored* dataset frame at the start step, which
     ``_evaluate_from_dataset`` writes into ``infos['pixels']``. This is the frame the planner sees on
     its very first solve, before the env has been stepped once.
  3. ``World._extract_init_goal``'s ``goal`` — the *stored* dataset frame at ``start + goal_offset``.
     This is the target the world model's cost is computed against, re-injected on every step. Leaving
     it official would let the model compare our frame to their frame, which is a different experiment.

(2) and (3) are re-rendered from the dataset's own ``state`` column, which is the point: the stored
images are redundant with the numbers stored beside them.

Nothing about the policy, the solver, the seeds, the episode selection, the transform pipeline or the
success criterion is touched. The success criterion in particular is computed from the 7-D pymunk state
(``PushT.eval_state``: block/agent position within 20 px and angle within pi/9) and never reads a pixel,
so the renderer cannot move the metric except through the planner's decisions — which is exactly the
channel this experiment is trying to measure.
"""
from __future__ import annotations

import numpy as np
import torch


# gym_pusht/swm PushT geometry constants that stable-envs expresses as multipliers on the reference.
# swm stores absolute sizes; the T is hard-coded to 30 inside ``add_tee`` regardless of the variation.
SWM_AGENT_SCALE = 40.0     # add_circle: radius = 0.375 * scale = 15
SWM_GOAL_SCALE = 40.0


class _Renderer:
    """stable-envs PushT, wrapped so it answers the two shapes the swm eval asks for."""

    MODES = ("direct", "resize512", "blank")

    def __init__(self, size: int = 224, mode: str = "direct", supersample: int = 1,
                 device: str = "cuda"):
        import stable_envs as se

        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.size = int(size)
        self.mode = mode
        # 'resize512' reproduces the reference's own downsampling path (rasterise at the 512 pymunk
        # canvas, then cv2.resize) so that a disagreement can be attributed to the rasteriser rather
        # than to the resampling policy. 'direct' is what the repo actually advertises.
        self.raster_size = 512 if mode == "resize512" else self.size
        self.env = None if mode == "blank" else se.make(
            "PushT", num_envs=1, size=self.raster_size, supersample=supersample, device=device)
        self.n_calls = 0

    def __call__(self, states, variation: dict | None = None) -> np.ndarray:
        """``(N,5)`` (or ``(5,)``) -> ``(N,size,size,3)`` uint8 numpy, HWC, RGB."""
        s = torch.as_tensor(np.asarray(states, dtype=np.float32)).reshape(-1, 5)
        self.n_calls += 1
        if self.mode == "blank":
            # Falsifiability control. Success in this eval is computed from the pymunk state and never
            # reads a pixel, so "the success rate did not change when I changed the renderer" is only
            # evidence if the success rate *can* change when the renderer changes. Blank frames carry
            # zero information about the scene; if the score survives them, the comparison is vacuous
            # and every other number on this page is worthless.
            return np.full((s.shape[0], self.size, self.size, 3), 255, np.uint8)
        img = self.env.render(s, size=self.raster_size, variation=variation).cpu().numpy()
        if self.mode == "resize512":
            import cv2
            img = np.stack([cv2.resize(f, (self.size, self.size)) for f in img])
        return img


def light_color(c) -> np.ndarray:
    """pymunk's ``DrawOptions.light_color`` followed by ``SpaceDebugColor.as_int()``.

    ``as_int`` *rounds*; truncating instead is an off-by-one on every interior pixel of the block, which
    is invisible by eye and fatal to a parity claim.
    """
    return np.round(np.minimum(1.2 * np.asarray(c, np.float32), 255.0))


def _variation_from_env(env) -> dict:
    """Read the swm env's live variation values into stable-envs' variation dict.

    The colour mapping is where a naive port goes wrong. pymunk's debug draw does not paint a shape in
    its own colour: ``draw_circle`` paints the disc in ``fill_color`` and an inner disc of radius
    ``r - 4`` in ``light_color(fill_color)``; ``draw_polygon`` paints the *interior* in
    ``light_color(fill_color)`` and the outline in ``fill_color``. So swm's single ``block.color``
    (LightSlateGray) is the block's *outline*, and its fill is the 1.2x-lightened version. stable-envs
    exposes fill and outline separately, so the two have to be derived here rather than passed through.

    Passing the live values rather than assuming defaults means a domain-randomised run stays faithful,
    and a run whose appearance is *not* default fails loudly instead of silently rendering the default
    palette and calling it parity.
    """
    vs = env.variation_space
    agent_scale = float(np.asarray(vs["agent"]["scale"].value).reshape(-1)[0])
    goal_scale = float(np.asarray(vs["goal"]["scale"].value).reshape(-1)[0])
    if env.shapes[int(vs["agent"]["shape"].value)] != "o" or \
       env.shapes[int(vs["block"]["shape"].value)] != "T":
        raise NotImplementedError(
            "swap_renderer only covers the circle-agent / T-block PushT; the env is configured with "
            f"agent={env.shapes[int(vs['agent']['shape'].value)]!r} "
            f"block={env.shapes[int(vs['block']['shape'].value)]!r}.")
    agent_c = np.asarray(vs["agent"]["color"].value, np.float32)
    block_c = np.asarray(vs["block"]["color"].value, np.float32)
    return {
        "agent.color": agent_c,                       # outer disc, painted in the shape colour
        "agent.inner_color": light_color(agent_c),    # inner disc of radius r-4
        "agent.scale": np.float32(agent_scale / SWM_AGENT_SCALE),
        "block.color": light_color(block_c),          # polygon interior
        "block.outline_color": block_c,               # fat-segment outline
        "goal.color": np.asarray(vs["goal"]["color"].value, np.float32),
        "goal.position": np.asarray(env.goal_pose[:2], np.float32),
        "goal.angle": np.float32(env.goal_pose[2]),
        "goal.scale": np.float32(goal_scale / SWM_GOAL_SCALE),
        "background.color": np.asarray(vs["background"]["color"].value, np.float32),
        "rendering.render_goal": np.float32(
            bool(vs["rendering"]["render_goal"].value) and bool(env.with_target)),
    }


def install(size: int = 224, mode: str = "direct", supersample: int = 1, device: str = "cuda",
            swap_live: bool = True, swap_dataset: bool = True) -> _Renderer:
    """Monkeypatch the three pixel sources. Returns the renderer (for call counting)."""
    import stable_worldmodel.world.world as swm_world
    from stable_worldmodel.envs.pusht import env as swm_pusht

    renderer = _Renderer(size=size, mode=mode, supersample=supersample, device=device)

    if swap_live:
        def _render_frame(self, mode_):  # noqa: ARG001 -- signature fixed by the caller
            st = self._get_obs()[:5]
            return renderer(st, _variation_from_env(self))[0]

        swm_pusht.PushT._render_frame = _render_frame

    if swap_dataset:
        _orig_extract = swm_world._extract_init_goal

        def _extract_init_goal(dataset, episodes_idx, start_steps, goal_offset):
            init_state, goal_state, dataset_videos = _orig_extract(
                dataset, episodes_idx, start_steps, goal_offset)
            # The stored frames are a function of the stored states; regenerate rather than read.
            # No variation is passed here, unlike the live path, because a stored frame carries no env
            # to read one off — the corpus was recorded at the stock appearance, which is exactly what
            # stable-envs' defaults are. A domain-randomised corpus would need its variation columns
            # threaded through, and would be a different experiment.
            if "pixels" in init_state and "state" in init_state:
                init_state["pixels"] = renderer(init_state["state"][:, :5])
            if "goal" in goal_state and "goal_state" in goal_state:
                goal_state["goal"] = renderer(goal_state["goal_state"][:, :5])
            return init_state, goal_state, dataset_videos

        swm_world._extract_init_goal = _extract_init_goal

    return renderer
