"""DM Control suite — eighteen cells in two tiers, one state vector each.

``stable_envs.envs.dmcontrol`` IS this package. This package split the 3,376-line module it used to be into
one small base, two renderer modules and one module per domain (plan §3, §12 B); every name the module
exported is re-exported here, so ``from stable_envs.envs import dmcontrol as D`` and every
``D.<anything>`` the tests, the probes and ``stable_worldmodel`` reach for keeps resolving, and the
registered classes still report ``stable_envs.envs.dmcontrol`` as their module — which is the path
they are reachable at, and the family ``registry.spec`` reports.

WHERE THINGS ARE
----------------

``base``      ``_DMCModel``: the model plumbing, the widened state, the appearance knobs, the visited
              distribution, the bridge to dm_control's own renderer. Properties of the DOMAIN.
``suite``     the PARITY renderer: three baked plates and an analytic ray-cast of the movers.
``trackcom``  the VISUAL renderer: the whole scene to MJWarp (CUDA) or ``cpu3d`` (CPU), every frame.
``_model``    what a compiled ``MjModel`` says, the caches, and the census of what an episode draws.
``_shading``  the lit-surface law both tiers evaluate, and the lights (including MuJoCo's headlight,
              which is not in the model).
``_tendon``   ``mjv_catenary`` and the capsule chain it becomes — ``ball_in_cup``'s string.
``_cells``    how a cell is DECLARED, and the provenance strings several of them share.
<domain>.py   one module per domain, holding that domain's declarations and its leaf classes. What a
              reader looking for "what does cheetah claim" opens.

Reference: ``dm_control.suite.load(domain, task).physics.render(h, w, camera_id=0)``, which is
``mujoco.Renderer`` at the default ``MjvOption`` (verified: 0/255 against a hand-built Renderer on
every domain probed). So the target is MuJoCo's own EGL raster of a lit 3-D scene, not a vector
drawing.

TWO TIERS, AND WHAT SPLITS THEM
-------------------------------

Camera 0 is a ``fixed`` model camera on twelve cells and a ``mode="trackcom"`` camera that rides the
body on six. That single fact decides the renderer and therefore the tier:

  * **PARITY** (``DMControlSuite``, twelve cells): the static half of the scene is a constant image,
    so it is baked through MuJoCo itself and only the movers are ray-cast. Interior-pixel agreement is
    exactly 1.000000 wherever the metric is defined.
  * **VISUAL** (``DMControlTrackCom``, six cells): with the camera on the body there is no static
    half to bake — hide every mover and the remaining frame still moves by 51-220/255 — so the whole
    scene goes to MJWarp every frame. Mean 3.5-8.1/255 against ``physics.render``, and the residual is
    two named texture mechanisms. These are NOT the official pixels; see ``DMControlTrackCom``.

WHAT THE STATE IS, AND WHY IT IS NOT ALWAYS ``qpos``
----------------------------------------------------

Measured, not asserted. Over 17 domains at 84 px: re-rendering
the same ``qpos`` changes 0 pixels, scaling ``qvel`` by 100 changes 0 pixels, reaching the same
``qpos`` after a different history changes 0 pixels, and EVERY ``qpos`` entry moves pixels in every
domain.

Within an episode that makes the state ``qpos``. Across episodes it does not, on four domains, because
``initialize_episode`` writes the episode's draw into the MODEL rather than into ``qpos``: ``reacher``
moves a target geom, ``finger-turn`` a target site, ``swimmer`` a target geom AND a target light.
Restoring ``qpos`` alone into a different episode leaves 171-219/255 of disagreement over 0.6-90% of
the frame; restoring ``qpos`` plus those 2-4 floats is **0/255** on every one. So the state is ``qpos`` plus the episode's draw, the draw is
DISCOVERED by diffing episodes rather than transcribed (``_episode_slots``), and ``state_names`` names
each entry.

The public :data:`~stable_envs.State` is that same row, grouped by what the two halves ARE:
``{"qpos": (nq,), "episode": (k,)}``, with the ``episode`` key absent on the eleven cells that draw
nothing. Nothing is hidden and nothing is reordered — ``pack``/``unpack`` round-trip the frozen corpus
layout bit for bit.

NO DYNAMICS
-----------

A declaration rather than an omission: these cells ship a renderer over ``dm_control``'s own physics
and there is no batched simulator in this package for any of them, so ``step``/``transition`` raise
``Env``'s precise refusal and ``has_dynamics`` is False. ``reset`` and ``render`` always work; to
produce trajectories, roll out the official CPU env and render the states it gives you.

The long-form research notes — why the plate + ray-cast design, where the parity residual lives, why
the interior claim does not hold at 256 px, and why MJWarp is not used for the fixed-camera cells —
are in ``docs/research/dmcontrol.md`` and in the modules that implement each of them.
"""

from __future__ import annotations

# The renderer modules and the base, re-exported under the names the package had.
# The domain modules. Importing them is what REGISTERS the eighteen cells, so the list is the
# registry: a domain not named here does not exist as far as ``stable_envs`` is concerned.
from . import (  # noqa: F401  # noqa: F401
    _cells,
    _model,
    _shading,
    _tendon,
    acrobot,
    ball_in_cup,
    base,
    cartpole,
    cheetah,
    finger,
    hopper,
    humanoid,
    manipulator,
    pendulum,
    point_mass,
    quadruped,
    reacher,
    suite,
    swimmer,
    trackcom,
    walker,
)
from ._cells import (  # noqa: F401
    _AT,
    _AT_MEASURED_WITH,
    _AT_VERSIONS,
    DOMAINS,
    PARITY_NOTE,
    TRACKCOM_DOMAINS,
    VISUAL_NOTE,
    parity_cell,
    trackcom_cell,
)
from ._model import (  # noqa: F401
    _COMPONENT,
    _ENV_CACHE,
    _GRID_MASKS,
    _GRID_SLOTS,
    _SCENE_CACHE,
    _SCENE_CACHE_MAX,
    _TC_CACHE,
    _TC_CACHE_MAX,
    EPISODE_ARRAYS,
    HIDDEN_GROUP,
    LIGHT_DIRECTIONAL,
    MJW_CHUNK,
    N_DRAWS,
    SPECULAR_SCALE,
    VISIBLE_GROUPS,
    _check_dm_control_works,
    _effective_alpha,
    _episode_slots,
    _grid_defaults,
    _grid_masks,
    _mat_rgb,
    _mat_spec_shin,
    _row_width,
    _slot_element,
    _slot_name,
    _suite,
    _tex_rgb,
)
from ._shading import _dot, _light_geometry, _Lights, _norm, _Scene, _shade, _upsample_rgb  # noqa: F401
from ._tendon import (  # noqa: F401
    _CAT_TOL,
    _MJMINVAL,
    _cat_residual,
    _chain_capsules,
    _chain_capsules_batch,
    _cosh_sinh,
    _find_tendon,
    _solve_catenary,
    _Tendon,
    _tendon_frame,
    _tendon_polyline,
    catenary,
)
from .acrobot import DMCAcrobotSwingup  # noqa: F401
from .ball_in_cup import DMCBallInCupCatch  # noqa: F401
from .base import _DMCModel  # noqa: F401
from .cartpole import DMCCartpoleBalance, DMCCartpoleSwingup  # noqa: F401
from .cheetah import DMCCheetahRun  # noqa: F401
from .finger import ALL_EXACT_FLOOR, DMCFingerSpin, DMCFingerTurnEasy, DMCFingerTurnHard  # noqa: F401
from .hopper import DMCHopperHop  # noqa: F401
from .humanoid import DMCHumanoidStand  # noqa: F401
from .manipulator import (  # noqa: F401
    _SWM_SMALL_TARGET,
    RENDER_INVARIANT_DIMS,
    SHAPE_KNOB_SPECS,
    STATE_DIMS_NOTE,
    DMCManipulatorBringBall,
)
from .pendulum import DMCPendulumSwingup  # noqa: F401
from .point_mass import DMCPointMassEasy  # noqa: F401
from .quadruped import (  # noqa: F401
    _QUADRUPED_AT,
    _QUADRUPED_AT_MEASURED_WITH,
    QUADRUPED_NOTE,
    DMCQuadrupedWalk,
    _swm_quadruped_env,
    _SWMQuadrupedModel,
)
from .reacher import DMCReacherEasy, DMCReacherHard  # noqa: F401
from .suite import DMControlSuite  # noqa: F401
from .swimmer import (  # noqa: F401
    _SWIMMER_AT,
    _SWIMMER_AT_MEASURED_WITH,
    _SWIMMER_AT_VERSIONS,
    SPECULAR_NOTE,
    TRACKCOM_RENDER_FLAGS,
    DMCSwimmerSwimmer6,
)
from .trackcom import DMControlTrackCom, _tc_context, _tc_frame, _TCtx  # noqa: F401
from .walker import DMCWalkerWalk  # noqa: F401

#: Cells whose registered SWM default is rendered at something other than the family's 84 px, so the
#: size the conformance suite builds them at is the size they are certified at. COLLECTED from the
#: cells rather than declared beside them: the fact is "this cell is certified at 224 px", and the
#: cell is where a reader looks for what a cell claims.
DEFAULT_SIZES: dict[str, int] = {
    n: int(c.default_size)
    for n, c in sorted((n, globals()[n]) for n in DOMAINS)
    if int(c.default_size) != _DMCModel.default_size
}

#: Cells refused on ``device="cpu"``, with the reason. Collected from ``CPU_UNCERTIFIED_REASON`` for
#: the same reason ``DEFAULT_SIZES`` is collected; ``DMControlTrackCom`` reads the cell's own
#: attribute and never this mapping, so the two cannot drift.
CPU_UNCERTIFIED: dict[str, str] = {
    n: c.CPU_UNCERTIFIED_REASON
    for n, c in sorted((n, globals()[n]) for n in (*DOMAINS, *TRACKCOM_DOMAINS))
    if getattr(c, "CPU_UNCERTIFIED_REASON", "")
}

#: Extra bases per trackcom cell, for the one whose model is not ``suite.load``'s. Collected, so it
#: cannot disagree with the class it describes.
TRACKCOM_BASES: dict[str, tuple[type, ...]] = {
    n: tuple(b for b in globals()[n].__bases__ if b is not DMControlTrackCom)
    for n in TRACKCOM_DOMAINS
    if len(globals()[n].__bases__) > 1
}

#: Every trackcom cell's measured numbers, in one table, for the certifier and the declaration tests.
#: The entries live in the domain modules; this is a view of them, keyed as the original table was.
VISUAL_FIDELITY: dict[str, dict] = {n: globals()[n].visual_fidelity for n in TRACKCOM_DOMAINS}
