"""DM Control renderers and independently recovered historical task semantics.

Each domain module declares its public class identity. Verified implementations
compose the preserved renderer with independent authoritative CPU simulators;
CUDA selects full-State and frame placement, not GPU physics. See the semantic
recovery matrix for exact task scopes. Unrecovered cells refuse public simulator
State/reset/step rather than returning compact renderer projections.

Full State includes each task's experimentally established continuation fields.
The private renderer representation remains qpos plus any episode model draw;
it is never sufficient evidence for simulator restoration. Renderer provenance
and historical measurements remain in the domain modules and research reports.
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
from .acrobot import DMCAcrobotBalance  # noqa: F401
from .ball_in_cup import DMCBallInCupCatch  # noqa: F401
from .base import _DMCModel  # noqa: F401
from .cartpole import DMCCartpoleBalance, DMCCartpoleSwingup  # noqa: F401
from .cheetah import DMCCheetahRun  # noqa: F401
from .finger import ALL_EXACT_FLOOR, DMCFingerSpin, DMCFingerTurnEasy, DMCFingerTurnHard  # noqa: F401
from .hopper import DMCHopperHop  # noqa: F401
from .humanoid import DMCHumanoidStand  # noqa: F401
from .manipulator import (  # noqa: F401
    _REGISTERED_SMALL_TARGET,
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
    _transcribed_quadruped_env,
    _TranscribedQuadrupedModel,
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

#: Cells whose certified default is rendered at something other than the family's 84 px, so the
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
