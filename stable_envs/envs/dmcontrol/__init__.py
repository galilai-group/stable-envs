"""DM Control classes combining renderers with optional CPU simulation adapters.

CUDA selects State and frame placement, not GPU physics. Public simulator
State includes continuation fields; qpos and episode-model render projections
cannot restore it. Domain imports register classes and supply fidelity metadata.
"""

from __future__ import annotations

# Import domain declarations to register their classes, then expose them at the package path.
from . import (  # noqa: F401
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
from .humanoid import DMCHumanoidWalk  # noqa: F401
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

# Nonstandard declared output sizes, collected from the parity classes.
DEFAULT_SIZES: dict[str, int] = {
    n: int(c.default_size)
    for n, c in sorted((n, globals()[n]) for n in DOMAINS)
    if int(c.default_size) != _DMCModel.default_size
}

# Explicit CPU refusals, collected from each class rather than maintained separately.
CPU_UNCERTIFIED: dict[str, str] = {
    n: c.CPU_UNCERTIFIED_REASON
    for n, c in sorted((n, globals()[n]) for n in (*DOMAINS, *TRACKCOM_DOMAINS))
    if getattr(c, "CPU_UNCERTIFIED_REASON", "")
}

# Additional mixins composed before the shared trackcom renderer.
TRACKCOM_BASES: dict[str, tuple[type, ...]] = {
    n: tuple(b for b in globals()[n].__bases__ if b is not DMControlTrackCom)
    for n in TRACKCOM_DOMAINS
    if len(globals()[n].__bases__) > 1
}

# Combined fidelity declarations for measurement tools and declaration checks.
VISUAL_FIDELITY: dict[str, dict] = {n: globals()[n].visual_fidelity for n in TRACKCOM_DOMAINS}
