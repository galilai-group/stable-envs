"""DM Control family: registers every domain class and collects the package-level tables."""

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
    DOMAINS,
    PARITY_NOTE,
    TRACKCOM_DOMAINS,
    VISUAL_NOTE,
    parity_cell,
    trackcom_cell,
)
from ._model import (  # noqa: F401
    _ENV_CACHE,
    _SCENE_CACHE,
    _TC_CACHE,
    EPISODE_ARRAYS,
    HIDDEN_GROUP,
    LIGHT_DIRECTIONAL,
    N_DRAWS,
    SPECULAR_SCALE,
    VISIBLE_GROUPS,
    _episode_slots,
)
from ._shading import _shade  # noqa: F401
from ._tendon import (  # noqa: F401
    _MJMINVAL,
    _chain_capsules,
    _chain_capsules_batch,
    _cosh_sinh,
    _solve_catenary,
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
    RENDER_INVARIANT_DIMS,
    SHAPE_KNOB_SPECS,
    STATE_DIMS_NOTE,
    DMCManipulatorBringBall,
)
from .pendulum import DMCPendulumSwingup  # noqa: F401
from .point_mass import DMCPointMassEasy  # noqa: F401
from .quadruped import (  # noqa: F401
    QUADRUPED_NOTE,
    DMCQuadrupedWalk,
    _transcribed_quadruped_env,
    _TranscribedQuadrupedModel,
)
from .reacher import DMCReacherEasy, DMCReacherHard  # noqa: F401
from .suite import DMControlSuite  # noqa: F401
from .swimmer import SPECULAR_NOTE, TRACKCOM_RENDER_FLAGS, DMCSwimmerSwimmer6  # noqa: F401
from .trackcom import DMControlTrackCom, _tc_context, _TCtx  # noqa: F401
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
