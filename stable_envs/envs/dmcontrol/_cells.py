from __future__ import annotations

from ...registry import register
from .suite import DMControlSuite
from .trackcom import DMControlTrackCom

# Generated classes are re-exported from this package, so their module path remains importable.
_MODULE = __name__.rsplit(".", 1)[0]

# Class name to (domain, task, camera), populated by registration below.
DOMAINS: dict[str, tuple[str, str, int]] = {}
# The corresponding table for trackcom renderers.
TRACKCOM_DOMAINS: dict[str, tuple[str, str, int]] = {}

_AT = (
    "84 px, 64 states per domain from sample_states (the domain's own initialize_episode plus a "
    "0-40 step random-action rollout), default appearance, H200 / driver 590.48.01 / MUJOCO_GL=egl, "
    "mujoco 3.11.0 + mujoco-warp 3.11.0, against dm_control's physics.render at camera 0"
)

# Reference pins gate source comparisons; candidate backend versions record measurement provenance.
_AT_VERSIONS = {"mujoco": "3.11.0"}
_AT_MEASURED_WITH = {"mujoco_warp": "3.11.0"}

VISUAL_NOTE = (
    "Body-tracking cameras use analytic rendering with filtered textures, floor reflections, "
    "and source-derived floor lighting. Geometry edges and body shading can still differ from "
    "source OpenGL. The visual_fidelity dictionary records the historical MuJoCo 3.11 benchmark; "
    "current source-pixel acceptance is measured separately by the maintained contract tests."
)

PARITY_NOTE = (
    "parity_interior_min is NOT relaxed: interior-pixel agreement measures exactly 1.000000 -- 0 bad "
    "pixels out of 9,834,291 -- over 201 independent draws of 64 states at the shipped "
    "default_supersample = 1. What is declared is parity_all_exact_min, and the reason is arithmetic "
    "rather than a tolerance -- all-pixel exactness is interior exactness WEIGHTED BY HOW MUCH OF THE "
    "FRAME MOVES, and finger-turn_easy's seven movers cover 11.7% of an 84 px frame against "
    "cartpole's 2.5%. Measured over those 201 draws: all-pixel 0.9103 at WORST and 0.9119 median, "
    "i.e. it CLEARS the shared 0.90 bar on every draw. 100.0% of every disagreeing pixel is inside the "
    "ray-cast mover pass and the baked plate is 1.00000 exact on the pixels it owns; the residual is a "
    "one-level rim on curved silhouettes, where MuJoCo shades a tessellated capsule per vertex and an "
    "analytic per-pixel evaluation lands within a few levels of it. finger-spin (0.9288 worst) and "
    "finger-turn_hard (0.9241 worst) are the same renderer with fewer mover pixels and declare "
    "nothing. Dropping the supersample IMPROVED all three -- at ss=2 the same draws gave 0.9053 / "
    "0.9249 / 0.9200 -- because an aliased edge is bit-identical to MuJoCo wherever MuJoCo's own MSAA "
    "happened to be fully covered, while a blended edge almost never is."
)


def parity_cell(cname: str, domain: str, task: str, camera: int = 0, *, bases: tuple[type, ...] = (), **extra) -> type:
    """Register a fixed-camera renderer class and add its domain tuple to DOMAINS.

    Prepend optional semantic/model mixins; extra attributes carry per-cell
    configuration and declarations.
    """
    cls = register(
        type(
            cname,
            tuple(bases) + (DMControlSuite,),
            dict(
                **extra,
                DOMAIN=domain,
                TASK=task,
                CAMERA=camera,
                name=f"dmc-{domain.replace('_', '-')}-{task.replace('_', '-')}",
                reference_id=f"dm_control/{domain}-{task}",
                __module__=_MODULE,
                __doc__=(
                    f"DM Control {domain}-{task} at camera {camera}. State is qpos plus whatever the "
                    f"episode draws into the model; the frame is a measured pure function of it."
                ),
            ),
        )
    )
    DOMAINS[cname] = (domain, task, camera)
    return cls


def trackcom_cell(
    cname: str,
    domain: str,
    task: str,
    camera: int,
    visual_fidelity: dict,
    render_flags: dict | None = None,
    note_suffix: str = "",
    bases: tuple[type, ...] = (),
    **extra,
) -> type:
    """Register a trackcom renderer class with fidelity metadata and optional mixins.

    Append note_suffix to the shared residual description; record the domain
    tuple in TRACKCOM_DOMAINS.
    """
    vf = visual_fidelity
    flags = dict(render_flags or {})
    note = VISUAL_NOTE + (" " + note_suffix if note_suffix else "")
    cls = register(
        type(
            cname,
            tuple(bases) + (DMControlTrackCom,),
            dict(
                **extra,
                DOMAIN=domain,
                TASK=task,
                CAMERA=camera,
                name=f"dmc-{domain.replace('_', '-')}-{task.replace('_', '-')}",
                reference_id=f"dm_control/{domain}-{task}",
                visual_fidelity=vf,
                RENDER_FLAGS=flags,
                visual_note=note,
                __module__=_MODULE,
                __doc__=(
                    f"DM Control {domain}-{task} at body-tracking camera {camera}.\n\n"
                    "VISUAL TIER: uses MJWarp on CUDA or cpu3d on CPU; source OpenGL is an explicit backend. "
                    "Pixels approximate the source rasterizer. See visual_note for measurement scope."
                ),
            ),
        )
    )
    TRACKCOM_DOMAINS[cname] = (domain, task, camera)
    return cls
