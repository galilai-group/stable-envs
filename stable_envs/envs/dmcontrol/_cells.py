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

VISUAL_NOTE = (
    "Body-tracking cameras use analytic rendering with filtered textures, floor reflections, "
    "and source-derived floor lighting. Geometry edges and body shading can still differ from "
    "source OpenGL. The visual_fidelity dictionary records the measured agreement and its "
    "measurement setup."
)

PARITY_NOTE = (
    "Interior-pixel agreement is 1.000000 (0 of 9,834,291 pixels differ) over 201 draws of 64 states "
    "at default_supersample = 1. All-pixel exactness is lower than on the other fixed-camera cells "
    "because it scales with the moving area: finger-turn_easy's seven movers cover "
    "11.7% of an 84 px frame against cartpole's 2.5%. Over those draws all-pixel exactness is 0.9103 "
    "worst and 0.9119 median. Every disagreeing pixel is in the ray-cast mover pass, and the baked "
    "plate is exact on the pixels it owns. The residual is a one-level rim on curved silhouettes, "
    "where MuJoCo shades a tessellated capsule per vertex and the analytic per-pixel evaluation lands "
    "within a few levels of it. At supersample 2 the finger cells score lower (turn_easy 0.9053, "
    "spin 0.9249, turn_hard 0.9200 worst), because a blended edge rarely matches MuJoCo's MSAA exactly."
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
                    "Uses MJWarp on CUDA or cpu3d on CPU; source OpenGL is an explicit backend. "
                    "Pixels approximate the source rasterizer. See visual_note for measurement scope."
                ),
            ),
        )
    )
    TRACKCOM_DOMAINS[cname] = (domain, task, camera)
    return cls
