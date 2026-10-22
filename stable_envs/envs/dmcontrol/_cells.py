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
