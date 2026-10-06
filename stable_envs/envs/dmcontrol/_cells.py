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
    "The frame is MJWarp's raytrace of the same scene, not dm_control's own raster. Two named "
    "mechanisms carry the residual, both measured per surface off MuJoCo's segmentation buffer: "
    '(1) the skybox is an 800x4800 gradient with mark="random" specks and MJWarp samples it with one '
    "unfiltered tap, which is 48-89% of the frame on cheetah/hopper/walker at mean 4.1-4.7/255 and is "
    "what holds SSIM down; (2) the checker floor is MINIFIED and MJWarp has no level-of-detail "
    "selection, so where MuJoCo's mipmap shows one grey (far-band luminance spread 0.1 on humanoid) "
    "MJWarp paints full contrast (9.7). Suitable for generating training data live -- the frame is a "
    "function of the state and the appearance knobs are per-environment tensors. NOT a substitute for "
    "a stored dataset when a policy will be evaluated on official dm_control pixels: the train and "
    "eval pixel distributions differ by the amount above and nothing here measures what that costs a "
    "policy. Closing it needs plane-texture level-of-detail in mujoco_warp's render kernel, which "
    "needs ray differentials it does not compute."
)

PARITY_NOTE = (
    "parity_interior_min is NOT relaxed: interior-pixel agreement measures exactly 1.000000 -- 0 bad "
    "pixels out of 9,834,291 -- over 201 independent draws of 64 states at the shipped "
    "default_supersample = 1. What is declared is parity_all_exact_min, and the reason is arithmetic "
    "rather than a tolerance -- all-pixel exactness is interior exactness WEIGHTED BY HOW MUCH OF THE "
    "FRAME MOVES, and finger-turn_easy's seven movers cover 11.7% of an 84 px frame against "
    "cartpole's 2.5%. Measured over those 201 draws: all-pixel 0.9103 at WORST and 0.9119 median, "
    "i.e. it CLEARS the shared 0.90 bar on every draw -- the floor is declared so that the margin is "
    "the scene's rather than luck's, and it is set from a draw count rather than from one large "
    "evaluation, because a rare pixel event reported off a single draw is how MujocoPusher came to "
    "declare a floor it then failed 15 times in 101. 100.0% of every disagreeing pixel is inside the "
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
                    f'DM Control {domain}-{task} at camera {camera}, a mode="trackcom" camera.\n\n'
                    f"VISUAL TIER -- these are NOT the official pixels. The frame is MJWarp's raytrace of "
                    f"dm_control's own scene at dm_control's own camera, which lands on MuJoCo's "
                    f"cam_xpos to 1e-8 m; measured against physics.render it is mean "
                    f"{vf['mean_abs']:.2f}/255 with {100 * vf['within8_frac']:.1f}% of pixels within "
                    f"8/255 and SSIM {vf['ssim']:.3f}. The residual is the unfiltered speckled skybox and "
                    f"the minified checker floor. Good for generating training data live; not "
                    f"interchangeable with a stored dataset of official pixels. See visual_note."
                ),
            ),
        )
    )
    TRACKCOM_DOMAINS[cname] = (domain, task, camera)
    return cls
