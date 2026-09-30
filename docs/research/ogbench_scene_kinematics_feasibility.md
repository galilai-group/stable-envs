# Scene rendering kinematics feasibility (2026-09-29)

This is a read-only follow-on investigation. The first Scene performance
candidate in `manipulation_opengl.py` is unchanged. The probe uses CPU MuJoCo
math and `MjvScene` construction on the local Mac; it creates no OpenGL context
and provides no GPU throughput or pixel-parity claim.

## Proposed minimum for the pinned scene

For the currently compiled Scene model and fixed default render options,
`mj_kinematics(model, data)` followed by `mj_camlight(model, data)` is sufficient
in the probe to prepare the scene after installing qpos and mocap poses.

Compiled model inspection found two fixed cameras, two fixed lights, no flexes,
no skins, no plugins, and one fixed joint tendon (two `mjWRAP_JOINT` entries).
There are no spatial tendon segments to draw. The registered `front_pixels`
camera and light modes remain fixed under the supported appearance changes.

MuJoCo 3.12.0 separates forward kinematics from inertia assembly, collisions,
constraint construction, transmission, velocity, actuation, and solving. The
latter operations are work performed by full forward dynamics that the fixed
RGB view does not request. See the pinned [forward implementation](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_forward.c).

The fixed camera/light modes consume body transforms. CoM-tracking modes also
need subtree centers of mass, so a general replacement would require
`mj_comPos`. This proposal is deliberately limited to this registered fixed
view. See [camera/light updates](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_core_smooth.c).

Spatial tendon visualization reads wrapping results; diagnostic displays such
as contacts and actuator forces consume additional physics outputs. Those
would invalidate a blanket claim that all MuJoCo rendering needs only two
calls. The registered default options and inspected model exclude these
requirements. See [scene construction](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_vis_visualize.c).

## Local evidence

`semantic_recovery_evidence/scene_kinematics_feasibility_20260929/probe.py` and
`result.json` preserve the experiment. The exact package versions are
MuJoCo 3.12.0 and OGBench 1.2.1. The cached OGBench source used is
`/Users/samib/.cache/uv/archive-v0/3TuLJmtHKIeo8JlKfw-FV/ogbench`.

For 128 normalized random qpos/mocap configurations, full-forward and minimal
updates produced identical values for all 1,110 inspected scene geom, light,
and camera attributes. Candidate qpos was not modified. The same candidate
Data was reused across configurations to exercise stale-data hazards.

Ten repetitions of those poses measured update-only means of 0.156905 ms
(SD 0.002585) for full forward and 0.003071 ms (SD 0.000133) for the two calls.
These local CPU microtimings include qpos/mocap array installation. They are
neither end-to-end rendering FPS nor an RTX 3090 prediction. This randomized
probe does not replace the accepted saved-State corpus or GPU image gate.

## Follow-on experiment

Use a renderer-private helper to set the final button geom and handle material
colors directly from `buttons` and `lock_color`, preserving the existing
CompatibleScene color rule. The upstream helper also changes joint damping
and invokes full forward; damping cannot change a rendered already-installed
qpos, and this private rendering model never steps. Do not modify upstream
simulation/compatibility helpers or public State schemas.

After pose/color installation, run the two kinematics calls. Assert the model
and visualization preconditions above, or fall back to full forward when
preconditions are not met. Test all four button states, color variations,
camera/light variations, and drawer/window/cube motion. Compare resulting
scene attributes and complete rendered frames to the first candidate and
source; run unprofiled B64/128/256 timings on CPU and RTX 3090. Do not cache
rendered poses or images merely because the timing corpus repeats.

## Honest preparation boundary

The current paired harness prepares one source instance per saved pose before
timing. `source_instance` in `profile_all.py` calls numeric source `install`,
including pose reconstruction/forward, and validates exact State recapture.
The source timer subsequently calls each already-prepared instance's public
render method and stacks/copies its result.

Our RenderState projection and disk loading also happen before timing, but
our public batched renderer installs qpos/mocap into its cached rendering Data
and derives transforms inside the timer. Thus both consume saved states and
produce matched images, but the timer includes pose preparation on our side
that has already happened on the source side. This should be disclosed as an
API-level rendering comparison against a prepared source, not described as
identical preparation boundaries or end-to-end simulation performance.

Keep the existing conservative comparison visible. If reporting a second
boundary, either time source installation plus rendering as a separate metric,
or expose an explicitly prepared renderer input and compare draw-only paths.
Neither change should silently replace existing numbers or exclude our
transform work while continuing to call the result arbitrary-State rendering.

## Completed follow-on

The separately staged candidate subsequently passed full-image GPU and CPU gates at B64/128/256 and was adopted. Source-normalized MSE is0 for the final CPU-output measurements, and baseline/candidate GPU frames differ by at most1 uint8 level. The final implementation keeps full-forward fallback for unsupported features. See `semantic_recovery_evidence/scene_render_kinematics_20260929/` and `renderer_performance_followup_20260929.md`; the earlier local update-only timings remain feasibility evidence, not rendering FPS.
