# Scene render overhead candidate (2026-09-29)

The completed RTX 3090 profile is preserved in
`semantic_recovery_evidence/ogb_work_profile_20260929/retry/30/diagnosis.json`.
Its three instrumented batch-256 renders spent approximately 164 ms/batch in
OpenGL drawing/readback, 60 ms in 512 `mj_forward` calls, 16 ms in renderer
lookup, 5 ms in scene update, and 64 ms in the enclosing renderer excluding
those measured sections. These wall timings include profiling overhead and
must not replace the uninstrumented paired FPS measurements.

The implementation in `stable_envs/envs/ogbench/manipulation_opengl.py` removes
four concrete sources of redundant work:

- Scene called `mj_forward` explicitly after `_apply_button_states`, even though
  pinned OGBench 1.2.1 `SceneEnv._apply_button_states` already calls it. The
  subsequent `CompatibleScene` override changes geom/material colors only.
  Scene now uses the upstream forward once; Cube retains its explicit forward.
- Each image constructed a TensorDict row, hashed five model fields, and looked
  up a renderer. The batch now groups exact bytes of all `MODEL_VALUES`, extracts
  pose/color arrays once, and performs one renderer lookup per configuration.
  It writes each image to its original batch index.
- Each image allocated an RGB array and then copied it, followed by another
  copy in the final stack. MuJoCo's public `Renderer.render(out=...)` now writes
  into a fresh contiguous batch array. Supersampling reuses one scratch image,
  preserving the same arithmetic mean and rounding.
- Installed-package version metadata was reparsed on every render. The pinned
  version check is now cached for the process lifetime.

This changes no geometry, materials, shaders, resolution, supersampling, State
schema, or source simulation logic. It does not remove the serial OpenGL draw
loop, and it still computes one full forward pass per image. The throughput
ceiling should therefore be assessed against the remaining measured draw and
readback time, rather than assuming a large GPU speedup.

Five CPU unit tests in `tests/unit/test_ogbench_opengl_batch.py` passed. They
exercise the actual batching code with graphics and optional environment
packages mocked: original output order across interleaved configurations,
exact byte grouping including signed zero, exactly one forward per image,
both Cube and Scene, supersampling, input nonmutation, and lifetime of returned
image storage. These tests do **not** establish real-render pixel parity.
The same-host RTX 3090 baseline/candidate gate and CPU timing are required
before promoting this candidate in the comparison table. Existing Scene
repeatability can differ by one uint8 intensity; record full-image drift rather
than presuming reordered draws are bit-exact.

Both the first cleanup and the follow-on guarded kinematics update passed the real RTX3090 and software-GL CPU B64/128/256 gates. The adopted implementation includes both changes; see `renderer_performance_followup_20260929.md` and the `scene_render_kinematics_20260929` evidence for final measurements. No quality settings changed.
