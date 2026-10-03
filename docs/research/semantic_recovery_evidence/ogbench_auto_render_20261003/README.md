# Cube/Scene automatic renderer selection accepted

All four independent device gates passed in array 6966805: 112 cases with exact
regression images and task/State outputs. Both SLURM tasks completed successfully.
The integrated 142-file source snapshot matches the tested manifest. The CPU
automatic selection is now activated; process routing remains local pending
its separate throughput and default-construction gates.

The previous public CPU default selected the CUDA-only mesh renderer even though
an explicit, verified software OpenGL renderer exists. The candidate selects
that existing renderer for backend="auto" on CPU. Explicit GPU-only requests
still fail on CPU; CUDA automatic selection and all rendering implementations
remain unchanged. No State schema or stepping API changes.

The local suite passes 13 tests with no skips. It checks the named State boundary,
CPU dispatch and explicit backend refusal, configuration and grouping, and owned
outputs. A pre-existing mock fixture depended on import order; it now explicitly
binds its mocked pixel validator. This is not a GPU image acceptance claim.

Each registration's Oscar gate compares 28 cases against the frozen prior renderer:
full and compact State at 64px and native resolution, output buffers, default and
varied models, one/eight actions, default and sparse/reverse/empty selectors,
exact task and State outputs, and retained frame ownership. CPU uses software GL;
CUDA checks the unchanged default mesh path. Six alternating-order resident-State
render samples per mode record mean/SD. These are bounded regression measurements,
not a new upstream fidelity certification or large-batch performance table.

The first array 6966778 failed before frame checks: Xvfb announced a display again
after the readiness pipe had closed. Both raw failures are preserved. Array
6966805 uses the same production candidate, keeping the pipe open until server
exit. Do not infer acceptance from submission or SLURM status; inspect reports.

To reconstruct each frozen source, start from base f2d918743f5bc6727f77fc978b5d78730ea59083,
apply the version's source_overrides.json, and check every source_manifest.json
file hash. The control renderer is that base commit's manipulation_renderer.py,
with its independent digest in the manifest. The source checkout is under code/;
place the control under reference/ and manifest beside code/. run.sh records the
complete cluster launch. Existing accepted evidence remains untouched.

Reproduce comparison.csv with summarize_ogbench_render_defaults.py accepted
NEW_OUTPUT --source FROZEN_STAGE/code. The strict summarizer checks all four
reports, exact case coverage, renderer device proof, ownership, source/control
hashes, six finite samples per mode and the reported mean/SD. Timing is B2 at
native image resolution, with resident State and warmed renderers. It is not
an online pipeline benchmark or a speedup against the original simulator.

The semantic matrix remains 39 VERIFIED / 3 INCOMPLETE. This closes default
renderer selection, not another semantic promotion or renderer-fidelity claim.
