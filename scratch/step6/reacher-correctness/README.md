# Step 6B probes — Reacher visibility + the fidelity guard

Every number in `examples/results/step6/reacher-correctness/{design,findings}.md` and every
number in the docstrings this patch added comes from one of these. They run on a Mac with
no EGL, so they set `MUJOCO_GL=glfw` and `device="cpu"`; on Oscar drop both.

    PYTHONPATH=$PWD MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step6/reacher-correctness/<probe>.py [args]

| probe | what it answers |
|---|---|
| `probe_hide.py` | does MuJoCo's OWN renderer stop drawing an element when we zero its own rgba alpha, and when we zero its material's? (both: 19 px / 43 px changed, and per-element is the finer instrument) |
| `probe_alpha.py` | on every parity cell, is each visible element's alpha governed by its material or by its own rgba? |
| `probe_slots.py` | which elements does each cell's `initialize_episode` DRAW — i.e. which cells get a `visibility.*` leaf |
| `probe_perrow.py` | per-row visibility, plus permutation, sub-batch and render-purity invariance under a per-row style |
| `probe_guard.py` | `check_task_object_visibility` over independent draws and over the frozen Step-4 corpus — the numbers `TASK_OBJECT_PRESENCE_MARGIN` is set from |
| `probe_checks.py` | the mandatory conformance battery on the affected cells, on CPU |
| `local_digests.py` | default-style frame digests on the frozen corpora, for the before/after pixel-identity argument |
| `probe_static_leak.py` | the review-found render-purity bug: does a cached scene's STATIC primitive set come from its own bake or from the last one? (before the fix: 18 px / max 215 on DMCReacherHard, 88 px / max 140 on DMCFingerTurnHard; after: 0) |
| `probe_mode_equivalence.py` | are the Step-5 `offscreen` workaround and the Step-6 `hidden` representation the same picture? (max \|d\| 0 at 84 and 224 px on both reacher cells) |
