"""Do the Step-5 ``offscreen`` workaround and the Step-6 ``hidden`` representation draw the same frame?

It is not obvious that they should. ``offscreen`` still BUILDS the target geom, transforms it and
casts against it -- it only puts it 10 m away -- so in principle it can still return a reflection in
the floor or block a shadow ray. ``hidden`` removes it from the scene entirely. Whether the first
leaks anything into the picture is a question about this scene's geometry and lighting, and the way
to answer it is to render both and subtract.

The answer matters for the Step-5 semantic conclusion: if they are identical, the downstream arm
recorded under ``offscreen`` is the arm that would be recorded under ``hidden``.

    PYTHONPATH=$PWD MUJOCO_GL=glfw python probe_mode_equivalence.py DMCReacherHard DMCReacherEasy
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np

import stable_envs as se

#: swap_renderer.OFFSCREEN, kept here so this probe does not need the swm-facing harness on the path.
OFFSCREEN = (10.0, 10.0, 0.01)

for name in sys.argv[1:] or ["DMCReacherHard", "DMCReacherEasy"]:
    st = np.load(f"examples/results/step4/states/{name}.npz")["states"][:16]
    for size in (84, 224):
        e = se.make(name, batch=len(st), size=size, device="cpu")
        hidden = e.render(st, variation={"visibility.target": 0}).cpu().numpy()

        # the Step-5 mechanism: overwrite the episode-drawn columns with a far-away position
        off = st.copy()
        for j, (arr, k) in enumerate(type(e)._slots()):
            comp = int(k % 3) if arr.endswith("_pos") else 0
            off[:, e.nq + j] = OFFSCREEN[comp]
        placed_far = e.render(off).cpu().numpy()

        d = np.abs(hidden.astype(int) - placed_far.astype(int)).max(-1)
        print(f"{name:18} {size:4}px  hidden vs offscreen: max={int(d.max())} "
              f"px={int((d > 0).sum())} of {d.size}")
