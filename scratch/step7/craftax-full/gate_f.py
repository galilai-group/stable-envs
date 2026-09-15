"""Gate F: the shape and info contract this cell offers a stable_worldmodel consumer.

``swm/CraftaxPixels-v1`` is registered at ``obs_shape=(130, 110, 3)`` -- hard-coded in
``stable_worldmodel/envs/craftax/craftax.py`` because craftax's own declared ``observation_space``
was not trusted -- and its observation is ``render_craftax_pixels(state) / 255``. What this records is
that the facade's frame is that shape, that its ``info`` carries the keys the shard expects, and that
``step`` raises by design: this env ships a renderer and no batched simulator, so its Gate F path is
``swap_renderer``, which keeps the official env's dynamics and replaces only its pixels.

``stable_worldmodel`` is in no local venv, so the run through ``examples/swm_compat/`` belongs to the
Oscar leg; everything checkable without it is checked here and the swm side is read from the source
tree at ``/Users/samib/stable-worldmodel``.

    PYTHONPATH=. python scratch/step7/craftax-full/gate_f.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "examples", "results", "step7", "craftax-full", "gate_f")
SWM = "/Users/samib/stable-worldmodel"


def swm_contract():
    p = os.path.join(SWM, "stable_worldmodel", "envs", "craftax", "craftax.py")
    if not os.path.exists(p):
        return {"error": f"stable_worldmodel source not found at {SWM}"}
    src = open(p).read()
    m = re.search(r"class CraftaxPixelsWrapper.*?obs_shape=\((\d+), (\d+), (\d+)\)", src, re.S)
    reg = os.path.join(SWM, "stable_worldmodel", "envs", "__init__.py")
    return {
        "source": p,
        "env_name": "Craftax-Pixels-v1" if "Craftax-Pixels-v1" in src else None,
        "auto_reset": "auto_reset=False" in src,
        "obs_shape": [int(m.group(i)) for i in (1, 2, 3)] if m else None,
        "obs_is_render_over_255": "np.clip(self._last_obs, 0.0, 1.0) * 255.0" in src,
        "registered_id": "swm/CraftaxPixels-v1" in open(reg).read(),
        "step_is_jitted": "jax.jit(_step" in src,
        "action_repeat": None,
    }


def main():
    from stable_envs.compat.worldmodel import StableEnvGym
    from stable_envs.envs.craftax_full import CraftaxFull

    rec = {"swm": swm_contract()}
    e = StableEnvGym("CraftaxFull", device="cpu")
    obs, info = e.reset(seed=0)
    frame = e.render()
    rec["facade"] = {
        "obs_shape": list(np.asarray(obs).shape), "obs_dtype": str(np.asarray(obs).dtype),
        "render_shape": list(frame.shape), "render_dtype": str(frame.dtype),
        "info_keys": sorted(info.keys()),
        "observation_space": str(e.observation_space),
    }
    try:
        e.step(0)
        rec["facade"]["step"] = "returned -- unexpected for a renderer-only cell"
    except NotImplementedError as ex:
        rec["facade"]["step"] = f"NotImplementedError (by design): {str(ex)[:160]}"
    rec["shapes_agree"] = (rec["swm"].get("obs_shape") == rec["facade"]["render_shape"])
    rec["note"] = ("the facade's observation is the 544-slot render window, not the picture; the "
                   "PICTURE is what must match swm, and render() returns exactly its (130, 110, 3). "
                   "step() raises because this cell has no batched simulator -- the Gate F path is "
                   "swap_renderer, as for CraftaxClassic.")
    rec["owed_on_oscar"] = ("the run through examples/swm_compat/run_swm_workflow.py, which needs "
                            "stable_worldmodel; it is in $SCR/venvs/se-step5 on Oscar and in no "
                            "local venv")
    for size, want in ((10, [130, 110, 3]), (16, [208, 176, 3])):
        env = CraftaxFull(batch=2, device="cpu", size=size)
        got = list(env.render(CraftaxFull.sample_states(env, 2)).shape[1:])
        rec.setdefault("sizes", {})[str(size)] = {"got": got, "want": want, "ok": got == want}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "gate_f_local.json")
    with open(p, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    sys.exit(main())
