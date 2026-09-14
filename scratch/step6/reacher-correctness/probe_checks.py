import os, sys, json
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np, stable_envs as se
from stable_envs.testing import conformance as C

for name in sys.argv[1:]:
    e = se.make(name, batch=8, device="cpu")
    st = np.load(f"examples/results/step4/states/{name}.npz")["states"][:8]
    sty = C.uniform_style(e, len(st), groups=2, seed=0)
    e.rng = np.random.default_rng(4041)
    r = {}
    r["variations_do_something"] = C.check_variations_do_something(e)
    e.set_style(None); e.rng = np.random.default_rng(4041)
    r["per_env_variation"] = C.check_per_env_variation(e)
    e.set_style(None); e.rng = np.random.default_rng(4041)
    r["style_is_state"] = C.check_style_is_state(e)
    e.set_style(None); e.rng = np.random.default_rng(4041)
    r["default_style_is_official"] = C.check_default_style_is_official(e)
    e.set_style(None); e.rng = np.random.default_rng(4041)
    r["state_dims_matter"] = C.check_state_dims_matter(e)
    e.set_style(None); e.rng = np.random.default_rng(4041)
    r["order_independence"] = C.check_order_independence(e, st, sty)
    e.set_style(None); e.rng = np.random.default_rng(4041)
    r["batch_size_invariance"] = C.check_batch_size_invariance(e, st, sty)
    e.set_style(None); e.rng = np.random.default_rng(4041)
    r["render_is_pure"] = C.check_render_is_pure(e, st, sty)
    e.set_style(None); e.rng = np.random.default_rng(4041)
    r["colors_apply_exactly"] = C.check_colors_apply_exactly(e)
    print(name, json.dumps(r, default=str))
