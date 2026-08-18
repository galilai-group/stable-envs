"""MinAtar: what is the content, and what is the reference?"""
import re
import jax, numpy as np, pgx

for eid in ("minatar-breakout", "minatar-asterix", "minatar-freeway",
            "minatar-seaquest", "minatar-space_invaders"):
    env = pgx.make(eid)
    st = jax.jit(env.init)(jax.random.PRNGKey(0))
    obs = np.asarray(st.observation)
    print(f"{eid:26s} observation={obs.shape} {obs.dtype} "
          f"bytes={obs.size * obs.dtype.itemsize:6d} channels={obs.shape[-1]}")
try:
    import matplotlib
    print("matplotlib", matplotlib.__version__)
except Exception as e:
    print("matplotlib MISSING:", type(e).__name__, e)
import inspect
from pgx.minatar import utils
src = inspect.getsource(utils.visualize_minatar)
print("\nvisualize_minatar uses:",
      sorted(set(re.findall(r"(plt\.\w+|ax\.\w+|sys\.\w+)", src))))
