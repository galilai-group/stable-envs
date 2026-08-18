"""Final state check: what is registered, what is not, and the headline numbers."""
import sys
import numpy as np
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import stable_envs as se
from stable_envs.envs import pgx as P
from stable_envs.testing import conformance as C

print(f"registry size (se-pgx venv) = {len(se.REGISTRY)}")
print(f"pgx module import error     = {se.envs.import_errors().get('pgx', 'NONE - imported ok')}")
print(f"registered pgx ids          = {sorted(n for n in se.REGISTRY if n.startswith('Pgx'))}")
print(f"built but NOT registered    = {P.NOT_REGISTERED}")
print(f"PgxGo19x19 constructible    = {hasattr(P, 'PgxGo19x19')}, in REGISTRY = {'PgxGo19x19' in se.REGISTRY}")
print()
for nm in ("PgxTicTacToe", "PgxConnectFour", "PgxOthello", "PgxGo9x9", "PgxGo19x19"):
    cls = getattr(P, nm)
    env = cls(batch=8, device="cuda")
    st = env.sample_states(16, rng=np.random.default_rng(0))
    off = cls.reference_render(cls.reference_env(size=env.size), st)
    r = C.pixel_agreement(env.render(st).cpu().numpy(), off)
    reg = "REGISTERED    " if nm in se.REGISTRY else "not registered"
    print(f"{nm:16s} {reg} size={env.size:4d} dim={env.state_dim:3d} "
          f"interior={r['interior_exact_frac']:.6f} all={r['exact_frac']:.6f} "
          f"mean={r['mean_abs']:.3f} style_dim={env.style_dim} knobs={env.variation_space.names()}")
