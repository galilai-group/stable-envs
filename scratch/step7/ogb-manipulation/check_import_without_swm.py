"""Import-without-stable_worldmodel guard, run in se6-venv where swm is genuinely absent."""
import sys
sys.path.insert(0, "/Users/samib/stable-envs-step7-ogb-manipulation")
import stable_envs as se
import stable_envs.envs as E
assert "stable_worldmodel" not in sys.modules, "this check is meaningless if swm is importable"
try:
    import stable_worldmodel  # noqa
    raise SystemExit("SKIP: stable_worldmodel is importable in this interpreter")
except ImportError:
    pass
err = E.import_errors().get("ogb_manip")
assert err is None, f"ogb_manip must import without stable_worldmodel; got {err}"
assert "OGBCube" in se.REGISTRY and "OGBScene" in se.REGISTRY, "both ids must register"
assert len(se.REGISTRY) == 78, f"registry should be 78, got {len(se.REGISTRY)}"
for n in ("OGBCube", "OGBScene"):
    c = se.REGISTRY[n]
    assert c.fidelity_tier == "visual" and c.visual_fidelity["mean_abs"] > 0
    assert "VISUAL TIER" in (c.__doc__ or "").upper()
print("OK: registry 78, both cells registered, no stable_worldmodel needed")
