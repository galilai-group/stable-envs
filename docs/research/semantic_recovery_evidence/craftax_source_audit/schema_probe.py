import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/craftax-mpl")
import dataclasses
import importlib.metadata as md
import json
import platform
import time
from pathlib import Path

import jax
import numpy as np

h = Path("/private/tmp/stable-envs-semantic-recovery/docs/research/semantic_recovery_evidence/craftax_source_audit")
profile = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "versions": {
        n: md.version(n)
        for n in ["craftax", "jax", "jaxlib", "flax", "gymnax", "chex", "numpy", "torch", "tensordict"]
    },
    "devices": [str(x) for x in jax.devices()],
    "jax_enable_x64": jax.config.jax_enable_x64,
    "jax_default_prng_impl": jax.config.jax_default_prng_impl,
    "jax_threefry_partitionable": jax.config.jax_threefry_partitionable,
}
(h / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
from craftax.craftax_env import make_craftax_env_from_name

result = {}
for classic in [True, False]:
    name = "Craftax-" + ("Classic-" if classic else "") + "Symbolic-v1"
    print("BEGIN", name, flush=True)
    env = make_craftax_env_from_name(name, False)
    t = time.perf_counter()
    obs, s = env.reset(jax.random.PRNGKey(0))
    jax.block_until_ready((obs, s))
    print("RESET", time.perf_counter() - t, flush=True)

    def schema(obj, prefix=()):
        if dataclasses.is_dataclass(obj):
            return sum([schema(getattr(obj, f.name), prefix + (f.name,)) for f in dataclasses.fields(obj)], [])
        if isinstance(obj, tuple):
            return sum([schema(v, prefix + (str(i),)) for i, v in enumerate(obj)], [])
        if obj is None:
            return [{"key": list(prefix), "constant": None}]
        a = np.asarray(obj)
        return [
            {
                "key": ["engine", *prefix],
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "min": float(a.min()),
                "max": float(a.max()),
                "bytes": a.nbytes,
            }
        ]

    fam = "Classic" if classic else "Full"
    result[fam] = {
        "env_class": type(env).__module__ + "." + type(env).__name__,
        "params": dataclasses.asdict(env.default_params),
        "static": dataclasses.asdict(env.static_env_params),
        "schema": schema(s),
        "num_actions": env.num_actions,
        "symbolic_shape": list(obs.shape),
        "reset_cold_seconds": time.perf_counter() - t,
        "key_dtype": str(s.state_rng.dtype),
        "key_shape": list(s.state_rng.shape),
        "key_impl": str(jax.random.key_impl(s.state_rng)),
    }
    (h / "candidate-engine-schema.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        h / (fam + "-initial.npz"),
        **{"/".join(j["key"]): np.asarray(getattr(s, j["key"][-1])) for j in []},
        symbolic=np.asarray(obs),
    )
    print("SCHEMA", fam, len(result[fam]["schema"]), sum(x.get("bytes", 0) for x in result[fam]["schema"]), flush=True)
print("CRAFTAX_SCHEMA_PROBE_PASS", flush=True)
