"""Controlled independent Cube/Scene oracle shared by active acceptance tests."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def numpy_equal(a, b):
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def load_oracle():
    path = ROOT / "docs/research/semantic_recovery_evidence/ogb_rng_feasibility/ownership.py"
    spec = importlib.util.spec_from_file_location("ogb_rng_design_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_oracle(env, raw, oracle):
    diff = oracle.previous.differences(oracle.previous.engine(env), oracle.previous.engine(raw))
    assert not diff, diff
    for method in ("compute_observation", "get_reset_info", "get_step_info"):
        diff = oracle.previous.differences(getattr(env, method)(), getattr(raw, method)())
        assert not diff, (method, diff)
    for field in ("dof_damping", "site_pos", "geom_rgba", "mat_rgba", "geom_size", "body_mass", "body_inertia"):
        assert np.array_equal(getattr(env.model, field), getattr(raw.model, field)), field
    assert env.np_random.bit_generator.state == raw.np_random.bit_generator.state
    numpy_equal(env.task_rng.get_state(), raw.task_rng.get_state())
    assert env.action_space.np_random.bit_generator.state == raw.action_space.np_random.bit_generator.state
    assert env.variation_parent.bit_generator.state == raw.variation_parent.bit_generator.state
