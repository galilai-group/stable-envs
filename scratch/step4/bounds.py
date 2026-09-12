"""Where each Step 4 pass/fail bound comes from. Extracted from the file that OWNS it, never retyped.

Gate A is "entirely the existing conformance suite" (Step 2, section 4), which means the bar an env
is held to already exists and lives in ``tests/``. Copying those numbers into the certifier would
create a second copy that can drift from the first silently -- and a drifted copy is worse than no
check, because the report would keep certifying against a bar the suite no longer applies.

So every bound is read out of the source that declares it, at run time:

    tests/gpu/test_conformance.py       INTERIOR_EXACT_MIN, ALL_EXACT_MIN, MEAN_ABS_MAX  (module constants)
                                        the backend-agreement floor                       (inline assert)
    tests/gpu/test_reproducibility.py   DOWNSAMPLE_EXACT_FRAC, DOWNSAMPLE_MARGIN          (module constants)
                                        the two step-vs-reference bounds                  (inline assert)
    the env's own declaration           parity_interior_min, parity_all_exact_min, visual_fidelity
    scratch/certification.json          the representation tripwire, frozen by Step 2

Read by ``ast``, not by import: ``tests/gpu/*`` skips at module level without a CUDA device, and the
certifier is meant to run anywhere. ``tests/unit/test_step4.py`` asserts that what comes out of here
still matches what those files say, so an extraction that goes blind is a test failure rather than a
quietly missing bound.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFORMANCE_TEST = ROOT / "tests" / "gpu" / "test_conformance.py"
REPRO_TEST = ROOT / "tests" / "gpu" / "test_reproducibility.py"


def module_constants(path: Path) -> dict:
    """Every module-level literal assignment in ``path``. The ones that are not literals are skipped."""
    out: dict = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and node.value is not None:
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass
    return out


def assert_bound(path: Path, func: str, needle: str) -> float | None:
    """The number an ``assert`` inside ``func`` compares ``needle`` against.

    For the bounds that are written inline rather than named -- the backend-agreement floor is
    ``assert r["exact_frac"] >= 0.99`` inside its own test, and there is no constant to read. Returns
    ``None`` rather than a guess if the shape of the assert changes, so a stale extraction shows up as
    an absent bound instead of as a wrong one.
    """
    src = Path(path).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == func):
            continue
        for a in ast.walk(node):
            if not (isinstance(a, ast.Assert) and isinstance(a.test, ast.Compare)):
                continue
            left = ast.get_source_segment(src, a.test.left) or ""
            if needle not in left:
                continue
            try:
                return float(ast.literal_eval(a.test.comparators[0]))
            except (ValueError, SyntaxError, TypeError):
                return None
    return None


def load() -> dict:
    """Every bound Step 4 applies, with the file and symbol it came from beside it."""
    conf = module_constants(CONFORMANCE_TEST)
    repro = module_constants(REPRO_TEST)
    cite_c, cite_r = "tests/gpu/test_conformance.py", "tests/gpu/test_reproducibility.py"
    return {
        "parity_interior_min_default": {"value": conf.get("INTERIOR_EXACT_MIN"),
                                        "from": f"{cite_c}:INTERIOR_EXACT_MIN"},
        "parity_all_exact_min_default": {"value": conf.get("ALL_EXACT_MIN"),
                                         "from": f"{cite_c}:ALL_EXACT_MIN"},
        "parity_mean_abs_max": {"value": conf.get("MEAN_ABS_MAX"), "from": f"{cite_c}:MEAN_ABS_MAX"},
        "backend_agreement_exact_frac_min": {
            "value": assert_bound(CONFORMANCE_TEST, "test_backend_agreement", 'r["exact_frac"]'),
            "from": f"{cite_c}:test_backend_agreement"},
        "step_agent_max_err": {
            "value": assert_bound(CONFORMANCE_TEST, "test_gpu_step_tracks_the_reference_simulator",
                                  'r["agent_max_err"]'),
            "from": f"{cite_c}:test_gpu_step_tracks_the_reference_simulator"},
        "step_block_median_err": {
            "value": assert_bound(CONFORMANCE_TEST, "test_gpu_step_tracks_the_reference_simulator",
                                  'r["block_median_err"]'),
            "from": f"{cite_c}:test_gpu_step_tracks_the_reference_simulator"},
        "downsample_exact_frac": {"value": repro.get("DOWNSAMPLE_EXACT_FRAC", {}),
                                  "from": f"{cite_r}:DOWNSAMPLE_EXACT_FRAC"},
        "downsample_margin": {"value": repro.get("DOWNSAMPLE_MARGIN"),
                              "from": f"{cite_r}:DOWNSAMPLE_MARGIN"},
        "downsample_measured_on": {"value": repro.get("MEASURED_ON"), "from": f"{cite_r}:MEASURED_ON"},
    }


if __name__ == "__main__":
    import json
    b = load()
    d = b["downsample_exact_frac"]["value"]
    print(json.dumps({k: (f"{len(v['value'])} envs" if isinstance(v["value"], dict) else v["value"])
                      for k, v in b.items()}, indent=1))
    print(f"downsample floors recorded for {len(d)} envs")
