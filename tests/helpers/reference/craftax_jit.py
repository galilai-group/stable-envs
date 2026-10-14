"""Measured gap between the jitted stable-worldmodel Craftax wrapper and the un-jitted renderer.

Certified Craftax frames are compared against the un-jitted official renderer. A jitted
wrapper of the same renderer can differ by at most 1 uint8 because ``jax.jit`` fuses
float32 arithmetic differently (up to 1.5e-05). Test/regression data only.
"""

from __future__ import annotations

CRAFTAX_JIT_GAP: dict[str, dict] = {
    "CraftaxClassic": {
        "swm_wrapper_jits": ["reset", "step"],
        "swm_wrapper_source": "stable_worldmodel/envs/craftax/craftax.py:CraftaxWrapper.__init__ "
        "(self._reset_fn = jax.jit(_reset, ...); self._step_fn = "
        "jax.jit(_step, ...))",
        "certified_reference": "un-jitted craftax 1.6.1 (see reference_env)",
        "live_swm_reference": "jitted craftax 1.6.1, via CraftaxWrapper",
        "certified_exact_frac": 1.0,
        "certified_max_abs_diff": 0,
        "live_swm_exact_frac_7px": 0.997854,
        "live_swm_exact_frac_16px": 0.997225,
        "max_abs_diff_swm_wrapper_vs_certified": 1,
        "mechanism": "jax.jit fuses the renderer's float32 arithmetic differently; the jitted "
        "reference disagrees with its own un-jitted self by up to 1.5e-05, which a "
        "sleeping or lit row's desaturation can turn into a uint8 rounding flip",
        "note": "not hidden: this is a stated, bounded (<=1 uint8 step) discrepancy between the "
        "certified claim and what a live stable_worldmodel workflow observes, not a "
        "renderer defect here.",
    },
    "CraftaxFull": {
        "swm_wrapper_jits": ["reset", "step"],
        "swm_wrapper_source": "stable_worldmodel/envs/craftax/craftax.py:CraftaxWrapper.__init__ "
        "(self._reset_fn = jax.jit(_reset, ...); self._step_fn = "
        "jax.jit(_step, ...))",
        "certified_reference": "un-jitted craftax 1.6.1 (see reference_env)",
        "live_swm_reference": "jitted craftax 1.6.1, via CraftaxWrapper",
        "certified_exact_frac": 1.0,
        "certified_max_abs_diff": 0,
        "live_swm_exact_frac_probe_pool": "48/49 states uint8-identical jitted vs un-jitted; 1/49 "
        "differed on 0.0023% of pixels, max |diff| 1 (see parity_note above)",
        "max_abs_diff_swm_wrapper_vs_certified": 1,
        "note": "not hidden: a stated discrepancy between the certified claim and what a live "
        "stable_worldmodel workflow observes, not a renderer defect here.",
    },
}
