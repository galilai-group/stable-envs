"""Test-only reference tables that tie stable-envs classes to stable-worldmodel.

They map the 42 SWM ids in the frozen cell matrix to registered classes, hold the Gymnasium
ids used to build reference envs, and map SWM variation spaces onto stable-envs styles. The
installed package does not use them.
"""

from .certification import (
    CERT_GYM_IDS,
    certification_gym_id,
    has_reference_renderer,
    install_on_classes,
)
from .craftax_jit import CRAFTAX_JIT_GAP
from .crosswalk import (
    HISTORICAL_CLASS_NAMES,
    SWM_IDS,
    SYMBOLIC_IDS,
    canonical_from_frozen,
    canonical_names,
    class_for_frozen_cell,
    frozen_from_canonical,
    swm_ids_for_class,
)
from .swap_renderer import pusht_state, swap_renderer
from .variation import (
    VARIATION_MAPS,
    VariationMap,
    check_mapping,
    style_from_swm_env,
    style_from_variation,
    swm_variation_space,
    variation_from_style,
    variation_map,
)

__all__ = [
    "CERT_GYM_IDS",
    "CRAFTAX_JIT_GAP",
    "HISTORICAL_CLASS_NAMES",
    "SWM_IDS",
    "SYMBOLIC_IDS",
    "certification_gym_id",
    "has_reference_renderer",
    "install_on_classes",
    "VARIATION_MAPS",
    "VariationMap",
    "canonical_from_frozen",
    "canonical_names",
    "check_mapping",
    "class_for_frozen_cell",
    "frozen_from_canonical",
    "pusht_state",
    "style_from_swm_env",
    "style_from_variation",
    "swap_renderer",
    "swm_ids_for_class",
    "swm_variation_space",
    "variation_from_style",
    "variation_map",
]
