This directory was produced BEFORE cpu3d.QUANT existed, when the backend rounded to nearest; its
--frac-bits 0 run therefore records the ROUNDING quantiser ablation (rename from m1_local_frac0).
Regenerated below with --quant round; the texture-weight ablation is m1_local_frac0/ (frac_bits 0, trunc).
