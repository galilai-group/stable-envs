# check_resolution_consistency on the bound's protocol vs the frozen corpus, all bounded configs

Bound: `tests/gpu/test_reproducibility.py::DOWNSAMPLE_EXACT_FRAC`, margin 0.005. Protocol: `H200, driver 590.48.01, torch 2.11+cu130, triton 3.6, mujoco 3.9; 8 states per env from sample_states(rng=default_rng(render_worker.STATE_SEED)), default appearance`.
Interpreters: {"se-step5": {"gpu": "NVIDIA GeForce RTX 3090", "host": "gpu2105.oscar.ccv.brown.edu", "job": "6346226"}, "se-step6": {"gpu": "NVIDIA GeForce RTX 3090", "host": "gpu2105.oscar.ccv.brown.edu", "job": "6346226"}, "se-step4-swm": {"gpu": "NVIDIA GeForce RTX 3090", "host": "gpu2105.oscar.ccv.brown.edu", "job": "6346226"}}

| config | recorded bound | floor (bound - margin) | protocol (8 states, STATE_SEED) | frozen corpus (64) | corpus first 8 | protocol - bound | corpus - bound | verdict on protocol | verdict on corpus | venv |
|---|---|---|---|---|---|---|---|---|---|---|
| AntmazeGiantNavigate | 0.708287 | 0.703287 | 0.705290 | 0.705220 | 0.705198 | -0.002997 | -0.003067 | pass | pass | se-step6 |
| AntmazeLargeExplore | 0.722699 | 0.717699 | 0.720643 | 0.720633 | 0.720668 | -0.002056 | -0.002066 | pass | pass | se-step6 |
| AntmazeMediumExplore | 0.722771 | 0.717771 | 0.721074 | 0.720991 | 0.720950 | -0.001697 | -0.001780 | pass | pass | se-step6 |
| AntmazeTeleportExplore | 0.717594 | 0.712594 | 0.715805 | 0.715892 | 0.715942 | -0.001789 | -0.001702 | pass | pass | se-step6 |
| CraftaxClassic | 1.000000 | 0.995000 | 1.000000 | 1.000000 | 1.000000 | +0.000000 | +0.000000 | pass | pass | se-step5 |
| DMCCartpoleSwingup | 0.282065 | 0.277065 | 0.277618 | 0.277461 | 0.277370 | -0.004447 | -0.004604 | pass | pass | se-step5 |
| DMCCheetahRun | 0.533850 | 0.528850 | 0.533854 | 0.530395 | 0.532207 | +0.000004 | -0.003455 | pass | pass | se-step5 |
| DMCHopperHop | 0.550220 | 0.545220 | 0.550223 | 0.552615 | 0.551357 | +0.000003 | +0.002395 | pass | pass | se-step5 |
| DMCHumanoidStand | 0.603650 | 0.598650 | 0.603653 | 0.609543 | 0.617684 | +0.000003 | +0.005893 | pass | pass | se-step5 |
| DMCWalkerWalk | 0.628950 | 0.623950 | 0.628951 | 0.618771 | 0.624043 | +0.000001 | -0.010179 | pass | FAIL | se-step5 |
| HumanoidmazeGiantNavigate | 0.712624 | 0.707624 | 0.709627 | 0.709592 | 0.709605 | -0.002997 | -0.003032 | pass | pass | se-step6 |
| HumanoidmazeLargeNavigate | 0.725895 | 0.720895 | 0.723827 | 0.723839 | 0.723860 | -0.002068 | -0.002056 | pass | pass | se-step6 |
| HumanoidmazeMediumNavigate | 0.723503 | 0.718503 | 0.721799 | 0.721773 | 0.721814 | -0.001704 | -0.001730 | pass | pass | se-step6 |
| PointmazeGiantNavigate | 0.708274 | 0.703274 | 0.705287 | 0.705225 | 0.705253 | -0.002987 | -0.003049 | pass | pass | se-step6 |
| PointmazeLargeNavigate | 0.722781 | 0.717781 | 0.720726 | 0.720705 | 0.720616 | -0.002055 | -0.002076 | pass | pass | se-step6 |
| PointmazeMediumNavigate | 0.723306 | 0.718306 | 0.721602 | 0.721455 | 0.721468 | -0.001704 | -0.001851 | pass | pass | se-step6 |
| PointmazeTeleportNavigate | 0.717602 | 0.712602 | 0.715815 | 0.716019 | 0.716062 | -0.001787 | -0.001583 | pass | pass | se-step6 |
| PushT | 0.897163 | 0.892163 | 0.897163 | 0.897546 | 0.898966 | -0.000000 | +0.000383 | pass | pass | se-step5 |
