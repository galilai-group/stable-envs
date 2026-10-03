# All twelve DMC process comparisons complete

This consolidates the ten registrations from 6961930 with independently completed
Acrobot/Pendulum retries 6964379. All 24 CPU/software-GL and CUDA/EGL gates pass:
12 cases each, exact State/task/frame parity, owned outputs, RNG/internal-reset
continuation, physical/visual variations and verified rendering overlap.

Every source hash resolves to f0716916cf285f969a1159f6904eac254a926dc6. The
strict 288-row comparison includes B64/256, T1/8, local/P4/P8 and six samples per
cell, with mean and sample SD. Each ratio is same-host, same eight-CPU allocation;
all GPU jobs used RTX 3090. P8/local ratios span 1.92–7.10x across the recorded
workloads. CPU physics is used for both output devices. These are library public
stepping comparisons, not renderer FPS, GPU physics or upstream-source speedups.

Reproduce with summarize_dmc_process.py results NEW_OUTPUT --source FROZEN_CHECKOUT
--timings. Its default now has every required domain present. The original ten
reports and failed first attempts remain unchanged in their previous folders.

Quadruped's separate legacy-versus-column exporter CUDA frame failure remains
unresolved despite its process-versus-local success. It is excluded from the
next default-routing candidate. Eleven eligible registrations still require an
unpatched public default gate before the local constructors may be activated.
