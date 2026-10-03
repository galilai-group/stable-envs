# First five DMC pipeline closures

Acrobot, Pendulum, Cartpole, Reacher and Cheetah independently pass the complete
CPU/CUDA sweep in array 6968815 tasks 0–4. Terminal accounting resolves each raw
job ID to the application report and host. Each device has 18 exact State/task/
frame cases, 24 component measurements and 36 schedule timing cells. Six samples
per cell cover B64/256/1024, T1/8, no/final/all frames, retaining all States.

The joined tables contain 240 component rows and 180 paired pipeline rows.
CPU uses software OpenGL; CUDA uses NVIDIA graphics on RTX 3090. Component
stepping includes task packaging, IPC and transfer; it is not bare physics.
The control defers the same fused worker requests until receive, so speedup
isolates scheduling overlap. Construction, restoration and warmup are excluded.
Image MSE is zero against that library control, not a new upstream parity gate.

At B1024/T8/all frames the GPU online means range from 13,470 to 22,939 FPS.
Overlap ratios range from 1.018 to 1.335 at that configuration. CPU/GPU means,
SDs and every measured cell remain in the CSVs; source speedups are not claimed.
The other six family tasks and same-device source comparisons remain open.

Reproduce per-domain tables using summarize_dmc_pipeline.py with the frozen
family source_manifest.json. SHA256SUMS.json paths below are relative to the
parent completed directory. Prior submission evidence remains unchanged.
