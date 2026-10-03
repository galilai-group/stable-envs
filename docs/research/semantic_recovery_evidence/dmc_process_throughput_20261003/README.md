# RTX 3090 DMC process throughput, ten-registration closure

Array `6961930` independently completed CPU/software-GL and CUDA/EGL public
gates and B64/256, T1/8 local/P4/P8 paired measurements for ten registrations.
`comparison.csv` contains 240 rows, six samples each, mean and sample SD.
Acrobot and Pendulum failed display setup before the gates; their failures
were preserved in `../dmc_baseline_closure_20261002/process-failures/`. Retry
`6964379` is pending. They are deliberately absent from this CSV.

Source hashes in every completed report match commit
`f0716916cf285f969a1159f6904eac254a926dc6`, including its shared dispatcher.
Use that commit when validating this evidence, not the later masking candidate.
The frozen archive manifest/launcher are preserved in
`../dmc_process_20261002/v3/`. All twenty independent device gates were checked
for 12 cases, exact frames, ownership/replay/variations, actual overlap and
software-GL/NVIDIA renderer proofs. Every timing cell and six samples is present.

Reproduce with `summarize_dmc_process.py --timings --source <f0716916 checkout>`
and explicitly select the ten domain directories present here via `--domains`.
The default command requires all twelve and correctly rejects this partial cohort.

All comparisons are paired on their recorded host with the same eight-CPU
allocation; absolute rates across different hosts are not controlled task-only
comparisons. Physics is CPU for both output devices. These are public steps/s
with all States and frames disabled, not renderer FPS or upstream speedups.
The column-export acceptance array `6961949` and two missing process tasks must
close before selecting the full-family default. Higher-batch, render/pipeline
and same-device upstream comparisons remain outstanding.
