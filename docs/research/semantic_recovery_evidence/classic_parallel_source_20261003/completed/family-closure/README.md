# Five completed serial/parallel source comparisons: one parent Torch thread

All five registrations independently complete array 6971034 with exit 0:0.
Terminal raw IDs and hosts match each report. Strict summaries validate both
output devices, B64/256 T1/8 no/final/all frames, exact source-parallel versus
source-serial State/task/images and output ownership, source-file hashes and
all six samples. All checked images versus our renderer have normalized MSE 0.
The joined table has 120 comparison rows from 360 six-sample timing cells.
These are bounded seed-17 online trajectories, not a new broad 1000-state corpus.

This completed campaign fixed the parent Torch renderer to one thread while
source used either one or eight worker processes, all within the same eight-core
allocation. It therefore underuses our native CPU rasterizer. The separate
CPU-thread sweep demonstrates large gains with eight parent threads and exact
outputs; a new source campaign will use that configuration. These results stay
unchanged and must not be presented as the best available CPU throughput.

Pendulum B256/T8/all: serial CPU source 273 ± 3 FPS, parallel CPU source
1,500 ± 47, ours CPU 524 ± 1. With CUDA output, the parallel CPU source plus
CUDA delivery reaches 1,402 ± 9, ours GPU 12,190 ± 20 (8.69x). Source pygame
has no native GPU renderer. No universal CPU/GPU speedup is claimed. Ours also
returns mandatory complete final State; source oracle export is outside timing.

SHA256SUMS.json paths are relative to the parent completed directory.
