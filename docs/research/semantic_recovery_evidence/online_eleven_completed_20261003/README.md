# Online goal: eleven of 35 comparisons validated

Adds DMC Acrobot, DMC Pendulum and SimplePointMaze to the previous eight collected
registrations. All three jobs completed with exit 0. Strict collectors reproduce
the remote CSVs byte-for-byte, validating six timing samples, source/candidate
hashes, full State/task equivalence, ownership, matched-frame MSE and same-host
strategy/device provenance. No production changes or semantic promotions.

At B64/T8/all frames, DMC Acrobot is 2.573× faster on CPU and 1.908× on GPU than
the best same-device source strategy; DMC Pendulum is 2.964× CPU and 1.997× GPU.
SimplePointMaze is 4.366× CPU and 23.371× CUDA-output; its source has no GPU
renderer, so the latter compares source CPU rendering plus CUDA delivery.
All three satisfy the per-frame normalized MSE <=0.001 gate. These are complete
online timings, with mandatory final State and no intermediate State history.
All raw means, SDs and no/final/all-frame cells are retained under `tables/`.

The [coverage CSV](coverage.csv) is a fixed 42-registration snapshot. Of the 35
accepted non-Craftax registrations: 11 have bounded CPU/CUDA rendered comparisons,
three PickAndPlace registrations have only no-frame stepping comparisons, nine
Fetch rendered comparisons failed their CPU image gate, OGBMaze failed source
P1/P2 GPU image equivalence, two OGBench source image gates remain running, and
nine DMC direct-source comparisons remain missing. Internal library controls and
offline renderer-only measurements do not populate this count.

Among the eleven measured representative all-frame rows, eight show clear mean
speedups on both compared paths; MountainCarControl is effectively tied on CPU
(1.0015×), CartPoleControl is 0.8564× CPU and MountainCarContinuousControl is
0.7988× CPU. This is not eleven universally optimized environments. Classic rows
use B256; DMC/Navigation rows use B64, always T8/all frames. Eight source renderers
have no native GPU path; the three completed DMC registrations use actual NVIDIA
source rendering and same-device comparisons. Overall goal remains incomplete.

Reproduce this CSV from the committed prior checkpoint with `reconcile.py --output
<new CSV>`. The strict collectors are `summarize_dmc_source.py` (B64) and
`summarize_navigation_source.py` (CPU threads 8), using their respective frozen
candidate/source bundles from the earlier prepared/submitted checkpoints.
