# Nine completed DMC pipeline registrations

Array 6968815 tasks 0–8 independently pass CPU/CUDA validation: Acrobot,
Pendulum, Cartpole, Reacher, Cheetah, Hopper, Walker, BallInCup and Finger.
The earlier five- and eight-registration closures remain unchanged. Finger
completed as raw job 6970381 on gpu2107 in 35:30 with exit 0:0. Strict summaries
verify source hashes, eight-CPU affinity, CPU/software and CUDA/NVIDIA GL,
complete State/task/frame comparisons, B64/256/1024 T1/8 and six timing samples.

The joined tables contain 432 component rows and 324 paired online rows.
All intermediate States are retained with no/final/all selected frames.
The deferred control executes the same fused requests; ratios measure overlap,
not upstream speedup. Task-step components include IPC/output transfer. Zero
frame MSE is against the accepted library control, not a fresh source gate.
Manipulator and Humanoid remain pending; Quadruped remains excluded.

SHA256SUMS.json paths are relative to the parent completed directory.
