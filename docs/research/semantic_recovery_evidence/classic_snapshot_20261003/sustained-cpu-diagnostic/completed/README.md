# Sustained CPU diagnostic closure

Both 6971014 tasks complete with exit 0:0 on RTX 3090 hosts (Pendulum raw6971015
on gpu2102, Acrobot raw6971014 on gpu2104). Each has eight independent-process
reports with exact full-trajectory and final-State fingerprints, six windows of
128 consecutive transitions per lane, B256 and the same normal-GC configuration.
The strict summaries retain every window and every reset duration separately.

Pendulum's T8/no-history/no-frame case improves 1.554x to 46,635 ± 192 steps/s;
Acrobot improves 1.148x to 53,608 ± 2,023. Final-frame/all-State cases range
0.993–1.019x and 0.998–1.005x respectively. No source speedup is implied.

These sustained workloads do not close every earlier isolated-call regression.
The original continuous MountainCar CPU B256/T1/no-State/final-frame cell has
a retained 0.731 ratio across six slow candidate samples; its CPU/CUDA B1024/T8/
all-State/final-frame cells also remain slower. Acrobot has retained all-State/
no-frame and other smaller-cell regressions. The original workload is being
rechecked in fresh strategy processes before enabling the shared candidate.
Production remains unchanged. No prior report or sample is removed.
