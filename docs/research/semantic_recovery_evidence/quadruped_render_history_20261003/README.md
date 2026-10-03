# Quadruped renderer-history trace pending

Job 6967610 runs a diagnostic on gpu2109, the original failing RTX 3090 host.
The frozen exporter gate and its exact assertions are unchanged. The driver
wraps the rendering boundary to record compact State/style/pristine-input hashes,
MuJoCo scene geometry/cameras/lights, kinematics and images. It groups repeated
byte-identical renderer inputs and reports which fields differ when pixels do.
Snapshots own their arrays; a local smoke verified this and camera-field change
detection. No rendering flags or production code are modified.

The original failure and its first diagnostic remain untouched in
`../quadruped_export_diagnostic_20261003/`. This run writes to a fresh scratch
directory. A reproduced assertion should leave the job FAILED with diagnostic
artifacts; that is evidence to inspect, not acceptance. Instrumentation could
affect reproducibility, and non-reproduction alone cannot close the original gate.
Quadruped automatic process selection remains disabled.

Remote stage: `/oscar/scratch/sboughan/se-semantic-recovery/quadruped-render-history-v1-20261003`.
